import os
import argparse
import logging
import yaml
import sys
from pathlib import Path

from src.data.dataprocess import DataProcessor
from src.experts.expert_models import ShortChainExpert, MediumChainExpert, LongChainExpert
from src.gating.expert_router import ExpertRouter
from src.training.training import GRPOTrainer
from src.inference.inference_engine import InferenceEngine
import datetime

# 设置日志
def setup_logging(log_level="INFO", dataset_name=None, mode=None):
    """设置日志配置"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 使用时间戳、数据集名称和模式创建更有信息量的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建文件名
    filename_parts = ["medp_cv"]
    if mode:
        filename_parts.append(mode)
    if dataset_name:
        filename_parts.append(dataset_name)
    filename_parts.append(timestamp)
    
    # 组合文件名
    log_filename = "_".join(filename_parts) + ".log"
    log_file = log_dir / log_filename
    
    # 创建根日志记录器
    root_logger = logging.getLogger()
    # 清除所有现有处理程序
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # 设置根日志记录器级别为最低级别（这样信息会传递给所有处理程序）
    root_logger.setLevel(logging.DEBUG)
    
    # 创建文件处理程序，级别为DEBUG
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台处理程序，级别为INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理程序添加到根日志记录器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)  # 将HTTP相关日志设为WARNING级别
    logging.getLogger("httpx").setLevel(logging.WARNING)     # 将HTTP相关日志设为WARNING级别
    logging.info(f"日志配置完成，日志文件: {log_file}")
    return log_file

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MEDP-CV: 多专家动态提示生成与可信度投票")
    
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["preprocess", "train", "evaluate", "inference"],
                        help="运行模式")
    parser.add_argument("--dataset", type=str, default=None,
                        help="要处理的数据集名称")
    parser.add_argument("--question", type=str, default=None,
                        help="推理模式下的输入问题")
    parser.add_argument("--options", type=str, default=None,
                        help="推理模式下的选项（如有）")
    
    return parser.parse_args()

def preprocess_data(config_path, dataset_name=None):
    """预处理数据并构建专家库"""
    preprocessor = DataProcessor(config_path)
    
    if dataset_name:
        # 处理指定数据集
        preprocessor.preprocess_dataset(dataset_name)
        
        # 为该数据集生成专家示例
        expert_libraries = preprocessor.generate_expert_examples(dataset_name)
    else:
        # 处理所有数据集
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        for dataset_config in config['data']['datasets']:
            dataset_name = dataset_config['name']
            try:
                preprocessor.preprocess_dataset(dataset_name)
            except Exception as e:
                logging.error(f"处理数据集 {dataset_name} 时出错: {e}")
        
        # 为所有数据集生成专家示例
        expert_libraries = preprocessor.generate_expert_examples()
    
    logging.info("数据预处理和专家库构建完成")
    return expert_libraries

def train_model(config_path, dataset_name):
    """训练门控网络"""
    if not dataset_name:
        logging.error("训练模式需要指定数据集")
        return
    trainer = GRPOTrainer(config_path)
    trainer.train(dataset_name)
    logging.info(f"在数据集 {dataset_name} 上完成门控网络训练")

def evaluate_model(config_path, dataset_name):
    """评估模型性能"""
    if not dataset_name:
        logging.error("评估模式需要指定数据集")
        return
    trainer = GRPOTrainer(config_path)
    metrics = trainer.evaluate(dataset_name)
    
    logging.info(f"在数据集 {dataset_name} 上的评估指标:")
    for metric, value in metrics.items():
        if isinstance(value, dict):
            logging.info(f"  {metric}:")
            for k, v in value.items():
                logging.info(f"    {k}: {v}")
        else:
            logging.info(f"  {metric}: {value}")

def run_inference(config_path, question, options=None):
    """运行推理"""
    if not question:
        logging.error("推理模式需要输入问题")
        return
    
    # 创建专家路由器
    router = ExpertRouter(config_path)
    
    # 创建推理引擎
    engine = InferenceEngine(config_path)
    
    # 执行推理
    result = engine.reason(question, options, router=router)
    
    # 打印结果
    logging.info("推理结果:")
    logging.info(f"问题: {question}")
    if options:
        logging.info(f"选项: {options}")
    
    for i, expert_result in enumerate(result['expert_results']):
        expert_type = expert_result['expert_type'].replace('_', ' ').title()
        logging.info(f"\n专家 {i+1} ({expert_type}):")
        logging.info(f"思维链:\n{expert_result['chain_of_thought']}")
        logging.info(f"答案: {expert_result['answer']}")
        logging.info(f"置信度: {expert_result['confidence']:.4f}")
    
    logging.info(f"\n最终答案: {result['final_answer']}")
    logging.info(f"总体置信度: {result['confidence']:.4f}")
    
    if 'consistency' in result:
        logging.info(f"一致性: {result['consistency']:.4f}")
    
    # 返回结果对象以便进一步使用
    return result

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志，并获取日志文件路径
    log_file = setup_logging(
        log_level=args.log_level,
        dataset_name=args.dataset,
        mode=args.mode
    )
    
    logging.info(f"开始运行 MEDP-CV，模式: {args.mode}")
    logging.info(f"日志文件: {log_file}")
    
    # 如果指定了数据集，更新配置文件中的当前数据集
    if args.dataset:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # 设置当前数据集
            config['data']['current_dataset'] = args.dataset
            
            # 保存更新后的配置
            with open(args.config, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logging.info(f"已更新配置，当前数据集: {args.dataset}")
        except Exception as e:
            logging.error(f"更新配置失败: {e}")
    
    if args.mode == "preprocess":
        preprocess_data(args.config, args.dataset)
    
    elif args.mode == "train":
        train_model(args.config, args.dataset)
    
    elif args.mode == "evaluate":
        evaluate_model(args.config, args.dataset)
    
    elif args.mode == "inference":
        run_inference(args.config, args.question, args.options)
    
    logging.info("运行完成")

if __name__ == "__main__":
    main()