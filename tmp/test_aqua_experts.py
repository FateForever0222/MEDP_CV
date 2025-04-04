import os
import logging
import pandas as pd
import time
import argparse
from tqdm import tqdm

from src.llm.llm_interface import LLMInterface
from src.utils.text_utils import extract_cot_and_answer, format_for_csv

# 设置日志
def setup_logging(log_file=None):
    """设置日志配置"""
    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/aqua_expert_test_{timestamp}.log"
    
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 提高第三方库日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def format_options(options):
    """将选项字典格式化为字符串"""
    if isinstance(options, str):
        # 尝试解析JSON字符串
        try:
            import json
            options_dict = json.loads(options.replace("'", "\""))
            return ", ".join([f"{k}: {v}" for k, v in options_dict.items()])
        except:
            return options
    elif isinstance(options, dict):
        return ", ".join([f"{k}: {v}" for k, v in options.items()])
    return str(options)

def test_aqua_with_experts():
    """测试AQuA数据集上的专家模式"""
    # 创建LLM接口
    llm = LLMInterface("config/config.yaml")
    
    # 加载AQuA测试集
    test_file = "data/processed/AQuA_test.csv"
    if not os.path.exists(test_file):
        logger.error(f"测试文件不存在: {test_file}")
        return
    
    df = pd.read_csv(test_file)
    logger.info(f"已加载AQuA测试集，共{len(df)}个问题")
    
    # 随机抽取5个样本
    test_samples = df.sample(5, random_state=42)
    
    # 定义专家提示模板
    expert_templates = {
        "短链专家": "Let's think this through step by step, but keep it brief.",
        "中链专家": "Let's think step by step.",
        "长链专家": "Let's analyze this in detail step by step."
    }
    
    results = []
    
    # 对每个测试样本
    for idx, row in tqdm(test_samples.iterrows(), total=len(test_samples), desc="测试进度"):
        question = row['question']
        options = row.get('options', "")
        correct_answer = row.get('answer', "")
        
        # 格式化选项
        formatted_options = format_options(options)
        
        logger.info(f"\n\n问题 #{idx}:")
        logger.info(f"问题: {question}")
        logger.info(f"选项: {formatted_options}")
        logger.info(f"正确答案: {correct_answer}")
        logger.info("-" * 80)
        
        # 对每种专家模板
        for expert_name, template in expert_templates.items():
            logger.info(f"\n使用'{expert_name}':")
            
            # 构建提示
            prompt = f"Question: {question}\n"
            if formatted_options:
                prompt += f"Options: {formatted_options}\n"
            prompt += f"{template}\n"
            
            try:
                # 调用LLM
                start_time = time.time()
                response = llm.generate(prompt)
                elapsed_time = time.time() - start_time
                
                # 提取答案
                cot, extracted_answer = extract_cot_and_answer(response, "AQuA")
                
                # 检查答案是否正确
                is_correct = extracted_answer.upper() == correct_answer.upper()
                
                # 记录结果
                logger.info(f"模型回答:\n{response}")
                logger.info(f"提取的答案: {extracted_answer}")
                logger.info(f"是否正确: {is_correct}")
                logger.info(f"耗时: {elapsed_time:.2f}秒")
                logger.info("-" * 40)
                
                # 保存结果
                results.append({
                    'question_id': idx,
                    'question': question,
                    'options': formatted_options,
                    'correct_answer': correct_answer,
                    'expert': expert_name,
                    'template': template,
                    'response': response,
                    'extracted_answer': extracted_answer,
                    'is_correct': is_correct,
                    'response_time': elapsed_time,
                    'response_length': len(response),
                    'cot_length': len(cot)
                })
                
            except Exception as e:
                logger.error(f"处理问题时出错: {e}")
        
        logger.info("=" * 80)
    
    # 保存结果到CSV
    results_df = pd.DataFrame(results)
    output_file = "results/aqua_expert_comparison.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.info(f"详细结果已保存到: {output_file}")
    
    # 简单分析
    logger.info("\n==== 结果分析 ====")
    
    # 每种专家的准确率
    expert_accuracy = results_df.groupby('expert')['is_correct'].mean()
    logger.info("\n专家准确率:")
    for expert, accuracy in expert_accuracy.items():
        logger.info(f"{expert}: {accuracy:.2%}")
    
    # 平均回答长度
    expert_length = results_df.groupby('expert')['response_length'].mean()
    logger.info("\n平均回答长度:")
    for expert, length in expert_length.items():
        logger.info(f"{expert}: {length:.1f}字符")
    
    # 平均思维链长度
    expert_cot = results_df.groupby('expert')['cot_length'].mean()
    logger.info("\n平均思维链长度:")
    for expert, length in expert_cot.items():
        logger.info(f"{expert}: {length:.1f}字符")
    
    return results_df

if __name__ == "__main__":
    # 设置日志
    logger = setup_logging()
    logger.info("开始AQuA数据集专家比较测试")
    
    # 运行测试
    test_aqua_with_experts()
    
    logger.info("测试完成!")