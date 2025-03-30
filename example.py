import logging
import sys
from pathlib import Path

from src.data.data_loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.experts.short_chain_expert import ShortChainExpert
from src.experts.medium_chain_expert import MediumChainExpert
from src.experts.long_chain_expert import LongChainExpert
from src.gating.router import DynamicRouter
from src.inference.reasoning_pipeline import ReasoningPipeline
from src.training.grpo_trainer import GRPOTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 配置文件路径
CONFIG_PATH = "config/config.yaml"

def example_preprocessing():
    """数据预处理示例"""
    print("\n===== 数据预处理示例 =====")
    
    preprocessor = DataPreprocessor(CONFIG_PATH)
    
    # 预处理单个数据集
    dataset_name = "CSQA"  # 可以根据需要更改为其他数据集
    print(f"预处理数据集: {dataset_name}")
    
    processed_data = preprocessor.preprocess_dataset(dataset_name)
    print(f"处理后数据示例：\n{processed_data.head()}")
    
    # 生成专家示例库
    print("\n生成专家示例库...")
    expert_libraries = preprocessor.generate_expert_examples(dataset_name)
    
    for expert_type, library in expert_libraries.items():
        print(f"\n{expert_type} 专家库大小: {len(library)}")
        if not library.empty:
            print(f"示例：\n{library.iloc[0]['question']}")
            print(f"思维链：\n{library.iloc[0]['chain_of_thought'][:200]}...")

def example_experts_and_routing():
    """专家模型和路由示例"""
    print("\n===== 专家模型和路由示例 =====")
    
    # 创建专家模型
    short_chain_expert = ShortChainExpert(CONFIG_PATH)
    medium_chain_expert = MediumChainExpert(CONFIG_PATH)
    long_chain_expert = LongChainExpert(CONFIG_PATH)
    
    # 测试问题
    simple_question = "如果一个袋子里有3个红球和2个蓝球，随机抽取一个球，抽到红球的概率是多少？"
    medium_question = "一个盒子里有红球5个，蓝球8个，绿球4个，随机抽取2个球，求抽到的两个球颜色不同的概率。"
    complex_question = "一个箱子中有10个球，其中3个红色，4个蓝色，3个绿色。小明随机取出3个球，然后小红也随机取出3个球，最后小刚取出剩下的4个球。求小明取出的球中红球比例最大的概率。"
    
    # 使用短链专家生成提示
    print("\n短链专家生成的提示:")
    short_prompt = short_chain_expert.generate_prompt(simple_question)
    print(short_prompt[:300] + "...")
    
    # 使用中链专家生成提示
    print("\n中链专家生成的提示:")
    medium_prompt = medium_chain_expert.generate_prompt(medium_question)
    print(medium_prompt[:300] + "...")
    
    # 使用长链专家生成提示
    print("\n长链专家生成的提示:")
    long_prompt = long_chain_expert.generate_prompt(complex_question)
    print(long_prompt[:300] + "...")
    
    # 创建动态路由器
    router = DynamicRouter(CONFIG_PATH)
    
    # 测试路由
    print("\n测试动态路由:")
    
    print(f"\n简单问题: {simple_question}")
    experts, weights = router.route(simple_question)
    print(f"选中的专家: {[e.expert_type for e in experts]}")
    print(f"专家权重: {weights.squeeze().tolist()}")
    
    print(f"\n中等问题: {medium_question}")
    experts, weights = router.route(medium_question)
    print(f"选中的专家: {[e.expert_type for e in experts]}")
    print(f"专家权重: {weights.squeeze().tolist()}")
    
    print(f"\n复杂问题: {complex_question}")
    experts, weights = router.route(complex_question)
    print(f"选中的专家: {[e.expert_type for e in experts]}")
    print(f"专家权重: {weights.squeeze().tolist()}")
    
    # 测试自适应路由
    print("\n测试自适应路由:")
    
    print(f"\n简单问题的自适应路由:")
    experts, weights = router.adaptive_routing(simple_question)
    print(f"选中的专家: {[e.expert_type for e in experts]}")
    print(f"专家权重: {weights.squeeze().tolist()}")

def example_inference():
    """推理流水线示例"""
    print("\n===== 推理流水线示例 =====")
    
    # 创建推理流水线
    pipeline = ReasoningPipeline(CONFIG_PATH)
    
    # 创建动态路由器
    router = DynamicRouter(CONFIG_PATH)
    
    # 测试问题
    question = "如果一个盒子里有3个红球，4个蓝球和5个绿球，随机取出2个球，求取出的两个球都是红球的概率。"
    
    # 执行推理
    print(f"\n问题: {question}")
    
    # 获取路由结果
    experts, weights = router.adaptive_routing(question)
    print(f"选中的专家: {[e.expert_type for e in experts]}")
    
    # 使用推理流水线
    result = pipeline.reason_with_experts(question, None, experts)
    
    # 打印结果
    print("\n推理结果:")
    for i, expert_result in enumerate(result['expert_results']):
        expert_type = expert_result['expert_type'].replace('_', ' ').title()
        print(f"\n专家 {i+1} ({expert_type}):")
        print(f"思维链的前100字符:\n{expert_result['chain_of_thought'][:100]}...")
        print(f"答案: {expert_result['answer']}")
        print(f"置信度: {expert_result['confidence']:.4f}")
    
    print(f"\n最终答案: {result['final_answer']}")
    print(f"总体置信度: {result['confidence']:.4f}")

def main():
    """主函数，运行所有示例"""
    print("======= MEDP-CV 模型框架示例 =======")
    
    # 运行数据预处理示例
    try:
        example_preprocessing()
    except Exception as e:
        print(f"预处理示例失败: {e}")
    
    # 运行专家和路由示例
    try:
        example_experts_and_routing()
    except Exception as e:
        print(f"专家与路由示例失败: {e}")
    
    # 运行推理示例
    try:
        example_inference()
    except Exception as e:
        print(f"推理示例失败: {e}")
    
    print("\n所有示例完成!")

if __name__ == "__main__":
    main()