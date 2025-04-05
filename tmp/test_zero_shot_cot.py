import os,sys
import json
import logging
import re
import pandas as pd
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os
from src.llm.llm_interface import LLMInterface
from src.utils.text_utils import extract_cot_and_answer

# 设置日志
def setup_logging(log_file="logs/aqua_test_results.log"):
    """设置日志配置"""
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
    
    # 提高第三方库和网络请求相关日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def format_options(options):
    """将选项字典格式化为字符串"""
    if isinstance(options, str):
        return options
    elif isinstance(options, dict):
        return ", ".join([f"{k}: {v}" for k, v in options.items()])
    return ""

def load_aqua_test_set(file_path):
    """加载AQuA测试集"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"测试集文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"已加载AQuA测试集，共{len(df)}个问题")
    return df

def run_test(llm, test_df, output_file, max_samples=None):
    """运行测试"""
    results = []
    correct_count = 0
    
    # 限制测试样本数量
    if max_samples and max_samples < len(test_df):
        test_df = test_df.sample(max_samples, random_state=42)
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="测试进度"):
        question = row['question']
        options = row['options']
        correct_answer = row['answer']
        
        # 格式化选项
        formatted_options = format_options(options)
        
        # 构建提示
        prompt = f"Question: {question}\n"
        if formatted_options:
            prompt += f"Options: {formatted_options}\n"
        prompt += "Let's think step by step."
        
        # 记录问题
        logger.info(f"问题 #{idx}:")
        logger.info(f"问题: {question}")
        logger.info(f"选项: {formatted_options}")
        logger.info(f"正确答案: {correct_answer}")
        
        try:
            # 调用LLM
            start_time = time.time()
            response = llm.generate(prompt)
            elapsed_time = time.time() - start_time
            
            # 提取答案
            cot, extracted_answer = extract_cot_and_answer(response, "AQuA")
            
            # 检查答案是否正确
            is_correct = extracted_answer.upper() == correct_answer.upper()
            if is_correct:
                correct_count += 1
            
            # 记录结果
            logger.info(f"模型回答: {response}")
            logger.info(f"提取的答案: {extracted_answer}")
            logger.info(f"是否正确: {is_correct}")
            logger.info(f"耗时: {elapsed_time:.2f}秒")
            logger.info("-" * 80)
            
            # 保存结果
            results.append({
                'question': question,
                'options': formatted_options,
                'correct_answer': correct_answer,
                'response': response,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'time': elapsed_time
            })
            
        except Exception as e:
            logger.error(f"处理问题时出错: {e}")
    
    # 计算准确率
    accuracy = correct_count / len(test_df) if len(test_df) > 0 else 0
    logger.info(f"测试完成! 准确率: {accuracy:.2%} ({correct_count}/{len(test_df)})")
    
    # 保存结果到CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    logger.info(f"详细结果已保存到: {output_file}")
    
    return accuracy, results

def main():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT测试 - AQuA数据集")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--test-file", type=str, default="data/processed/AQuA_test.csv", help="测试集文件路径")
    parser.add_argument("--output", type=str, default="results/aqua_zero_shot_cot_results.csv", help="结果输出文件")
    parser.add_argument("--max-samples", type=int, default=None, help="最大测试样本数量")
    args = parser.parse_args()
    
    # 创建LLM接口
    llm = LLMInterface(args.config)
    
    # 输出目录
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试集
    test_df = load_aqua_test_set(args.test_file)
    
    # 运行测试
    accuracy, results = run_test(llm, test_df, args.output, args.max_samples)
    
    # 输出结果摘要
    print(f"\n测试结果摘要:")
    print(f"测试样本数: {len(test_df)}")
    print(f"准确率: {accuracy:.2%}")
    print(f"详细结果已保存到: {args.output}")

if __name__ == "__main__":
    # 设置日志
    logger = setup_logging()
    logger.info("开始zero-shot-CoT测试 - AQuA数据集")
    
    # 运行主函数
    main()
    
    logger.info("测试完成!")