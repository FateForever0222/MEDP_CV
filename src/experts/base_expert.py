import os
import pandas as pd
import yaml
import logging
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity

from src.data.data_loader import DataLoader
from src.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class BaseExpert(ABC):
    """
    专家模型基类，定义了专家模型的通用接口和功能
    """
    
    def __init__(self, expert_type: str, config_path: str = "config/config.yaml"):
        """
        初始化专家模型
        
        Args:
            expert_type: 专家类型（'short_chain', 'medium_chain', 'long_chain'）
            config_path: 配置文件路径
        """
        self.expert_type = expert_type
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.data_config = config['data']
        self.expert_config = config['experts']
        self.expert_libraries_config = self.data_config['expert_libraries']
        
        # 加载LLM接口
        self.llm = LLMInterface(config_path)
        
        # 加载数据
        self.data_loader = DataLoader(config_path)
        self.examples = self.data_loader.load_expert_library(expert_type)
        
        if self.examples.empty:
            logger.warning(f"No examples found for {expert_type} expert")
        else:
            logger.info(f"Loaded {len(self.examples)} examples for {expert_type} expert")
        
        # 初始化嵌入缓存
        self.embedding_cache = {}
    
    def generate_prompt(self, question: str, options: Optional[str] = None, 
                        num_examples: int = 3) -> str:
        """
        为给定问题生成提示
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            num_examples: 使用的示例数量
            
        Returns:
            生成的提示
        """
        if self.examples.empty:
            return self._create_zero_shot_prompt(question, options)
        
        # 检索最相似的示例
        similar_examples = self._retrieve_similar_examples(question, num_examples)
        
        # 使用检索到的示例构建few-shot提示
        prompt = self._create_few_shot_prompt(question, options, similar_examples)
        
        return prompt
    
    def _retrieve_similar_examples(self, question: str, num_examples: int) -> pd.DataFrame:
        """
        检索与输入问题最相似的示例
        
        Args:
            question: 输入问题
            num_examples: 要检索的示例数量
            
        Returns:
            包含相似示例的DataFrame
        """
        if len(self.examples) <= num_examples:
            return self.examples  # 如果示例总数少于要检索的数量，返回所有示例
        
        # 获取问题的嵌入
        if question not in self.embedding_cache:
            self.embedding_cache[question] = self._get_embedding(question)
        question_embedding = self.embedding_cache[question]
        
        # 获取每个示例的嵌入
        example_embeddings = []
        for _, row in self.examples.iterrows():
            example_q = row['question']
            if example_q not in self.embedding_cache:
                self.embedding_cache[example_q] = self._get_embedding(example_q)
            example_embeddings.append(self.embedding_cache[example_q])
        
        # 计算相似度
        similarities = cosine_similarity([question_embedding], example_embeddings)[0]
        
        # 获取最相似的示例索引
        similar_indices = np.argsort(similarities)[-num_examples:][::-1]
        
        # 返回最相似的示例
        return self.examples.iloc[similar_indices]
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        return self.llm.get_embedding(text)
    
    def _create_zero_shot_prompt(self, question: str, options: Optional[str] = None) -> str:
        """
        创建零样本提示
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            零样本提示
        """
        # 获取专家类型对应的提示模板
        if self.expert_type == 'short_chain':
            prompt_template = self.expert_libraries_config['short_chain']['prompt_template']
        elif self.expert_type == 'medium_chain':
            prompt_template = self.expert_libraries_config['medium_chain']['prompt_template']
        elif self.expert_type == 'long_chain':
            prompt_template = self.expert_libraries_config['long_chain']['prompt_template']
        else:
            prompt_template = "Let's think step by step."
        
        # 构建提示
        prompt = f"Question: {question}\n\n"
        if options:
            prompt += f"Options: {options}\n\n"
        prompt += f"{prompt_template}\n"
        
        return prompt
    
    def reason(self, question: str, options: Optional[str] = None) -> Tuple[str, float]:
        """
        对问题进行推理
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            (推理结果, 置信度)元组
        """
        # 生成提示
        prompt = self.generate_prompt(question, options)
        
        # 使用LLM进行推理
        response, confidence = self.llm.generate_with_confidence(prompt)
        
        return response, confidence
    
    @abstractmethod
    def get_expert_features(self, question: str, options: Optional[str] = None) -> torch.Tensor:
        """
        获取专家特征，用于门控网络的决策
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            特征张量
        """
        pass
        prompt += f"{prompt_template}\n"
        
        return prompt
    
    def _create_few_shot_prompt(self, question: str, options: Optional[str], 
                               similar_examples: pd.DataFrame) -> str:
        """
        使用检索到的示例创建few-shot提示
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            similar_examples: 相似示例DataFrame
            
        Returns:
            few-shot提示
        """
        prompt_template = similar_examples['prompt_template'].iloc[0]
        
        # 构建示例部分
        examples_text = ""
        for _, example in similar_examples.iterrows():
            examples_text += f"Question: {example['question']}\n\n"
            
            if 'options' in example and pd.notna(example['options']):
                examples_text += f"Options: {example['options']}\n\n"
            
            examples_text += f"{example['chain_of_thought']}\n\n"
            examples_text += f"Answer: {example['generated_answer']}\n\n"
            examples_text += "-" * 50 + "\n\n"
        
        # 构建最终提示
        prompt = examples_text
        prompt += f"Question: {question}\n\n"
        if options:
            prompt += f"Options: {options}\n\n"