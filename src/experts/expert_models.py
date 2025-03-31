import os
import pandas as pd
import yaml
import logging
import random
import torch
import torch.nn as nn
import numpy as np
import re
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
        
        return prompt


class ShortChainExpert(BaseExpert):
    """
    短链专家模型，擅长简单直接的推理，步骤数通常为1-3步
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化短链专家
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__('short_chain', config_path)
        
        # 短链专家的特殊初始化可以在这里添加
        logger.info("Initialized ShortChainExpert")
    
    def get_expert_features(self, question: str, options: Optional[str] = None) -> torch.Tensor:
        """
        获取专家特征，用于门控网络的决策
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            特征张量
        """
        # 为短链专家提取特征：
        # 1. 问题长度（短问题可能更适合短链推理）
        # 2. 问题复杂度（基于简单启发式）
        # 3. 专家库中相似问题的平均步骤数
        
        # 1. 问题长度特征
        words = question.split()
        question_length = len(words)
        normalized_length = min(1.0, question_length / 50.0)  # 归一化，假设50词以上为1.0
        
        # 2. 问题复杂度特征
        complexity_indicators = [
            'why', 'how', 'explain', 'analyze', 'compare', 'contrast',
            'evaluate', 'discuss', 'elaborate', 'describe', 'define'
        ]
        complexity_score = sum(1 for word in words if word.lower() in complexity_indicators)
        normalized_complexity = min(1.0, complexity_score / 5.0)  # 归一化
        
        # 检查是否为选择题
        is_multiple_choice = 1.0 if options else 0.0
        
        # 3. 相似问题的步骤数特征
        if not self.examples.empty:
            similar_examples = self._retrieve_similar_examples(question, 3)
            avg_steps = similar_examples['num_steps'].mean()
            # 短链专家偏好步骤少的问题
            steps_preference = max(0.0, 1.0 - (avg_steps - 1) / 6.0)  # 1步=1.0, 7步以上=0.0
        else:
            steps_preference = 0.5  # 没有示例时的默认值
        
        # 短链适合性特征
        features = [
            normalized_length,
            normalized_complexity,
            is_multiple_choice,
            steps_preference,
            1.0,  # 短链专家的身份特征
            0.0,  # 中链专家的身份特征
            0.0   # 长链专家的身份特征
        ]
        
        return torch.tensor(features, dtype=torch.float)


class MediumChainExpert(BaseExpert):
    """
    中链专家模型，擅长中等复杂度的推理，步骤数通常为4-6步
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化中链专家
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__('medium_chain', config_path)
        
        # 中链专家的特殊初始化可以在这里添加
        logger.info("Initialized MediumChainExpert")
    
    def get_expert_features(self, question: str, options: Optional[str] = None) -> torch.Tensor:
        """
        获取专家特征，用于门控网络的决策
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            特征张量
        """
        # 为中链专家提取特征：
        # 1. 问题长度（中等长度问题可能更适合中链推理）
        # 2. 问题复杂度（基于中等复杂度的启发式）
        # 3. 专家库中相似问题的平均步骤数
        
        # 1. 问题长度特征
        words = question.split()
        question_length = len(words)
        
        # 中等长度的问题最适合中链专家（使用高斯分布）
        # 峰值在25词左右
        normalized_length = np.exp(-0.5 * ((question_length - 25) / 15) ** 2)
        
        # 2. 问题复杂度特征
        complexity_indicators = [
            'why', 'how', 'explain', 'analyze', 'compare', 'contrast',
            'evaluate', 'discuss', 'elaborate', 'describe', 'define'
        ]
        complexity_score = sum(1 for word in words if word.lower() in complexity_indicators)
        
        # 中等复杂度的问题最适合中链专家（高斯分布，峰值在2-3个复杂词）
        normalized_complexity = np.exp(-0.5 * ((complexity_score - 2.5) / 2) ** 2)
        
        # 检查问题类型特征
        math_indicators = ['calculate', 'compute', 'solve', 'find', 'determine', 'value', 'sum', 'product']
        logic_indicators = ['if', 'then', 'either', 'or', 'not', 'and', 'follows', 'implies', 'conclusion']
        
        is_math_question = sum(1 for word in words if word.lower() in math_indicators) > 0
        is_logic_question = sum(1 for word in words if word.lower() in logic_indicators) > 0
        
        math_logic_score = 0.7 if (is_math_question or is_logic_question) else 0.3
        
        # 3. 相似问题的步骤数特征
        if not self.examples.empty:
            similar_examples = self._retrieve_similar_examples(question, 3)
            avg_steps = similar_examples['num_steps'].mean()
            
            # 中链专家偏好步骤在4-6范围的问题
            steps_preference = np.exp(-0.5 * ((avg_steps - 5) / 2) ** 2)
        else:
            steps_preference = 0.5  # 没有示例时的默认值
        
        # 中链适合性特征
        features = [
            normalized_length,
            normalized_complexity,
            math_logic_score,
            steps_preference, 
            0.0,  # 短链专家的身份特征
            1.0,  # 中链专家的身份特征
            0.0   # 长链专家的身份特征
        ]
        
        return torch.tensor(features, dtype=torch.float)


class LongChainExpert(BaseExpert):
    """
    长链专家模型，擅长复杂深入的推理，步骤数通常为7步或更多
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化长链专家
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__('long_chain', config_path)
        
        # 长链专家的特殊初始化可以在这里添加
        logger.info("Initialized LongChainExpert")
    
    def get_expert_features(self, question: str, options: Optional[str] = None) -> torch.Tensor:
        """
        获取专家特征，用于门控网络的决策
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            特征张量
        """
        # 为长链专家提取特征：
        # 1. 问题长度（长问题更适合长链推理）
        # 2. 问题复杂度（基于高复杂度的启发式）
        # 3. 专家库中相似问题的平均步骤数
        
        # 1. 问题长度特征
        words = question.split()
        question_length = len(words)
        
        # 长问题更适合长链专家（使用sigmoid函数）
        normalized_length = 1 / (1 + np.exp(-(question_length - 30) / 10))
        
        # 2. 问题复杂度特征
        # 复杂度指标：深度推理关键词、问题句数、条件数量等
        complexity_indicators = [
            'why', 'how', 'explain', 'analyze', 'compare', 'contrast',
            'evaluate', 'discuss', 'elaborate', 'describe', 'define',
            'synthesize', 'critique', 'justify', 'theorize', 'hypothesize',
            'deduce', 'derive', 'infer', 'prove', 'demonstrate'
        ]
        complexity_score = sum(1 for word in words if word.lower() in complexity_indicators)
        
        # 更多的复杂度指标更适合长链专家
        normalized_complexity = min(1.0, complexity_score / 5.0)
        
        # 计算问题中的句子数量
        sentences = re.split(r'[.!?]+', question)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # 更多的句子通常意味着更复杂的问题
        normalized_sentences = min(1.0, sentence_count / 3.0)
        
        # 计算问题中的条件数量（if, when, given that等）
        condition_indicators = ['if', 'when', 'given', 'assume', 'suppose', 'considering']
        condition_count = sum(1 for word in words if word.lower() in condition_indicators)
        normalized_conditions = min(1.0, condition_count / 2.0)
        
        # 分析问题是否需要多步推理
        multi_step_indicators = [
            'first', 'second', 'third', 'finally', 'next', 'then',
            'after', 'before', 'following', 'preceded by', 'subsequently'
        ]
        multi_step_score = sum(1 for word in words if word.lower() in multi_step_indicators)
        normalized_multi_step = min(1.0, multi_step_score / 3.0)
        
        # 3. 相似问题的步骤数特征
        if not self.examples.empty:
            similar_examples = self._retrieve_similar_examples(question, 3)
            avg_steps = similar_examples['num_steps'].mean()
            
            # 长链专家偏好步骤多的问题
            steps_preference = min(1.0, max(0.0, (avg_steps - 4) / 6.0))  # 10步及以上为1.0
        else:
            steps_preference = 0.5  # 没有示例时的默认值
        
        # 长链适合性特征
        features = [
            normalized_length,
            normalized_complexity,
            normalized_sentences,
            normalized_conditions,
            normalized_multi_step,
            steps_preference,
            0.0,  # 短链专家的身份特征
            0.0,  # 中链专家的身份特征
            1.0   # 长链专家的身份特征
        ]
        
        return torch.tensor(features, dtype=torch.float)
    
    def generate_prompt(self, question: str, options: Optional[str] = None, 
                        num_examples: int = 2) -> str:
        """
        为长链专家生成更详细的提示，可能包含更少但更相关的示例
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            num_examples: 使用的示例数量，默认为2（长链示例通常更长）
            
        Returns:
            生成的提示
        """
        # 对于长链专家，我们使用较少的示例但添加更详细的指导
        prompt = super().generate_prompt(question, options, num_examples)
        
        # 添加更详细的分析指导
        additional_guidance = (
            "Break down the problem into multiple detailed steps. "
            "Consider all relevant factors and potential approaches. "
            "Provide thorough reasoning for each step. "
            "Make sure to validate your conclusions and check for any errors or oversights. "
            "If appropriate, consider alternative solutions or perspectives. "
        )
        
        # 在问题之后添加额外指导
        prompt_parts = prompt.split("Question: " + question)
        if len(prompt_parts) == 2:
            prompt = prompt_parts[0] + "Question: " + question + "\n\n" + additional_guidance + prompt_parts[1]
        
        return prompt