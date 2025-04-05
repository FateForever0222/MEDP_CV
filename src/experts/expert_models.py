import os
import pandas as pd
import yaml
import logging
import random
import torch
import torch.nn as nn
import numpy as np
import re
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity

from src.data.dataprocess import DataProcessor
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
            dataset_name: 当前处理的数据集名称
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
        self.data_loader = DataProcessor(config_path)
        dataset_name = self.data_config.get('current_dataset')
        # 如果没有指定数据集，使用配置文件中的第一个数据集
        if dataset_name is None and 'datasets' in self.data_config and self.data_config['datasets']:
            dataset_name = self.data_config['datasets'][0]['name']
            logger.info(f"未指定数据集，使用配置中的第一个数据集: {dataset_name}")
        
        # 尝试加载专家库
        self.examples = pd.DataFrame()  # 默认为空DataFrame
        
        if dataset_name:
            try:
                logger.info(f"尝试从 {dataset_name} 加载 {expert_type} 专家库")
                self.examples = self.data_loader.load_expert_library(dataset_name, expert_type)
                if not self.examples.empty:
                    logger.info(f"从 {dataset_name} 加载了 {len(self.examples)} 个 {expert_type} 专家示例")
                else:
                    logger.warning(f"从 {dataset_name} 加载的 {expert_type} 专家库为空")
            except Exception as e:
                logger.error(f"加载 {dataset_name} 的 {expert_type} 专家库失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"未指定数据集，无法加载专家库")
        
        # 初始化嵌入缓存
        self.embedding_cache = {}
        self.example_embeddings_cache = {}
        
        # 加载持久化嵌入缓存
        self._load_embedding_cache()
    def _load_embedding_cache(self):
        """
        从磁盘加载嵌入缓存
        """
        cache_dir = Path("cache/embeddings")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.expert_type}_embeddings.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.example_embeddings_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.example_embeddings_cache)} cached embeddings for {self.expert_type}")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self.example_embeddings_cache = {}
    
    def _save_embedding_cache(self):
        """
        将嵌入缓存保存到磁盘
        """
        cache_dir = Path("cache/embeddings")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.expert_type}_embeddings.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.example_embeddings_cache, f)
            logger.info(f"Updated embedding cache for {self.expert_type}")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
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
        检索与输入问题最相似的示例，使用缓存提高性能
        
        Args:
            question: 输入问题
            num_examples: 要检索的示例数量
            
        Returns:
            包含相似示例的DataFrame
        """
        if len(self.examples) <= num_examples:
            logger.debug(f"示例总数少于要检索的数量，返回所有示例: {len(self.examples)}")
            return self.examples  # 如果示例总数少于要检索的数量，返回所有示例
        
        # 计算或获取问题的嵌入
        question_hash = str(hash(question))
        if question_hash not in self.embedding_cache:
            self.embedding_cache[question_hash] = self._get_embedding(question)
        question_embedding = self.embedding_cache[question_hash]
        
        # 计算或获取示例的嵌入
        missing_embeddings = False
        for idx, row in self.examples.iterrows():
            if idx not in self.example_embeddings_cache:
                missing_embeddings = True
                example_q = row['question']
                example_hash = str(hash(example_q))
                if example_hash not in self.embedding_cache:
                    self.embedding_cache[example_hash] = self._get_embedding(example_q)
                self.example_embeddings_cache[idx] = self.embedding_cache[example_hash]
        
        # 如果有新计算的嵌入，保存缓存
        if missing_embeddings:
            self._save_embedding_cache()
        
        # 计算相似度
        similarities = {}
        for idx, embedding in self.example_embeddings_cache.items():
            # 简单的余弦相似度计算
            dot_product = sum(a * b for a, b in zip(question_embedding, embedding))
            norm1 = sum(a * a for a in question_embedding) ** 0.5
            norm2 = sum(b * b for b in embedding) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                similarities[idx] = 0
            else:
                similarities[idx] = dot_product / (norm1 * norm2)
        
        # 获取最相似的示例索引
        similar_indices = sorted(similarities, key=similarities.get, reverse=True)[:num_examples]
        logger.debug(f"为问题 '{question[:30]}...' 检索到的相似示例索引: {similar_indices}")
        logger.debug(f"对应相似度: {[similarities[idx] for idx in similar_indices]}")
        # 返回最相似的示例
        return self.examples.loc[similar_indices]
    
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
            prompt_template = self.expert_libraries_config.get('short_chain', {}).get(
                'prompt_template', "Let's think this through step by step, but keep it brief")
        elif self.expert_type == 'medium_chain':
            prompt_template = self.expert_libraries_config.get('medium_chain', {}).get(
                'prompt_template', "Let's think step by step")
        elif self.expert_type == 'long_chain':
            prompt_template = self.expert_libraries_config.get('long_chain', {}).get(
                'prompt_template', "Let's analyze this in detail step by step")
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
        logger.debug(f"\n{self.expert_type} 提示:\n{prompt}")
        # 使用LLM进行推理
        response, confidence = self.llm.generate_with_confidence(prompt)
        logger.debug(f"{self.expert_type} 置信度: {confidence:.4f}")
        return response, confidence
    
    def get_expert_features(self, question: str, options: Optional[str] = None) -> torch.Tensor:
        """
        获取专家特征，用于门控网络的决策
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            特征张量
        """
        # 基本特征提取
        words = question.split()
        question_length = len(words)
        normalized_length = min(1.0, question_length / 50.0)  # 归一化
        
        # 一个识别问题性质的简单特征
        is_multiple_choice = 1.0 if options else 0.0
        
        # 返回一个基础特征向量，子类应扩展这个方法
        features = [
            normalized_length,
            is_multiple_choice,
            0.0,  # 保留位，用于子类添加额外特征
            0.0,  # 保留位，用于子类添加额外特征
            0.0,  # 短链专家身份
            0.0,  # 中链专家身份
            0.0   # 长链专家身份
        ]
        
        return torch.tensor(features, dtype=torch.float)
    
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
        prompt_template = ""
        if not similar_examples.empty and 'prompt_template' in similar_examples.columns:
            prompt_template = similar_examples['prompt_template'].iloc[0]
        else:
            # 根据专家类型选择默认模板
            if self.expert_type == 'short_chain':
                prompt_template = "Let's think this through step by step, but keep it brief"
            elif self.expert_type == 'medium_chain':
                prompt_template = "Let's think step by step"
            elif self.expert_type == 'long_chain':
                prompt_template = "Let's analyze this in detail step by step"
        
        # 添加清晰的指导语
        prompt = "I'll show you some examples of how to solve similar problems. These are just examples to illustrate the reasoning process. "
        prompt += f"After the examples, I'll give you a new question to solve using the {self.expert_type.replace('_', ' ')} approach.\n\n"
        prompt += "=== EXAMPLES (FOR REFERENCE ONLY) ===\n\n"
         # 构建示例部分
        for i, (_, example) in enumerate(similar_examples.iterrows()):
            prompt += f"Example {i+1}:\n"
            prompt += f"Question: {example['question']}\n\n"
            
            if 'options' in example and pd.notna(example['options']):
                prompt += f"Options: {example['options']}\n\n"
            
            if 'chain_of_thought' in example and pd.notna(example['chain_of_thought']):
                prompt += f"{example['chain_of_thought']}\n\n"
            
            if 'generated_answer' in example and pd.notna(example['generated_answer']):
                prompt += f"Answer: {example['generated_answer']}\n\n"
            
            prompt += "-" * 50 + "\n\n"
        
        # 添加清晰的分隔和新问题标记
        prompt += "=== NOW, ANSWER THE FOLLOWING NEW QUESTION ===\n\n"
        prompt += f"Question: {question}\n\n"
        if options:
            prompt += f"Options: {options}\n\n"
        prompt += f"{prompt_template} and only after completing your reasoning, provide your final answer starting with 'Answer:'.\n"
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
        # 从基类获取基本特征
        features = super().get_expert_features(question, options).tolist()
        
        # 为短链专家提取额外特征
        words = question.split()
        
        # 简单问题的指标词
        simple_indicators = ['what', 'when', 'where', 'who', 'which', 'name', 'identify']
        simple_score = sum(1 for word in words if word.lower() in simple_indicators) / max(1, len(words))
        features[2] = min(1.0, simple_score * 2)  # 归一化并放大特征
        
        # 短问题偏好
        brevity_score = max(0.0, 1.0 - len(words) / 30.0)  # 30词以上得分为0
        features[3] = brevity_score
        
        # 修改专家身份特征
        features[4] = 1.0  # 短链专家
        logger.debug(f"{self.expert_type} 基础特征: {features}")
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
        # 从基类获取基本特征
        features = super().get_expert_features(question, options).tolist()
        
        # 为中链专家提取额外特征
        words = question.split()
        
        # 中等复杂度问题的指标词
        mid_indicators = ['explain', 'describe', 'compare', 'how', 'why', 'calculate', 'solve']
        mid_score = sum(1 for word in words if word.lower() in mid_indicators) / max(1, len(words))
        features[2] = min(1.0, mid_score * 2)  # 归一化并放大特征
        
        # 中等长度问题偏好（10-30词之间得分最高）
        word_count = len(words)
        if word_count < 10:
            length_score = word_count / 10.0
        elif word_count <= 30:
            length_score = 1.0
        else:
            length_score = max(0.0, 1.0 - (word_count - 30) / 20.0)  # 50词以上得分为0
        features[3] = length_score
        
        # 修改专家身份特征
        features[5] = 1.0  # 中链专家
        
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
        # 从基类获取基本特征
        features = super().get_expert_features(question, options).tolist()
        
        # 为长链专家提取额外特征
        words = question.split()
        
        # 复杂问题的指标词
        complex_indicators = [
            'analyze', 'evaluate', 'synthesize', 'critique', 'justify', 
            'theorize', 'deduce', 'prove', 'demonstrate', 'hypothesize'
        ]
        complex_score = sum(1 for word in words if word.lower() in complex_indicators) / max(1, len(words))
        features[2] = min(1.0, complex_score * 3)  # 归一化并放大特征
        
        # 较长问题偏好
        word_count = len(words)
        length_score = min(1.0, max(0.0, (word_count - 20) / 30.0))  # 50词及以上得分为1.0
        features[3] = length_score
        
        # 修改专家身份特征
        features[6] = 1.0  # 长链专家
        
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
        # 对于长链专家，使用基类方法但减少示例数量
        prompt = super().generate_prompt(question, options, num_examples)
        
        return prompt