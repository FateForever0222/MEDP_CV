import torch
import numpy as np
import logging
import re
from typing import Dict, List, Tuple, Optional

from src.experts.base_expert import BaseExpert

logger = logging.getLogger(__name__)

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