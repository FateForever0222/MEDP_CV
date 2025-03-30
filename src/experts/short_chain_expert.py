import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from src.experts.base_expert import BaseExpert

logger = logging.getLogger(__name__)

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
        
        # 短链适合性分数
        # 短问题、低复杂度、选择题和以往步骤少的问题都更适合短链专家
        
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