import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from src.experts.base_expert import BaseExpert

logger = logging.getLogger(__name__)

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