import numpy as np
import yaml
import logging
import re
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class RewardModel:
    """
    奖励模型，用于计算专家推理的奖励
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化奖励模型
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.reward_config = config['training']['reward']
        
        # 奖励权重
        self.accuracy_weight = self.reward_config['accuracy_weight']
        self.confidence_weight = self.reward_config['confidence_weight']
        self.step_score_weight = self.reward_config['step_score_weight']
        
        # 是否使用非线性奖励计算
        self.use_non_linear = self.reward_config.get('use_non_linear', False)
        
        logger.info(f"Initialized RewardModel with weights: accuracy={self.accuracy_weight}, "
                   f"confidence={self.confidence_weight}, step_score={self.step_score_weight}")
    
    def calculate_reward(self, answer: str, correct_answer: Optional[str] = None, 
                        confidence: float = 0.0, step_count: int = 0,
                        expert_types: List[str] = None) -> float:
        """
        计算奖励
        
        Args:
            answer: 生成的答案
            correct_answer: 正确答案（如果有）
            confidence: 模型置信度
            step_count: 推理步骤数
            expert_types: 使用的专家类型列表
            
        Returns:
            奖励值
        """
        # 计算正确性奖励
        accuracy_score = self._calculate_accuracy_score(answer, correct_answer)
        
        # 计算置信度奖励
        confidence_score = confidence
        
        # 计算步骤合理性奖励
        step_score = self._calculate_step_score(step_count, expert_types)
        
        # 线性组合或非线性组合
        if self.use_non_linear:
            # 非线性组合 (对数形式)
            reward = np.log(
                self.accuracy_weight * accuracy_score + 
                self.confidence_weight * confidence_score * 
                self.step_score_weight * step_score + 
                1e-10  # 防止log(0)
            )
        else:
            # 线性组合
            reward = (
                self.accuracy_weight * accuracy_score +
                self.confidence_weight * confidence_score +
                self.step_score_weight * step_score
            )
        
        return float(reward)
    
    def _calculate_accuracy_score(self, answer: str, correct_answer: Optional[str]) -> float:
        """
        计算答案正确性得分
        
        Args:
            answer: 生成的答案
            correct_answer: 正确答案
            
        Returns:
            正确性得分(0或1)
        """
        if correct_answer is None:
            # 如果没有正确答案，给予中等分数
            return 0.5
        
        # 标准化答案: 删除标点符号，转换为小写
        def normalize(text):
            if not isinstance(text, str):
                text = str(text)
            return re.sub(r'[^\w\s]', '', text).lower().strip()
        
        norm_answer = normalize(answer)
        norm_correct = normalize(correct_answer)
        
        # 多选题的情况
        if len(norm_correct) <= 3 and norm_correct.isalpha():
            # 尝试从生成的答案中提取选项
            option_match = re.search(r'\b([A-Da-d])\b', answer)
            if option_match:
                extracted_option = option_match.group(1).upper()
                return 1.0 if extracted_option == norm_correct.upper() else 0.0
        
        # 检查答案是否包含正确答案或完全匹配
        if norm_answer == norm_correct or norm_correct in norm_answer:
            return 1.0
        else:
            return 0.0
    
    def _calculate_step_score(self, step_count: int, expert_types: List[str]) -> float:
        """
        计算步骤合理性得分
        
        Args:
            step_count: 推理步骤数
            expert_types: 使用的专家类型列表
            
        Returns:
            步骤合理性得分(0-1)
        """
        if not expert_types:
            return 0.5  # 默认中等分数
        
        # 检查步骤数是否与使用的专家类型一致
        scores = []
        
        for expert_type in expert_types:
            if expert_type == 'short_chain':
                # 短链专家期望1-3步
                score = max(0.0, 1.0 - abs(step_count - 2) / 3.0)
            elif expert_type == 'medium_chain':
                # 中链专家期望4-6步
                score = max(0.0, 1.0 - abs(step_count - 5) / 4.0)
            elif expert_type == 'long_chain':
                # 长链专家期望7步以上
                score = min(1.0, max(0.0, (step_count - 4) / 6.0))
            else:
                score = 0.5  # 未知专家类型
            
            scores.append(score)
        
        # 返回最高分数
        return max(scores)
    
    def update_config(self, new_config: Dict[str, float]) -> None:
        """
        更新奖励计算配置
        
        Args:
            new_config: 新配置
        """
        if 'accuracy_weight' in new_config:
            self.accuracy_weight = new_config['accuracy_weight']
        
        if 'confidence_weight' in new_config:
            self.confidence_weight = new_config['confidence_weight']
        
        if 'step_score_weight' in new_config:
            self.step_score_weight = new_config['step_score_weight']
        
        if 'use_non_linear' in new_config:
            self.use_non_linear = new_config['use_non_linear']
        
        logger.info(f"Updated reward config: {new_config}")
    
    def calculate_detailed_reward(self, answer: str, correct_answer: Optional[str] = None, 
                                 confidence: float = 0.0, step_count: int = 0,
                                 expert_types: List[str] = None) -> Dict[str, float]:
        """
        计算详细的奖励组成
        
        Args:
            answer: 生成的答案
            correct_answer: 正确答案（如果有）
            confidence: 模型置信度
            step_count: 推理步骤数
            expert_types: 使用的专家类型列表
            
        Returns:
            包含各部分奖励的字典
        """
        # 计算各部分奖励
        accuracy_score = self._calculate_accuracy_score(answer, correct_answer)
        confidence_score = confidence
        step_score = self._calculate_step_score(step_count, expert_types)
        
        # 计算总奖励
        total_reward = self.calculate_reward(
            answer, correct_answer, confidence, step_count, expert_types
        )
        
        return {
            'accuracy_score': accuracy_score,
            'confidence_score': confidence_score,
            'step_score': step_score,
            'total_reward': total_reward
        }