import yaml
import logging
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class ConfidenceCalculator:
    """
    置信度计算器，用于评估推理结果的可信度
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化置信度计算器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Initialized ConfidenceCalculator")
    
    def calculate_confidence(self, expert_results: List[Dict[str, Any]]) -> float:
        """
        计算推理结果的综合置信度
        
        Args:
            expert_results: 专家结果列表
            
        Returns:
            综合置信度(0-1)
        """
        if not expert_results:
            return 0.0
        
        if len(expert_results) == 1:
            # 单专家情况，使用LLM返回的置信度
            return expert_results[0]['confidence']
        
        # 多专家情况，考虑多种因素
        
        # 1. 专家置信度
        expert_confidences = [result['confidence'] for result in expert_results]
        avg_expert_confidence = sum(expert_confidences) / len(expert_confidences)
        
        # 2. 一致性
        answers = [result['answer'] for result in expert_results]
        consistency = self._calculate_consistency(answers)
        
        # 3. 步骤合理性
        step_scores = [self._evaluate_step_count(result['step_count'], result['expert_type']) 
                      for result in expert_results]
        avg_step_score = sum(step_scores) / len(step_scores)
        
        # 组合这些因素
        # 一致性是最重要的，其次是专家置信度，最后是步骤合理性
        combined_confidence = (
            0.5 * consistency + 
            0.3 * avg_expert_confidence + 
            0.2 * avg_step_score
        )
        
        return combined_confidence
    
    def _calculate_consistency(self, answers: List[str]) -> float:
        """
        计算答案的一致性
        
        Args:
            answers: 答案列表
            
        Returns:
            一致性分数(0-1)
        """
        if not answers or len(answers) == 1:
            return 1.0
        
        # 标准化答案
        normalized_answers = []
        for answer in answers:
            # 将答案转换为小写并删除标点符号
            norm = re.sub(r'[^\w\s]', '', answer.lower()).strip()
            normalized_answers.append(norm)
        
        # 统计最常见的答案
        from collections import Counter
        counter = Counter(normalized_answers)
        
        # 计算最常见答案的比例
        most_common = counter.most_common(1)[0]
        consistency = most_common[1] / len(normalized_answers)
        
        return consistency
    
    def _evaluate_step_count(self, step_count: int, expert_type: str) -> float:
        """
        评估步骤数是否与专家类型匹配
        
        Args:
            step_count: 步骤数
            expert_type: 专家类型
            
        Returns:
            匹配分数(0-1)
        """
        if expert_type == 'short_chain':
            # 短链专家期望1-3步
            if 1 <= step_count <= 3:
                return 1.0
            else:
                return max(0.0, 1.0 - abs(step_count - 2) / 3)
        
        elif expert_type == 'medium_chain':
            # 中链专家期望4-6步
            if 4 <= step_count <= 6:
                return 1.0
            else:
                return max(0.0, 1.0 - abs(step_count - 5) / 4)
        
        elif expert_type == 'long_chain':
            # 长链专家期望7步以上
            if step_count >= 7:
                return 1.0
            else:
                return max(0.0, (step_count - 3) / 4)
        
        else:
            return 0.5  # 未知专家类型
    
    def calculate_detailed_confidence(self, expert_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算详细的置信度指标
        
        Args:
            expert_results: 专家结果列表
            
        Returns:
            包含各置信度指标的字典
        """
        if not expert_results:
            return {'overall_confidence': 0.0}
        
        if len(expert_results) == 1:
            # 单专家情况
            expert_confidence = expert_results[0]['confidence']
            step_score = self._evaluate_step_count(
                expert_results[0]['step_count'], 
                expert_results[0]['expert_type']
            )
            
            return {
                'expert_confidence': expert_confidence,
                'step_score': step_score,
                'consistency': 1.0,
                'overall_confidence': 0.7 * expert_confidence + 0.3 * step_score
            }
        
        # 多专家情况
        
        # 1. 专家置信度
        expert_confidences = [result['confidence'] for result in expert_results]
        avg_expert_confidence = sum(expert_confidences) / len(expert_confidences)
        
        # 2. 一致性
        answers = [result['answer'] for result in expert_results]
        consistency = self._calculate_consistency(answers)
        
        # 3. 步骤合理性
        step_scores = [self._evaluate_step_count(result['step_count'], result['expert_type']) 
                      for result in expert_results]
        avg_step_score = sum(step_scores) / len(step_scores)
        
        # 计算综合置信度
        combined_confidence = (
            0.5 * consistency + 
            0.3 * avg_expert_confidence + 
            0.2 * avg_step_score
        )
        
        return {
            'expert_confidence': avg_expert_confidence,
            'step_score': avg_step_score,
            'consistency': consistency,
            'overall_confidence': combined_confidence
        }