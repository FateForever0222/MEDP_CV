
import yaml
import logging
import re
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

from src.experts.base_expert import BaseExpert
from src.llm.llm_interface import LLMInterface
from src.inference.confidence_calculator import ConfidenceCalculator
from src.inference.retry_mechanism import RetryMechanism

logger = logging.getLogger(__name__)

class ReasoningPipeline:
    """
    推理流水线，用于执行完整的推理过程
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化推理流水线
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.inference_config = self.config['inference']
        
        # 创建LLM接口
        self.llm = LLMInterface(config_path)
        
        # 创建置信度计算器
        self.confidence_calculator = ConfidenceCalculator(config_path)
        
        # 创建重试机制
        self.retry_mechanism = RetryMechanism(config_path)
        
        # 置信度阈值
        self.confidence_threshold = self.inference_config['confidence_threshold']
        
        logger.info(f"Initialized ReasoningPipeline with confidence threshold {self.confidence_threshold}")
    
    def reason(self, question: str, options: Optional[str] = None, 
               experts: Optional[List[BaseExpert]] = None,
               router=None) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            experts: 专家列表（如果为None则使用路由器动态选择）
            router: 动态路由器（如果专家为None则必须提供）
            
        Returns:
            推理结果字典
        """
        # 如果没有提供专家，使用路由器选择
        if experts is None:
            if router is None:
                raise ValueError("Either experts or router must be provided")
            experts, expert_weights = router.adaptive_routing(question, options)
        
        # 使用专家进行推理
        return self.reason_with_experts(question, options, experts)
    
    def reason_with_experts(self, question: str, options: Optional[str] = None,
                           experts: List[BaseExpert]) -> Dict[str, Any]:
        """
        使用指定的专家进行推理
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            experts: 专家列表
            
        Returns:
            推理结果字典
        """
        # 结果存储
        expert_results = []
        expert_confidences = []
        
        # 使用每个专家进行推理
        for expert in experts:
            response, confidence = expert.reason(question, options)
            
            # 提取思维链和最终答案
            cot, answer = self._extract_cot_and_answer(response)
            
            # 计算步骤数
            step_count = self._count_reasoning_steps(cot)
            
            expert_results.append({
                'expert_type': expert.expert_type,
                'chain_of_thought': cot,
                'answer': answer,
                'confidence': confidence,
                'step_count': step_count
            })
            expert_confidences.append(confidence)
        
        # 计算最终答案和置信度
        final_answer, confidence = self._determine_final_answer(expert_results)
        
        # 检查是否需要重试
        if confidence < self.confidence_threshold:
            retry_result = self.retry_mechanism.retry(
                question=question,
                options=options,
                experts=experts,
                initial_results=expert_results
            )
            
            if retry_result:
                # 使用重试结果
                final_answer = retry_result['answer']
                confidence = retry_result['confidence']
                expert_results.append(retry_result)
        
        # 构建结果
        result = {
            'question': question,
            'options': options,
            'expert_results': expert_results,
            'final_answer': final_answer,
            'confidence': confidence,
            'step_count': max([r['step_count'] for r in expert_results]) if expert_results else 0
        }
        
        # 如果是多专家投票，添加一致性指标
        if len(experts) > 1:
            answers = [r['answer'] for r in expert_results]
            result['consistency'] = self._calculate_consistency(answers)
        
        return result
    
    def _extract_cot_and_answer(self, response: str) -> Tuple[str, str]:
        """
        从响应中提取思维链和最终答案
        
        Args:
            response: LLM响应
            
        Returns:
            (思维链, 最终答案)元组
        """
        # 尝试查找"Answer:"或"Therefore,"等标记最终答案的短语
        answer_markers = ["Answer:", "Therefore,", "So the answer is", "The answer is", "Hence,", "In conclusion,"]
        
        for marker in answer_markers:
            if marker in response:
                parts = response.split(marker, 1)
                return parts[0].strip(), parts[1].strip()
        
        # 如果没有找到标记，假设最后一行是答案
        lines = response.strip().split('\n')
        if len(lines) > 1:
            return '\n'.join(lines[:-1]).strip(), lines[-1].strip()
        else:
            return "", response.strip()
    
    def _count_reasoning_steps(self, cot: str) -> int:
        """
        计算思维链中的推理步骤数
        
        Args:
            cot: 思维链文本
            
        Returns:
            步骤数
        """
        # 通过寻找步骤标记来计数
        step_markers = [
            r"Step \d+", r"\d+\.", r"\(\d+\)", 
            "First", "Second", "Third", "Fourth", "Fifth", 
            "Next", "Then", "Finally"
        ]
        
        steps = 0
        lines = cot.split('\n')
        
        for line in lines:
            if any(re.search(marker, line, re.IGNORECASE) for marker in step_markers):
                steps += 1
        
        # 如果没有找到明确的步骤标记，则按段落计数
        if steps == 0:
            # 将文本分成段落，非空段落视为一个步骤
            paragraphs = [p for p in re.split(r'\n\s*\n', cot) if p.strip()]
            steps = len(paragraphs)
        
        return max(1, steps)  # 确保至少有1个步骤
    
    def _determine_final_answer(self, expert_results: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        根据专家结果确定最终答案和置信度
        
        Args:
            expert_results: 专家结果列表
            
        Returns:
            (最终答案, 置信度)元组
        """
        if not expert_results:
            return "", 0.0
        
        if len(expert_results) == 1:
            # 单专家情况，直接使用其结果
            return expert_results[0]['answer'], expert_results[0]['confidence']
        
        # 多专家情况，进行投票
        answers = [result['answer'] for result in expert_results]
        confidences = [result['confidence'] for result in expert_results]
        
        # 计票
        vote_counter = Counter(answers)
        
        # 如果存在多数答案，选择它
        most_common = vote_counter.most_common(1)[0]
        if most_common[1] > 1:
            majority_answer = most_common[0]
            # 计算支持该答案的专家的平均置信度
            supporting_confidences = [
                conf for ans, conf in zip(answers, confidences) if ans == majority_answer
            ]
            confidence = sum(supporting_confidences) / len(supporting_confidences)
            return majority_answer, confidence
        
        # 如果没有多数答案，选择置信度最高的专家的答案
        max_conf_idx = confidences.index(max(confidences))
        return answers[max_conf_idx], confidences[max_conf_idx]
    
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
        
        # 计算最常见答案的比例
        counter = Counter(normalized_answers)
        most_common = counter.most_common(1)[0]
        consistency = most_common[1] / len(normalized_answers)
        
        return consistency