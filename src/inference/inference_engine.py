import yaml
import logging
import re
import torch
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from src.utils.text_utils import extract_cot_and_answer, count_reasoning_steps
from src.experts.expert_models import BaseExpert
from src.llm.llm_interface import LLMInterface

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


class RetryMechanism:
    """
    简化的重试机制，只使用多专家策略处理低置信度推理结果
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化重试机制
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.inference_config = self.config['inference']
        
        # 最大重试次数
        self.max_retries = self.inference_config.get('max_retries', 1)
        
        # 创建LLM接口
        self.llm = LLMInterface(config_path)
        
        logger.info(f"Initialized simplified RetryMechanism with max_retries={self.max_retries}")
    
    def retry(self, question: str, options: Optional[str], experts: List[BaseExpert],
             initial_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        在低置信度情况下使用多专家策略重试
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            experts: 专家列表
            initial_results: 初始推理结果
            
        Returns:
            重试后的结果（如果成功），否则为None
        """
        logger.info(f"Starting retry for question: {question[:50]}...")
        
        # 收集初始答案和置信度
        initial_confidences = [result['confidence'] for result in initial_results]
        
        # 使用多专家综合重试
        retry_result = self._retry_with_more_experts(question, options, experts, initial_results)
        
        # 检查重试是否成功
        if retry_result and retry_result['confidence'] > max(initial_confidences, default=0):
            logger.info(f"Retry successful with multi-expert strategy")
            return retry_result
        
        # 重试失败，返回初始结果中置信度最高的
        if initial_results:
            max_conf_idx = initial_confidences.index(max(initial_confidences))
            logger.info("Retry failed, returning best initial result")
            return initial_results[max_conf_idx]
        
        return None
    
    def _retry_with_more_experts(self, question: str, options: Optional[str],
                                experts: List[BaseExpert], initial_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        使用多专家综合提示重试
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            experts: 专家列表
            initial_results: 初始推理结果
            
        Returns:
            重试结果（如果成功），否则为None
        """
        # 提取所有专家的思维链
        chains_of_thought = [r['chain_of_thought'] for r in initial_results]
        
        # 创建综合提示
        prompt = f"Question: {question}\n\n"
        if options:
            prompt += f"Options: {options}\n\n"
        
        # 添加各专家的思维链，但不包括答案
        for i, cot in enumerate(chains_of_thought):
            expert_type = initial_results[i]['expert_type'].replace('_', ' ').title()
            prompt += f"{expert_type} reasoning:\n{cot}\n\n"
        
        # 添加综合指导
        prompt += (
            "Now, considering all the reasoning approaches above, "
            "please provide your own step-by-step reasoning "
            "and determine the final answer. Be systematic and thorough."
        )
        
        # 使用综合提示进行最终推理
        response, confidence = self.llm.generate_with_confidence(prompt)
        
        # 提取思维链和答案
        cot, answer = self._extract_cot_and_answer(response)
        
        # 计算步骤数
        step_count = self._count_reasoning_steps(cot)
        
        return {
            'expert_type': 'combined_experts',
            'chain_of_thought': cot,
            'answer': answer,
            'confidence': confidence,
            'step_count': step_count
        }
    
    def _extract_cot_and_answer(self, response: str) -> Tuple[str, str]:
        return extract_cot_and_answer(response)
    
    def _count_reasoning_steps(self, cot: str) -> int:
        return count_reasoning_steps(cot)


class InferenceEngine:
    """
    推理引擎，整合推理流水线、置信度计算和重试机制
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化推理引擎
        
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
        
        logger.info(f"Initialized InferenceEngine with confidence threshold {self.confidence_threshold}")
    
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
            experts, expert_weights = router.route(question, options)
        
        # 使用专家进行推理
        return self.reason_with_experts(question, experts, options)
    
    def reason_with_experts(self, question: str,
                           experts: List[BaseExpert], options: Optional[str] = None) -> Dict[str, Any]:
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
        return self.retry_mechanism._extract_cot_and_answer(response)
    
    def _count_reasoning_steps(self, cot: str) -> int:
        """
        计算思维链中的推理步骤数
        
        Args:
            cot: 思维链文本
            
        Returns:
            步骤数
        """
        return self.retry_mechanism._count_reasoning_steps(cot)
    
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
        return self.confidence_calculator._calculate_consistency(answers)
    
    def calculate_detailed_confidence(self, expert_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算详细的置信度指标
        
        Args:
            expert_results: 专家结果列表
            
        Returns:
            包含各置信度指标的字典
        """
        return self.confidence_calculator.calculate_detailed_confidence(expert_results)