import yaml
import logging
import random
from typing import Dict, List, Tuple, Optional, Any

from src.experts.base_expert import BaseExpert
from src.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class RetryMechanism:
    """
    重试机制，用于处理低置信度推理结果
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
        self.max_retries = self.inference_config['max_retries']
        
        # 重试策略
        self.retry_strategies = self.inference_config['retry_strategies']
        
        # 创建LLM接口
        self.llm = LLMInterface(config_path)
        
        logger.info(f"Initialized RetryMechanism with max_retries={self.max_retries}")
    
    def retry(self, question: str, options: Optional[str], experts: List[BaseExpert],
             initial_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        在低置信度情况下尝试重试
        
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
        initial_answers = [result['answer'] for result in initial_results]
        initial_confidences = [result['confidence'] for result in initial_results]
        
        # 尝试不同的重试策略
        for retry_attempt in range(self.max_retries):
            # 选择重试策略
            strategy = self._select_strategy(retry_attempt)
            logger.info(f"Retry attempt {retry_attempt+1}/{self.max_retries} using strategy: {strategy}")
            
            # 应用重试策略
            if strategy == 'change_expert':
                # 选择不同的专家
                retry_result = self._retry_with_different_expert(question, options, experts, initial_results)
            
            elif strategy == 'adjust_prompt':
                # 调整提示
                retry_result = self._retry_with_adjusted_prompt(question, options, experts[0], initial_results)
            
            elif strategy == 'increase_experts':
                # 增加专家数量
                retry_result = self._retry_with_more_experts(question, options, experts, initial_results)
            
            else:
                logger.warning(f"Unknown retry strategy: {strategy}")
                continue
            
            # 检查重试是否成功
            if retry_result and retry_result['confidence'] > max(initial_confidences, default=0):
                logger.info(f"Retry successful with strategy {strategy}")
                return retry_result
        
        # 重试失败，返回初始结果中置信度最高的
        if initial_results:
            max_conf_idx = initial_confidences.index(max(initial_confidences))
            logger.info("Retry failed, returning best initial result")
            return initial_results[max_conf_idx]
        
        return None
    
    def _select_strategy(self, retry_attempt: int) -> str:
        """
        选择重试策略
        
        Args:
            retry_attempt: 当前重试次数
            
        Returns:
            重试策略名称
        """
        if retry_attempt < len(self.retry_strategies):
            # 按照配置中的顺序使用策略
            return self.retry_strategies[retry_attempt]
        else:
            # 随机选择策略
            return random.choice(self.retry_strategies)
    
    def _retry_with_different_expert(self, question: str, options: Optional[str],
                                     experts: List[BaseExpert], initial_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        使用不同的专家重试
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            experts: 可用专家列表
            initial_results: 初始推理结果
            
        Returns:
            重试结果（如果成功），否则为None
        """
        if len(experts) <= 1:
            # 没有其他专家可用
            return None
        
        # 找出初始结果中表现最差的专家类型
        initial_confidences = {r['expert_type']: r['confidence'] for r in initial_results}
        worst_expert_type = min(initial_confidences, key=initial_confidences.get)
        
        # 找到不同类型的专家
        available_experts = [e for e in experts if e.expert_type != worst_expert_type]
        if not available_experts:
            return None
        
        # 选择置信度最高的不同类型专家
        retry_expert = available_experts[0]
        for expert in available_experts[1:]:
            if expert.expert_type in initial_confidences and initial_confidences[expert.expert_type] > initial_confidences.get(retry_expert.expert_type, 0):
                retry_expert = expert
        
        # 使用选定的专家重试
        response, confidence = retry_expert.reason(question, options)
        
        # 提取思维链和答案
        cot, answer = self._extract_cot_and_answer(response)
        
        # 计算步骤数
        step_count = self._count_reasoning_steps(cot)
        
        return {
            'expert_type': retry_expert.expert_type,
            'chain_of_thought': cot,
            'answer': answer,
            'confidence': confidence,
            'step_count': step_count
        }
    
    def _retry_with_adjusted_prompt(self, question: str, options: Optional[str],
                                   expert: BaseExpert, initial_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        通过调整提示重试
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            expert: 专家
            initial_results: 初始推理结果
            
        Returns:
            重试结果（如果成功），否则为None
        """
        # 生成更详细的指导提示
        # 基于初始结果添加具体指导
        guidance = self._generate_guidance(question, initial_results)
        
        # 创建增强提示
        if options:
            enhanced_prompt = f"Question: {question}\n\nOptions: {options}\n\n{guidance}"
        else:
            enhanced_prompt = f"Question: {question}\n\n{guidance}"
        
        # 使用增强提示进行推理
        response, confidence = self.llm.generate_with_confidence(enhanced_prompt)
        
        # 提取思维链和答案
        cot, answer = self._extract_cot_and_answer(response)
        
        # 计算步骤数
        step_count = self._count_reasoning_steps(cot)
        
        return {
            'expert_type': expert.expert_type,
            'chain_of_thought': cot,
            'answer': answer,
            'confidence': confidence,
            'step_count': step_count
        }
    
    def _retry_with_more_experts(self, question: str, options: Optional[str],
                                experts: List[BaseExpert], initial_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        使用更多专家的综合提示重试
        
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
    
    def _generate_guidance(self, question: str, initial_results: List[Dict[str, Any]]) -> str:
        """
        基于初始结果生成指导
        
        Args:
            question: 输入问题
            initial_results: 初始推理结果
            
        Returns:
            指导文本
        """
        # 分析问题类型
        question_lower = question.lower()
        
        if any(term in question_lower for term in ['calculate', 'compute', 'solve', 'find the value']):
            # 数学问题指导
            guidance = (
                "This is a mathematical problem. Please:\n"
                "1. Identify all the given values and what you need to find.\n"
                "2. Determine the appropriate formulas or equations to use.\n"
                "3. Solve step-by-step, showing all your calculations.\n"
                "4. Verify your answer by checking units and reasonableness.\n"
                "5. Clearly state your final answer with appropriate units if applicable.\n"
            )
        
        elif any(term in question_lower for term in ['explain why', 'reason for', 'cause of']):
            # 解释型问题指导
            guidance = (
                "This question asks for an explanation. Please:\n"
                "1. Clearly identify what needs to be explained.\n"
                "2. Consider multiple possible factors or causes.\n"
                "3. Evaluate each factor with logical reasoning.\n"
                "4. Use specific evidence or principles to support your explanation.\n"
                "5. Provide a comprehensive conclusion that directly answers the question.\n"
            )
        
        elif any(term in question_lower for term in ['compare', 'contrast', 'difference', 'similarity']):
            # 比较型问题指导
            guidance = (
                "This is a comparative question. Please:\n"
                "1. Identify the entities being compared.\n"
                "2. Establish clear criteria for comparison.\n"
                "3. Systematically analyze similarities and differences for each criterion.\n"
                "4. Consider relative importance of different factors.\n"
                "5. Draw a balanced conclusion based on your comparison.\n"
            )
        
        else:
            # 通用指导
            guidance = (
                "Please approach this question methodically:\n"
                "1. Break down the problem into clear components.\n"
                "2. Consider all relevant information and variables.\n"
                "3. Use logical reasoning to analyze each component.\n"
                "4. Check for errors or inconsistencies in your reasoning.\n"
                "5. Synthesize your findings into a comprehensive answer.\n"
            )
        
        # 添加基于初始结果的具体指导
        if initial_results:
            # 提取可能的答案
            answers = [r['answer'] for r in initial_results]
            if len(set(answers)) > 1:
                # 存在不一致的答案，添加特别指导
                guidance += (
                    "\nNote: This question has proven challenging with multiple possible answers. "
                    "Please be especially thorough in your reasoning and carefully justify your conclusion.\n"
                )
        
        return guidance
    
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
        import re
        
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