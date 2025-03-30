import os
import json
import yaml
import logging
import requests
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class LLMInterface:
    """
    LLM接口类，用于与本地或远程的大语言模型进行交互
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化LLM接口
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['llm']
        
        self.model_name = self.config['model_name']
        self.api_url = self.config['api_url']
        self.max_tokens = self.config['max_tokens']
        self.temperature = self.config['temperature']
        self.timeout = self.config['timeout']
        
        logger.info(f"Initialized LLM interface for {self.model_name}")
    
    def generate_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            temperature: 温度参数(可选)，如果为None则使用配置值
            
        Returns:
            生成的文本
        """
        if temperature is None:
            temperature = self.temperature
        
        try:
            response = self._call_api(prompt, temperature)
            return self._extract_text_from_response(response)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def generate_with_confidence(self, prompt: str, temperature: Optional[float] = None) -> tuple:
        """
        生成文本并返回置信度
        
        Args:
            prompt: 输入提示
            temperature: 温度参数(可选)
            
        Returns:
            (生成的文本, 置信度)元组
        """
        if temperature is None:
            temperature = self.temperature
        
        try:
            response = self._call_api(prompt, temperature)
            text = self._extract_text_from_response(response)
            confidence = self._estimate_confidence(response)
            return text, confidence
        except Exception as e:
            logger.error(f"Error generating text with confidence: {e}")
            raise
    
    def _call_api(self, prompt: str, temperature: float) -> Dict:
        """
        调用LLM API
        
        Args:
            prompt: 输入提示
            temperature: 温度参数
            
        Returns:
            API响应的JSON对象
        """
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            'prompt': prompt,
            'max_tokens': self.max_tokens,
            'temperature': temperature,
            'model': self.model_name
        }
        
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避
    
    def _extract_text_from_response(self, response: Dict) -> str:
        """
        从API响应中提取生成的文本
        
        Args:
            response: API响应的JSON对象
            
        Returns:
            生成的文本
        """
        # 根据不同API响应格式进行提取
        if 'choices' in response and len(response['choices']) > 0:
            # OpenAI/Llama格式
            if 'text' in response['choices'][0]:
                return response['choices'][0]['text'].strip()
            elif 'message' in response['choices'][0]:
                return response['choices'][0]['message']['content'].strip()
        elif 'generated_text' in response:
            # HuggingFace格式
            return response['generated_text'].strip()
        elif 'completion' in response:
            # Claude格式
            return response['completion'].strip()
        
        # 如果上述都不匹配，尝试返回整个响应的字符串形式
        logger.warning(f"Unrecognized API response format: {response}")
        return str(response)
    
    def _estimate_confidence(self, response: Dict) -> float:
        """
        估计模型的输出置信度
        
        Args:
            response: API响应的JSON对象
            
        Returns:
            估计的置信度(0.0-1.0)
        """
        # 如果API直接提供置信度或概率信息
        if 'choices' in response and len(response['choices']) > 0:
            if 'logprobs' in response['choices'][0]:
                # 使用logprobs计算平均置信度
                logprobs = response['choices'][0]['logprobs']['token_logprobs']
                if logprobs:
                    # 过滤None值
                    valid_logprobs = [lp for lp in logprobs if lp is not None]
                    if valid_logprobs:
                        # 将logprobs转换为概率再平均 (e^logprob)
                        import math
                        probs = [math.exp(min(lp, 0)) for lp in valid_logprobs]  # 防止数值溢出
                        return sum(probs) / len(probs)
        
        # 如果没有直接的置信度信息，使用自定义启发式方法
        
        # 1. 如果温度为0，假设较高置信度
        if 'temperature' in response and response['temperature'] == 0:
            return 0.9
        
        # 2. 如果使用采样且有多个候选项，根据候选项之间的差异估计置信度
        if 'choices' in response and len(response['choices']) > 1:
            # 实际实现中，可能需要比较不同选择的内容差异
            return 0.7  # 默认中等置信度
        
        # 3. 默认返回中等置信度
        return 0.8
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        # 一个简单的实现，实际应用中应替换为真实的嵌入模型调用
        embedding_url = self.api_url.replace('completions', 'embeddings')
        
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                'input': text,
                'model': f"{self.model_name}-embedding"  # 假设嵌入模型命名
            }
            
            response = requests.post(
                embedding_url,
                headers=headers,
                data=json.dumps(data),
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                return result['data'][0]['embedding']
            else:
                raise ValueError(f"Unexpected embedding response format: {result}")
        
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # 如果API调用失败，返回一个全零的示例向量
            return [0.0] * 768  # 假设768维嵌入
    
    def batch_generate(self, prompts: List[str], temperature: Optional[float] = None) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            temperature: 温度参数(可选)
            
        Returns:
            生成的文本列表
        """
        results = []
        for prompt in prompts:
            try:
                results.append(self.generate_text(prompt, temperature))
            except Exception as e:
                logger.error(f"Error in batch generation for prompt: {prompt[:50]}...: {e}")
                results.append("")
        return results

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新LLM配置
        
        Args:
            new_config: 新配置参数
        """
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
                setattr(self, key, value)
        
        logger.info(f"Updated LLM config: {new_config}")