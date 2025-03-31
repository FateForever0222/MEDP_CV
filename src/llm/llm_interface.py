import json
import yaml
import logging
import requests
import time
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class LLMInterface:
    """
    LLM接口类，用于与ollama进行交互
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
        self.api_url = "http://localhost:11434/api/generate"  # ollama的API
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
            return response
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
            text = self._call_api(prompt, temperature)
            # 由于ollama没有提供置信度，我们使用一个固定值
            confidence = 0.8  
            return text, confidence
        except Exception as e:
            logger.error(f"Error generating text with confidence: {e}")
            raise
    
    def _call_api(self, prompt: str, temperature: float) -> str:
        """
        调用ollama API
        
        Args:
            prompt: 输入提示
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature
        }
        
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                return result['response']
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避
    
    def get_embedding(self, text: str) -> List[float]:
        """
        使用SentenceTransformer获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        try:
            # 加载模型（首次运行会下载模型）
            from sentence_transformers import SentenceTransformer
            
            # 使用静态变量存储模型实例以避免重复加载
            if not hasattr(self, 'embedding_model'):
                # 使用与您之前相同的模型（如果您指定了特定模型，请替换此处）
                self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
            # 获取嵌入
            embedding = self.embedding_model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error in SentenceTransformer embedding: {e}")
            # 使用备选方案
            import numpy as np
            
            # 使用简单的词袋+哈希方法
            text = text.lower()
            import re
            words = re.findall(r'\w+', text)
            
            # 创建固定维度的向量
            dim = 300
            vec = np.zeros(dim)
            
            # 对每个词进行简单编码
            for word in words:
                # 使用词的哈希值来确定向量的位置和值
                h = hash(word) % dim
                vec[h] += 1
            
            # 归一化
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            return vec.tolist()
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