import yaml
import logging
import re
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

# 导入ollama库
import ollama

# 导入sentence_transformers用于嵌入
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class LLMInterface:
    """
    LLM接口，用于与Ollama托管的语言模型交互
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化LLM接口
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载LLM配置
        self.llm_config = self.config.get('llm', {})
        
        # 设置默认模型
        self.model_name = self.llm_config.get('model_name', 'llama2')
        
        # 生成参数
        self.gen_params = {
            'temperature': self.llm_config.get('temperature', 0.7),
            'top_p': self.llm_config.get('top_p', 0.9),
            'num_predict': self.llm_config.get('max_tokens', 2048),
            'presence_penalty': self.llm_config.get('presence_penalty', 0.0),
            'frequency_penalty': self.llm_config.get('frequency_penalty', 0.0),
        }
        
        # 初始化sentence-transformer作为嵌入模型
        self.embedding_model_name = self.llm_config.get('embedding_model', 'all-MiniLM-L6-v2')
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"加载嵌入模型: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            self.embedding_model = None
        
        logger.info(f"Initialized LLMInterface with model {self.model_name}")
    
    def generate(self, prompt: str) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 输入提示
            
        Returns:
            生成的响应
        """
        try:
            logger.debug(f"向 {self.model_name} 发送请求...")
            start_time = time.time()
            
            # 使用ollama Python库生成
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options=self.gen_params
            )
            
            text = response.get('response', '')
            
            elapsed_time = time.time() - start_time
            logger.debug(f"请求完成，耗时: {elapsed_time:.2f}秒")
            
            return text
        
        except Exception as e:
            logger.error(f"生成过程中出错: {e}")
            return f"生成过程中出错: {str(e)}"
    
    def generate_with_confidence(self, prompt: str) -> Tuple[str, float]:
        """
        生成文本响应并让模型自己估算置信度
        
        Args:
            prompt: 输入提示
            
        Returns:
            (生成的响应, 置信度)元组
        """
        # 修改提示，要求模型在回答后给出置信度
        confidence_prompt = f"{prompt}\n\nAfter you provide your answer, please rate your confidence in your answer on a scale from 0.0 to 1.0, where 0.0 means completely uncertain and 1.0 means absolutely certain. Format your confidence as 'Confidence: [0.0-1.0]' on a new line after your answer."
        
        full_response = self.generate(confidence_prompt)
        
        # 尝试从响应中提取答案和置信度
        confidence_pattern = r"Confidence:\s*(0?\.\d+|1\.0)"
        confidence_match = re.search(confidence_pattern, full_response, re.IGNORECASE)
        
        if confidence_match:
            # 提取置信度
            confidence = float(confidence_match.group(1))
            # 移除置信度部分以获取纯答案
            answer = re.sub(r"\n*Confidence:\s*0?\.\d+\s*$", "", full_response, flags=re.IGNORECASE)
        else:
            # 如果没有找到置信度格式，使用默认置信度
            answer = full_response
            confidence = 0.7  # 默认中等置信度
            logger.warning("模型未提供置信度，使用默认值0.7")
        
        return answer, confidence
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量，使用sentence-transformers
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        if self.embedding_model is None:
            logger.error("嵌入模型未加载，无法获取嵌入")
            # 返回零向量作为回退
            return [0.0] * 384  # all-MiniLM-L6-v2的默认维度
        
        try:
            # 使用sentence-transformers获取嵌入
            embedding = self.embedding_model.encode(text)
            
            # 转换为Python列表并返回
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"获取嵌入时出错: {e}")
            # 返回零向量作为回退
            return [0.0] * 384