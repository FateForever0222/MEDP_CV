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
        self.model_path = self.llm_config.get('embedding_model_path', './model/all-MiniLM-L6-v2/')
        try:
            # 加载本地模型
            self.embedding_model = SentenceTransformer(self.model_path)
            logger.info(f"加载嵌入模型: {self.model_path}")
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
        # 改进提示语，鼓励更客观的置信度评估
        confidence_prompt = (
            f"{prompt}\n\n"
            f"After you provide your answer, please rate your confidence in your answer on a scale from 0.0 to 1.0, "
            f"where 0.0 means completely uncertain and 1.0 means absolutely certain. "
            f"Be critical in your assessment, and provide a single specific number, not a range. "
            f"Format your confidence as 'Confidence: X.X' on a new line after your answer."
        )
        
        full_response = self.generate(confidence_prompt)
        
        # 添加详细日志，记录完整响应
        # logger.debug(f"=== LLM 完整响应 ===\n{full_response}\n===================")
        
        # 尝试从响应中提取答案和置信度 - 增强版正则表达式
        confidence_patterns = [
            r"Confidence:\s*(0?\.\d+|1\.0)",  # 匹配 Confidence: 0.8
            r"Confidence:\s*\[(0?\.\d+|1\.0)\]",  # 匹配 Confidence: [0.8]
            r"Confidence:\s*\[(0?\.\d+|1\.0)-(0?\.\d+|1\.0)\]"  # 匹配 Confidence: [0.8-0.9]
        ]
        
        confidence = None
        for pattern in confidence_patterns:
            match = re.search(pattern, full_response, re.IGNORECASE)
            if match:
                if len(match.groups()) == 1:
                    # 单一值格式
                    confidence = float(match.group(1))
                    break
                elif len(match.groups()) == 2:
                    # 范围格式，取平均值
                    low = float(match.group(1))
                    high = float(match.group(2))
                    confidence = (low + high) / 2
                    break
        
        # 移除置信度部分以获取纯答案
        answer = re.sub(r"\n*Confidence:.*$", "", full_response, flags=re.IGNORECASE | re.DOTALL)
        
        # 如果没有找到置信度格式，使用默认置信度
        if confidence is None:
            confidence = 0.7  # 默认中等置信度
            logger.debug("模型未提供置信度或格式无法识别，使用默认值0.7")
        
        return answer.strip(), confidence 
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