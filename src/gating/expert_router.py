import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import yaml
from typing import Dict, List, Tuple, Optional, Union, Any

from src.experts.expert_models import BaseExpert, ShortChainExpert, MediumChainExpert, LongChainExpert

logger = logging.getLogger(__name__)

class MLPGatingNetwork(nn.Module):
    """
    基于MLP的门控网络，用于专家选择
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_experts: int = 3, 
                dropout: float = 0.2, temperature: float = 1.0):
        """
        初始化门控网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_experts: 专家数量
            dropout: Dropout概率
            temperature: Softmax温度参数
        """
        super(MLPGatingNetwork, self).__init__()
        
        self.temperature = temperature
        
        # 构建多层感知机
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 输出层，为每个专家生成一个权重
        self.output_layer = nn.Linear(prev_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算每个专家的权重
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            专家权重 [batch_size, num_experts]
        """
        # 通过MLP处理特征
        hidden = self.mlp(x)
        
        # 输出每个专家的权重（尚未归一化）
        logits = self.output_layer(hidden)
        
        # 使用带温度的softmax归一化权重
        weights = F.softmax(logits / self.temperature, dim=-1)
        
        return weights


class ExpertRouter:
    """
    专家路由器，用于动态选择最合适的专家，优先选择单一专家以减少资源消耗
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化专家路由器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 门控网络配置
        self.gating_config = self.config.get('gating', {})
        self.input_dim = self.gating_config.get('input_dim', 7)  # 默认输入维度，与专家特征维度匹配
        self.hidden_dims = self.gating_config.get('hidden_dims', [64, 32])
        self.model_path = os.path.join("models", "gating_network.pt")
        self.uncertainty_threshold = self.gating_config.get('uncertainty_threshold', 0.9)
        
        # 创建专家实例
        try:
            self.experts = {
                0: ShortChainExpert(config_path),
                1: MediumChainExpert(config_path),
                2: LongChainExpert(config_path)
            }
            
            # 专家名称映射
            self.expert_names = {
                0: "短链专家",
                1: "中链专家",
                2: "长链专家"
            }
            
            logger.info(f"Successfully initialized all experts")
        except Exception as e:
            logger.error(f"Error initializing experts: {e}")
            # 创建空专家映射，避免完全失败
            self.experts = {}
            self.expert_names = {}
        
        # 创建门控网络
        self.gating_network = MLPGatingNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            num_experts=len(self.experts) if self.experts else 3,
            dropout=self.gating_config.get('dropout', 0.2),
            temperature=self.gating_config.get('temperature', 1.0)
        )
        
        # 加载预训练模型（如果存在）
        self._load_pretrained_model()
        
        logger.info(f"Initialized ExpertRouter with {len(self.experts)} experts")
    
    def _load_pretrained_model(self) -> None:
        """加载预训练的门控网络模型（如果存在）"""
        if os.path.exists(self.model_path):
            try:
                self.gating_network.load_state_dict(torch.load(self.model_path))
                logger.info(f"Loaded pre-trained gating network from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}")
                logger.info("Using initialized weights")
        else:
            logger.info("No pre-trained model found, using initialized weights")
    
    def _get_combined_features(self, question: str, options: Optional[str] = None) -> torch.Tensor:
        """
        从所有专家获取特征并组合
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            组合特征张量
        """
        features_list = []
        
        for expert_id, expert in self.experts.items():
            try:
                expert_features = expert.get_expert_features(question, options)
                features_list.append(expert_features)
            except Exception as e:
                logger.warning(f"Error getting features from expert {expert_id}: {e}")
                # 创建一个空特征向量作为替代
                empty_features = torch.zeros(self.input_dim, dtype=torch.float)
                features_list.append(empty_features)
        
        if not features_list:
            # 如果没有成功获取任何特征，返回全零特征
            return torch.zeros(1, self.input_dim, dtype=torch.float)
        
        # 计算平均特征
        combined_features = torch.stack(features_list).mean(dim=0).unsqueeze(0)
        
        return combined_features
    
    def route(self, question: str, options: Optional[str] = None, 
             force_single: bool = True) -> Tuple[List[BaseExpert], torch.Tensor]:
        """
        根据问题特征路由到合适的专家
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            force_single: 是否强制选择单一专家
            
        Returns:
            (选中的专家列表, 专家权重)元组
        """
        if not self.experts:
            logger.error("No experts available for routing")
            raise ValueError("No experts available for routing")
        
        # 获取组合特征
        try:
            features = self._get_combined_features(question, options)
        except Exception as e:
            logger.error(f"Error getting features: {e}")
            # 出错时随机选择一个专家
            expert_id = np.random.choice(list(self.experts.keys()))
            return [self.experts[expert_id]], torch.tensor([1.0, 0.0, 0.0])
        
        # 获取专家权重
        with torch.no_grad():
            weights = self.gating_network(features)
        
        if force_single:
            # 强制选择单一专家
            top_expert_idx = weights.argmax(dim=-1).item()
            selected_expert_ids = [top_expert_idx]
        else:
            # 计算熵作为不确定性度量
            epsilon = 1e-10  # 防止log(0)
            entropy = -torch.sum(weights * torch.log(weights + epsilon))
            max_entropy = torch.log(torch.tensor(len(self.experts), dtype=torch.float))
            normalized_entropy = entropy / max_entropy
            
            # 只有在非常高的不确定性时才选择多个专家
            if normalized_entropy > self.uncertainty_threshold:
                _, top_indices = torch.topk(weights, k=min(2, len(self.experts)), dim=-1)
                selected_expert_ids = top_indices.squeeze().tolist()
                if not isinstance(selected_expert_ids, list):
                    selected_expert_ids = [selected_expert_ids]
            else:
                top_expert_idx = weights.argmax(dim=-1).item()
                selected_expert_ids = [top_expert_idx]
        
        # 获取所选专家实例
        selected_experts = []
        for idx in selected_expert_ids:
            if idx in self.experts:
                selected_experts.append(self.experts[idx])
        
        if not selected_experts:
            # 如果没有成功选择任何专家，选择第一个可用的专家
            logger.warning("No experts selected, using fallback expert")
            first_expert_id = list(self.experts.keys())[0]
            selected_experts = [self.experts[first_expert_id]]
        
        expert_names = [self.expert_names.get(idx, f"专家{idx}") for idx in selected_expert_ids if idx in self.experts]
        logger.info(f"Routed question to experts: {expert_names}")
        
        return selected_experts, weights.squeeze()
    
    def reason(self, question: str, options: Optional[str] = None) -> Tuple[str, float, str]:
        """
        使用最合适的专家对问题进行推理
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            (推理结果, 置信度, 使用的专家名称)元组
        """
        if not self.experts:
            error_msg = "No experts available for reasoning"
            logger.error(error_msg)
            return error_msg, 0.0, "无专家可用"
        
        try:
            # 路由到最合适的专家
            selected_experts, weights = self.route(question, options)
            
            if len(selected_experts) == 1:
                # 单一专家模式
                expert = selected_experts[0]
                response, confidence = expert.reason(question, options)
                expert_idx = list(self.experts.keys())[list(self.experts.values()).index(expert)]
                expert_name = self.expert_names.get(expert_idx, f"专家{expert_idx}")
                
                return response, confidence, expert_name
            else:
                # 多专家模式（极少发生）
                responses = []
                confidences = []
                
                for expert in selected_experts:
                    try:
                        resp, conf = expert.reason(question, options)
                        responses.append(resp)
                        confidences.append(conf)
                    except Exception as e:
                        logger.warning(f"Error during expert reasoning: {e}")
                        responses.append("推理错误")
                        confidences.append(0.0)
                
                if not responses:
                    return "所有专家推理失败", 0.0, "推理失败"
                
                # 选择置信度最高的响应
                best_idx = np.argmax(confidences)
                expert = selected_experts[best_idx]
                expert_idx = list(self.experts.keys())[list(self.experts.values()).index(expert)]
                expert_name = self.expert_names.get(expert_idx, f"专家{expert_idx}")
                
                return responses[best_idx], confidences[best_idx], expert_name
                
        except Exception as e:
            logger.error(f"Error in reasoning process: {e}")
            # 出错时使用第一个可用的专家
            try:
                first_expert = list(self.experts.values())[0]
                response, confidence = first_expert.reason(question, options)
                expert_idx = list(self.experts.keys())[0]
                expert_name = self.expert_names.get(expert_idx, f"专家{expert_idx}")
                
                logger.info(f"Using fallback expert {expert_name} due to routing error")
                return response, confidence, expert_name
            except Exception as e2:
                logger.error(f"Fallback reasoning also failed: {e2}")
                return "推理过程出错", 0.0, "错误"
    
    def update_model(self, state_dict: Dict) -> None:
        """
        更新门控网络权重并保存
        
        Args:
            state_dict: 新的权重状态字典
        """
        try:
            self.gating_network.load_state_dict(state_dict)
            
            # 保存更新后的模型
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(state_dict, self.model_path)
            logger.info(f"Updated and saved gating network weights to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
    
    def get_all_experts(self) -> Dict[int, BaseExpert]:
        """
        获取所有专家
        
        Returns:
            专家字典
        """
        return self.experts