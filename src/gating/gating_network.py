import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Union

from src.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class GatingNetwork(nn.Module):
    """
    门控网络，用于动态选择最合适的专家
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
        super(GatingNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_experts = num_experts
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
    
    def get_expert_weights(self, features: torch.Tensor) -> torch.Tensor:
        """
        获取专家权重
        
        Args:
            features: 输入特征 [batch_size, input_dim]
            
        Returns:
            专家权重 [batch_size, num_experts]
        """
        with torch.no_grad():
            return self.forward(features)
    
    def select_experts(self, features: torch.Tensor, top_k: int = 1) -> Tuple[List[int], torch.Tensor]:
        """
        选择top_k个专家
        
        Args:
            features: 输入特征 [batch_size, input_dim]
            top_k: 要选择的专家数量
            
        Returns:
            (选中的专家索引列表, 专家权重)元组
        """
        # 获取专家权重
        weights = self.get_expert_weights(features)
        
        # 选择top_k个专家
        top_k_weights, top_k_indices = torch.topk(weights, k=min(top_k, self.num_experts), dim=-1)
        
        # 转换为列表
        selected_experts = top_k_indices.squeeze().cpu().tolist()
        if top_k == 1:
            selected_experts = [selected_experts]
        
        return selected_experts, weights


class TransformerGatingNetwork(nn.Module):
    """
    基于Transformer的门控网络，可以处理序列特征
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 3,
                nhead: int = 4, num_layers: int = 2, dropout: float = 0.2, 
                temperature: float = 1.0):
        """
        初始化Transformer门控网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_experts: 专家数量
            nhead: 多头注意力头数
            num_layers: Transformer层数
            dropout: Dropout概率
            temperature: Softmax温度参数
        """
        super(TransformerGatingNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.temperature = temperature
        
        # 输入特征线性投影到hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层，为每个专家生成一个权重
        self.output_layer = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，计算每个专家的权重
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim] 或 [batch_size, input_dim]
            mask: 注意力掩码 (可选)
            
        Returns:
            专家权重 [batch_size, num_experts]
        """
        # 处理非序列输入
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # 投影到hidden_dim
        projected = self.input_projection(x)
        
        # 通过Transformer编码器
        encoded = self.transformer_encoder(projected, src_key_padding_mask=mask)
        
        # 取最后一个编码器的输出作为特征表示
        # 可以使用平均池化或仅使用第一个token
        pooled = encoded.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 输出每个专家的权重
        logits = self.output_layer(pooled)
        
        # 使用带温度的softmax归一化权重
        weights = F.softmax(logits / self.temperature, dim=-1)
        
        return weights
    
    def get_expert_weights(self, features: torch.Tensor) -> torch.Tensor:
        """
        获取专家权重
        
        Args:
            features: 输入特征
            
        Returns:
            专家权重 [batch_size, num_experts]
        """
        with torch.no_grad():
            return self.forward(features)
    
    def select_experts(self, features: torch.Tensor, top_k: int = 1) -> Tuple[List[int], torch.Tensor]:
        """
        选择top_k个专家
        
        Args:
            features: 输入特征
            top_k: 要选择的专家数量
            
        Returns:
            (选中的专家索引列表, 专家权重)元组
        """
        # 获取专家权重
        weights = self.get_expert_weights(features)
        
        # 选择top_k个专家
        top_k_weights, top_k_indices = torch.topk(weights, k=min(top_k, self.num_experts), dim=-1)
        
        # 转换为列表
        selected_experts = top_k_indices.squeeze().cpu().tolist()
        if top_k == 1 and not isinstance(selected_experts, list):
            selected_experts = [selected_experts]
        
        return selected_experts, weights


def create_gating_network(config_path: str = "config/config.yaml") -> Union[GatingNetwork, TransformerGatingNetwork]:
    """
    根据配置创建门控网络
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        门控网络实例
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    gating_config = config['gating']
    model_type = gating_config['model_type']
    
    if model_type == 'mlp':
        return GatingNetwork(
            input_dim=7,  # 根据专家特征维度调整
            hidden_dims=gating_config['hidden_dims'],
            num_experts=3,  # 当前有3种专家
            dropout=gating_config['dropout'],
            temperature=gating_config['temperature']
        )
    elif model_type == 'transformer':
        return TransformerGatingNetwork(
            input_dim=7,  # 根据专家特征维度调整
            hidden_dim=gating_config['hidden_dims'][0],
            num_experts=3,  # 当前有3种专家
            nhead=4,
            num_layers=2,
            dropout=gating_config['dropout'],
            temperature=gating_config['temperature']
        )
    else:
        raise ValueError(f"Unknown gating model type: {model_type}")