import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class ExpertRouter:
    """
    专家路由器，用于动态选择最合适的专家，优先选择单一专家以减少资源消耗
    """
    
    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [64, 32], 
                 num_experts: int = 3, model_path: str = "models/gating_network.pt"):
        """
        初始化专家路由器
        
        Args:
            input_dim: 输入特征维度，默认为10
            hidden_dims: 隐藏层维度列表
            num_experts: 专家数量，默认为3（短链、中链、长链）
            model_path: 预训练模型路径
        """
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.model_path = model_path
        
        # 创建门控网络
        self.gating_network = MLPGatingNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_experts=num_experts,
            dropout=0.2,
            temperature=1.0  # 较低的温度会使选择更加"尖锐"，偏向单一专家
        )
        
        # 加载预训练模型（如果存在）
        self._load_pretrained_model()
        
        # 专家名称映射
        self.expert_names = {
            0: "短链专家",
            1: "中链专家",
            2: "长链专家"
        }
        
        logger.info("Initialized ExpertRouter with MLP gating network")
    
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
    
    def route(self, features: torch.Tensor, force_single: bool = True) -> Tuple[List[int], torch.Tensor]:
        """
        路由到最合适的专家
        
        Args:
            features: 输入特征 [batch_size, input_dim]
            force_single: 是否强制选择单一专家
            
        Returns:
            (选中的专家索引列表, 专家权重)元组
        """
        # 获取专家权重
        with torch.no_grad():
            weights = self.gating_network(features)
        
        if force_single:
            # 强制选择单一专家
            top_expert_idx = weights.argmax(dim=-1).item()
            selected_experts = [top_expert_idx]
        else:
            # 计算熵作为不确定性度量
            epsilon = 1e-10  # 防止log(0)
            entropy = -torch.sum(weights * torch.log(weights + epsilon))
            max_entropy = torch.log(torch.tensor(self.num_experts, dtype=torch.float))
            normalized_entropy = entropy / max_entropy
            
            # 仅在非常高的不确定性时选择多个专家
            if normalized_entropy > 0.9:  # 设置非常高的阈值，使多专家选择变得罕见
                _, top_indices = torch.topk(weights, k=2, dim=-1)
                selected_experts = top_indices.squeeze().tolist()
                if not isinstance(selected_experts, list):
                    selected_experts = [selected_experts]
            else:
                top_expert_idx = weights.argmax(dim=-1).item()
                selected_experts = [top_expert_idx]
        
        expert_names = [self.expert_names[idx] for idx in selected_experts]
        logger.info(f"Routed to experts: {expert_names}")
        return selected_experts, weights
    
    def update_model(self, state_dict: Dict) -> None:
        """
        更新门控网络权重并保存
        
        Args:
            state_dict: 新的权重状态字典
        """
        self.gating_network.load_state_dict(state_dict)
        
        # 保存更新后的模型
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(state_dict, self.model_path)
        logger.info(f"Updated and saved gating network weights to {self.model_path}")


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


def extract_problem_features(question: str, options: Optional[str] = None) -> torch.Tensor:
    """
    从问题中提取特征用于路由
    
    Args:
        question: 问题文本
        options: 选项文本（如果有）
        
    Returns:
        问题特征张量
    """
    # 这里实现特征提取逻辑
    # 示例特征：问题长度、数字个数、特殊符号个数等
    
    # 1. 问题长度（标准化）
    question_length = len(question) / 100  # 标准化为0-1范围
    
    # 2. 问题中的数字数量
    num_count = sum(c.isdigit() for c in question) / 10
    
    # 3. 问题中的特殊符号数量
    symbol_count = sum(c in "+-*/^=()[]{}%$#@" for c in question) / 10
    
    # 4. 问题中的单词数
    word_count = len(question.split()) / 20
    
    # 5. 是否包含选项
    has_options = 1.0 if options else 0.0
    
    # 6. 问题类型得分（示例）
    math_score = (num_count + symbol_count) / 2  # 数学题得分
    logic_score = symbol_count  # 逻辑题得分
    common_score = (1 - math_score - symbol_count) * 0.5  # 常识题得分
    
    # 7. 估计步骤数
    # 复杂度启发式：词数+数字数+符号数
    complexity = word_count + num_count + symbol_count
    estimated_steps = min(complexity * 2, 1.0)  # 标准化到0-1
    
    # 8-10. 额外特征（可根据实际需求添加）
    extra_feature1 = 0.5  # 占位
    extra_feature2 = 0.5  # 占位
    extra_feature3 = 0.5  # 占位
    
    # 组合特征
    features = torch.tensor([
        question_length, 
        num_count,
        symbol_count,
        word_count,
        has_options,
        math_score,
        logic_score,
        common_score,
        estimated_steps,
        extra_feature1
    ], dtype=torch.float32).unsqueeze(0)  # 添加批次维度
    
    return features