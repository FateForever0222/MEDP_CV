import torch
import numpy as np
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

from src.experts.base_expert import BaseExpert
from src.experts.short_chain_expert import ShortChainExpert
from src.experts.medium_chain_expert import MediumChainExpert
from src.experts.long_chain_expert import LongChainExpert
from src.gating.gating_network import GatingNetwork, TransformerGatingNetwork, create_gating_network

logger = logging.getLogger(__name__)

class DynamicRouter:
    """
    动态路由器，负责将问题路由到最合适的专家
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化动态路由器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载门控网络配置
        self.gating_config = self.config['gating']
        self.inference_config = self.config['inference']
        
        # 创建专家模型
        self.experts = {
            0: ShortChainExpert(config_path),
            1: MediumChainExpert(config_path),
            2: LongChainExpert(config_path)
        }
        self.expert_names = {
            0: "短链专家",
            1: "中链专家",
            2: "长链专家"
        }
        
        # 创建门控网络
        self.gating_network = create_gating_network(config_path)
        
        # 加载预训练的门控网络（如果存在）
        self._load_gating_network()
        
        logger.info("Initialized DynamicRouter with experts: ShortChainExpert, MediumChainExpert, LongChainExpert")
    
    def _load_gating_network(self) -> None:
        """
        加载预训练的门控网络权重（如果存在）
        """
        import os
        
        model_path = os.path.join("models", "gating_network.pt")
        if os.path.exists(model_path):
            try:
                self.gating_network.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded pre-trained gating network from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained gating network: {e}")
        else:
            logger.info("No pre-trained gating network found, using initialized weights")
    
    def route(self, question: str, options: Optional[str] = None, 
             top_k: int = 1) -> Tuple[List[BaseExpert], torch.Tensor]:
        """
        根据问题特征路由到合适的专家
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            top_k: 要选择的专家数量
            
        Returns:
            (选中的专家列表, 专家权重)元组
        """
        # 获取所有专家的特征
        features = self._get_combined_features(question, options)
        
        # 使用门控网络选择专家
        expert_indices, expert_weights = self.gating_network.select_experts(features, top_k)
        
        # 获取选中的专家
        selected_experts = [self.experts[idx] for idx in expert_indices]
        
        logger.info(f"Routed question to experts: {[self.expert_names[idx] for idx in expert_indices]}")
        return selected_experts, expert_weights
    
    def _get_combined_features(self, question: str, options: Optional[str] = None) -> torch.Tensor:
        """
        获取组合特征用于门控决策
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            组合特征张量
        """
        # 收集所有专家的特征
        features_list = []
        for expert_idx, expert in self.experts.items():
            # 获取专家特征
            expert_features = expert.get_expert_features(question, options)
            features_list.append(expert_features)
        
        # 确保所有特征长度一致
        max_length = max(f.shape[0] for f in features_list)
        padded_features = []
        
        for features in features_list:
            if features.shape[0] < max_length:
                padding = torch.zeros(max_length - features.shape[0], dtype=features.dtype)
                padded = torch.cat([features, padding])
            else:
                padded = features
            padded_features.append(padded)
        
        # 组合特征
        combined_features = torch.stack(padded_features).mean(dim=0).unsqueeze(0)
        return combined_features
    
    def adaptive_routing(self, question: str, options: Optional[str] = None) -> Tuple[List[BaseExpert], torch.Tensor]:
        """
        自适应路由，根据问题的不确定性动态调整选择的专家数量
        
        Args:
            question: 输入问题
            options: 选项（如果有）
            
        Returns:
            (选中的专家列表, 专家权重)元组
        """
        # 获取所有专家的特征
        features = self._get_combined_features(question, options)
        
        # 获取专家权重
        with torch.no_grad():
            expert_weights = self.gating_network.get_expert_weights(features).squeeze()
        
        # 计算权重熵，作为不确定性度量
        # 熵越高，权重分布越均匀，表示不确定性越高
        epsilon = 1e-10  # 防止log(0)
        entropy = -torch.sum(expert_weights * torch.log(expert_weights + epsilon))
        
        # 归一化熵，理论最大值为log(num_experts)
        max_entropy = torch.log(torch.tensor(len(self.experts), dtype=torch.float))
        normalized_entropy = entropy / max_entropy
        
        # 根据熵决定选择的专家数量
        # 熵高（不确定）时选择更多专家，熵低（确定）时选择更少专家
        if normalized_entropy > 0.8:
            top_k = 3  # 高不确定性，选择所有专家
        elif normalized_entropy > 0.5:
            top_k = 2  # 中等不确定性，选择两个专家
        else:
            top_k = 1  # 低不确定性，选择一个专家
        
        # 使用门控网络选择专家
        expert_indices, expert_weights = self.gating_network.select_experts(features, top_k)
        
        # 获取选中的专家
        selected_experts = [self.experts[idx] for idx in expert_indices]
        
        logger.info(f"Adaptively routed question (uncertainty={normalized_entropy:.2f}) to {top_k} experts: {[self.expert_names[idx] for idx in expert_indices]}")
        return selected_experts, expert_weights
    
    def update_gating_network(self, state_dict: Dict) -> None:
        """
        更新门控网络权重
        
        Args:
            state_dict: 新的权重状态字典
        """
        self.gating_network.load_state_dict(state_dict)
        logger.info("Updated gating network weights")
        
        # 保存更新后的模型
        import os
        os.makedirs("models", exist_ok=True)
        torch.save(state_dict, os.path.join("models", "gating_network.pt"))
    
    def get_all_experts(self) -> Dict[int, BaseExpert]:
        """
        获取所有专家
        
        Returns:
            专家字典
        """
        return self.experts