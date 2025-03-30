import torch
import numpy as np
import logging
from typing import List, Union, Any

logger = logging.getLogger(__name__)

class GroupSampler:
    """
    组采样器，用于GRPO训练中的专家组合采样
    """
    
    def __init__(self, num_groups: int = 8, noise_std: float = 0.1):
        """
        初始化组采样器
        
        Args:
            num_groups: 采样组数
            noise_std: 噪声标准差
        """
        self.num_groups = num_groups
        self.noise_std = noise_std
        
        logger.info(f"Initialized GroupSampler with {num_groups} groups and noise std {noise_std}")
    
    def sample_groups(self, gating_network: Any, features: torch.Tensor) -> List[torch.Tensor]:
        """
        采样多组专家权重
        
        Args:
            gating_network: 门控网络
            features: 输入特征
            
        Returns:
            专家权重组列表
        """
        # 获取当前策略下的基准权重
        with torch.no_grad():
            base_weights = gating_network(features)
        
        # 采样多组权重
        groups = []
        
        # 首先添加基准权重（无噪声）
        groups.append(base_weights.clone())
        
        # 然后添加带噪声的权重
        for _ in range(self.num_groups - 1):
            # 生成高斯噪声
            noise = torch.randn_like(base_weights) * self.noise_std
            
            # 添加噪声到基准权重
            noisy_weights = base_weights + noise
            
            # 确保权重非负
            noisy_weights = torch.relu(noisy_weights)
            
            # 归一化权重
            normalized_weights = noisy_weights / (torch.sum(noisy_weights) + 1e-10)
            
            groups.append(normalized_weights)
        
        return groups
    
    def sample_groups_cluster(self, gating_network: Any, features: torch.Tensor, 
                              num_clusters: int = 3) -> List[torch.Tensor]:
        """
        采样多组专家权重，基于聚类的变种
        
        Args:
            gating_network: 门控网络
            features: 输入特征
            num_clusters: 聚类数
            
        Returns:
            专家权重组列表
        """
        # 获取当前策略下的基准权重
        with torch.no_grad():
            base_weights = gating_network(features)
        
        # 采样多组权重
        groups = []
        
        # 首先添加基准权重（无噪声）
        groups.append(base_weights.clone())
        
        # 为每个聚类创建代表性权重
        for cluster in range(num_clusters):
            # 创建偏向某个专家类型的权重
            cluster_weights = torch.zeros_like(base_weights)
            cluster_weights[0, cluster % base_weights.shape[1]] = 1.0
            groups.append(cluster_weights)
        
        # 添加混合权重
        remaining_groups = self.num_groups - len(groups)
        for _ in range(remaining_groups):
            # 在基准权重和聚类权重之间插值
            alpha = np.random.beta(0.5, 0.5)  # Beta分布，偏向极端值
            cluster_idx = np.random.randint(1, len(groups))
            
            mixed_weights = alpha * base_weights + (1 - alpha) * groups[cluster_idx]
            
            # 添加少量噪声
            noise = torch.randn_like(mixed_weights) * (self.noise_std / 2)
            mixed_weights = mixed_weights + noise
            
            # 确保权重非负
            mixed_weights = torch.relu(mixed_weights)
            
            # 归一化权重
            normalized_weights = mixed_weights / (torch.sum(mixed_weights) + 1e-10)
            
            groups.append(normalized_weights)
        
        return groups
    
    def update_config(self, num_groups: int = None, noise_std: float = None) -> None:
        """
        更新配置
        
        Args:
            num_groups: 新的采样组数
            noise_std: 新的噪声标准差
        """
        if num_groups is not None:
            self.num_groups = num_groups
            logger.info(f"Updated num_groups to {num_groups}")
        
        if noise_std is not None:
            self.noise_std = noise_std
            logger.info(f"Updated noise_std to {noise_std}")