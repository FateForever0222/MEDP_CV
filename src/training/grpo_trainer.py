import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
import logging
import random
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
from tqdm import tqdm

from src.gating.gating_network import GatingNetwork, TransformerGatingNetwork, create_gating_network
from src.gating.router import DynamicRouter
from src.training.reward_model import RewardModel
from src.training.group_sampling import GroupSampler
from src.data.data_loader import DataLoader
from src.inference.reasoning_pipeline import ReasoningPipeline

logger = logging.getLogger(__name__)

class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) 训练器
    用于优化门控网络的策略
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化GRPO训练器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载训练配置
        self.training_config = self.config['training']['grpo']
        self.reward_config = self.config['training']['reward']
        
        # 创建门控网络
        self.gating_network = create_gating_network(config_path)
        
        # 创建动态路由器
        self.router = DynamicRouter(config_path)
        
        # 替换路由器中的门控网络，确保一致性
        self.router.gating_network = self.gating_network
        
        # 创建奖励模型
        self.reward_model = RewardModel(config_path)
        
        # 创建组采样器
        self.group_sampler = GroupSampler(
            num_groups=self.training_config['num_groups'],
            noise_std=self.training_config['noise_std']
        )
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.gating_network.parameters(),
            lr=self.training_config['learning_rate']
        )
        
        # 创建推理流水线
        self.pipeline = ReasoningPipeline(config_path)
        
        # 数据加载器
        self.data_loader = DataLoader(config_path)
        
        # 训练状态
        self.best_accuracy = 0.0
        self.best_reward = 0.0
        self.patience_counter = 0
        
        logger.info("Initialized GRPO Trainer")
    
    def train(self, dataset_name: str, num_epochs: Optional[int] = None) -> None:
        """
        训练门控网络
        
        Args:
            dataset_name: 数据集名称
            num_epochs: 训练轮数（如果为None则使用配置中的值）
        """
        if num_epochs is None:
            num_epochs = self.training_config['max_epochs']
        
        # 加载训练数据
        try:
            train_data = self.data_loader.load_processed_data(f"{dataset_name}_train")
            logger.info(f"Loaded {len(train_data)} training examples from {dataset_name}")
        except FileNotFoundError:
            logger.error(f"Training data for {dataset_name} not found")
            return
        
        # 创建保存模型的目录
        os.makedirs("models", exist_ok=True)
        
        # 开始训练
        logger.info(f"Starting GRPO training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 打乱数据
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            
            # 限制每个epoch的样本数，以加快训练
            max_samples = min(500, len(train_data))  # 每个epoch最多使用500个样本
            epoch_data = train_data.iloc[:max_samples]
            
            total_loss = 0.0
            total_reward = 0.0
            correct_predictions = 0
            
            # 使用tqdm显示进度条
            for idx, row in tqdm(epoch_data.iterrows(), total=len(epoch_data), desc=f"Epoch {epoch+1}/{num_epochs}"):
                question = row['question']
                options = row['options'] if 'options' in row and pd.notna(row['options']) else None
                correct_answer = row['answer'] if 'answer' in row else None
                
                if not question or pd.isna(question):
                    continue
                
                # 训练单个样本
                loss, reward, is_correct = self._train_single_sample(question, options, correct_answer)
                
                total_loss += loss
                total_reward += reward
                if is_correct:
                    correct_predictions += 1
            
            # 计算epoch指标
            avg_loss = total_loss / len(epoch_data)
            avg_reward = total_reward / len(epoch_data)
            accuracy = correct_predictions / len(epoch_data)
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, "
                       f"Accuracy: {accuracy:.4f}, Time: {epoch_time:.2f}s")
            
            # 检查是否需要保存模型
            if accuracy > self.best_accuracy:
                logger.info(f"New best accuracy: {accuracy:.4f} (previous: {self.best_accuracy:.4f})")
                self.best_accuracy = accuracy
                torch.save(self.gating_network.state_dict(), os.path.join("models", "best_gating_network.pt"))
                self.patience_counter = 0
            elif avg_reward > self.best_reward:
                logger.info(f"New best reward: {avg_reward:.4f} (previous: {self.best_reward:.4f})")
                self.best_reward = avg_reward
                torch.save(self.gating_network.state_dict(), os.path.join("models", "best_reward_gating_network.pt"))
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # 早停
            if self.patience_counter >= self.training_config['early_stopping_patience']:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # 训练结束，加载最佳模型
        try:
            best_model_path = os.path.join("models", "best_gating_network.pt")
            self.gating_network.load_state_dict(torch.load(best_model_path))
            self.router.update_gating_network(self.gating_network.state_dict())
            logger.info(f"Loaded best model with accuracy {self.best_accuracy:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load best model: {e}")
    
    def _train_single_sample(self, question: str, options: Optional[str], 
                            correct_answer: Optional[str]) -> Tuple[float, float, bool]:
        """
        训练单个样本
        
        Args:
            question: 问题
            options: 选项（如果有）
            correct_answer: 正确答案（如果有）
            
        Returns:
            (损失, 奖励, 是否正确)元组
        """
        # 获取问题特征
        features = self.router._get_combined_features(question, options)
        
        # 使用组采样生成多组专家权重
        expert_groups = self.group_sampler.sample_groups(self.gating_network, features)
        
        # 计算每组的奖励
        group_rewards = []
        group_results = []
        best_result = None
        is_correct = False
        
        for group_idx, group_weights in enumerate(expert_groups):
            # 根据权重选择专家
            selected_experts = []
            for expert_idx, weight in enumerate(group_weights.squeeze().tolist()):
                if weight > 0.1:  # 仅选择权重大于阈值的专家
                    selected_experts.append(self.router.experts[expert_idx])
            
            # 确保至少选择一个专家
            if not selected_experts:
                max_idx = torch.argmax(group_weights).item()
                selected_experts = [self.router.experts[max_idx]]
            
            # 使用选定的专家进行推理
            try:
                result = self.pipeline.reason_with_experts(question, options, selected_experts)
                answer = result['final_answer']
                confidence = result['confidence']
                step_count = result.get('step_count', 0)
                
                # 计算该组的奖励
                reward = self.reward_model.calculate_reward(
                    answer=answer,
                    correct_answer=correct_answer,
                    confidence=confidence,
                    step_count=step_count,
                    expert_types=[expert.expert_type for expert in selected_experts]
                )
                
                group_rewards.append(reward)
                group_results.append(result)
                
                # 检查是否为最佳结果
                if best_result is None or reward > max(group_rewards[:-1], default=0):
                    best_result = result
                    is_correct = result.get('is_correct', False)
            
            except Exception as e:
                logger.error(f"Error reasoning with experts for group {group_idx}: {e}")
                group_rewards.append(0.0)
                group_results.append(None)
        
        # 计算每组的相对优势
        advantages = self._calculate_advantages(group_rewards)
        
        # 更新门控网络
        loss = self._update_policy(features, expert_groups, advantages)
        
        # 返回平均损失、平均奖励和是否正确
        return loss, sum(group_rewards) / len(group_rewards), is_correct
    
    def _calculate_advantages(self, rewards: List[float]) -> List[float]:
        """
        计算每组的相对优势
        
        Args:
            rewards: 每组的奖励列表
            
        Returns:
            优势列表
        """
        rewards = np.array(rewards)
        
        # 如果所有奖励相同，返回零优势
        if np.all(rewards == rewards[0]):
            return [0.0] * len(rewards)
        
        # 计算均值和标准差
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # 避免除以零
        if std_reward < 1e-8:
            std_reward = 1.0
        
        # 计算标准化的优势
        advantages = (rewards - mean_reward) / std_reward
        
        return advantages.tolist()
    
    def _update_policy(self, features: torch.Tensor, expert_groups: List[torch.Tensor], 
                      advantages: List[float]) -> float:
        """
        使用PPO更新策略
        
        Args:
            features: 输入特征
            expert_groups: 专家权重组列表
            advantages: 每组的优势列表
            
        Returns:
            损失值
        """
        self.optimizer.zero_grad()
        
        # 当前策略下的专家权重
        current_weights = self.gating_network(features)
        
        # 计算策略损失
        policy_loss = 0.0
        kl_loss = 0.0
        
        for group_weights, advantage in zip(expert_groups, advantages):
            if advantage == 0.0:
                continue
                
            # 计算比率
            ratio = torch.sum(current_weights * group_weights) / torch.sum(group_weights * group_weights)
            
            # 策略梯度损失
            policy_term = ratio * advantage
            policy_loss -= policy_term
            
            # KL散度正则化
            kl_div = F.kl_div(
                F.log_softmax(current_weights, dim=-1),
                F.softmax(group_weights, dim=-1),
                reduction='sum'
            )
            kl_loss += kl_div
        
        # 总损失
        kl_coef = self.training_config['kl_coef']
        total_loss = policy_loss + kl_coef * kl_loss
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.gating_network.parameters(), 1.0)
        
        # 更新参数
        self.optimizer.step()
        
        return total_loss.item()
    
    def evaluate(self, dataset_name: str) -> Dict[str, float]:
        """
        评估门控网络
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            包含评估指标的字典
        """
        # 加载测试数据
        try:
            test_data = self.data_loader.load_processed_data(f"{dataset_name}_test")
            logger.info(f"Loaded {len(test_data)} test examples from {dataset_name}")
        except FileNotFoundError:
            logger.error(f"Test data for {dataset_name} not found")
            return {}
        
        # 评估指标
        metrics = {
            'accuracy': 0.0,
            'avg_reward': 0.0,
            'avg_confidence': 0.0,
            'expert_usage': defaultdict(int)
        }
        
        # 限制评估样本数，以加快评估
        max_samples = min(200, len(test_data))  # 最多评估200个样本
        eval_data = test_data.iloc[:max_samples]
        
        correct = 0
        total_reward = 0.0
        total_confidence = 0.0
        
        # 使用tqdm显示进度条
        for idx, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc="Evaluating"):
            question = row['question']
            options = row['options'] if 'options' in row and pd.notna(row['options']) else None
            correct_answer = row['answer'] if 'answer' in row else None
            
            if not question or pd.isna(question):
                continue
            
            # 使用自适应路由进行推理
            selected_experts, expert_weights = self.router.adaptive_routing(question, options)
            
            # 记录专家使用情况
            for expert in selected_experts:
                metrics['expert_usage'][expert.expert_type] += 1
            
            # 使用推理流水线
            result = self.pipeline.reason_with_experts(question, options, selected_experts)
            
            # 更新指标
            if result.get('is_correct', False):
                correct += 1
            
            total_reward += self.reward_model.calculate_reward(
                answer=result['final_answer'],
                correct_answer=correct_answer,
                confidence=result['confidence'],
                step_count=result.get('step_count', 0),
                expert_types=[expert.expert_type for expert in selected_experts]
            )
            
            total_confidence += result['confidence']
        
        # 计算最终指标
        metrics['accuracy'] = correct / len(eval_data)
        metrics['avg_reward'] = total_reward / len(eval_data)
        metrics['avg_confidence'] = total_confidence / len(eval_data)
        
        # 将专家使用量转换为百分比
        total_usage = sum(metrics['expert_usage'].values())
        if total_usage > 0:
            for expert_type in metrics['expert_usage']:
                metrics['expert_usage'][expert_type] = metrics['expert_usage'][expert_type] / total_usage
        
        # 记录评估结果
        logger.info(f"Evaluation results on {dataset_name}:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Average reward: {metrics['avg_reward']:.4f}")
        logger.info(f"Average confidence: {metrics['avg_confidence']:.4f}")
        logger.info(f"Expert usage: {dict(metrics['expert_usage'])}")
        
        return metrics