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
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from src.data.dataprocess import DataProcessor
from src.gating.expert_router import ExpertRouter
from src.inference.inference_engine import InferenceEngine as ReasoningPipeline
from src.utils.text_utils import normalize_answer

logger = logging.getLogger(__name__)

class RewardModel:
    """
    奖励模型，用于计算专家推理的奖励
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化奖励模型
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 加载奖励配置
        self.reward_config = config.get('training', {}).get('reward', {})
        
        # 奖励权重
        self.accuracy_weight = self.reward_config.get('accuracy_weight', 0.7)
        self.confidence_weight = self.reward_config.get('confidence_weight', 0.2)
        self.step_score_weight = self.reward_config.get('step_score_weight', 0.1)
        
        # 是否使用非线性奖励计算
        self.use_non_linear = self.reward_config.get('use_non_linear', False)
        
        logger.info(f"Initialized RewardModel with weights: accuracy={self.accuracy_weight}, "
                  f"confidence={self.confidence_weight}, step_score={self.step_score_weight}")
    
    def calculate_reward(self, answer: str, correct_answer: Optional[str] = None, 
                       confidence: float = 0.0, step_count: int = 0,
                       expert_types: List[str] = None) -> float:
        """
        计算奖励
        
        Args:
            answer: 生成的答案
            correct_answer: 正确答案（如果有）
            confidence: 模型置信度
            step_count: 推理步骤数
            expert_types: 使用的专家类型列表
            
        Returns:
            奖励值
        """
        # 计算正确性奖励
        accuracy_score = self._calculate_accuracy_score(answer, correct_answer)
        
        # 计算置信度奖励
        confidence_score = confidence
        
        # 计算步骤合理性奖励
        step_score = self._calculate_step_score(step_count, expert_types)
        
        # 线性组合或非线性组合
        if self.use_non_linear:
            # 非线性组合 (对数形式)
            reward = np.log(
                self.accuracy_weight * accuracy_score + 
                self.confidence_weight * confidence_score * 
                self.step_score_weight * step_score + 
                1e-10  # 防止log(0)
            )
        else:
            # 线性组合
            reward = (
                self.accuracy_weight * accuracy_score +
                self.confidence_weight * confidence_score +
                self.step_score_weight * step_score
            )
        
        return float(reward)
    
    def _calculate_accuracy_score(self, answer: str, correct_answer: Optional[str]) -> float:
        """
        计算答案正确性得分
        
        Args:
            answer: 生成的答案
            correct_answer: 正确答案
                
        Returns:
            正确性得分(0或1)
        """
        if correct_answer is None:
            # 如果没有正确答案，给予0分
            return 0.0
        
        # 使用文本处理工具获取标准化的答案
        norm_answer = normalize_answer(answer)
        norm_correct = normalize_answer(correct_answer)
        
        # 简单比较：相同为1，不同为0
        return 1.0 if norm_answer == norm_correct else 0.0
    
    def _calculate_step_score(self, step_count: int, expert_types: List[str]) -> float:
        """
        计算步骤合理性得分
        
        Args:
            step_count: 推理步骤数
            expert_types: 使用的专家类型列表
            
        Returns:
            步骤合理性得分(0-1)
        """
        if not expert_types:
            return 0.5  # 默认中等分数
        
        # 检查步骤数是否与使用的专家类型一致
        scores = []
        
        for expert_type in expert_types:
            if expert_type == 'short_chain':
                # 短链专家期望1-3步
                score = max(0.0, 1.0 - abs(step_count - 2) / 3.0)
            elif expert_type == 'medium_chain':
                # 中链专家期望4-6步
                score = max(0.0, 1.0 - abs(step_count - 5) / 4.0)
            elif expert_type == 'long_chain':
                # 长链专家期望7步以上
                score = min(1.0, max(0.0, (step_count - 4) / 6.0))
            else:
                score = 0.5  # 未知专家类型
            
            scores.append(score)
        
        # 返回最高分数
        return max(scores) if scores else 0.5


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
        self.config_path = config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载训练配置
        self.training_config = self.config.get('training', {}).get('grpo', {})
        
        # 创建路由器（包含门控网络）
        self.router = ExpertRouter(config_path)
        
        # 创建奖励模型
        self.reward_model = RewardModel(config_path)
        
        # 创建组采样器
        self.group_sampler = GroupSampler(
            num_groups=self.training_config.get('num_groups', 8),
            noise_std=self.training_config.get('noise_std', 0.1)
        )
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.router.gating_network.parameters(),
            lr=self.training_config.get('learning_rate', 0.001)
        )
        
        # 创建推理流水线
        self.pipeline = ReasoningPipeline(config_path)
        
        # 数据加载器
        self.data_loader = DataProcessor(config_path)
        
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
            num_epochs = self.training_config.get('max_epochs', 10)
        
        # 加载训练数据
        try:
            train_data = self.data_loader.load_processed_data(f"{dataset_name}_train")
            logger.info(f"Loaded {len(train_data)} training examples from {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading training data for {dataset_name}: {e}")
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
            max_samples = min(self.training_config.get('samples_per_epoch', 100), len(train_data))
            epoch_data = train_data.iloc[:max_samples]
            
            total_loss = 0.0
            total_reward = 0.0
            correct_predictions = 0
            
            # 使用tqdm显示进度条
            for idx, row in tqdm(epoch_data.iterrows(), total=len(epoch_data), desc=f"Epoch {epoch+1}/{num_epochs}"):
                question = row['question']
                options = row.get('options') if 'options' in row and pd.notna(row.get('options')) else None
                correct_answer = row.get('answer') if 'answer' in row else None
                
                if pd.isna(question) or not question:
                    continue
                
                # 训练单个样本
                try:
                    loss, reward, is_correct = self._train_single_sample(question, options, correct_answer)
                    
                    total_loss += loss
                    total_reward += reward
                    if is_correct:
                        correct_predictions += 1
                except Exception as e:
                    logger.error(f"Error training sample {idx}: {e}")
                    continue
            
            # 计算epoch指标
            avg_loss = total_loss / max(1, len(epoch_data))
            avg_reward = total_reward / max(1, len(epoch_data))
            accuracy = correct_predictions / max(1, len(epoch_data))
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, "
                      f"Accuracy: {accuracy:.4f}, Time: {epoch_time:.2f}s")
            
            # 检查是否需要保存模型
            if accuracy > self.best_accuracy:
                logger.info(f"New best accuracy: {accuracy:.4f} (previous: {self.best_accuracy:.4f})")
                self.best_accuracy = accuracy
                torch.save(self.router.gating_network.state_dict(), os.path.join("models", "best_gating_network.pt"))
                self.patience_counter = 0
            elif avg_reward > self.best_reward:
                logger.info(f"New best reward: {avg_reward:.4f} (previous: {self.best_reward:.4f})")
                self.best_reward = avg_reward
                torch.save(self.router.gating_network.state_dict(), os.path.join("models", "best_reward_gating_network.pt"))
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # 早停
            patience = self.training_config.get('early_stopping_patience', 3)
            if self.patience_counter >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # 训练结束，加载最佳模型
        try:
            best_model_path = os.path.join("models", "best_gating_network.pt")
            state_dict = torch.load(best_model_path)
            self.router.update_model(state_dict)
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
        try:
            # 获取问题特征
            features = self.router._get_combined_features(question, options)
            
            logger.debug(f"问题特征: {features.squeeze().tolist()}")
            try:
                expert_groups = self.group_sampler.sample_groups(
                    self.router.gating_network, 
                    features,
                )
                logger.debug(f"生成了 {len(expert_groups)} 组专家权重")
            except Exception as e:
                logger.error(f"组采样错误: {e}")
                logger.error(f"专家数量: {len(self.router.experts)}")
                return 0.0, 0.0, False
            # 检查专家组是否为空
            if not expert_groups:
                logger.error("未生成任何专家组")
                return 0.0, 0.0, False
            # 记录专家组权重
            logger.debug(f"生成了 {len(expert_groups)} 组专家权重:")
            for i, group in enumerate(expert_groups):
                logger.debug(f"组 {i} 权重: {group.squeeze().tolist()}")
            
            # 计算每组的奖励
            group_rewards = []
            group_results = []
            best_result = None
            is_correct = False
            logger.debug(f"\n===== 问题: {question} =====")
            if options:
                logger.debug(f"选项: {options}")
            logger.debug(f"正确答案: {correct_answer}")
            for group_idx, group_weights in enumerate(expert_groups):
                # 记录当前组权重
                logger.debug(f"\n组 {group_idx} 权重: {group_weights.squeeze().tolist()}")
                # 根据权重选择专家
                selected_experts = []
                expert_indices = []
                max_weight = max(group_weights.squeeze().tolist())
                relative_threshold = max_weight * 0.8
                for expert_idx, weight in enumerate(group_weights.squeeze().tolist()):
                    if weight > relative_threshold:  # 仅选择权重大于最大权重的专家
                        if expert_idx in self.router.experts:
                            selected_experts.append(self.router.experts[expert_idx])
                            expert_indices.append(expert_idx)
                # 记录选择的专家
                expert_names = [self.router.expert_names.get(idx, f"专家{idx}") for idx in expert_indices]
                logger.debug(f"选择的专家: {expert_names}")
                # 确保至少选择一个专家
                if not selected_experts:
                    max_idx = torch.argmax(group_weights).item()
                    logger.debug(f"没有选择专家，默认选择权重最高的专家: {max_idx}")
                    if max_idx in self.router.experts:
                        selected_experts = [self.router.experts[max_idx]]
                    else:
                        # 无效的专家索引，使用第一个可用的专家
                        if self.router.experts:
                            selected_experts = [next(iter(self.router.experts.values()))]
                        else:
                            logger.error("No experts available")
                            continue
                
                # 使用选定的专家进行推理
                try:
                    result = self.pipeline.reason_with_experts(question, selected_experts, options)
                    answer = result.get('final_answer', '')
                    confidence = result.get('confidence', 0.0)
                    step_count = result.get('step_count', 0)
                    logger.debug(f"推理结果: {answer}")
                    logger.debug(f"置信度: {confidence:.4f}")
                    logger.debug(f"步骤数: {step_count}")
                    # 计算该组的奖励
                    reward = self.reward_model.calculate_reward(
                        answer=answer,
                        correct_answer=correct_answer,
                        confidence=confidence,
                        step_count=step_count,
                        expert_types=[expert.expert_type for expert in selected_experts]
                    )
                    logger.debug(f"奖励: {reward:.4f}")
                    group_rewards.append(reward)
                    group_results.append(result)
                    
                    # 检查是否为最佳结果
                    if best_result is None or reward > max(group_rewards[:-1], default=0):
                        best_result = result
                        is_correct = result.get('is_correct', False)
                        logger.debug(f"更新最佳结果，当前答案是否正确: {is_correct}")
                
                except Exception as e:
                    logger.error(f"组 {group_idx} 推理出错: {e}")
                    group_rewards.append(0.0)
                    group_results.append(None)
            
            # 如果所有组都失败了，返回零损失
            if not group_rewards:
                logger.warning(f"所有组都推理失败，返回零损失")
                return 0.0, 0.0, False
            
            # 计算每组的相对优势
            advantages = self._calculate_advantages(group_rewards)
            logger.debug(f"各组奖励: {group_rewards}")
            logger.debug(f"各组优势: {advantages}")
                    
            # 更新门控网络
            try:
                loss = self._update_policy(features, expert_groups, advantages)
                logger.debug(f"策略更新，损失: {loss:.6f}")
            except Exception as e:
                logger.error(f"更新策略出错: {e}")
                loss = 0.0
            # 返回平均损失、平均奖励和是否正确
            avg_reward = sum(group_rewards) / len(group_rewards)
            logger.debug(f"样本训练完成，平均奖励: {avg_reward:.4f}, 损失: {loss:.6f}, 是否正确: {is_correct}")
            return loss, avg_reward, is_correct
        except Exception as e:
            logger.error(f"训练样本时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0, 0.0, False
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
        使用GRPO更新策略
        
        Args:
            features: 输入特征
            expert_groups: 专家权重组列表
            advantages: 每组的优势列表
            
        Returns:
            损失值
        """
        self.optimizer.zero_grad()
        
        # 当前策略下的专家权重
        current_weights = self.router.gating_network(features)
        
        # 计算策略损失
        policy_loss = torch.tensor(0.0, requires_grad=True)
        kl_loss = 0.0
        
        for group_weights, advantage in zip(expert_groups, advantages):
            if advantage == 0.0:
                continue
                
            # 计算比率 (重要性采样权重)
            eps = 1e-8  # 避免数值不稳定
            ratio = torch.sum(current_weights * group_weights) / (torch.sum(group_weights * group_weights) + eps)
            
            # 策略梯度损失
            policy_term = ratio * advantage
            policy_loss -= policy_term
            
            # KL散度正则化
            kl_div = F.kl_div(
                F.log_softmax(current_weights, dim=-1),
                F.softmax(group_weights.detach(), dim=-1),
                reduction='sum'
            )
            kl_loss += kl_div
        
        # 总损失
        kl_coef = self.training_config.get('kl_coef', 0.2)
        total_loss = policy_loss + kl_coef * kl_loss
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.router.gating_network.parameters(), 1.0)
        
        # 更新参数
        self.optimizer.step()
        
        return total_loss.item()
    
    def evaluate(self, dataset_name: str, max_samples: int = 100) -> Dict[str, Any]:
        """
        评估门控网络
        
        Args:
            dataset_name: 数据集名称
            max_samples: 最大评估样本数
            
        Returns:
            包含评估指标的字典
        """
        # 加载测试数据
        try:
            test_data = self.data_loader.load_processed_data(f"{dataset_name}_test")
            logger.info(f"Loaded {len(test_data)} test examples from {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading test data for {dataset_name}: {e}")
            return {}
        
        # 评估指标
        metrics = {
            'accuracy': 0.0,
            'avg_reward': 0.0,
            'avg_confidence': 0.0,
            'expert_usage': defaultdict(int)
        }
        
        # 限制评估样本数
        max_samples = min(max_samples, len(test_data))
        eval_data = test_data.iloc[:max_samples]
        
        correct = 0
        total_reward = 0.0
        total_confidence = 0.0
        total_samples = 0
        
        # 使用tqdm显示进度条
        for idx, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc="Evaluating"):
            question = row['question']
            options = row.get('options') if 'options' in row and pd.notna(row.get('options')) else None
            correct_answer = row.get('answer') if 'answer' in row else None
            
            if pd.isna(question) or not question:
                continue
            
            try:
                # 使用路由器选择专家
                selected_experts, weights = self.router.route(question, options)
                
                # 记录专家使用情况
                for expert in selected_experts:
                    expert_type = getattr(expert, 'expert_type', 'unknown')
                    metrics['expert_usage'][expert_type] += 1
                
                # 使用推理流水线
                result = self.pipeline.reason_with_experts(question, selected_experts, options)
                
                # 更新指标
                if result.get('is_correct', False):
                    correct += 1
                
                total_reward += self.reward_model.calculate_reward(
                    answer=result.get('final_answer', ''),
                    correct_answer=correct_answer,
                    confidence=result.get('confidence', 0.0),
                    step_count=result.get('step_count', 0),
                    expert_types=[getattr(expert, 'expert_type', 'unknown') for expert in selected_experts]
                )
                
                total_confidence += result.get('confidence', 0.0)
                total_samples += 1
                
            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {e}")
                continue
        
        # 计算最终指标
        if total_samples > 0:
            metrics['accuracy'] = correct / total_samples
            metrics['avg_reward'] = total_reward / total_samples
            metrics['avg_confidence'] = total_confidence / total_samples
        
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