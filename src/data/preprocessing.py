import os
import re
import numpy as np
import pandas as pd
import yaml
import logging
import hashlib
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import train_test_split

from src.data.data_loader import DataLoader
from src.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path: str = "config/config.yaml", llm_interface=None):
        """
        初始化数据预处理器
        
        Args:
            config_path: 配置文件路径
            llm_interface: LLM接口实例，如果为None则创建新实例
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.data_config = config['data']
        self.eval_config = config['evaluation']
        
        self.data_loader = DataLoader(config_path)
        self.llm_interface = llm_interface or LLMInterface(config_path)
        
        # 专家库配置
        self.expert_config = self.data_config['expert_libraries']
        self.short_chain_config = self.expert_config['short_chain']
        self.medium_chain_config = self.expert_config['medium_chain']
        self.long_chain_config = self.expert_config['long_chain']
        
        # 去重和质量过滤配置
        self.similarity_threshold = self.data_config['deduplication']['similarity_threshold']
        self.quality_filter_enabled = self.data_config['quality_filter']['enabled']
    
    def preprocess_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        预处理单个数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            预处理后的DataFrame
        """
        logger.info(f"Preprocessing dataset: {dataset_name}")
        
        # 加载数据集
        df = self.data_loader.load_dataset(dataset_name)
        
        # 标准化列名
        df = self.standardize_columns(df, dataset_name)
        
        # 拆分训练和测试集
        train_df, test_df = train_test_split(
            df, 
            test_size=self.eval_config['test_split_ratio'],
            random_state=42
        )
        
        # 保存处理后的数据
        self.data_loader.save_processed_data(f"{dataset_name}_train", train_df)
        self.data_loader.save_processed_data(f"{dataset_name}_test", test_df)
        
        logger.info(f"Preprocessed {dataset_name}: {len(train_df)} train, {len(test_df)} test examples")
        return train_df
    
    def standardize_columns(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        标准化数据集的列名
        
        Args:
            df: 原始DataFrame
            dataset_name: 数据集名称
        
        Returns:
            标准化后的DataFrame
        """
        # 推断问题、答案和选项列
        columns = df.columns
        
        # 尝试寻找问题列
        question_cols = [col for col in columns if any(q in col.lower() for q in ['question', 'query', 'problem'])]
        # 尝试寻找答案列
        answer_cols = [col for col in columns if any(a in col.lower() for a in ['answer', 'label', 'target'])]
        # 尝试寻找选项列
        options_cols = [col for col in columns if any(o in col.lower() for o in ['options', 'choices', 'candidates'])]
        
        # 创建新的标准化DataFrame
        std_df = pd.DataFrame()
        
        # 添加问题列
        if question_cols:
            std_df['question'] = df[question_cols[0]]
        else:
            raise ValueError(f"Cannot identify question column in dataset {dataset_name}")
        
        # 添加答案列
        if answer_cols:
            std_df['answer'] = df[answer_cols[0]]
        else:
            logger.warning(f"No answer column found in dataset {dataset_name}")
        
        # 添加选项列(如果存在)
        if options_cols:
            std_df['options'] = df[options_cols[0]]
        
        # 添加数据集类型信息
        dataset_config = next((d for d in self.data_config['datasets'] if d['name'] == dataset_name), None)
        if dataset_config:
            std_df['task_type'] = dataset_config['type']
        
        # 添加ID列
        std_df['id'] = [f"{dataset_name}_{i}" for i in range(len(std_df))]
        
        return std_df
    
    def generate_expert_examples(self, dataset_name: str = None) -> Dict[str, pd.DataFrame]:
        """
        为每种专家类型生成思维链示例
        
        Args:
            dataset_name: 特定数据集名称，如果为None则处理所有数据集
            
        Returns:
            字典，键为专家类型，值为专家库DataFrame
        """
        expert_libraries = {
            'short_chain': pd.DataFrame(),
            'medium_chain': pd.DataFrame(),
            'long_chain': pd.DataFrame()
        }
        
        # 确定要处理的数据集
        if dataset_name:
            datasets = {dataset_name: self.data_loader.load_processed_data(f"{dataset_name}_train")}
        else:
            # 加载所有训练集
            datasets = {}
            for dataset_config in self.data_config['datasets']:
                name = dataset_config['name']
                try:
                    datasets[name] = self.data_loader.load_processed_data(f"{name}_train")
                except FileNotFoundError:
                    logger.warning(f"Processed data for {name} not found, preprocessing...")
                    datasets[name] = self.preprocess_dataset(name)
        
        # 为每个数据集生成专家示例
        for name, df in datasets.items():
            logger.info(f"Generating expert examples for dataset: {name}")
            
            # 对每个数据集，随机选择样本进行思维链生成
            sample_size = min(100, len(df))  # 每个数据集最多使用100个样本
            samples = df.sample(sample_size, random_state=42)
            
            # 为每个专家类型生成思维链
            for expert_type in expert_libraries.keys():
                examples = self._generate_chain_of_thought(samples, expert_type)
                expert_libraries[expert_type] = pd.concat([expert_libraries[expert_type], examples])
        
        # 去重和质量过滤
        for expert_type, library_df in expert_libraries.items():
            filtered_library = self._deduplicate_examples(library_df)
            if self.quality_filter_enabled:
                filtered_library = self._quality_filter(filtered_library, expert_type)
            
            # 保存专家库
            self.data_loader.save_expert_library(expert_type, filtered_library)
            expert_libraries[expert_type] = filtered_library
            
            logger.info(f"Created {expert_type} library with {len(filtered_library)} examples")
        
        return expert_libraries
    
    def _generate_chain_of_thought(self, df: pd.DataFrame, expert_type: str) -> pd.DataFrame:
        """
        为特定专家类型生成思维链示例
        
        Args:
            df: 数据集DataFrame
            expert_type: 专家类型 ('short_chain', 'medium_chain', 'long_chain')
            
        Returns:
            包含思维链的DataFrame
        """
        # 获取专家提示模板
        if expert_type == 'short_chain':
            prompt_template = self.short_chain_config['prompt_template']
        elif expert_type == 'medium_chain':
            prompt_template = self.medium_chain_config['prompt_template']
        elif expert_type == 'long_chain':
            prompt_template = self.long_chain_config['prompt_template']
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
        
        results = []
        
        for _, row in df.iterrows():
            question = row['question']
            answer = row['answer'] if 'answer' in row else None
            options = row['options'] if 'options' in row else None
            
            # 构建完整提示
            prompt = f"Question: {question}\n\n"
            if options is not None and isinstance(options, str):
                prompt += f"Options: {options}\n\n"
            prompt += f"{prompt_template}\n"
            
            # 调用LLM生成思维链
            try:
                response = self.llm_interface.generate_text(prompt)
                
                # 提取思维链和最终答案
                cot, final_answer = self._extract_cot_and_answer(response)
                
                results.append({
                    'question': question,
                    'options': options,
                    'prompt_template': prompt_template,
                    'chain_of_thought': cot,
                    'generated_answer': final_answer,
                    'correct_answer': answer,
                    'is_correct': self._check_answer_correctness(final_answer, answer) if answer else None,
                    'expert_type': expert_type,
                    'num_steps': self._count_reasoning_steps(cot)
                })
            except Exception as e:
                logger.error(f"Error generating CoT for question '{question[:30]}...': {e}")
        
        return pd.DataFrame(results)
    
    def _extract_cot_and_answer(self, response: str) -> Tuple[str, str]:
        """
        从LLM响应中提取思维链和最终答案
        
        Args:
            response: LLM响应文本
            
        Returns:
            (思维链, 最终答案)的元组
        """
        # 尝试查找"Answer:"或"Therefore,"等标记最终答案的短语
        answer_markers = ["Answer:", "Therefore,", "So the answer is", "The answer is", "Hence,", "In conclusion,"]
        
        for marker in answer_markers:
            if marker in response:
                parts = response.split(marker, 1)
                return parts[0].strip(), parts[1].strip()
        
        # 如果没有找到标记，假设最后一行是答案
        lines = response.strip().split('\n')
        if len(lines) > 1:
            return '\n'.join(lines[:-1]).strip(), lines[-1].strip()
        else:
            return "", response.strip()
    
    def _count_reasoning_steps(self, cot: str) -> int:
        """
        计算思维链中的推理步骤数
        
        Args:
            cot: 思维链文本
            
        Returns:
            步骤数
        """
        # 通过寻找步骤标记来计数
        step_markers = [
            r"Step \d+", r"\d+\.", r"\(\d+\)", 
            "First", "Second", "Third", "Fourth", "Fifth", 
            "Next", "Then", "Finally"
        ]
        
        steps = 0
        lines = cot.split('\n')
        
        for line in lines:
            if any(re.search(marker, line, re.IGNORECASE) for marker in step_markers):
                steps += 1
        
        # 如果没有找到明确的步骤标记，则按段落计数
        if steps == 0:
            # 将文本分成段落，非空段落视为一个步骤
            paragraphs = [p for p in re.split(r'\n\s*\n', cot) if p.strip()]
            steps = len(paragraphs)
        
        return max(1, steps)  # 确保至少有1个步骤
    
    def _check_answer_correctness(self, generated: str, correct: str) -> bool:
        """
        检查生成的答案是否正确
        
        Args:
            generated: 生成的答案
            correct: 正确答案
            
        Returns:
            布尔值，表示答案是否正确
        """
        if not correct or not generated:
            return False
        
        # 标准化答案：删除标点符号，转换为小写
        def normalize(answer):
            if not isinstance(answer, str):
                answer = str(answer)
            return re.sub(r'[^\w\s]', '', answer).lower().strip()
        
        norm_generated = normalize(generated)
        norm_correct = normalize(correct)
        
        # 多选题的情况：可能是A/B/C/D或完整答案
        if len(norm_correct) <= 3 and norm_correct.isalpha():
            # 尝试从生成的答案中提取选项
            option_match = re.search(r'\b([A-Da-d])\b', generated)
            if option_match:
                extracted_option = option_match.group(1).upper()
                return extracted_option == norm_correct.upper()
        
        # 检查答案是否包含正确答案或完全匹配
        return norm_generated == norm_correct or norm_correct in norm_generated
    
    def _deduplicate_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用MinHash去除相似示例
        
        Args:
            df: 专家示例DataFrame
            
        Returns:
            去重后的DataFrame
        """
        if len(df) <= 1:
            return df
        
        # 创建LSH索引
        lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=128)
        
        # 为每个示例创建MinHash
        minhashes = {}
        for idx, row in df.iterrows():
            # 组合问题和思维链作为去重的基础
            text = f"{row['question']} {row['chain_of_thought']}"
            minhash = MinHash(num_perm=128)
            for token in self._tokenize(text):
                minhash.update(token.encode('utf-8'))
            minhashes[idx] = minhash
            lsh.insert(idx, minhash)
        
        # 找出重复项
        duplicates = set()
        for idx, minhash in minhashes.items():
            if idx in duplicates:
                continue
                
            # 查找相似项
            similar_idxs = lsh.query(minhash)
            if len(similar_idxs) > 1:  # 找到相似项
                # 保留步骤数符合要求的示例
                similar_rows = df.loc[similar_idxs]
                best_idx = self._select_best_example(similar_rows)
                
                # 将其他相似项标记为重复
                for similar_idx in similar_idxs:
                    if similar_idx != best_idx:
                        duplicates.add(similar_idx)
                        
        # 移除重复项
        return df.drop(index=list(duplicates)).reset_index(drop=True)
    
    def _select_best_example(self, similar_rows: pd.DataFrame) -> int:
        """
        从相似示例中选择最好的一个
        
        Args:
            similar_rows: 相似示例的DataFrame
            
        Returns:
            最佳示例的索引
        """
        # 优先选择正确的答案
        correct_rows = similar_rows[similar_rows['is_correct'] == True]
        if not correct_rows.empty:
            candidates = correct_rows
        else:
            candidates = similar_rows
        
        # 根据专家类型选择步骤数最合适的样本
        expert_type = candidates['expert_type'].iloc[0]  # 假设所有行的专家类型相同
        
        if expert_type == 'short_chain':
            # 对于短链专家，选择步骤数在1-3之间的
            step_range = self.short_chain_config['step_range']
            valid_rows = candidates[(candidates['num_steps'] >= step_range[0]) & 
                                   (candidates['num_steps'] <= step_range[1])]
        elif expert_type == 'medium_chain':
            # 对于中链专家，选择步骤数在4-6之间的
            step_range = self.medium_chain_config['step_range']
            valid_rows = candidates[(candidates['num_steps'] >= step_range[0]) & 
                                   (candidates['num_steps'] <= step_range[1])]
        elif expert_type == 'long_chain':
            # 对于长链专家，选择步骤数≥7的
            step_range = self.long_chain_config['step_range']
            valid_rows = candidates[candidates['num_steps'] >= step_range[0]]
        else:
            valid_rows = candidates
        
        # 如果没有符合步骤要求的行，则返回到原始候选集
        if valid_rows.empty:
            valid_rows = candidates
        
        # 返回第一个有效行的索引
        return valid_rows.index[0]
    
    def _quality_filter(self, df: pd.DataFrame, expert_type: str) -> pd.DataFrame:
        """
        根据质量要求过滤示例
        
        Args:
            df: 专家示例DataFrame
            expert_type: 专家类型
            
        Returns:
            过滤后的DataFrame
        """
        # 获取专家类型对应的步骤范围
        if expert_type == 'short_chain':
            step_range = self.short_chain_config['step_range']
        elif expert_type == 'medium_chain':
            step_range = self.medium_chain_config['step_range']
        elif expert_type == 'long_chain':
            step_range = self.long_chain_config['step_range']
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
        
        # 过滤步骤数不符合要求的示例
        filtered_df = df[(df['num_steps'] >= step_range[0]) & (df['num_steps'] <= step_range[1])]
        
        # 如果过滤后的结果太少，则放宽标准
        if len(filtered_df) < len(df) * 0.5:
            logger.warning(f"Quality filter removed too many examples for {expert_type}, relaxing criteria")
            if expert_type == 'long_chain':
                # 对于长链专家，只要求步骤数≥下限
                filtered_df = df[df['num_steps'] >= step_range[0]]
            else:
                # 对于其他专家，允许步骤数超出上限
                filtered_df = df[df['num_steps'] >= step_range[0]]
        
        # 优先保留答案正确的示例
        correct_df = filtered_df[filtered_df['is_correct'] == True]
        if len(correct_df) >= len(df) * 0.3:  # 如果正确答案比例足够高
            return correct_df
        else:
            return filtered_df
            
    def _tokenize(self, text: str) -> List[str]:
        """
        简单的文本分词
        
        Args:
            text: 待分词文本
            
        Returns:
            分词列表
        """
        # 将文本转换为小写，并删除标点符号
        text = re.sub(r'[^\w\s]', '', text.lower())
        # 按空白字符分词
        return text.split()