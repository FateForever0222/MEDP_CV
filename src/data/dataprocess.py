import os
import re
import json
import pandas as pd
import yaml
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 确保导入tqdm
from src.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理器：加载数据集、生成专家库"""
    
    def __init__(self, config_path: str = "config/config.yaml", llm_interface=None):
        """初始化数据处理器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.data_config = config['data']
        self.eval_config = config['evaluation']
        
        # 路径设置
        self.processed_path = Path(self.data_config['processed_path'])
        self.expert_libraries_path = Path(self.data_config['expert_libraries']['path'])
        
        # 确保路径存在
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.expert_libraries_path.mkdir(parents=True, exist_ok=True)
        
        # 创建LLM接口
        self.llm_interface = llm_interface or LLMInterface(config_path)
        
        # 专家库配置
        self.expert_config = self.data_config['expert_libraries']
        self.short_chain_config = self.expert_config['short_chain']
        self.medium_chain_config = self.expert_config['medium_chain']
        self.long_chain_config = self.expert_config['long_chain']
        
        # 每个专家类型的示例数量
        self.examples_per_expert = self.expert_config.get('examples_per_expert', 80)
        
        # 去重和质量过滤配置
        self.similarity_threshold = self.data_config['deduplication']['similarity_threshold']
        self.quality_filter_enabled = self.data_config['quality_filter']['enabled']
        
        logger.info("数据处理器初始化完成")

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """加载指定的数据集"""
        dataset_name = dataset_name.upper()
        dataset_path = Path(self.data_config['raw_path'])
        
        logger.info(f"加载数据集: {dataset_name}")
        
        if dataset_name == "AQUA":
            return self._load_aqua_dataset(dataset_path / "AQuA" / "test.json")
        elif dataset_name == "COIN" or dataset_name == "COIN_FLIP":
            return self._load_coin_dataset(dataset_path / "coin_flip" / "coin_flip.json")
        elif dataset_name == "CSQA" or dataset_name == "COMMONSENSEQA":
            return self._load_csqa_dataset(dataset_path / "CommonsenseQA" / "train_rand_split.jsonl")
        elif dataset_name == "LETTER" or dataset_name == "LAST_LETTERS":
            return self._load_letter_dataset(dataset_path / "last_letters" / "last_letters.json")
        elif dataset_name == "MULTIARITH":
            return self._load_multiarith_dataset(dataset_path / "MultiArith" / "MultiArith.json")
        elif dataset_name == "STRATEGYQA":
            return self._load_strategyqa_dataset(dataset_path / "StrategyQA" / "task.json")
        else:
            raise ValueError(f"未支持的数据集: {dataset_name}")
    
    def _load_aqua_dataset(self, file_path: Path) -> pd.DataFrame:
        """加载AQuA数据集"""
        questions = []
        answers = []
        options_list = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                question = data["question"].strip()
                
                # 处理选项
                options = {}
                for idx, option in enumerate(data["options"]):
                    label = chr(ord('A') + idx)
                    options[label] = option
                
                correct = data["correct"]
                
                questions.append(question)
                answers.append(correct)
                options_list.append(options)
        
        return pd.DataFrame({
            "question": questions,
            "answer": answers,
            "options": options_list
        })
    
    def _load_coin_dataset(self, file_path: Path) -> pd.DataFrame:
        """加载Coin Flip数据集"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            examples = data.get("examples", data)
            
            questions = []
            answers = []
            
            for example in examples:
                if "question" in example and "answer" in example:
                    questions.append(example["question"])
                    answers.append(example["answer"])
        
        return pd.DataFrame({
            "question": questions,
            "answer": answers
        })
    
    def _load_csqa_dataset(self, file_path: Path) -> pd.DataFrame:
        """加载CSQA数据集"""
        questions = []
        answers = []
        options_list = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_data = json.loads(line)
                
                # 提取问题和选项
                question = json_data["question"]["stem"].strip()
                choices = json_data["question"]["choices"]
                
                # 格式化选项
                options = {}
                for choice in choices:
                    options[choice["label"]] = choice["text"]
                
                questions.append(question)
                answers.append(json_data["answerKey"])
                options_list.append(options)
        
        return pd.DataFrame({
            "question": questions,
            "answer": answers,
            "options": options_list
        })
    
    def _load_letter_dataset(self, file_path: Path) -> pd.DataFrame:
        """加载Last Letters数据集"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            examples = data.get("examples", data)
            
            questions = []
            answers = []
            
            for example in examples:
                if "question" in example and "answer" in example:
                    questions.append(example["question"])
                    answers.append(example["answer"])
        
        return pd.DataFrame({
            "question": questions,
            "answer": answers
        })
    
    def _load_multiarith_dataset(self, file_path: Path) -> pd.DataFrame:
        """加载MultiArith数据集"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            questions = []
            answers = []
            
            for example in data:
                question = example["sQuestion"].strip()
                answer = str(example["lSolutions"][0])
                if answer.endswith(".0"):
                    answer = answer[:-2]
                
                questions.append(question)
                answers.append(answer)
        
        return pd.DataFrame({
            "question": questions,
            "answer": answers
        })
    
    def _load_strategyqa_dataset(self, file_path: Path) -> pd.DataFrame:
        """加载StrategyQA数据集"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            examples = data.get("examples", data)
            
            questions = []
            answers = []
            
            for example in examples:
                if "input" in example:
                    question = example["input"].strip()
                    if "target_scores" in example:
                        answer = "yes" if example["target_scores"]["Yes"] == 1 else "no"
                    else:
                        answer = example.get("answer", "")
                else:
                    question = example.get("question", "").strip()
                    answer = example.get("answer", "")
                
                questions.append(question)
                answers.append(answer)
        
        return pd.DataFrame({
            "question": questions,
            "answer": answers
        })
    
    def preprocess_dataset(self, dataset_name: str) -> pd.DataFrame:
        """预处理单个数据集"""
        logger.info(f"预处理数据集: {dataset_name}")
        
        # 加载数据集
        df = self.load_dataset(dataset_name)
        
        # 拆分训练和测试集
        train_df, test_df = train_test_split(
            df, 
            test_size=self.eval_config['test_split_ratio'],
            random_state=42
        )
        
        # 添加数据集标识
        train_df['dataset'] = dataset_name
        test_df['dataset'] = dataset_name
        
        # 保存处理后的数据
        self.save_processed_data(f"{dataset_name}_train", train_df)
        self.save_processed_data(f"{dataset_name}_test", test_df)
        
        logger.info(f"预处理完成 {dataset_name}: {len(train_df)} 训练样本, {len(test_df)} 测试样本")
        return train_df
    
    def load_processed_data(self, dataset_name: str) -> pd.DataFrame:
        """加载处理后的数据集"""
        processed_path = self.processed_path / f"{dataset_name}.csv"
        if not processed_path.exists():
            raise FileNotFoundError(f"找不到处理后的数据: {processed_path}")
        
        return pd.read_csv(processed_path)
    
    def save_processed_data(self, dataset_name: str, df: pd.DataFrame) -> None:
        """保存处理后的数据集"""
        processed_path = self.processed_path / f"{dataset_name}.csv"
        df.to_csv(processed_path, index=False)
        logger.info(f"保存处理后的数据到 {processed_path}")
    
    def load_expert_library(self, dataset_name: str, expert_type: str) -> pd.DataFrame:
        """加载指定数据集和专家类型的专家库"""
        library_path = self.expert_libraries_path / f"{dataset_name}_{expert_type}_library.csv"
        if not library_path.exists():
            logger.warning(f"{dataset_name} 的 {expert_type} 专家库在 {library_path} 不存在")
            return pd.DataFrame()
        
        return pd.read_csv(library_path)
    
    def save_expert_library(self, dataset_name: str, expert_type: str, library_df: pd.DataFrame) -> None:
        """保存专家库到CSV文件"""
        library_path = self.expert_libraries_path / f"{dataset_name}_{expert_type}_library.csv"
        library_df.to_csv(library_path, index=False)
        logger.info(f"保存了 {len(library_df)} 个示例到 {dataset_name} 的 {expert_type} 专家库")
    
    def generate_expert_examples(self, dataset_name: str = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """为每个数据集的每种专家类型生成思维链示例"""
        expert_types = ['short_chain', 'medium_chain', 'long_chain']
        all_expert_libraries = {}
        
        # 确定要处理的数据集
        if dataset_name:
            datasets_to_process = [dataset_name]
        else:
            # 处理所有数据集
            datasets_to_process = [config['name'] for config in self.data_config['datasets']]
        
        # 为每个数据集生成专家示例
        for name in datasets_to_process:
            logger.info(f"为数据集 {name} 生成专家示例")
            
            try:
                # 尝试加载预处理数据
                train_df = self.load_processed_data(f"{name}_train")
            except FileNotFoundError:
                logger.warning(f"预处理数据集 {name} 不存在，开始预处理...")
                train_df = self.preprocess_dataset(name)
            
            # 为当前数据集创建专家库字典
            dataset_libraries = {}
            
            # 对当前数据集，随机选择样本进行思维链生成
            sample_size = min(self.examples_per_expert, len(train_df))
            samples = train_df.sample(sample_size, random_state=42)
            
            # 为每个专家类型生成思维链
            for expert_type in expert_types:
                examples = self._generate_chain_of_thought(samples, expert_type, name)
                
                # 去重和质量过滤
                filtered_examples = self._deduplicate_examples(examples)
                if self.quality_filter_enabled:
                    filtered_examples = self._quality_filter(filtered_examples, expert_type)
                
                # 保存专家库
                self.save_expert_library(name, expert_type, filtered_examples)
                dataset_libraries[expert_type] = filtered_examples
                
                logger.info(f"为数据集 {name} 创建 {expert_type} 专家库，包含 {len(filtered_examples)} 个示例")
            
            # 将当前数据集的专家库添加到总集合
            all_expert_libraries[name] = dataset_libraries
        
        return all_expert_libraries
    
    def _generate_chain_of_thought(self, df: pd.DataFrame, expert_type: str, dataset_name: str) -> pd.DataFrame:
        """为特定专家类型生成思维链示例"""
        # 获取专家提示模板
        if expert_type == 'short_chain':
            prompt_template = self.short_chain_config['prompt_template']
        elif expert_type == 'medium_chain':
            prompt_template = self.medium_chain_config['prompt_template']
        elif expert_type == 'long_chain':
            prompt_template = self.long_chain_config['prompt_template']
        else:
            raise ValueError(f"未知的专家类型: {expert_type}")
        
        results = []
        
        for _, row in tqdm(df.iterrows(), 
                     total=len(df), 
                     desc=f"生成{expert_type}示例",
                     ncols=80):
            question = row['question']
            answer = row['answer'] if 'answer' in row and not pd.isna(row['answer']) else None
            options = row['options'] if 'options' in row and not pd.isna(row['options']) else None
            
            # 构建完整提示
            prompt = f"Question: {question}\n\n"
            if options is not None:
                if isinstance(options, str):
                    prompt += f"Options: {options}\n\n"
                elif isinstance(options, dict):
                    options_text = ", ".join([f"{k}: {v}" for k, v in options.items()])
                    prompt += f"Options: {options_text}\n\n"
            
            prompt += f"{prompt_template}\n"
            
            # 调用LLM生成思维链
            try:
                response = self.llm_interface.generate(prompt)
                
                # 提取思维链和最终答案
                cot, final_answer = self._extract_cot_and_answer(response)
                
                results.append({
                    'dataset': dataset_name,
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
                logger.error(f"为数据集 {dataset_name} 问题生成思维链时出错: '{question[:30]}...': {e}")
        
        return pd.DataFrame(results)
    
    def _extract_cot_and_answer(self, response: str) -> Tuple[str, str]:
        """从LLM响应中提取思维链和最终答案"""
        # 尝试查找标记最终答案的短语
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
        """计算思维链中的推理步骤数"""
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
            paragraphs = [p for p in re.split(r'\n\s*\n', cot) if p.strip()]
            steps = len(paragraphs)
        
        return max(1, steps)  # 确保至少有1个步骤
    
    def _check_answer_correctness(self, generated: str, correct: str) -> bool:
        """检查生成的答案是否正确"""
        if not correct or not generated:
            return False
        
        # 标准化答案
        def normalize(answer):
            if not isinstance(answer, str):
                answer = str(answer)
            return re.sub(r'[^\w\s]', '', answer).lower().strip()
        
        norm_generated = normalize(generated)
        norm_correct = normalize(correct)
        
        # 多选题的情况
        if len(norm_correct) <= 3 and norm_correct.isalpha():
            option_match = re.search(r'\b([A-Da-d])\b', generated)
            if option_match:
                extracted_option = option_match.group(1).upper()
                return extracted_option == norm_correct.upper()
        
        # 检查答案是否匹配
        return norm_generated == norm_correct or norm_correct in norm_generated
    
    def _tokenize(self, text: str) -> List[str]:
        """简单的文本分词"""
        return [token.strip().lower() for token in re.sub(r'[^\w\s]', ' ', text).split()]
    
    def _deduplicate_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用MinHash去除相似示例"""
        if len(df) <= 1:
            return df
        
        # 创建LSH索引
        lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=128)
        
        # 为每个示例创建MinHash
        minhashes = {}
        for idx, row in df.iterrows():
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
            
            similar_idxs = lsh.query(minhash)
            if len(similar_idxs) > 1:
                similar_rows = df.loc[similar_idxs]
                best_idx = self._select_best_example(similar_rows)
                
                for similar_idx in similar_idxs:
                    if similar_idx != best_idx:
                        duplicates.add(similar_idx)
        
        # 移除重复项
        return df.drop(index=list(duplicates)).reset_index(drop=True)
    
    def _select_best_example(self, similar_rows: pd.DataFrame) -> int:
        """从相似示例中选择最好的一个"""
        # 优先选择正确的答案
        correct_rows = similar_rows[similar_rows['is_correct'] == True]
        if not correct_rows.empty:
            candidates = correct_rows
        else:
            candidates = similar_rows
        
        # 根据专家类型选择步骤数最合适的样本
        expert_type = candidates['expert_type'].iloc[0]
        
        if expert_type == 'short_chain':
            step_range = self.short_chain_config['step_range']
            valid_rows = candidates[(candidates['num_steps'] >= step_range[0]) & 
                                  (candidates['num_steps'] <= step_range[1])]
        elif expert_type == 'medium_chain':
            step_range = self.medium_chain_config['step_range']
            valid_rows = candidates[(candidates['num_steps'] >= step_range[0]) & 
                                  (candidates['num_steps'] <= step_range[1])]
        elif expert_type == 'long_chain':
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
        """基于步骤数和其他质量指标进行过滤"""
        if df.empty:
            return df
        
        # 根据专家类型设置步骤范围
        if expert_type == 'short_chain':
            step_range = self.short_chain_config['step_range']
            min_steps, max_steps = step_range[0], step_range[1]
        elif expert_type == 'medium_chain':
            step_range = self.medium_chain_config['step_range']
            min_steps, max_steps = step_range[0], step_range[1]
        elif expert_type == 'long_chain':
            step_range = self.long_chain_config['step_range']
            min_steps, max_steps = step_range[0], float('inf')
        else:
            return df
        
        # 基于步骤数过滤
        filtered_df = df[df['num_steps'] >= min_steps]
        if max_steps < float('inf'):
            filtered_df = filtered_df[filtered_df['num_steps'] <= max_steps]
        
        # 过滤掉没有答案或思维链的示例
        filtered_df = filtered_df[filtered_df['chain_of_thought'].notna() & (filtered_df['chain_of_thought'] != '')]
        filtered_df = filtered_df[filtered_df['generated_answer'].notna() & (filtered_df['generated_answer'] != '')]
        
        # 如果过滤后为空，返回原始DataFrame
        if filtered_df.empty:
            logger.warning(f"质量过滤后没有剩余示例，返回原始数据")
            return df
        
        logger.info(f"质量过滤: {len(df)} -> {len(filtered_df)} 个示例")
        return filtered_df