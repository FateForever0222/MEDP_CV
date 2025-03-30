import os
import json
import pandas as pd
import yaml
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化数据加载器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['data']
        
        self.datasets_config = self.config['datasets']
        self.processed_path = Path(self.config['processed_path'])
        self.expert_libraries_path = Path(self.config['expert_libraries']['path'])
        
        # 确保路径存在
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.expert_libraries_path.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        加载指定的数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            DataFrame包含问题和答案
        """
        # 查找数据集配置
        dataset_config = next((d for d in self.datasets_config if d['name'] == dataset_name), None)
        if not dataset_config:
            raise ValueError(f"Dataset {dataset_name} not found in config")
        
        dataset_path = Path(dataset_config['path'])
        dataset_type = dataset_config['type']
        
        # 根据数据集类型选择加载方法
        if os.path.exists(dataset_path / "train.json"):
            return self._load_json_dataset(dataset_path / "train.json")
        elif os.path.exists(dataset_path / "train.csv"):
            return self._load_csv_dataset(dataset_path / "train.csv")
        elif os.path.exists(dataset_path / "train.jsonl"):
            return self._load_jsonl_dataset(dataset_path / "train.jsonl")
        else:
            raise FileNotFoundError(f"No supported dataset file found in {dataset_path}")
    
    def _load_json_dataset(self, file_path: Path) -> pd.DataFrame:
        """加载JSON格式数据集"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 尝试标准化数据格式
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            return pd.DataFrame(data['data'])
        else:
            # 处理各种可能的JSON结构
            flattened_data = []
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        flattened_data.extend(value)
                    elif isinstance(value, dict):
                        flattened_data.append(value)
            
            if flattened_data:
                return pd.DataFrame(flattened_data)
            
            raise ValueError(f"Unsupported JSON structure in {file_path}")
    
    def _load_csv_dataset(self, file_path: Path) -> pd.DataFrame:
        """加载CSV格式数据集"""
        return pd.read_csv(file_path)
    
    def _load_jsonl_dataset(self, file_path: Path) -> pd.DataFrame:
        """加载JSONL格式数据集"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return pd.DataFrame(data)
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有配置中的数据集
        
        Returns:
            字典，键为数据集名称，值为DataFrame
        """
        datasets = {}
        for dataset_config in self.datasets_config:
            try:
                dataset_name = dataset_config['name']
                logger.info(f"Loading dataset: {dataset_name}")
                datasets[dataset_name] = self.load_dataset(dataset_name)
                logger.info(f"Loaded {len(datasets[dataset_name])} examples from {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_config['name']}: {e}")
        
        return datasets
    
    def load_expert_library(self, expert_type: str) -> pd.DataFrame:
        """
        加载指定类型的专家库
        
        Args:
            expert_type: 专家类型 ('short_chain', 'medium_chain', 'long_chain')
            
        Returns:
            DataFrame包含专家示例
        """
        library_path = self.expert_libraries_path / f"{expert_type}_library.csv"
        if not library_path.exists():
            logger.warning(f"Expert library for {expert_type} not found at {library_path}")
            return pd.DataFrame()
        
        return pd.read_csv(library_path)
    
    def save_expert_library(self, expert_type: str, library_df: pd.DataFrame) -> None:
        """
        保存专家库到CSV文件
        
        Args:
            expert_type: 专家类型
            library_df: 专家库DataFrame
        """
        library_path = self.expert_libraries_path / f"{expert_type}_library.csv"
        library_df.to_csv(library_path, index=False)
        logger.info(f"Saved {len(library_df)} examples to expert library: {library_path}")
    
    def load_processed_data(self, dataset_name: str) -> pd.DataFrame:
        """
        加载处理后的数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            处理后的DataFrame
        """
        processed_path = self.processed_path / f"{dataset_name}.csv"
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data not found: {processed_path}")
        
        return pd.read_csv(processed_path)
    
    def save_processed_data(self, dataset_name: str, df: pd.DataFrame) -> None:
        """
        保存处理后的数据集
        
        Args:
            dataset_name: 数据集名称
            df: 处理后的DataFrame
        """
        processed_path = self.processed_path / f"{dataset_name}.csv"
        df.to_csv(processed_path, index=False)
        logger.info(f"Saved processed data to {processed_path}")