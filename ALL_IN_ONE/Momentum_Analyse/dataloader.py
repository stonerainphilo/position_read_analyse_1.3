import numpy as np
import pandas as pd
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import h5py
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')
class BlockDataLoader:
    """
    为LLP计算设计的块数据加载器
    支持按需加载和抽样
    """
    
    def __init__(self, blocks_dir: str):
        self.blocks_dir = Path(blocks_dir)
        self.index = self._load_index()
        
    def _load_index(self) -> pd.DataFrame:
        """加载全局索引"""
        index_file = self.blocks_dir / 'global_index.parquet'
        if index_file.exists():
            return pd.read_parquet(index_file)
        else:
            # 从JSON加载
            json_file = self.blocks_dir / 'global_index.json'
            with open(json_file, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data['block_index']).T
    
    def get_blocks_by_momentum(self, 
                              min_energy: float = None,
                              max_energy: float = None,
                              momentum_direction: Tuple[float, float, float] = None,
                              tolerance: float = 0.1) -> List[str]:
        """
        根据动量条件筛选块
        
        Args:
            min_energy: 最小能量
            max_energy: 最大能量
            momentum_direction: 动量方向向量
            tolerance: 方向容差
            
        Returns:
            符合条件的块ID列表
        """
        filtered_blocks = []
        
        for _, row in self.index.iterrows():
            # 能量筛选
            if min_energy is not None and row['momentum_mean_e'] < min_energy:
                continue
            if max_energy is not None and row['momentum_mean_e'] > max_energy:
                continue
            
            # 方向筛选
            if momentum_direction is not None:
                block_momentum = np.array([row['momentum_mean_px'], 
                                          row['momentum_mean_py'], 
                                          row['momentum_mean_pz']])
                block_direction = block_momentum / np.linalg.norm(block_momentum)
                target_direction = np.array(momentum_direction)
                target_direction = target_direction / np.linalg.norm(target_direction)
                
                dot_product = np.dot(block_direction, target_direction)
                if dot_product < (1 - tolerance):
                    continue
            
            filtered_blocks.append(row['block_id'])
        
        return filtered_blocks
    
    def sample_from_block(self, 
                         block_id: str, 
                         n_samples: int = 1000,
                         strategy: str = 'random') -> pd.DataFrame:
        """
        从块中抽样粒子
        
        Args:
            block_id: 块ID
            n_samples: 抽样数量
            strategy: 抽样策略 ('random', 'importance', 'stratified')
            
        Returns:
            抽样粒子数据
        """
        # 加载块数据
        block_file = self.blocks_dir / 'blocks' / block_id / 'particles.parquet'
        if not block_file.exists():
            block_file = self.blocks_dir / 'blocks' / block_id / 'particles.h5'
            if not block_file.exists():
                raise FileNotFoundError(f"Block data not found: {block_id}")
        
        # 加载数据
        if block_file.suffix == '.parquet':
            block_data = pd.read_parquet(block_file)
        else:  # HDF5
            with h5py.File(block_file, 'r') as f:
                data_dict = {col: f[col][:] for col in f.keys()}
                block_data = pd.DataFrame(data_dict)
        
        # 抽样
        if n_samples >= len(block_data):
            return block_data
        
        if strategy == 'random':
            # 随机抽样
            sampled = block_data.sample(n=n_samples, random_state=42)
        
        elif strategy == 'importance':
            # 重要性抽样：基于动量大小
            momentum_magnitude = np.sqrt(block_data['px']**2 + 
                                        block_data['py']**2 + 
                                        block_data['pz']**2)
            weights = momentum_magnitude / momentum_magnitude.sum()
            sampled_indices = np.random.choice(
                len(block_data), 
                size=n_samples, 
                replace=False, 
                p=weights
            )
            sampled = block_data.iloc[sampled_indices]
        
        elif strategy == 'stratified':
            # 分层抽样：按能量分层
            n_strata = min(5, len(block_data) // 20)
            if n_strata < 2:
                sampled = block_data.sample(n=n_samples, random_state=42)
            else:
                block_data['energy_stratum'] = pd.qcut(block_data['e'], n_strata, labels=False)
                sampled = block_data.groupby('energy_stratum').apply(
                    lambda x: x.sample(n=min(n_samples // n_strata, len(x)), random_state=42)
                ).reset_index(drop=True)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        return sampled
    
    def get_block_statistics(self, block_id: str) -> Dict:
        """获取块的统计信息"""
        stats_file = self.blocks_dir / 'blocks' / block_id / 'statistics.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {}