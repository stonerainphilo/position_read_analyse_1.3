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
# class BlockDataLoader:
#     """
#     为LLP计算设计的块数据加载器
#     支持按需加载和抽样
#     """
    
#     def __init__(self, blocks_dir: str):
#         self.blocks_dir = Path(blocks_dir)
#         self.index = self._load_index()
        
#     def _load_index(self) -> pd.DataFrame:
#         """加载全局索引"""
#         index_file = self.blocks_dir / 'global_index.parquet'
#         if index_file.exists():
#             return pd.read_parquet(index_file)
#         else:
#             # 从JSON加载
#             json_file = self.blocks_dir / 'global_index.json'
#             with open(json_file, 'r') as f:
#                 data = json.load(f)
#             return pd.DataFrame(data['block_index']).T
    
#     def get_blocks_by_momentum(self, 
#                               min_energy: float = None,
#                               max_energy: float = None,
#                               momentum_direction: Tuple[float, float, float] = None,
#                               tolerance: float = 0.1) -> List[str]:
#         """
#         根据动量条件筛选块
        
#         Args:
#             min_energy: 最小能量
#             max_energy: 最大能量
#             momentum_direction: 动量方向向量
#             tolerance: 方向容差
            
#         Returns:
#             符合条件的块ID列表
#         """
#         filtered_blocks = []
        
#         for _, row in self.index.iterrows():
#             # 能量筛选
#             if min_energy is not None and row['momentum_mean_e'] < min_energy:
#                 continue
#             if max_energy is not None and row['momentum_mean_e'] > max_energy:
#                 continue
            
#             # 方向筛选
#             if momentum_direction is not None:
#                 block_momentum = np.array([row['momentum_mean_px'], 
#                                           row['momentum_mean_py'], 
#                                           row['momentum_mean_pz']])
#                 block_direction = block_momentum / np.linalg.norm(block_momentum)
#                 target_direction = np.array(momentum_direction)
#                 target_direction = target_direction / np.linalg.norm(target_direction)
                
#                 dot_product = np.dot(block_direction, target_direction)
#                 if dot_product < (1 - tolerance):
#                     continue
            
#             filtered_blocks.append(row['block_id'])
        
#         return filtered_blocks
    
#     def sample_from_block(self, 
#                          block_id: str, 
#                          n_samples: int = 1000,
#                          strategy: str = 'random') -> pd.DataFrame:
#         """
#         从块中抽样粒子
        
#         Args:
#             block_id: 块ID
#             n_samples: 抽样数量
#             strategy: 抽样策略 ('random', 'importance', 'stratified')
            
#         Returns:
#             抽样粒子数据
#         """
#         # 加载块数据
#         block_file = self.blocks_dir / 'blocks' / block_id / 'particles.parquet'
#         if not block_file.exists():
#             block_file = self.blocks_dir / 'blocks' / block_id / 'particles.h5'
#             if not block_file.exists():
#                 raise FileNotFoundError(f"Block data not found: {block_id}")
        
#         # 加载数据
#         if block_file.suffix == '.parquet':
#             block_data = pd.read_parquet(block_file)
#         else:  # HDF5
#             with h5py.File(block_file, 'r') as f:
#                 data_dict = {col: f[col][:] for col in f.keys()}
#                 block_data = pd.DataFrame(data_dict)
        
#         # 抽样
#         if n_samples >= len(block_data):
#             return block_data
        
#         if strategy == 'random':
#             # 随机抽样
#             sampled = block_data.sample(n=n_samples, random_state=42)
        
#         elif strategy == 'importance':
#             # 重要性抽样：基于动量大小
#             momentum_magnitude = np.sqrt(block_data['px']**2 + 
#                                         block_data['py']**2 + 
#                                         block_data['pz']**2)
#             weights = momentum_magnitude / momentum_magnitude.sum()
#             sampled_indices = np.random.choice(
#                 len(block_data), 
#                 size=n_samples, 
#                 replace=False, 
#                 p=weights
#             )
#             sampled = block_data.iloc[sampled_indices]
        
#         elif strategy == 'stratified':
#             # 分层抽样：按能量分层
#             n_strata = min(5, len(block_data) // 20)
#             if n_strata < 2:
#                 sampled = block_data.sample(n=n_samples, random_state=42)
#             else:
#                 block_data['energy_stratum'] = pd.qcut(block_data['e'], n_strata, labels=False)
#                 sampled = block_data.groupby('energy_stratum').apply(
#                     lambda x: x.sample(n=min(n_samples // n_strata, len(x)), random_state=42)
#                 ).reset_index(drop=True)
        
#         else:
#             raise ValueError(f"Unknown sampling strategy: {strategy}")
        
#         return sampled
    
#     def get_block_statistics(self, block_id: str) -> Dict:
#         """获取块的统计信息"""
#         stats_file = self.blocks_dir / 'blocks' / block_id / 'statistics.json'
#         if stats_file.exists():
#             with open(stats_file, 'r') as f:
#                 return json.load(f)
#         return {}
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
            
            # 转换JSON数据为DataFrame
            index_data = []
            for block_id, info in data.get('block_index', {}).items():
                info['block_id'] = block_id
                index_data.append(info)
            
            return pd.DataFrame(index_data) if index_data else pd.DataFrame()
    
    def sample_from_block(self, 
                         block_id: str, 
                         n_samples: int = 100,
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
        # 查找块数据文件
        block_dir = self.blocks_dir / 'blocks' / block_id
        
        # 查找数据文件
        h5_file = block_dir / 'particles.h5'
        parquet_file = block_dir / 'particles.parquet'
        csv_file = block_dir / 'particles.csv.gz'
        
        # 加载数据
        if parquet_file.exists():
            block_data = pd.read_parquet(parquet_file)
        elif h5_file.exists():
            # 修正HDF5加载代码
            try:
                with h5py.File(h5_file, 'r') as f:
                    # 获取所有数据集的键
                    keys = list(f.keys())
                    
                    # 创建字典存储数据
                    data_dict = {}
                    for key in keys:
                        # 跳过元数据组
                        if key == 'metadata':
                            continue
                        # 确保键是字符串
                        if isinstance(key, str):
                            data_dict[key] = f[key][:]
                        elif isinstance(key, bytes):
                            data_dict[key.decode('utf-8')] = f[key][:]
                    
                    block_data = pd.DataFrame(data_dict)
            except Exception as e:
                raise ValueError(f"Error loading HDF5 file {h5_file}: {e}")
        elif csv_file.exists():
            block_data = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError(f"No particle data found for block {block_id}")
        
        # 如果数据为空或抽样数大于数据量，返回所有数据
        if len(block_data) == 0:
            return pd.DataFrame()
        if n_samples >= len(block_data):
            return block_data
        
        # 抽样
        if strategy == 'random':
            # 随机抽样
            sampled = block_data.sample(n=min(n_samples, len(block_data)), 
                                       random_state=42, replace=False)
        
        elif strategy == 'importance':
            # 重要性抽样：基于动量大小
            if 'px' in block_data.columns and 'py' in block_data.columns and 'pz' in block_data.columns:
                momentum_magnitude = np.sqrt(block_data['px']**2 + 
                                            block_data['py']**2 + 
                                            block_data['pz']**2)
                weights = momentum_magnitude / momentum_magnitude.sum()
                
                # 确保权重和为1
                weights = weights.fillna(0)
                if weights.sum() == 0:
                    weights = np.ones(len(block_data)) / len(block_data)
                
                sampled_indices = np.random.choice(
                    len(block_data), 
                    size=min(n_samples, len(block_data)), 
                    replace=False, 
                    p=weights
                )
                sampled = block_data.iloc[sampled_indices].reset_index(drop=True)
            else:
                # 如果没有动量信息，使用随机抽样
                sampled = block_data.sample(n=min(n_samples, len(block_data)), 
                                           random_state=42, replace=False)
        
        elif strategy == 'stratified':
            # 分层抽样：按能量分层
            if 'e' in block_data.columns:
                n_strata = min(5, len(block_data) // 20)
                if n_strata < 2:
                    sampled = block_data.sample(n=min(n_samples, len(block_data)), 
                                               random_state=42, replace=False)
                else:
                    # 使用分位数分层
                    block_data = block_data.copy()
                    block_data['energy_stratum'] = pd.qcut(block_data['e'], 
                                                          n_strata, 
                                                          labels=False, 
                                                          duplicates='drop')
                    
                    # 确保每层都有标签
                    if block_data['energy_stratum'].nunique() < 2:
                        sampled = block_data.sample(n=min(n_samples, len(block_data)), 
                                                   random_state=42, replace=False)
                    else:
                        sampled = block_data.groupby('energy_stratum').apply(
                            lambda x: x.sample(n=min(n_samples // n_strata, len(x)), 
                                             random_state=42, replace=False)
                        ).reset_index(drop=True)
            else:
                sampled = block_data.sample(n=min(n_samples, len(block_data)), 
                                           random_state=42, replace=False)
        
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
    
    def get_all_blocks(self) -> List[str]:
        """获取所有可用的块ID"""
        blocks_dir = self.blocks_dir / 'blocks'
        if blocks_dir.exists():
            return [d.name for d in blocks_dir.iterdir() if d.is_dir()]
        return []
    
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
        if self.index.empty:
            return []
        
        filtered_blocks = []
        
        for _, row in self.index.iterrows():
            # 检查必要的列是否存在
            if 'momentum_mean_e' not in row:
                continue
                
            # 能量筛选
            if min_energy is not None and row['momentum_mean_e'] < min_energy:
                continue
            if max_energy is not None and row['momentum_mean_e'] > max_energy:
                continue
            
            # 方向筛选
            if momentum_direction is not None:
                required_cols = ['momentum_mean_px', 'momentum_mean_py', 'momentum_mean_pz']
                if not all(col in row for col in required_cols):
                    continue
                    
                block_momentum = np.array([row['momentum_mean_px'], 
                                          row['momentum_mean_py'], 
                                          row['momentum_mean_pz']])
                
                # 避免零动量
                if np.linalg.norm(block_momentum) == 0:
                    continue
                    
                block_direction = block_momentum / np.linalg.norm(block_momentum)
                target_direction = np.array(momentum_direction)
                
                # 避免零向量
                if np.linalg.norm(target_direction) == 0:
                    continue
                    
                target_direction = target_direction / np.linalg.norm(target_direction)
                
                dot_product = np.dot(block_direction, target_direction)
                if dot_product < (1 - tolerance):
                    continue
            
            filtered_blocks.append(row['block_id'])
        
        return filtered_blocks