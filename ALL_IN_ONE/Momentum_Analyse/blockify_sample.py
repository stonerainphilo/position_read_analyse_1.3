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

@dataclass
class BlockConfig:
    """分块配置参数"""
    # 动量空间分块
    momentum_n_clusters: int = 5000  # 动量聚类数量
    momentum_features: List[str] = None  # 用于聚类的动量特征
    
    # 位置空间分块
    position_bin_size: float = 100.0  # 位置分块大小(mm)
    use_position_clustering: bool = False  # 是否使用位置聚类
    position_n_clusters: int = 5  # 位置聚类数量
    
    # 存储配置
    max_particles_per_file: int = 10000  # 每个文件最大粒子数
    compression: str = 'gzip'  # 压缩格式
    
    def __post_init__(self):
        if self.momentum_features is None:
            self.momentum_features = ['px', 'py', 'pz', 'e']

@dataclass
class ParticleBlock:
    """粒子块数据结构"""
    block_id: str  # 块ID，格式: "pos_X_Y_Z_mom_K"
    particles: pd.DataFrame  # 该块所有粒子数据
    
    # 块统计信息
    position_stats: Dict  # 位置统计
    momentum_stats: Dict  # 动量统计
    metadata: Dict  # 元数据
    
    def __post_init__(self):
        """计算块的统计信息"""
        if self.particles.empty:
            return
            
        # 位置统计
        pos_cols = ['decay_x', 'decay_y', 'decay_z']
        if all(col in self.particles.columns for col in pos_cols):
            pos_data = self.particles[pos_cols]
            self.position_stats = {
                'mean': pos_data.mean().tolist(),
                'std': pos_data.std().tolist(),
                'min': pos_data.min().tolist(),
                'max': pos_data.max().tolist(),
                'covariance': pos_data.cov().values.tolist(),
                'histogram': self._compute_histogram(pos_data, bins=20)
            }
        
        # 动量统计
        mom_cols = ['px', 'py', 'pz', 'e']
        if all(col in self.particles.columns for col in mom_cols):
            mom_data = self.particles[mom_cols]
            self.momentum_stats = {
                'mean': mom_data.mean().tolist(),
                'std': mom_data.std().tolist(),
                'min': mom_data.min().tolist(),
                'max': mom_data.max().tolist(),
                'covariance': mom_data.cov().values.tolist(),
                'energy_histogram': self._compute_histogram(mom_data[['e']], bins=20),
                'momentum_magnitude': np.sqrt(np.sum(mom_data[['px', 'py', 'pz']]**2, axis=1)).tolist()
            }
    
    def _compute_histogram(self, data: pd.DataFrame, bins: int = 20) -> Dict:
        """计算直方图"""
        histograms = {}
        for col in data.columns:
            hist, edges = np.histogram(data[col], bins=bins)
            histograms[col] = {
                'counts': hist.tolist(),
                'edges': edges.tolist(),
                'density': (hist / (np.diff(edges) * hist.sum())).tolist() if hist.sum() > 0 else hist.tolist()
            }
        return histograms
    
    def to_dict(self) -> Dict:
        """转换为字典，便于序列化"""
        return {
            'block_id': self.block_id,
            'particle_count': len(self.particles),
            'position_stats': self.position_stats,
            'momentum_stats': self.momentum_stats,
            'metadata': self.metadata
        }

class HierarchicalParticleBlocking:
    """
    分层粒子分块系统
    按四动量聚类分块，同时完整保存位置分布
    """
    
    def __init__(self, 
                 data_path: Union[str, Path, pd.DataFrame],
                 output_dir: str = './particle_blocks',
                 config: Optional[BlockConfig] = None):
        """
        初始化分块系统
        
        Args:
            data_path: 粒子数据路径或DataFrame
            output_dir: 输出目录
            config: 分块配置
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置
        self.config = config or BlockConfig()
        
        # 加载数据
        if isinstance(data_path, (str, Path)):
            print(f"Loading data from {data_path}...")
            # 自动检测文件格式
            if str(data_path).endswith('.csv') or str(data_path).endswith('.csv.gz'):
                self.data = pd.read_csv(data_path)
            elif str(data_path).endswith('.parquet'):
                self.data = pd.read_parquet(data_path)
            elif str(data_path).endswith('.h5'):
                with h5py.File(data_path, 'r') as f:
                    self.data = pd.DataFrame(f['particles'][:])
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        elif isinstance(data_path, pd.DataFrame):
            self.data = data_path.copy()
        else:
            raise ValueError("data_path must be a file path or DataFrame")
        
        print(f"Loaded {len(self.data)} particles")
        print(f"Data columns: {list(self.data.columns)}")
        
        # 验证必需的列
        required_columns = ['decay_x', 'decay_y', 'decay_z', 'px', 'py', 'pz', 'e']
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # 初始化存储
        self.blocks: Dict[str, ParticleBlock] = {}
        self.block_index = {}
        self.global_stats = {}
        
    def create_momentum_clusters(self) -> np.ndarray:
        """
        按四动量进行聚类
        Returns: 聚类标签数组
        """
        print("Creating momentum clusters...")
        
        # 提取动量特征
        momentum_data = self.data[self.config.momentum_features].values
        
        # 标准化动量数据
        momentum_mean = momentum_data.mean(axis=0)
        momentum_std = momentum_data.std(axis=0)
        momentum_std[momentum_std == 0] = 1  # 避免除零
        
        momentum_normalized = (momentum_data - momentum_mean) / momentum_std
        
        # 使用MiniBatchKMeans处理大数据
        if len(momentum_data) > 10000:
            kmeans = MiniBatchKMeans(
                n_clusters=self.config.momentum_n_clusters,
                batch_size=1000,
                random_state=42,
                n_init=3
            )
        else:
            kmeans = KMeans(
                n_clusters=self.config.momentum_n_clusters,
                random_state=42,
                n_init=10
            )
        
        momentum_labels = kmeans.fit_predict(momentum_normalized)
        
        # 保存聚类中心
        self.momentum_cluster_centers = kmeans.cluster_centers_ * momentum_std + momentum_mean
        
        print(f"Created {self.config.momentum_n_clusters} momentum clusters")
        
        # 计算每个聚类的统计信息
        for i in range(self.config.momentum_n_clusters):
            cluster_mask = (momentum_labels == i)
            cluster_size = cluster_mask.sum()
            print(f"  Cluster {i}: {cluster_size} particles ({cluster_size/len(self.data):.1%})")
        
        return momentum_labels
    
    def create_position_blocks(self, momentum_labels: np.ndarray) -> Dict[str, pd.DataFrame]:
        """
        在每个动量聚类内，按位置创建分块
        
        Args:
            momentum_labels: 动量聚类标签
            
        Returns:
            分块数据字典 {block_id: particles_dataframe}
        """
        print("\nCreating position blocks within momentum clusters...")
        
        # 添加动量标签到数据
        self.data['momentum_cluster'] = momentum_labels
        
        # 初始化块字典
        blocks_dict = {}
        block_counter = 0
        
        # 对每个动量聚类
        for mom_cluster in range(self.config.momentum_n_clusters):
            cluster_data = self.data[self.data['momentum_cluster'] == mom_cluster]
            
            if len(cluster_data) == 0:
                continue
            
            print(f"\nProcessing momentum cluster {mom_cluster} ({len(cluster_data)} particles)")
            
            if self.config.use_position_clustering:
                # 使用位置聚类
                pos_data = cluster_data[['decay_x', 'decay_y', 'decay_z']].values
                
                # 标准化位置数据
                pos_mean = pos_data.mean(axis=0)
                pos_std = pos_data.std(axis=0)
                pos_std[pos_std == 0] = 1
                pos_normalized = (pos_data - pos_mean) / pos_std
                
                # 位置聚类
                pos_kmeans = KMeans(
                    n_clusters=min(self.config.position_n_clusters, len(cluster_data)),
                    random_state=42
                )
                position_labels = pos_kmeans.fit_predict(pos_normalized)
                
                # 创建位置块
                for pos_cluster in np.unique(position_labels):
                    block_mask = (position_labels == pos_cluster)
                    block_data = cluster_data.iloc[block_mask].copy()
                    
                    if len(block_data) > 0:
                        block_id = f"mom_{mom_cluster:03d}_pos_{pos_cluster:03d}"
                        blocks_dict[block_id] = block_data
                        block_counter += 1
                        
                        print(f"  Position cluster {pos_cluster}: {len(block_data)} particles")
            
            else:
                # 使用规则网格分块
                pos_data = cluster_data[['decay_x', 'decay_y', 'decay_z']]
                
                # 计算网格范围
                pos_min = pos_data.min()
                pos_max = pos_data.max()
                pos_range = pos_max - pos_min
                
                # 计算每个维度的网格数
                n_bins = np.ceil(pos_range / self.config.position_bin_size).astype(int)
                n_bins = np.maximum(n_bins, 1)  # 至少1个bin
                
                # 创建网格标签
                for i, coord in enumerate(['decay_x', 'decay_y', 'decay_z']):
                    bins = np.linspace(pos_min[i], pos_max[i], n_bins[i] + 1)
                    cluster_data[f'{coord}_bin'] = pd.cut(cluster_data[coord], bins, labels=False)
                
                # 遍历所有网格单元
                for x_bin in range(n_bins[0]):
                    for y_bin in range(n_bins[1]):
                        for z_bin in range(n_bins[2]):
                            mask = (
                                (cluster_data['decay_x_bin'] == x_bin) &
                                (cluster_data['decay_y_bin'] == y_bin) &
                                (cluster_data['decay_z_bin'] == z_bin)
                            )
                            
                            block_data = cluster_data[mask].copy()
                            
                            if len(block_data) > 0:
                                block_id = f"mom_{mom_cluster:03d}_pos_{x_bin:03d}_{y_bin:03d}_{z_bin:03d}"
                                blocks_dict[block_id] = block_data
                                block_counter += 1
        
        print(f"\nCreated {block_counter} total blocks")
        return blocks_dict
    
    def create_blocks(self) -> Dict[str, ParticleBlock]:
        """
        创建分块系统的主函数
        """
        print("=" * 60)
        print("Starting hierarchical particle blocking")
        print("=" * 60)
        
        # 步骤1: 动量聚类
        momentum_labels = self.create_momentum_clusters()
        
        # 步骤2: 位置分块
        blocks_dict = self.create_position_blocks(momentum_labels)
        
        # 步骤3: 创建ParticleBlock对象
        print("\nCreating ParticleBlock objects...")
        total_particles = 0
        
        for block_id, block_data in blocks_dict.items():
            # 移除临时列
            for col in block_data.columns:
                if col.endswith('_bin') or col == 'momentum_cluster':
                    block_data = block_data.drop(columns=[col])
            
            # 创建ParticleBlock
            particle_block = ParticleBlock(
                block_id=block_id,
                particles=block_data,
                position_stats={},
                momentum_stats={},
                metadata={
                    'creation_time': pd.Timestamp.now().isoformat(),
                    'original_particle_count': len(block_data),
                    'momentum_features': self.config.momentum_features.copy()
                }
            )
            
            self.blocks[block_id] = particle_block
            total_particles += len(block_data)
            
            # 更新块索引
            self.block_index[block_id] = {
                'particle_count': len(block_data),
                'position_mean': particle_block.position_stats.get('mean', [0, 0, 0]),
                'momentum_mean': particle_block.momentum_stats.get('mean', [0, 0, 0, 0]),
                'file_path': f"blocks/{block_id}/particles.h5"
            }
        
        # 计算全局统计
        self._compute_global_stats()
        
        print(f"\nTotal blocks created: {len(self.blocks)}")
        print(f"Total particles in blocks: {total_particles} ({(total_particles/len(self.data)):.1%} of original)")
        print("=" * 60)
        
        return self.blocks
    
    def _compute_global_stats(self):
        """计算全局统计信息"""
        print("Computing global statistics...")
        
        # 收集所有块的统计信息
        all_position_means = []
        all_momentum_means = []
        all_particle_counts = []
        
        for block_id, block in self.blocks.items():
            if block.position_stats and 'mean' in block.position_stats:
                all_position_means.append(block.position_stats['mean'])
            if block.momentum_stats and 'mean' in block.momentum_stats:
                all_momentum_means.append(block.momentum_stats['mean'])
            all_particle_counts.append(len(block.particles))
        
        if all_position_means:
            all_position_means = np.array(all_position_means)
            self.global_stats['position_mean'] = all_position_means.mean(axis=0).tolist()
            self.global_stats['position_std'] = all_position_means.std(axis=0).tolist()
        
        if all_momentum_means:
            all_momentum_means = np.array(all_momentum_means)
            self.global_stats['momentum_mean'] = all_momentum_means.mean(axis=0).tolist()
            self.global_stats['momentum_std'] = all_momentum_means.std(axis=0).tolist()
        
        self.global_stats['total_blocks'] = len(self.blocks)
        self.global_stats['total_particles'] = sum(all_particle_counts)
        self.global_stats['avg_particles_per_block'] = np.mean(all_particle_counts)
        self.global_stats['std_particles_per_block'] = np.std(all_particle_counts)
    
    def save_blocks(self, format: str = 'hdf5'):
        """
        保存分块数据到磁盘
        
        Args:
            format: 存储格式 ('hdf5', 'parquet', 'csv')
        """
        print(f"\nSaving blocks to {self.output_dir}...")
        
        # 创建目录结构
        blocks_dir = self.output_dir / 'blocks'
        blocks_dir.mkdir(exist_ok=True)
        
        # 保存每个块
        for block_id, block in self.blocks.items():
            block_dir = blocks_dir / block_id
            block_dir.mkdir(exist_ok=True)
            
            # 保存粒子数据
            if format == 'hdf5':
                self._save_hdf5(block_dir, block)
            elif format == 'parquet':
                self._save_parquet(block_dir, block)
            elif format == 'csv':
                self._save_csv(block_dir, block)
            
            # 保存块统计信息
            stats_file = block_dir / 'statistics.json'
            with open(stats_file, 'w') as f:
                json.dump(block.to_dict(), f, indent=2)
        
        # 保存全局索引
        self._save_global_index()
        
        # 保存配置
        config_file = self.output_dir / 'blocking_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"Blocks saved successfully to {self.output_dir}")
    
    def _save_hdf5(self, block_dir: Path, block: ParticleBlock):
        """保存为HDF5格式"""
        file_path = block_dir / 'particles.h5'
        with h5py.File(file_path, 'w') as f:
            # 保存粒子数据
            for col in block.particles.columns:
                f.create_dataset(col, data=block.particles[col].values, compression='gzip')
            
            # 保存元数据
            metadata_group = f.create_group('metadata')
            for key, value in block.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata_group.attrs[key] = value
    
    def _save_parquet(self, block_dir: Path, block: ParticleBlock):
        """保存为Parquet格式"""
        file_path = block_dir / 'particles.parquet'
        block.particles.to_parquet(file_path, compression='gzip')
    
    def _save_csv(self, block_dir: Path, block: ParticleBlock):
        """保存为CSV格式"""
        file_path = block_dir / 'particles.csv.gz'
        block.particles.to_csv(file_path, index=False, compression='gzip')
    
    def _save_global_index(self):
        """保存全局索引"""
        index_data = {
            'block_index': self.block_index,
            'global_stats': self.global_stats,
            'momentum_cluster_centers': self.momentum_cluster_centers.tolist() if hasattr(self, 'momentum_cluster_centers') else [],
            'creation_time': pd.Timestamp.now().isoformat()
        }
        
        index_file = self.output_dir / 'global_index.json'
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        # 也保存为Parquet以便快速查询
        index_df = pd.DataFrame([
            {
                'block_id': block_id,
                'particle_count': info['particle_count'],
                'position_mean_x': info['position_mean'][0],
                'position_mean_y': info['position_mean'][1],
                'position_mean_z': info['position_mean'][2],
                'momentum_mean_px': info['momentum_mean'][0],
                'momentum_mean_py': info['momentum_mean'][1],
                'momentum_mean_pz': info['momentum_mean'][2],
                'momentum_mean_e': info['momentum_mean'][3],
                'file_path': info['file_path']
            }
            for block_id, info in self.block_index.items()
        ])
        
        index_df.to_parquet(self.output_dir / 'global_index.parquet', index=False)
    
    def load_block(self, block_id: str) -> ParticleBlock:
        """
        加载单个块
        
        Args:
            block_id: 块ID
            
        Returns:
            ParticleBlock对象
        """
        block_dir = self.output_dir / 'blocks' / block_id
        
        if not block_dir.exists():
            raise ValueError(f"Block {block_id} not found")
        
        # 加载粒子数据
        data_files = list(block_dir.glob('particles.*'))
        if not data_files:
            raise FileNotFoundError(f"No particle data found for block {block_id}")
        
        data_file = data_files[0]
        
        if data_file.suffix == '.h5':
            with h5py.File(data_file, 'r') as f:
                data_dict = {col: f[col][:] for col in f.keys() if col != 'metadata'}
                particles = pd.DataFrame(data_dict)
        elif data_file.suffix == '.parquet':
            particles = pd.read_parquet(data_file)
        elif data_file.suffix in ['.csv', '.gz']:
            particles = pd.read_csv(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")
        
        # 加载统计信息
        stats_file = block_dir / 'statistics.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        else:
            stats = {}
        
        # 重新创建ParticleBlock
        block = ParticleBlock(
            block_id=block_id,
            particles=particles,
            position_stats=stats.get('position_stats', {}),
            momentum_stats=stats.get('momentum_stats', {}),
            metadata=stats.get('metadata', {})
        )
        
        return block
    
    def get_block_summary(self) -> pd.DataFrame:
        """
        获取块摘要信息
        """
        summary_data = []
        
        for block_id, block in self.blocks.items():
            summary_data.append({
                'block_id': block_id,
                'particle_count': len(block.particles),
                'position_mean_x': block.position_stats.get('mean', [0, 0, 0])[0],
                'position_mean_y': block.position_stats.get('mean', [0, 0, 0])[1],
                'position_mean_z': block.position_stats.get('mean', [0, 0, 0])[2],
                'position_std_x': block.position_stats.get('std', [0, 0, 0])[0],
                'position_std_y': block.position_stats.get('std', [0, 0, 0])[1],
                'position_std_z': block.position_stats.get('std', [0, 0, 0])[2],
                'momentum_mean_px': block.momentum_stats.get('mean', [0, 0, 0, 0])[0],
                'momentum_mean_py': block.momentum_stats.get('mean', [0, 0, 0, 0])[1],
                'momentum_mean_pz': block.momentum_stats.get('mean', [0, 0, 0, 0])[2],
                'momentum_mean_e': block.momentum_stats.get('mean', [0, 0, 0, 0])[3],
                'momentum_std_px': block.momentum_stats.get('std', [0, 0, 0, 0])[0],
                'momentum_std_py': block.momentum_stats.get('std', [0, 0, 0, 0])[1],
                'momentum_std_pz': block.momentum_stats.get('std', [0, 0, 0, 0])[2],
                'momentum_std_e': block.momentum_stats.get('std', [0, 0, 0, 0])[3]
            })
        
        return pd.DataFrame(summary_data)

# 使用示例
if __name__ == "__main__":
    # 配置分块参数
    config = BlockConfig(
        momentum_n_clusters=3000,  # 8个动量聚类
        position_bin_size=50.0,  # 50mm的位置网格
        use_position_clustering=False,  # 使用规则网格
        compression='gzip'
    )
    
    # 创建分块系统
    blocker = HierarchicalParticleBlocking(
        data_path="/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Decay_B2025-12-03_2HDM_B_test/B_521_pos.csv",
        output_dir="/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks",
        config=config
    )
    
    # 创建分块
    blocks = blocker.create_blocks()
    
    # 保存分块数据
    blocker.save_blocks(format='csv')  # 或 'hdf5', 'csv'
    
    # 获取摘要信息
    summary = blocker.get_block_summary()
    print(f"\nBlock Summary:")
    print(f"Total blocks: {len(summary)}")
    print(f"Total particles: {summary['particle_count'].sum()}")
    print(f"Average particles per block: {summary['particle_count'].mean():.1f}")
    
    # 保存摘要
    summary.to_csv("/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/block_summary.csv", index=False)
    
    # 示例：加载特定块
    example_block_id = "mom_000_pos_000_000_000"
    try:
        loaded_block = blocker.load_block(example_block_id)
        print(f"\nLoaded block {example_block_id}:")
        print(f"  Particles: {len(loaded_block.particles)}")
        print(f"  Position mean: {loaded_block.position_stats.get('mean', 'N/A')}")
        print(f"  Momentum mean: {loaded_block.momentum_stats.get('mean', 'N/A')}")
    except Exception as e:
        print(f"Error loading block: {e}")