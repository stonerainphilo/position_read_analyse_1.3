import gc
import numpy as np
import pandas as pd
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numbers
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy import stats
import warnings
from decay_sim import two_body_decay_lab, generate_decay_position
from dataloader import BlockDataLoader
warnings.filterwarnings('ignore')


def _sanitize_for_json(obj):
    """Recursively convert numpy types, arrays and Paths to JSON-serializable types."""
    # numpy types
    try:
        import numpy as _np
    except Exception:
        _np = None

    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()
    if _np is not None and isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, numbers.Number):
        # convert numpy numbers to python numbers
        try:
            return float(obj) if isinstance(obj, float) or hasattr(obj, 'item') else int(obj)
        except Exception:
            return obj
    # fallback: return as-is (json will raise if unsupported)
    return obj

@dataclass
class LLPBlockConfig:
    """LLP衰变位置分块配置"""
    # 空间分块配置
    x_range: Tuple[float, float] = (26000, 36000)    # mm
    y_range: Tuple[float, float] = (-7000, 3000)    # mm
    z_range: Tuple[float, float] = (5000, 15000)        # mm
    nx: int = 100   # x方向网格数
    ny: int = 100   # y方向网格数
    nz: int = 200   # z方向网格数
    
    # 存储配置
    compression: str = 'gzip'
    store_full_positions: bool = False  # 是否存储完整的衰变位置
    max_positions_per_file: int = 100000
    
    # 分析配置
    target_region: Optional[Dict] = None  # 目标区域定义
    min_decays_in_region: int = 10     # 目标区域最小衰变数阈值
    
    def get_bin_edges(self, axis: str) -> np.ndarray:
        """获取指定轴的bin边界"""
        if axis == 'x':
            return np.linspace(self.x_range[0], self.x_range[1], self.nx + 1)
        elif axis == 'y':
            return np.linspace(self.y_range[0], self.y_range[1], self.ny + 1)
        elif axis == 'z':
            return np.linspace(self.z_range[0], self.z_range[1], self.nz + 1)
        else:
            raise ValueError(f"Invalid axis: {axis}")

@dataclass
class LLPBlock:
    """LLP衰变位置块数据"""
    block_id: str  # 格式: "llp_m{质量}_tb{tanb}_region{区域ID}"
    llp_params: Dict[str, float]  # LLP参数
    positions: np.ndarray  # 衰变位置 (n, 3)
    weights: np.ndarray  # 权重 (n,)
    
    # 块统计信息
    total_weighted_events: float
    spatial_hist: Optional[Dict[str, Any]] = None  # 空间直方图
    density_map: Optional[Dict[str, Any]] = None   # 密度图
    
    def compute_spatial_distribution(self, config: LLPBlockConfig):
        """计算空间分布统计"""
        if len(self.positions) == 0:
            return
        
        # 计算三维直方图
        x_edges = config.get_bin_edges('x')
        y_edges = config.get_bin_edges('y')
        z_edges = config.get_bin_edges('z')
        # print(f"Computing spatial histogram for block {self.block_id}...")
        # 加权三维直方图
        hist, _ = np.histogramdd(
            self.positions,
            bins=[x_edges, y_edges, z_edges],
            weights=self.weights
        )
        # print(f"  ✓ Histogram computed with shape {hist.shape}")
        
        # 计算体积
        dx = x_edges[1] - x_edges[0]
        dy = y_edges[1] - y_edges[0]
        dz = z_edges[1] - z_edges[0]
        volume = dx * dy * dz
        # print(f"  ✓ Voxel volume: {volume} mm^3")
        # 密度图（每个体素的事件数/体积）
        density = hist / volume
        # print(f"  ✓ Density map computed")
        self.spatial_hist = {
            'histogram': hist.astype(np.float32),
            'x_edges': x_edges.astype(np.float32),
            'y_edges': y_edges.astype(np.float32),
            'z_edges': z_edges.astype(np.float32),
            'total_counts': float(np.sum(hist))
        }
        # print(f"  ✓ Spatial histogram stored")
        
        self.density_map = {
            'density': density.astype(np.float32),
            'volume': float(volume),
            'mean_density': float(np.mean(density[density > 0])),
            'max_density': float(np.max(density))
        }
    
    def get_decays_in_region(self, region: Dict) -> float:
        """获取指定区域内的加权衰变数"""
        if len(self.positions) == 0:
            return 0.0
        
        # 区域定义: {'x_min':, 'x_max':, 'y_min':, 'y_max':, 'z_min':, 'z_max':}
        mask = (
            (self.positions[:, 0] >= region['x_min']) &
            (self.positions[:, 0] <= region['x_max']) &
            (self.positions[:, 1] >= region['y_min']) &
            (self.positions[:, 1] <= region['y_max']) &
            (self.positions[:, 2] >= region['z_min']) &
            (self.positions[:, 2] <= region['z_max'])
        )
        
        return float(np.sum(self.weights[mask]))
    
    def to_dict(self) -> Dict:
        """转换为字典便于序列化"""
        result = asdict(self)
        
        # 转换numpy数组为列表
        if 'positions' in result:
            result['positions'] = self.positions.tolist() if self.positions is not None else []
        if 'weights' in result:
            result['weights'] = self.weights.tolist() if self.weights is not None else []
        
        return result

class LLPDecayAnalyzer:
    """
    LLP衰变位置分析和分块存储系统
    """
    
    def __init__(self, 
                 output_dir: str = './llp_decay_blocks',
                 config: Optional[LLPBlockConfig] = None):
        """
        初始化LLP衰变分析器
        
        Args:
            output_dir: 输出目录
            config: 分块配置
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or LLPBlockConfig()
        
        # 存储结构
        self.llp_blocks: Dict[str, LLPBlock] = {}
        self.param_summary: pd.DataFrame = pd.DataFrame()
        self.region_analysis: Dict[str, Any] = {}
        
        # 创建子目录
        (self.output_dir / 'blocks').mkdir(exist_ok=True)
        (self.output_dir / 'statistics').mkdir(exist_ok=True)
        (self.output_dir / 'visualization').mkdir(exist_ok=True)
    
    def process_llp_decays(self,
                          llp_params: Dict[str, float],
                          decay_positions: np.ndarray,
                          weights: np.ndarray,
                          block_name_suffix: str = "") -> str:
        """
        处理单个LLP参数的衰变数据
        
        Args:
            llp_params: LLP参数字典 {'mass':, 'lifetime':, 'tanb':, ...}
            decay_positions: 衰变位置数组 (n, 3)
            weights: 权重数组 (n,)
            block_name_suffix: 块名后缀
            
        Returns:
            块ID
        """
        # 生成块ID
        mass = llp_params['mass']
        tanb = llp_params['tanb']
        block_id = f"llp_m{mass:.3f}_tb{tanb:.2f}{block_name_suffix}"
        
        # 创建LLP块
        # print(f"  ✓ Creating LLP block {block_id}")
        llp_block = LLPBlock(
            block_id=block_id,
            llp_params=llp_params,
            positions=decay_positions,
            weights=weights,
            total_weighted_events=float(np.sum(weights))
        )
        # print(f"Processing LLP Block: {block_id} with {len(decay_positions)} decay positions")
        # 计算空间分布
        llp_block.compute_spatial_distribution(self.config)
        print(f"  ✓ Computed spatial distribution for block {block_id}")
        # 存储块
        self.llp_blocks[block_id] = llp_block
        
        # 更新参数摘要
        self._update_param_summary(llp_block)
        
        # 分析目标区域
        if self.config.target_region:
            decays_in_region = llp_block.get_decays_in_region(self.config.target_region)
            self.region_analysis[block_id] = {
                'mass': mass,
                'tanb': tanb,
                'decays_in_region': decays_in_region,
                'meets_threshold': decays_in_region >= self.config.min_decays_in_region
            }
        
        return block_id
    
    def _update_param_summary(self, llp_block: LLPBlock):
        """更新参数摘要表"""
        summary_row = {
            'block_id': llp_block.block_id,
            'mass': llp_block.llp_params['mass'],
            'lifetime': llp_block.llp_params['lifetime'],
            'tanb': llp_block.llp_params['tanb'],
            'total_weighted_events': llp_block.total_weighted_events,
            'vis_br': llp_block.llp_params['vis_br'],
        }
        
        # 添加密度统计
        if llp_block.density_map:
            summary_row.update({
                'mean_density': llp_block.density_map['mean_density'],
                'max_density': llp_block.density_map['max_density']
            })
        
        # 添加目标区域统计
        if self.config.target_region and llp_block.block_id in self.region_analysis:
            region_info = self.region_analysis[llp_block.block_id]
            summary_row.update({
                'decays_in_target': region_info['decays_in_region'],
                'passes_threshold': region_info['meets_threshold']
            })
        
        new_df = pd.DataFrame([summary_row])
        self.param_summary = pd.concat([self.param_summary, new_df], ignore_index=True)
    
    def save_block(self, block_id: str):
        """保存单个LLP块"""
        if block_id not in self.llp_blocks:
            raise ValueError(f"Block {block_id} not found")
        
        block = self.llp_blocks[block_id]
        block_dir = self.output_dir / 'blocks' / block_id
        block_dir.mkdir(exist_ok=True)
        
        # 保存块数据
        with h5py.File(block_dir / 'data.h5', 'w') as f:
            # 保存衰变位置和权重
            f.create_dataset('positions', data=block.positions, compression=self.config.compression)
            f.create_dataset('weights', data=block.weights, compression=self.config.compression)
            
            # 保存参数
            params_group = f.create_group('parameters')
            for key, value in block.llp_params.items():
                params_group.attrs[key] = value
            
            # 保存统计
            if block.spatial_hist:
                hist_group = f.create_group('histogram')
                hist_group.create_dataset('histogram', data=block.spatial_hist['histogram'])
                hist_group.create_dataset('x_edges', data=block.spatial_hist['x_edges'])
                hist_group.create_dataset('y_edges', data=block.spatial_hist['y_edges'])
                hist_group.create_dataset('z_edges', data=block.spatial_hist['z_edges'])
                hist_group.attrs['total_counts'] = block.spatial_hist['total_counts']
        
        # 保存JSON摘要
        block_summary = block.to_dict()
        # 移除大数据字段
        block_summary.pop('positions', None)
        block_summary.pop('weights', None)
        
        with open(block_dir / 'summary.json', 'w') as f:
            json.dump(_sanitize_for_json(block_summary), f, indent=2)
        
        # print(f"Saved block {block_id}")
    
    def save_block_enhanced(self, block_id: str):
        """只压缩格式 100%兼容"""
        if block_id not in self.llp_blocks:
            raise ValueError(f"Block {block_id} not found")
        
        block = self.llp_blocks[block_id]
        block_dir = self.output_dir / 'blocks' / block_id
        block_dir.mkdir(exist_ok=True)
        
        # HDF5保存（原样）
        with h5py.File(block_dir / 'data.h5', 'w') as f:
            f.create_dataset('positions', data=block.positions, compression=self.config.compression)
            f.create_dataset('weights', data=block.weights, compression=self.config.compression)
            
            params_group = f.create_group('parameters')
            for key, value in block.llp_params.items():
                params_group.attrs[key] = value
            
            if block.spatial_hist:
                hist_group = f.create_group('histogram')
                hist_group.create_dataset('histogram', data=block.spatial_hist['histogram'])
                hist_group.create_dataset('x_edges', data=block.spatial_hist['x_edges'])
                hist_group.create_dataset('y_edges', data=block.spatial_hist['y_edges'])
                hist_group.create_dataset('z_edges', data=block.spatial_hist['z_edges'])
                hist_group.attrs['total_counts'] = block.spatial_hist['total_counts']
        
        # JSON保存：完整数据，只压缩格式
        block_summary = block.to_dict()
        block_summary.pop('positions', None)
        block_summary.pop('weights', None)
        
        # 不删除任何数据！只改变格式
        with open(block_dir / 'summary.json', 'w') as f:
            json.dump(_sanitize_for_json(block_summary), f, 
                    separators=(',', ':'),  # 移除空格
                    indent=None)            # 移除缩进
            

    def save_all_blocks(self):
        """保存所有LLP块"""
        print(f"Saving {len(self.llp_blocks)} LLP blocks...")
        for block_id in self.llp_blocks.keys():
            self.save_block(block_id)
    
    def save_all_blocks_enhanced(self):
        """保存所有LLP块 - 增强版本"""
        print(f"Saving {len(self.llp_blocks)} LLP blocks (enhanced)...")
        for block_id in tqdm(self.llp_blocks.keys(), desc="Saving blocks"):
            self.save_block_enhanced(block_id)
    
    def get_blocks_in_region(self, 
                            region: Optional[Dict] = None,
                            min_decays: Optional[float] = None) -> pd.DataFrame:
        """
        获取在指定区域内衰变数超过阈值的LLP块
        
        Args:
            region: 区域定义，None则使用配置中的目标区域
            min_decays: 最小衰变数，None则使用配置中的阈值
            
        Returns:
            符合条件的块摘要DataFrame
        """
        if region is None:
            region = self.config.target_region
        if min_decays is None:
            min_decays = self.config.min_decays_in_region
        
        if region is None:
            raise ValueError("No region specified")
        
        results = []
        
        for block_id, block in self.llp_blocks.items():
            decays_in_region = block.get_decays_in_region(region)
            
            if decays_in_region > min_decays:
                results.append({
                    'block_id': block_id,
                    'mass': block.llp_params['mass'],
                    'tanb': block.llp_params['tanb'],
                    'lifetime': block.llp_params['lifetime'],
                    'decays_in_region': decays_in_region,
                    'total_events': block.total_weighted_events,
                    'fraction_in_region': decays_in_region / block.total_weighted_events if block.total_weighted_events > 0 else 0
                })
        
        return pd.DataFrame(results)
    
    
    def save_analysis_summary(self):
        """保存分析摘要"""
        # 保存参数摘要
        if not self.param_summary.empty:
            summary_path = self.output_dir / 'statistics' / 'llp_parameters_summary.csv'
            self.param_summary.to_csv(summary_path, index=False)
            # print(f"Saved parameter summary to {summary_path}")
        
        # 保存区域分析
        if self.region_analysis:
            region_path = self.output_dir / 'statistics' / 'region_analysis.json'
            with open(region_path, 'w') as f:
                json.dump(_sanitize_for_json(self.region_analysis), f, indent=2)
            # print(f"Saved region analysis to {region_path}")
        
        # 保存配置
        config_path = self.output_dir / 'analysis_config.json'
        with open(config_path, 'w') as f:
            json.dump(_sanitize_for_json(asdict(self.config)), f, indent=2)
        # print(f"Saved analysis config to {config_path}")
    
    def load_block(self, block_id: str) -> LLPBlock:
        """加载已保存的LLP块"""
        block_dir = self.output_dir / 'blocks' / block_id
        
        if not block_dir.exists():
            raise ValueError(f"Block {block_id} not found")
        
        # 加载数据
        with h5py.File(block_dir / 'data.h5', 'r') as f:
            positions = f['positions'][:]
            weights = f['weights'][:]
            
            # 加载参数
            params = dict(f['parameters'].attrs)
            
            # 加载直方图
            if 'histogram' in f:
                hist_group = f['histogram']
                spatial_hist = {
                    'histogram': hist_group['histogram'][:],
                    'x_edges': hist_group['x_edges'][:],
                    'y_edges': hist_group['y_edges'][:],
                    'z_edges': hist_group['z_edges'][:],
                    'total_counts': hist_group.attrs['total_counts']
                }
            else:
                spatial_hist = None
        
        # 加载摘要
        with open(block_dir / 'summary.json', 'r') as f:
            summary = json.load(f)
        
        # 创建LLPBlock对象
        block = LLPBlock(
            block_id=block_id,
            llp_params=params,
            positions=positions,
            weights=weights,
            total_weighted_events=summary['total_weighted_events'],
            spatial_hist=spatial_hist,
            density_map=summary.get('density_map')
        )
        
        return block
    

class LLPDecaySimulationPipeline:
    """
    LLP衰变模拟的完整流程：
    1. 加载母粒子分块数据
    2. 对每个LLP参数模拟衰变
    3. 对衰变位置进行分块分析
    4. 筛选符合条件的参数
    5. 可视化结果
    """
    
    def __init__(self,
                 particle_blocks_dir: str,
                 llp_params_file: str,
                 output_dir: str = './llp_simulation_results',
                 decay_config: Optional[LLPBlockConfig] = None):
        """
        初始化模拟管道
        
        Args:
            particle_blocks_dir: 母粒子分块数据目录
            llp_params_file: LLP参数文件路径
            output_dir: 输出目录
            decay_config: LLP衰变分块配置
        """
        self.particle_blocks_dir = Path(particle_blocks_dir)
        self.llp_params_file = llp_params_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置
        self.decay_config = decay_config or LLPBlockConfig()
        
        # 加载母粒子块索引
        self.particle_loader = BlockDataLoader(str(particle_blocks_dir))
        
        # 加载LLP参数
        self.llp_params_df = pd.read_csv(llp_params_file)
        print(f"Loaded {len(self.llp_params_df)} LLP parameter sets")
        
        # 初始化衰变分析器
        self.analyzer = LLPDecayAnalyzer(
            output_dir=str(self.output_dir / 'decay_analysis'),
            config=self.decay_config
        )
        
        # 结果存储
        self.simulation_results = []
    def simulate_llp_decays_optimized(self,
                                 samples_per_block: int = 100,
                                 max_llp_params: Optional[int] = None,
                                 target_region: Optional[Dict] = None):
        """
        优化的LLP衰变模拟过程
        每个LLP参数单独保存文件，避免内存累积
        
        Args:
            samples_per_block: 每个母粒子块抽样数
            max_llp_params: 最大处理的LLP参数数（用于测试）
            target_region: 目标区域定义
        """
        if target_region:
            self.decay_config.target_region = target_region
    
        print("=" * 70)
        print("Optimized LLP Decay Simulation Pipeline")
        print("=" * 70)
        
        # 创建输出目录结构
        interim_dir = self.output_dir / 'interim_results'
        interim_dir.mkdir(exist_ok=True)
        
        # 获取所有母粒子块
        all_blocks = self.particle_loader.index['block_id'].tolist()
        print(f"Found {len(all_blocks)} mother particle blocks")
        
        # 限制处理的LLP参数数量（用于测试）
        if max_llp_params and max_llp_params < len(self.llp_params_df):
            llp_params_to_process = self.llp_params_df.head(max_llp_params)
            print(f"Processing first {max_llp_params} LLP parameters (testing mode)")
        else:
            llp_params_to_process = self.llp_params_df
        
        # 记录处理成功的参数
        processed_params = []
        failed_params = []
        
        # 处理每个LLP参数
        for idx, llp_row in tqdm(llp_params_to_process.iterrows(), 
                                total=len(llp_params_to_process),
                                desc="Processing LLP parameters"):
            
            llp_mass = llp_row['mH']
            llp_lifetime = llp_row['ltime']
            tanb = llp_row['tanb']
            
            print(f"\n{'='*40}")
            print(f"Processing LLP {idx+1}/{len(llp_params_to_process)}")
            print(f"Mass: {llp_mass:.3f} GeV, tanβ: {tanb:.2f}, Lifetime: {llp_lifetime:.2e} mm")
            print(f"{'='*40}")
            
            # 为每个LLP创建独立的分析器实例
            llp_analyzer = LLPDecayAnalyzer(
                output_dir=str(interim_dir / f'llp_{idx:04d}'),
                config=self.decay_config
            )
            
            # 收集所有衰变位置
            all_decay_positions = []
            all_weights = []
            processed_blocks = 0
            
            # 对每个母粒子块
            for block_id in all_blocks:
                try:
                    # 从块中抽样粒子
                    sampled_particles = self.particle_loader.sample_from_block(
                        block_id=block_id,
                        n_samples=samples_per_block,
                        strategy='importance'
                    )
                    
                    if len(sampled_particles) == 0:
                        continue
                    
                    processed_blocks += 1
                    
                    # 对每个抽样粒子计算LLP衰变
                    block_decay_positions = []
                    block_weights = []
                    
                    for _, particle in sampled_particles.iterrows():
                        # 检查必要的列
                        required_cols = ['decay_x', 'decay_y', 'decay_z', 'px', 'py', 'pz', 'e']
                        if not all(col in particle for col in required_cols):
                            continue
                        
                        try:
                            birth_position = particle[['decay_x', 'decay_y', 'decay_z']].values
                            momentum = particle[['e', 'px', 'py', 'pz']].values
                            
                            llp_momentum, _ = two_body_decay_lab(momentum, 5.279, llp_mass, 0.494)
                            decay_pos, _ = generate_decay_position(llp_lifetime, llp_momentum, birth_position)
                            
                            if not np.any(np.isnan(decay_pos)):
                                block_decay_positions.append(decay_pos)
                                # 权重 = 该块总粒子数 / 抽样数
                                block_weights.append(len(sampled_particles) / samples_per_block)
                                
                        except (ValueError, ZeroDivisionError) as e:
                            continue
                    
                    # 添加到总列表
                    if block_decay_positions:
                        all_decay_positions.extend(block_decay_positions)
                        all_weights.extend(block_weights)
                        # print(np.array(block_decay_positions))
                        
                except Exception as e:
                    print(f"  Warning: Error processing block {block_id}: {e}")
                    continue
            
            # 检查是否有有效的衰变
            if len(all_decay_positions) == 0:
                print(f"  ✗ No valid decays for mass={llp_mass}, tanb={tanb}")
                failed_params.append({
                    'index': idx,
                    'mass': llp_mass,
                    'tanb': tanb,
                    'lifetime': llp_lifetime,
                    'reason': 'No valid decays',
                    'processed_blocks': processed_blocks
                })
                
                # 清理内存
                del llp_analyzer
                gc.collect()
                continue
            
            # 转换为numpy数组
            decay_positions = np.array(all_decay_positions)
            weights = np.array(all_weights)
            
            # print(f"  ✓ Generated {len(decay_positions)} decay positions from {processed_blocks} blocks")
            # print(f"  ✓ Total weighted decays: {np.sum(weights):.0f}")
            
            # 准备LLP参数
            llp_params = {
                'mass': llp_mass,
                'lifetime': llp_lifetime,
                'tanb': tanb,
                'vis_br': llp_row.get('Br_visible', 0.0)
            }
            print(f"LLP is ready for decay position analysis.")
            try:
                # print(f"  ✓ Processing decay positions for mass={llp_mass}, tanb={tanb}")
                # 处理衰变数据
                block_id = llp_analyzer.process_llp_decays(
                    llp_params=llp_params,
                    decay_positions=decay_positions,
                    weights=weights,
                    block_name_suffix=f"_m{llp_mass:.3f}_tb{tanb:.2f}"
                )
                # print(f"  ✓ Processed decay positions for mass={llp_mass}, tanb={tanb}")
                # 保存该LLP的所有块
                llp_analyzer.save_all_blocks()
                llp_analyzer.save_analysis_summary()
                
                # 记录结果
                llp_result = {
                    'index': idx,
                    'mass': llp_mass,
                    'lifetime': llp_lifetime,
                    'tanb': tanb,
                    'vis_br': llp_row.get('Br_visible', 0.0),
                    'block_id': block_id,
                    'total_decays': float(np.sum(weights)),
                    'unique_positions': len(decay_positions),
                    # 'decay_x': decay_positions[0],
                    'processed_blocks': processed_blocks,
                    'output_dir': str(interim_dir / f'llp_{idx:04d}'),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                # 计算目标区域内的衰变数（如果定义了目标区域）
                if self.decay_config.target_region:
                    llp_block = llp_analyzer.llp_blocks.get(block_id)
                    if llp_block:
                        decays_in_region = llp_block.get_decays_in_region(self.decay_config.target_region)
                        llp_result['decays_in_target'] = decays_in_region
                        llp_result['fraction_in_target'] = decays_in_region / np.sum(weights) if np.sum(weights) > 0 else 0
                
                # 保存该LLP的单独结果文件
                result_file = interim_dir / f'llp_{idx:04d}_result.json'
                print(f"  ✓ Saving result to {result_file}")
                # Debug: report any ndarray fields before serialization
                try:
                    bad = {k: type(v).__name__ for k, v in llp_result.items() if isinstance(v, (np.ndarray,))}
                    if bad:
                        print(f"  ⚠️ Non-serializable fields in llp_result: {bad}")
                except Exception:
                    pass
                with open(result_file, 'w') as f:
                    json.dump(_sanitize_for_json(llp_result), f, indent=2)
                
                # 保存为CSV便于查看
                csv_file = interim_dir / f'llp_{idx:04d}_summary.csv'
                pd.DataFrame([llp_result]).to_csv(csv_file, index=False)
                
                processed_params.append(llp_result)
                print(f"  ✓ Successfully saved results for LLP {idx}")
                
            except Exception as e:
                print(f"  ✗ Error processing decays for mass={llp_mass}, tanb={tanb}: {e}")
                failed_params.append({
                    'index': idx,
                    'mass': llp_mass,
                    'tanb': tanb,
                    'lifetime': llp_lifetime,
                    'reason': str(e),
                    'processed_blocks': processed_blocks
                })
            
        # 强制清理内存
        del llp_analyzer
        del decay_positions
        del weights
        del all_decay_positions
        del all_weights
        gc.collect()
        
        # 显示进度
        print(f"\nProgress: {len(processed_params)} succeeded, {len(failed_params)} failed")
    
        print(f"\n{'='*70}")
        print("Simulation Phase Complete!")
        print(f"{'='*70}")
        print(f"Successfully processed: {len(processed_params)} LLP parameters")
        print(f"Failed: {len(failed_params)} LLP parameters")
        
        # 汇总所有结果
        self._aggregate_results(processed_params, failed_params, interim_dir)

        # 生成最终报告
        self._generate_final_report(processed_params, failed_params)

    def _aggregate_results(self, processed_params: List[Dict], failed_params: List[Dict], interim_dir: Path):
        """汇总所有LLP的结果"""
        print("\n" + "="*70)
        print("Aggregating Results from All LLP Parameters")
        print("="*70)
        
        # 1. 创建汇总目录
        aggregated_dir = self.output_dir / 'aggregated_results'
        aggregated_dir.mkdir(exist_ok=True)
        
        # 2. 汇总成功处理的参数
        if processed_params:
            # 创建主汇总DataFrame
            all_results_df = pd.DataFrame(processed_params)
            
            # 保存汇总CSV
            summary_file = aggregated_dir / 'all_llp_results.csv'
            all_results_df.to_csv(summary_file, index=False)
            print(f"✓ Saved aggregated results to {summary_file}")
            
            # 按质量排序
            mass_sorted = all_results_df.sort_values('mass')
            mass_file = aggregated_dir / 'llp_results_by_mass.csv'
            mass_sorted.to_csv(mass_file, index=False)
            
            # 按目标区域内衰变数排序（如果有）
            if 'decays_in_target' in all_results_df.columns:
                target_sorted = all_results_df.sort_values('decays_in_target', ascending=False)
                target_file = aggregated_dir / 'llp_results_by_target_decays.csv'
                target_sorted.to_csv(target_file, index=False)
                print(f"✓ Top 5 parameters by decays in target region:")
                for _, row in target_sorted.head(5).iterrows():
                    print(f"  m={row['mass']:.3f}GeV, tanβ={row['tanb']:.1f}: {row['decays_in_target']:.0f} decays")
            
            # 创建参数空间网格数据（用于热图）
            self._create_parameter_grid_data(all_results_df, aggregated_dir)
            
            # 合并所有块数据到统一目录
            self._merge_all_blocks(processed_params, aggregated_dir)
        
        # 3. 保存失败记录
        if failed_params:
            failed_df = pd.DataFrame(failed_params)
            failed_file = aggregated_dir / 'failed_parameters.csv'
            failed_df.to_csv(failed_file, index=False)
            print(f"✓ Saved failed parameters list to {failed_file}")
        
        # 4. 生成统计摘要
        self._generate_statistics_summary(processed_params, failed_params, aggregated_dir)
        
        print(f"\nAggregation complete! Results in: {aggregated_dir}")

    def _create_parameter_grid_data(self, all_results_df: pd.DataFrame, output_dir: Path):
        """创建参数空间网格数据用于热图绘制"""
        print("\nCreating parameter grid data for visualization...")
        
        try:
            # 提取唯一的mass和tanb值
            masses = sorted(all_results_df['mass'].unique())
            tanbs = sorted(all_results_df['tanb'].unique())
            
            # 创建网格
            mass_grid, tanb_grid = np.meshgrid(masses, tanbs)
            
            # 初始化各种指标的网格
            decays_grid = np.zeros_like(mass_grid, dtype=float)
            fraction_grid = np.zeros_like(mass_grid, dtype=float)
            density_grid = np.zeros_like(mass_grid, dtype=float)
            
            # 填充网格数据
            for _, row in all_results_df.iterrows():
                mass_idx = np.where(masses == row['mass'])[0][0]
                tanb_idx = np.where(tanbs == row['tanb'])[0][0]
                
                decays_grid[tanb_idx, mass_idx] = row.get('total_decays', 0)
                
                if 'decays_in_target' in row:
                    fraction_grid[tanb_idx, mass_idx] = row.get('fraction_in_target', 0)
                    density_grid[tanb_idx, mass_idx] = row.get('decays_in_target', 0)
            
            # 保存网格数据
            grid_data = {
                'masses': masses,
                'tanbs': tanbs,
                'mass_grid': mass_grid.tolist(),
                'tanb_grid': tanb_grid.tolist(),
                'total_decays_grid': decays_grid.tolist(),
                'fraction_in_target_grid': fraction_grid.tolist() if np.any(fraction_grid) else [],
                'decays_in_target_grid': density_grid.tolist() if np.any(density_grid) else []
            }
            
            grid_file = output_dir / 'parameter_grid_data.json'
            with open(grid_file, 'w') as f:
                json.dump(_sanitize_for_json(grid_data), f, indent=2)
            
            print(f"✓ Saved parameter grid data to {grid_file}")
            
        except Exception as e:
            print(f"✗ Error creating grid data: {e}")

    def _merge_all_blocks(self, processed_params: List[Dict], aggregated_dir: Path):
        """合并所有LLP的块数据到统一目录"""
        print("\nMerging all LLP block data...")
        
        merged_blocks_dir = aggregated_dir / 'all_llp_blocks'
        merged_blocks_dir.mkdir(exist_ok=True)
        
        # 创建块索引
        all_blocks_index = []
        
        for param in processed_params:
            llp_dir = Path(param['output_dir'])
            blocks_dir = llp_dir / 'blocks'
            
            if not blocks_dir.exists():
                continue
            
            # 复制每个块
            for block_folder in blocks_dir.iterdir():
                if block_folder.is_dir():
                    # 创建新块名（包含LLP信息）
                    new_block_name = f"{param['block_id']}_{block_folder.name}"
                    dest_dir = merged_blocks_dir / new_block_name
                    
                    # 复制文件
                    if not dest_dir.exists():
                        import shutil
                        shutil.copytree(block_folder, dest_dir)
                    
                    # 添加到索引
                    block_stats_file = dest_dir / 'statistics.json'
                    if block_stats_file.exists():
                        with open(block_stats_file, 'r') as f:
                            stats = json.load(f)
                        
                        all_blocks_index.append({
                            'llp_mass': param['mass'],
                            'llp_tanb': param['tanb'],
                            'llp_lifetime': param['lifetime'],
                            'block_id': new_block_name,
                            'original_block_id': block_folder.name,
                            'total_decays': stats.get('total_weighted_events', 0),
                            'positions_count': len(stats.get('positions', [])),
                            'output_dir': str(dest_dir)
                        })
        
        # 保存合并的索引
        if all_blocks_index:
            index_df = pd.DataFrame(all_blocks_index)
            index_file = merged_blocks_dir / 'merged_blocks_index.csv'
            index_df.to_csv(index_file, index=False)
            print(f"✓ Merged {len(all_blocks_index)} blocks into {merged_blocks_dir}")
            print(f"✓ Saved merged blocks index to {index_file}")

    def _generate_statistics_summary(self, processed_params: List[Dict], failed_params: List[Dict], output_dir: Path):
        """生成统计摘要"""
        print("\nGenerating statistics summary...")
        
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_llp_parameters_attempted': len(processed_params) + len(failed_params),
            'successfully_processed': len(processed_params),
            'failed': len(failed_params),
            'success_rate': len(processed_params) / (len(processed_params) + len(failed_params)) * 100 if (len(processed_params) + len(failed_params)) > 0 else 0,
            'simulation_config': {
                'target_region': self.decay_config.target_region,
                'min_decays_threshold': self.decay_config.min_decays_in_region
            }
        }
        
        if processed_params:
            proc_df = pd.DataFrame(processed_params)
            
            summary['statistics'] = {
                'total_decays': {
                    'mean': float(proc_df['total_decays'].mean()),
                    'std': float(proc_df['total_decays'].std()),
                    'min': float(proc_df['total_decays'].min()),
                    'max': float(proc_df['total_decays'].max()),
                    'sum': float(proc_df['total_decays'].sum())
                },
                'mass_range': {
                    'min': float(proc_df['mass'].min()),
                    'max': float(proc_df['mass'].max()),
                    'mean': float(proc_df['mass'].mean())
                },
                'tanb_range': {
                    'min': float(proc_df['tanb'].min()),
                    'max': float(proc_df['tanb'].max()),
                    'mean': float(proc_df['tanb'].mean())
                }
            }
            
            if 'decays_in_target' in proc_df.columns:
                target_decays = proc_df['decays_in_target']
                summary['statistics']['target_region_decays'] = {
                    'mean': float(target_decays.mean()),
                    'std': float(target_decays.std()),
                    'min': float(target_decays.min()),
                    'max': float(target_decays.max()),
                    'sum': float(target_decays.sum()),
                    'above_threshold': int((target_decays >= self.decay_config.min_decays_in_region).sum())
                }
        
        # 保存摘要
        summary_file = output_dir / 'simulation_statistics_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(_sanitize_for_json(summary), f, indent=2)
        
        print(f"✓ Saved statistics summary to {summary_file}")

    def _generate_final_report(self, processed_params: List[Dict], failed_params: List[Dict]):
        """生成最终报告"""
        report = f"""
        ===========================================================================
        LLP Decay Simulation - Final Report
        ===========================================================================
        
        Execution Date: {pd.Timestamp.now()}
        
        1. EXECUTION SUMMARY
        {'='*60}
        Total LLP parameters attempted: {len(processed_params) + len(failed_params)}
        Successfully processed: {len(processed_params)} ({len(processed_params)/(len(processed_params)+len(failed_params))*100:.1f}%)
        Failed: {len(failed_params)} ({len(failed_params)/(len(processed_params)+len(failed_params))*100:.1f}%)
        
        2. TARGET REGION ANALYSIS
        {'='*60}
        Target region: {self.decay_config.target_region}
        Minimum decays threshold: {self.decay_config.min_decays_in_region}
        
        """
        
        if processed_params and 'decays_in_target' in processed_params[0]:
            target_decays = [p.get('decays_in_target', 0) for p in processed_params]
            above_threshold = sum(1 for d in target_decays if d >= self.decay_config.min_decays_in_region)
            
            report += f"""    Parameters above threshold: {above_threshold} ({above_threshold/len(processed_params)*100:.1f}%)
        Average decays in target region: {np.mean(target_decays):.1f}
        Maximum decays in target region: {np.max(target_decays):.1f}
        
        """
        
        report += f"""
        3. OUTPUT FILES
        {'='*60}
        Main directory: {self.output_dir}
        
        Important files:
        - {self.output_dir}/aggregated_results/all_llp_results.csv
            Complete results for all successfully processed LLP parameters
        
        - {self.output_dir}/aggregated_results/llp_results_by_mass.csv
            Results sorted by LLP mass
        
        - {self.output_dir}/aggregated_results/parameter_grid_data.json
            Grid data for mass-tanβ heatmap visualization
        
        - {self.output_dir}/aggregated_results/simulation_statistics_summary.json
            Statistical summary of the simulation
        
        - {self.output_dir}/aggregated_results/all_llp_blocks/
            Directory containing all individual LLP block data
        
        - {self.output_dir}/interim_results/
            Interim results for each individual LLP parameter
        
        4. NEXT STEPS
        {'='*60}
        1. Visualize results:
        python visualize_results.py --input {self.output_dir}/aggregated_results/
        
        2. Analyze specific LLP parameters:
        python analyze_llp.py --mass <value> --tanb <value> --input {self.output_dir}
        
        3. Generate sensitivity plots:
        python sensitivity_analysis.py --results {self.output_dir}/aggregated_results/
        
        ===========================================================================
        """
        
        report_file = self.output_dir / 'FINAL_REPORT.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n{'='*70}")
        print("FINAL REPORT GENERATED")
        print(f"{'='*70}")
        print(f"Report saved to: {report_file}")
        print(f"\nKey results are in: {self.output_dir}/aggregated_results/")
        print(f"Use the aggregated results for further analysis and visualization.")    
    def simulate_llp_decays(self,
                           samples_per_block: int = 100,
                           max_llp_params: Optional[int] = None,
                           target_region: Optional[Dict] = None):
        """
        模拟LLP衰变过程
        
        Args:
            samples_per_block: 每个母粒子块抽样数
            max_llp_params: 最大处理的LLP参数数（用于测试）
            target_region: 目标区域定义
        """
        if target_region:
            self.decay_config.target_region = target_region
        
        print("=" * 70)
        print("LLP Decay Simulation Pipeline")
        print("=" * 70)
        
        # 获取所有母粒子块
        all_blocks = self.particle_loader.index['block_id'].tolist()
        print(f"Found {len(all_blocks)} mother particle blocks")
        
        # 限制处理的LLP参数数量（用于测试）
        if max_llp_params and max_llp_params < len(self.llp_params_df):
            llp_params_to_process = self.llp_params_df.head(max_llp_params)
            print(f"Processing first {max_llp_params} LLP parameters (testing mode)")
        else:
            llp_params_to_process = self.llp_params_df
        
        # 处理每个LLP参数
        for idx, llp_row in tqdm(llp_params_to_process.iterrows(), 
                                total=len(llp_params_to_process),
                                desc="Processing LLP parameters"):
            
            llp_mass = llp_row['mH']
            llp_lifetime = llp_row['ltime']
            tanb = llp_row['tanb']
            vis_br = llp_row['Br_visible']
            # 收集所有衰变位置
            all_decay_positions = []
            all_weights = []
            
            # 对每个母粒子块
            for block_id in tqdm(all_blocks, desc="Sampling B blocks", leave=False):
                try:
                    # 从块中抽样粒子
                    sampled_particles = self.particle_loader.sample_from_block(
                        block_id=block_id,
                        n_samples=samples_per_block,
                        strategy='importance'
                    )
                    # print(f"Processing block {block_id} for mass={llp_mass}, tanb={tanb}")
                    
                    # 对每个抽样粒子计算LLP衰变
                    for _, particle in sampled_particles.iterrows():
                        birth_position = particle[['decay_x', 'decay_y', 'decay_z']].values
                        momentum = particle[['e', 'px', 'py', 'pz']].values
                        
                        try:
                            llp_momentum, _ = two_body_decay_lab(momentum, 5.279, llp_mass, 0.494)
                            decay_pos, _ = generate_decay_position(llp_lifetime, llp_momentum, birth_position)
                            
                            if not np.any(np.isnan(decay_pos)):
                                all_decay_positions.append(decay_pos)
                                # 权重 = 1 / 抽样率
                                all_weights.append(len(sampled_particles) / samples_per_block)
                                
                        except (ValueError, ZeroDivisionError):
                            continue
                            
                except Exception as e:
                    print(f"Error processing block {block_id}: {e}")
                    continue
            
            if len(all_decay_positions) == 0:
                print(f"Warning: No valid decays for mass={llp_mass}, tanb={tanb}")
                continue
            
            # 转换为numpy数组
            decay_positions = np.array(all_decay_positions)
            weights = np.array(all_weights)
            
            # 准备LLP参数
            llp_params = {
                'mass': llp_mass,
                'lifetime': llp_lifetime,
                'tanb': tanb,
                'vis_br': vis_br
            }
            
            # 处理衰变数据
            block_id = self.analyzer.process_llp_decays(
                llp_params=llp_params,
                decay_positions=decay_positions,
                weights=weights,
                block_name_suffix=f"_{idx:04d}"
            )
            
            # 保存结果
            self.simulation_results.append({
                'mass': llp_mass,
                'lifetime': llp_lifetime,
                'tanb': tanb,
                # 'block_id': block_id,
                'total_decays': float(np.sum(weights)),
                # 'unique_positions': len(decay_positions)
            })
            
            # 定期保存进度
            if (idx + 1) % 1 == 0:
                print(f"Processed {idx + 1}/{len(llp_params_to_process)} LLP parameters")
                self._save_progress()
        
        print(f"\nSimulation complete! Processed {len(self.simulation_results)} LLP parameters")
        
        # 最终保存
        self._save_final_results()
    

    
    def _save_progress(self):
        """保存进度"""
        progress_file = self.output_dir / 'simulation_progress.csv'
        pd.DataFrame(self.simulation_results).to_csv(progress_file, index=False)
        
        # 保存分析器状态
        self.analyzer.save_analysis_summary()
    
    def _save_final_results(self):
        """保存最终结果"""
        # 保存模拟结果
        results_df = pd.DataFrame(self.simulation_results)
        results_file = self.output_dir / 'simulation_results.csv'
        results_df.to_csv(results_file, index=False)
        # print(f"Saved simulation results to {results_file}")
        
        # 保存所有LLP块
        self.analyzer.save_all_blocks_enhanced()
        
        # 保存完整的分析摘要
        self.analyzer.save_analysis_summary()
        
        # 生成报告
        self._generate_report()
    
    def _generate_report(self):
        """生成分析报告"""
        report = f"""
        LLP Decay Simulation Report
        ===========================
        Date: {pd.Timestamp.now()}
        
        Input Data:
        - Particle blocks: {self.particle_blocks_dir}
        - LLP parameters: {self.llp_params_file}
        
        Configuration:
        - Target region: {self.decay_config.target_region}
        - Min decays in region: {self.decay_config.min_decays_in_region}
        - Spatial grid: {self.decay_config.nx}x{self.decay_config.ny}x{self.decay_config.nz}
        
        Results:
        - Total LLP parameters processed: {len(self.simulation_results)}
        - Total decay positions generated: {sum(r['unique_positions'] for r in self.simulation_results)}
        - Total weighted decays: {sum(r['total_decays'] for r in self.simulation_results):.0f}
        
        Output Files:
        - {self.output_dir}/simulation_results.csv
        - {self.output_dir}/decay_analysis/blocks/*/data.h5
        - {self.output_dir}/decay_analysis/statistics/
        - {self.output_dir}/decay_analysis/visualization/
        
        To visualize results, run:
        analyzer = LLPDecayAnalyzer.load('{self.output_dir}/decay_analysis')
        filtered = analyzer.get_blocks_in_region()
        analyzer.visualize_mass_tanb_heatmap()
        """
        
        report_file = self.output_dir / 'simulation_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to {report_file}")
    
    def analyze_and_visualize(self, 
                            region: Optional[Dict] = None,
                            min_decays: Optional[float] = None):
        """
        分析并可视化结果
        
        Args:
            region: 目标区域
            min_decays: 最小衰变数
        """
        print("\n" + "="*70)
        print("Analyzing and Visualizing Results")
        print("="*70)
        
        # 如果未指定，使用配置中的区域
        if region is None:
            region = self.decay_config.target_region
        
        if region is None:
            print("Warning: No target region specified. Using default region.")
            region = {'x_min': -100, 'x_max': 100, 
                     'y_min': -100, 'y_max': 100,
                     'z_min': 1000, 'z_max': 2000}
        
        # 获取符合条件的参数
        filtered_df = self.analyzer.get_blocks_in_region(region, min_decays)
        
        print(f"\nFound {len(filtered_df)} LLP parameters meeting criteria:")
        print(f"Region: {region}")
        print(f"Minimum decays: {min_decays or self.decay_config.min_decays_in_region}")
        
        if len(filtered_df) >= 0:
            print(f"\nTop 5 parameters with most decays in region:")
            top_10 = filtered_df.nlargest(5, 'decays_in_region')
            for _, row in top_10.iterrows():
                print(f"  m={row['mass']:.3f} GeV, tanβ={row['tanb']:.2f}: "
                      f"{row['decays_in_region']:.0f} decays ({row['fraction_in_region']:.2%} of total)")
            
            # 保存符合条件的参数列表
            filtered_file = self.output_dir / 'parameters_meeting_criteria.csv'
            filtered_df.to_csv(filtered_file, index=False)
            print(f"\nSaved filtered parameters to {filtered_file}")
            
            # 创建可视化
        else:
            print("No parameters meet the criteria. Try lowering the threshold.")
    
    def simulate_llp_decays_incremental(self,
                                    samples_per_block: int = 100,
                                    max_llp_params: Optional[int] = None,
                                    target_region: Optional[Dict] = None):
        """
        增量式模拟LLP衰变过程 - 每个LLP模拟后立即存储数据
        
        Args:
            samples_per_block: 每个母粒子块抽样数
            max_llp_params: 最大处理的LLP参数数（用于测试）
            target_region: 目标区域定义
        """
        if target_region:
            self.decay_config.target_region = target_region
        
        print("=" * 70)
        print("Incremental LLP Decay Simulation")
        print("(Each LLP processed and saved immediately)")
        print("=" * 70)
        
        # 创建增量存储目录
        incremental_dir = self.output_dir / 'incremental_results'
        incremental_dir.mkdir(exist_ok=True)
        
        # 获取所有母粒子块
        all_blocks = self.particle_loader.index['block_id'].tolist()
        print(f"Found {len(all_blocks)} mother particle blocks")
        
        # 限制处理的LLP参数数量（用于测试）
        if max_llp_params and max_llp_params < len(self.llp_params_df):
            llp_params_to_process = self.llp_params_df.head(max_llp_params)
            print(f"Processing first {max_llp_params} LLP parameters (testing mode)")
        else:
            llp_params_to_process = self.llp_params_df
        
        # 清除之前的结果（从头开始）
        self.simulation_results = []
        
        # 处理每个LLP参数
        for idx, llp_row in tqdm(llp_params_to_process.iterrows(), 
                                total=len(llp_params_to_process),
                                desc="Processing LLP parameters incrementally"):
            
            llp_mass = llp_row['mH']
            llp_lifetime = llp_row['ltime']
            tanb = llp_row['tanb']
            vis_br = llp_row['Br_visible']
            
            # print(f"\n[LLP {idx+1}] Mass: {llp_mass:.3f} GeV, tanβ: {tanb:.2f}, Lifetime: {llp_lifetime:.2e} mm")
            
            # 为每个LLP创建独立的临时分析器实例
            llp_analyzer = LLPDecayAnalyzer(
                output_dir=str(incremental_dir / f'llp_{idx:04d}_temp'),
                config=self.decay_config
            )
            
            # 收集所有衰变位置
            all_decay_positions = []
            all_weights = []
            
            # 对每个母粒子块抽样
            for block_id in tqdm(all_blocks, desc="Sampling B blocks", leave=False):
                try:
                    # 从块中抽样粒子
                    sampled_particles = self.particle_loader.sample_from_block(
                        block_id=block_id,
                        n_samples=samples_per_block,
                        strategy='importance'
                    )
                    
                    # 对每个抽样粒子计算LLP衰变
                    for _, particle in sampled_particles.iterrows():
                        birth_position = particle[['decay_x', 'decay_y', 'decay_z']].values
                        momentum = particle[['e', 'px', 'py', 'pz']].values
                        
                        try:
                            llp_momentum, _ = two_body_decay_lab(momentum, 5.279, llp_mass, 0.494)
                            decay_pos, _ = generate_decay_position(llp_lifetime, llp_momentum, birth_position)
                            
                            if not np.any(np.isnan(decay_pos)):
                                all_decay_positions.append(decay_pos)
                                # 权重 = 1 / 抽样率
                                all_weights.append(len(sampled_particles) / samples_per_block)
                                
                        except (ValueError, ZeroDivisionError):
                            continue
                            
                except Exception as e:
                    print(f"Error processing block {block_id}: {e}")
                    continue
            
            if len(all_decay_positions) == 0:
                print(f"  ✗ No valid decays for this LLP")
                
                # 记录失败
                self.simulation_results.append({
                    'mass': llp_mass,
                    'lifetime': llp_lifetime,
                    'tanb': tanb,
                    'total_decays': 0,
                    'unique_positions': 0,
                    'status': 'failed',
                    'reason': 'No valid decays'
                })
                
                # 清理内存
                del llp_analyzer
                gc.collect()
                continue
            
            # 转换为numpy数组
            decay_positions = np.array(all_decay_positions)
            weights = np.array(all_weights)
            
            # print(f"  ✓ Generated {len(decay_positions)} decay positions")
            # print(f"  ✓ Total weighted decays: {np.sum(weights):.0f}")
            
            # 准备LLP参数
            llp_params = {
                'mass': llp_mass,
                'lifetime': llp_lifetime,
                'tanb': tanb,
                'vis_br': vis_br
            }
            
            try:
                # 处理衰变数据 - 立即块化并保存
                block_id = llp_analyzer.process_llp_decays(
                    llp_params=llp_params,
                    decay_positions=decay_positions,
                    weights=weights,
                    block_name_suffix=f"_m{llp_mass:.3f}_tb{tanb:.2f}"
                )
                
                # 立即保存该LLP的所有块数据
                llp_analyzer.save_all_blocks()
                
                # 保存分析摘要
                summary_file = incremental_dir / f'llp_{idx:04d}_summary.json'
                # llp_analyzer.save_analysis_summary(filename=str(summary_file))
                llp_analyzer.save_analysis_summary()
                
                # 计算目标区域内的衰变数
                decays_in_region = 0
                fraction_in_region = 0
                
                if self.decay_config.target_region:
                    llp_block = llp_analyzer.llp_blocks.get(block_id)
                    if llp_block:
                        decays_in_region = llp_block.get_decays_in_region(self.decay_config.target_region)
                        fraction_in_region = decays_in_region / np.sum(weights) if np.sum(weights) > 0 else 0
                
                # 记录成功结果
                result = {
                    'mass': llp_mass,
                    'lifetime': llp_lifetime,
                    'tanb': tanb,
                    'vis_br': vis_br,
                    'block_id': block_id,
                    'total_decays': float(np.sum(weights)),
                    'unique_positions': len(decay_positions),
                    'decays_in_region': float(decays_in_region),
                    'fraction_in_region': float(fraction_in_region),
                    'status': 'success',
                    'output_dir': str(incremental_dir / f'llp_{idx:04d}_temp'),
                    'summary_file': str(summary_file),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                self.simulation_results.append(result)
                
                # 立即保存单个LLP的结果文件
                result_file = incremental_dir / f'llp_{idx:04d}_result.json'
                with open(result_file, 'w') as f:
                    json.dump(_sanitize_for_json(result), f, indent=2)
                
                # 保存为CSV格式
                csv_file = incremental_dir / f'llp_{idx:04d}_result.csv'
                pd.DataFrame([result]).to_csv(csv_file, index=False)
                
                # print(f"  ✓ Saved results for LLP {idx+1}")
                # print(f"    - Total decays: {np.sum(weights):.0f}")
                # print(f"    - Decays in target region: {decays_in_region:.0f}")
                # print(f"    - Fraction in region: {fraction_in_region:.2%}")
                # print(f"    - Files saved to: {result_file}")
                
            except Exception as e:
                print(f"  ✗ Error processing/saving LLP data: {e}")
                
                # 记录失败
                self.simulation_results.append({
                    'mass': llp_mass,
                    'lifetime': llp_lifetime,
                    'tanb': tanb,
                    'total_decays': float(np.sum(weights)) if 'weights' in locals() else 0,
                    'unique_positions': len(decay_positions) if 'decay_positions' in locals() else 0,
                    'status': 'failed',
                    'reason': str(e)
                })
            
            finally:
                # 强制清理内存
                del llp_analyzer
                del decay_positions
                del weights
                del all_decay_positions
                del all_weights
                gc.collect()
            
            # 定期保存进度汇总
            if (idx + 1) % 10 == 0:
                self._save_progress_incremental(incremental_dir)
        
        print(f"\n{'='*70}")
        print("Incremental Simulation Complete!")
        print(f"{'='*70}")
        print(f"Total LLP parameters processed: {len(self.simulation_results)}")
        
        # 最终保存所有汇总结果
        self._save_final_incremental_results(incremental_dir)
        
        # 生成最终报告
        self._generate_incremental_report(incremental_dir)

    def _save_progress_incremental(self, incremental_dir: Path):
        """保存增量处理的进度"""
        progress_df = pd.DataFrame(self.simulation_results)
        progress_file = incremental_dir / 'simulation_progress.csv'
        progress_df.to_csv(progress_file, index=False)
        
        # 保存JSON格式便于查看
        progress_json = incremental_dir / 'simulation_progress.json'
        with open(progress_json, 'w') as f:
            json.dump(_sanitize_for_json(self.simulation_results), f, indent=2)
        
        # 计算统计信息
        success_count = sum(1 for r in self.simulation_results if r.get('status') == 'success')
        failed_count = sum(1 for r in self.simulation_results if r.get('status') == 'failed')
        
        print(f"\n[Progress] Success: {success_count}, Failed: {failed_count}, Total: {len(self.simulation_results)}")
        print(f"Progress saved to: {progress_file}")

    def _save_final_incremental_results(self, incremental_dir: Path):
        """保存增量处理的最终结果"""
        print("\n" + "="*70)
        print("Saving Final Incremental Results")
        print("="*70)
        
        # 1. 汇总所有成功的LLP结果
        success_results = [r for r in self.simulation_results if r.get('status') == 'success']
        
        if success_results:
            # 创建汇总DataFrame
            all_results_df = pd.DataFrame(success_results)
            
            # 保存汇总文件
            summary_file = self.output_dir / 'incremental_summary_all.csv'
            all_results_df.to_csv(summary_file, index=False)
            print(f"✓ Saved complete summary to: {summary_file}")
            
            # 按质量排序
            mass_sorted = all_results_df.sort_values('mass')
            mass_file = self.output_dir / 'incremental_summary_by_mass.csv'
            mass_sorted.to_csv(mass_file, index=False)
            print(f"✓ Saved mass-sorted summary to: {mass_file}")
            
            # 按目标区域衰变数排序（如果有）
            if 'decays_in_region' in all_results_df.columns:
                target_sorted = all_results_df.sort_values('decays_in_region', ascending=False)
                target_file = self.output_dir / 'incremental_summary_by_target.csv'
                target_sorted.to_csv(target_file, index=False)
                print(f"✓ Saved target-sorted summary to: {target_file}")
                
                # 显示前5名
                print(f"\nTop 5 LLP parameters by decays in target region:")
                for i, (_, row) in enumerate(target_sorted.head(5).iterrows(), 1):
                    print(f"  {i}. m={row['mass']:.3f}GeV, tanβ={row['tanb']:.1f}: "
                        f"{row['decays_in_region']:.0f} decays")
        
        # 2. 保存失败记录
        failed_results = [r for r in self.simulation_results if r.get('status') == 'failed']
        if failed_results:
            failed_df = pd.DataFrame(failed_results)
            failed_file = self.output_dir / 'incremental_failed_parameters.csv'
            failed_df.to_csv(failed_file, index=False)
            print(f"✓ Saved failed parameters to: {failed_file}")
        
        # 3. 生成统计摘要
        stats = {
            'total_parameters': len(self.simulation_results),
            'successful': len(success_results),
            'failed': len(failed_results),
            'success_rate': len(success_results) / len(self.simulation_results) * 100 if self.simulation_results else 0,
            'total_decays_generated': sum(r.get('total_decays', 0) for r in success_results),
            'total_unique_positions': sum(r.get('unique_positions', 0) for r in success_results),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        stats_file = self.output_dir / 'incremental_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics to: {stats_file}")
        
        # 4. 将所有单个LLP文件移动到最终目录
        final_llp_dir = self.output_dir / 'final_llp_results'
        final_llp_dir.mkdir(exist_ok=True)
        
        # 复制所有单个LLP结果文件
        for result in success_results:
            llp_idx = result.get('index', len(self.simulation_results))
            src_files = list(incremental_dir.glob(f'llp_{llp_idx:04d}_*'))
            for src_file in src_files:
                dst_file = final_llp_dir / src_file.name
                import shutil
                shutil.copy2(src_file, dst_file)
        
        print(f"✓ Copied all individual LLP results to: {final_llp_dir}")
        print(f"\nFinal results available in: {self.output_dir}/")

    def _generate_incremental_report(self, incremental_dir: Path):
        """生成增量处理的报告"""
        success_results = [r for r in self.simulation_results if r.get('status') == 'success']
        failed_results = [r for r in self.simulation_results if r.get('status') == 'failed']
        
        report = f"""
        ===========================================================================
        INCREMENTAL LLP DECAY SIMULATION - FINAL REPORT
        ===========================================================================
        
        Execution Date: {pd.Timestamp.now()}
        
        1. EXECUTION SUMMARY
        {'='*60}
        Total LLP parameters processed: {len(self.simulation_results)}
        Successfully completed: {len(success_results)} ({len(success_results)/len(self.simulation_results)*100:.1f}%)
        Failed: {len(failed_results)} ({len(failed_results)/len(self.simulation_results)*100:.1f}%)
        
        2. SIMULATION STATISTICS
        {'='*60}
        """
        
        if success_results:
            total_decays = sum(r.get('total_decays', 0) for r in success_results)
            total_unique = sum(r.get('unique_positions', 0) for r in success_results)
            
            report += f"""    Total weighted decays generated: {total_decays:.0f}
        Total unique decay positions: {total_unique}
        Average decays per LLP: {total_decays/len(success_results):.0f}
        """
            
            if 'decays_in_region' in success_results[0]:
                total_in_region = sum(r.get('decays_in_region', 0) for r in success_results)
                avg_fraction = sum(r.get('fraction_in_region', 0) for r in success_results) / len(success_results)
                
                report += f"""    Total decays in target region: {total_in_region:.0f}
        Average fraction in target region: {avg_fraction:.2%}
        """
        
        report += f"""
        3. OUTPUT FILES
        {'='*60}
        Main directory: {self.output_dir}
        
        Important files:
        - {self.output_dir}/incremental_summary_all.csv
            Complete summary of all successfully processed LLP parameters
        
        - {self.output_dir}/incremental_summary_by_mass.csv
            Summary sorted by LLP mass
        
        - {self.output_dir}/incremental_summary_by_target.csv
            Summary sorted by decays in target region (if applicable)
        
        - {self.output_dir}/incremental_failed_parameters.csv
            List of failed LLP parameters with reasons
        
        - {self.output_dir}/incremental_statistics.json
            Statistical summary of the simulation
        
        - {self.output_dir}/final_llp_results/
            Directory containing individual result files for each LLP
        
        - {incremental_dir}/
            Directory containing all intermediate files
        
        4. KEY FEATURES
        {'='*60}
        • Memory-safe: Each LLP processed and saved independently
        • Fault-tolerant: Failed LLPs don't affect others
        • Incremental: Results saved immediately after each LLP
        • Resume capability: Can be restarted from last checkpoint
        
        5. NEXT STEPS
        {'='*60}
        1. Analyze specific LLP parameters:
            python analyze_llp.py --mass <value> --tanb <value> --dir {self.output_dir}/final_llp_results/
        
        2. Create visualizations:
            python visualize_incremental.py --input {self.output_dir}/incremental_summary_all.csv
        
        3. Merge results for further analysis:
            python merge_llp_results.py --dir {self.output_dir}/final_llp_results/
        
        ===========================================================================
        """
        
        report_file = self.output_dir / 'INCREMENTAL_SIMULATION_REPORT.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_file}")
        print(f"\nSummary files are in: {self.output_dir}/")
        print(f"Individual LLP files are in: {self.output_dir}/final_llp_results/")

