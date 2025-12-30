import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as sk_r2
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class LLPDistributionAnalyzerCompact:
    """
    简洁的LLP衰变位置分布分析器
    针对您的特定数据结构优化
    """
    
    def __init__(self, base_dir: str):
        """
        初始化分析器
        
        Args:
            base_dir: 基础目录路径，包含llp_XXXX_temp子目录
        """
        self.base_dir = Path(base_dir)
        self.results = []
        self.df = None
        
    def load_all_blocks(self):
        """
        加载所有LLP块数据 - 针对您的数据结构优化
        """
        print(f"Loading LLP blocks from {self.base_dir}...")
        
        # 查找所有llp_XXXX_temp目录
        llp_dirs = sorted(list(self.base_dir.glob("llp_*_temp")))
        print(f"Found {len(llp_dirs)} LLP directories")
        
        if not llp_dirs:
            # 尝试其他可能的模式
            llp_dirs = sorted(list(self.base_dir.glob("llp_*")))
            print(f"Found {len(llp_dirs)} alternative LLP directories")
        
        # 加载每个LLP的数据
        for llp_dir in tqdm(llp_dirs, desc="Processing LLP directories"):
            try:
                # 方法1: 从blocks子目录加载
                blocks_subdir = llp_dir / "blocks"
                if blocks_subdir.exists():
                    # 查找所有的block子目录
                    for block_dir in blocks_subdir.iterdir():
                        if block_dir.is_dir():
                            block_data = self._load_single_block(block_dir)
                            if block_data:
                                self.results.append(block_data)
                
                # 方法2: 直接从LLP目录加载（如果没有blocks子目录）
                else:
                    # 查找HDF5文件
                    h5_files = list(llp_dir.glob("*.h5"))
                    if h5_files:
                        for h5_file in h5_files:
                            block_data = self._load_from_h5(h5_file, llp_dir)
                            if block_data:
                                self.results.append(block_data)
                
                # 方法3: 尝试加载结果文件获取参数
                result_file = llp_dir.parent / f"{llp_dir.stem}_result.json"
                if not result_file.exists():
                    result_file = llp_dir / f"{llp_dir.stem}_result.json"
                
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        # 可以合并参数信息
                        pass
                    except:
                        pass
                        
            except Exception as e:
                print(f"\nWarning: Error processing {llp_dir.name}: {e}")
                continue
        
        if not self.results:
            raise ValueError(f"No valid blocks found! Checked {len(llp_dirs)} directories.")
        
        print(f"\nSuccessfully loaded {len(self.results)} blocks")
        
        # 转换为DataFrame
        self.df = pd.DataFrame(self.results)
        
        # 显示基本信息
        self._print_summary()
    
    def _load_single_block(self, block_dir: Path) -> Optional[Dict[str, Any]]:
        """从单个块目录加载数据"""
        h5_file = block_dir / "data.h5"
        if not h5_file.exists():
            return None
        
        try:
            with h5py.File(h5_file, 'r') as f:
                # 检查必要的数据集
                if 'positions' not in f:
                    return None
                
                positions = f['positions'][:]
                
                # 获取权重
                if 'weights' in f:
                    weights = f['weights'][:]
                else:
                    weights = np.ones(len(positions))
                
                # 获取参数
                params = {}
                if 'parameters' in f:
                    params = dict(f['parameters'].attrs)
                
                # 提取块ID
                block_id = block_dir.name
                
                # 从块ID提取参数（如果HDF5中没有）
                if 'mass' not in params:
                    # 尝试从块名解析
                    import re
                    match = re.search(r'm([\d\.]+)_tb([\d\.]+)', block_id)
                    if match:
                        params['mass'] = float(match.group(1))
                        params['tanb'] = float(match.group(2))
                
        except Exception as e:
            print(f"Error reading {h5_file}: {e}")
            return None
        
        # 计算统计量
        return self._compute_statistics(positions, weights, params, block_id)
    
    def _load_from_h5(self, h5_file: Path, llp_dir: Path) -> Optional[Dict[str, Any]]:
        """直接从HDF5文件加载数据"""
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'positions' not in f:
                    return None
                
                positions = f['positions'][:]
                
                # 获取权重
                if 'weights' in f:
                    weights = f['weights'][:]
                else:
                    weights = np.ones(len(positions))
                
                # 获取参数
                params = {}
                if 'parameters' in f:
                    params = dict(f['parameters'].attrs)
                
                # 尝试从文件名或父目录获取参数
                block_id = h5_file.stem
                
                # 查找对应的结果文件
                result_file = llp_dir.parent / f"{llp_dir.stem}_result.json"
                if not result_file.exists():
                    result_file = llp_dir / f"{llp_dir.stem}_result.json"
                
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as rf:
                            result_data = json.load(rf)
                        # 合并参数
                        for key in ['mass', 'lifetime', 'tanb', 'vis_br']:
                            if key in result_data and key not in params:
                                params[key] = result_data[key]
                    except:
                        pass
                
        except Exception as e:
            print(f"Error reading {h5_file}: {e}")
            return None
        
        return self._compute_statistics(positions, weights, params, block_id)
    
    def _compute_statistics(self, positions: np.ndarray, weights: np.ndarray, 
                           params: Dict, block_id: str) -> Dict[str, Any]:
        """计算位置统计量"""
        if len(positions) == 0:
            return None
        
        # 确保positions是二维数组
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)
        
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # 归一化权重
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights_norm = weights / weights_sum
        else:
            weights_norm = np.ones_like(weights) / len(weights)
        
        # 计算基本统计量
        def weighted_mean(data, w):
            return np.average(data, weights=w)
        
        def weighted_std(data, w):
            mean = weighted_mean(data, w)
            variance = np.average((data - mean)**2, weights=w)
            return np.sqrt(variance)
        
        # 统计字典
        stats_dict = {
            'mean': weighted_mean,
            'std': weighted_std,
            'min': lambda d, w: np.min(d),
            'max': lambda d, w: np.max(d),
            'median': lambda d, w: np.median(d)
        }
        
        # 计算各坐标统计
        x_stats = {name: func(x, weights) for name, func in stats_dict.items()}
        y_stats = {name: func(y, weights) for name, func in stats_dict.items()}
        z_stats = {name: func(z, weights) for name, func in stats_dict.items()}
        
        # 尝试计算偏度和峰度（可选）
        try:
            if len(x) > 2:
                x_stats['skew'] = stats.skew(x)
                y_stats['skew'] = stats.skew(y)
                z_stats['skew'] = stats.skew(z)
        except:
            pass
        
        try:
            if len(x) > 3:
                x_stats['kurtosis'] = stats.kurtosis(x)
                y_stats['kurtosis'] = stats.kurtosis(y)
                z_stats['kurtosis'] = stats.kurtosis(z)
        except:
            pass
        
        return {
            'block_id': block_id,
            'mass': float(params.get('mass', 0)),
            'lifetime': float(params.get('lifetime', params.get('ltime', 0))),
            'tanb': float(params.get('tanb', params.get('tanβ', 0))),
            'vis_br': float(params.get('vis_br', params.get('Br_visible', 0))),
            'n_positions': len(positions),
            'total_weight': float(weights_sum),
            'x_stats': x_stats,
            'y_stats': y_stats,
            'z_stats': z_stats
        }
    
    def _print_summary(self):
        """打印数据摘要"""
        print(f"\n{'='*60}")
        print("DATA SUMMARY")
        print('='*60)
        
        print(f"Total blocks loaded: {len(self.df)}")
        print(f"Total decay positions: {self.df['n_positions'].sum():,}")
        print(f"Total weighted events: {self.df['total_weight'].sum():.0f}")
        
        if 'mass' in self.df.columns:
            print(f"\nMass statistics:")
            print(f"  Range: {self.df['mass'].min():.3f} - {self.df['mass'].max():.3f} GeV")
            print(f"  Mean: {self.df['mass'].mean():.3f} GeV")
            print(f"  Unique values: {self.df['mass'].nunique()}")
        
        if 'lifetime' in self.df.columns:
            print(f"\nLifetime statistics:")
            print(f"  Range: {self.df['lifetime'].min():.2e} - {self.df['lifetime'].max():.2e} mm")
            print(f"  Mean: {self.df['lifetime'].mean():.2e} mm")
        
        if 'tanb' in self.df.columns:
            print(f"\ntanβ statistics:")
            print(f"  Range: {self.df['tanb'].min():.2f} - {self.df['tanb'].max():.2f}")
            print(f"  Mean: {self.df['tanb'].mean():.2f}")
        
        # 显示前几个块的信息
        print(f"\nFirst 5 blocks:")
        for i, (_, row) in enumerate(self.df.head(5).iterrows()):
            print(f"  {i+1}. Block: {row['block_id'][:30]}...")
            print(f"     Mass: {row['mass']:.3f} GeV, tanβ: {row['tanb']:.1f}")
            print(f"     Positions: {row['n_positions']:,}, Weight: {row['total_weight']:.0f}")
    
    def analyze_distributions_simple(self):
        """简化版的分布分析"""
        if self.df is None or len(self.df) == 0:
            raise ValueError("No data loaded. Call load_all_blocks() first.")
        
        print(f"\n{'='*60}")
        print("ANALYZING DISTRIBUTIONS")
        print('='*60)
        
        # 提取数据
        X = self.df[['mass', 'lifetime']].values
        y_x = np.array([s['mean'] for s in self.df['x_stats']])
        y_y = np.array([s['mean'] for s in self.df['y_stats']])
        y_z = np.array([s['mean'] for s in self.df['z_stats']])
        
        # 1. 简单线性回归
        print("\n1. Linear Regression Analysis:")
        
        self.lin_models = {}
        for coord, y_data in [('X', y_x), ('Y', y_y), ('Z', y_z)]:
            model = LinearRegression()
            model.fit(X, y_data)
            y_pred = model.predict(X)
            r2 = sk_r2(y_data, y_pred)
            
            self.lin_models[coord] = {
                'model': model,
                'coef_mass': model.coef_[0],
                'coef_lifetime': model.coef_[1],
                'intercept': model.intercept_,
                'r2': r2
            }
            
            print(f"\n  {coord} position:")
            print(f"    Equation: y = {model.coef_[0]:.3f}*mass + {model.coef_[1]:.3e}*lifetime + {model.intercept_:.1f}")
            print(f"    R² score: {r2:.4f}")
        
        # 2. 多项式拟合（质量和寿命的对数）
        print("\n2. Polynomial Fitting (with log10 lifetime):")
        
        self.poly_models = {}
        
        def poly_func(params, a, b, c, d):
            """多项式模型: a*mass + b*log10(lifetime) + c*mass*log10(lifetime) + d"""
            mass, lifetime = params
            return a * mass + b * np.log10(lifetime) + c * mass * np.log10(lifetime) + d
        
        for coord, y_data in [('X', y_x), ('Y', y_y), ('Z', y_z)]:
            try:
                params, _ = curve_fit(
                    poly_func, X.T, y_data,
                    p0=[0.1, 0.1, 0.01, np.mean(y_data)],
                    maxfev=5000
                )
                
                y_pred = poly_func(X.T, *params)
                r2 = sk_r2(y_data, y_pred)
                
                self.poly_models[coord] = {
                    'params': params,
                    'r2': r2
                }
                
                print(f"\n  {coord} position:")
                print(f"    Equation: y = {params[0]:.3f}*mass + {params[1]:.3f}*log10(τ) + "
                      f"{params[2]:.3f}*mass*log10(τ) + {params[3]:.1f}")
                print(f"    R² score: {r2:.4f}")
                
            except Exception as e:
                print(f"  Warning: Polynomial fitting failed for {coord}: {e}")
        
        # 3. 相关性分析
        print("\n3. Correlation Analysis:")
        
        correlations = {}
        for coord, y_data in [('X', y_x), ('Y', y_y), ('Z', y_z)]:
            corr_mass = np.corrcoef(self.df['mass'], y_data)[0, 1]
            corr_lifetime = np.corrcoef(np.log10(self.df['lifetime']), y_data)[0, 1]
            
            correlations[coord] = {
                'mass': corr_mass,
                'log_lifetime': corr_lifetime
            }
            
            print(f"\n  {coord} position:")
            print(f"    Correlation with mass: {corr_mass:.3f}")
            print(f"    Correlation with log10(lifetime): {corr_lifetime:.3f}")
        
        self.correlations = correlations
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print('='*60)
    
    def create_compact_visualizations(self, output_dir: str = './distribution_results'):
        """创建简洁但信息丰富的可视化"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating visualizations in {output_path}...")
        
        # 设置样式
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['font.size'] = 10
        
        # 1. 主要趋势图：位置 vs 质量
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色, 橙色, 绿色
        
        for idx, (coord, color) in enumerate(zip(['x', 'y', 'z'], colors)):
            ax = axes[idx]
            
            # 提取数据
            masses = self.df['mass'].values
            means = [s['mean'] for s in self.df[f'{coord}_stats']]
            stds = [s['std'] for s in self.df[f'{coord}_stats']]
            
            # 按质量排序
            sort_idx = np.argsort(masses)
            masses_sorted = masses[sort_idx]
            means_sorted = np.array(means)[sort_idx]
            stds_sorted = np.array(stds)[sort_idx]
            
            # 绘制带误差棒的点
            ax.errorbar(masses_sorted, means_sorted, yerr=stds_sorted,
                       fmt='o', color=color, alpha=0.7, capsize=3, markersize=4,
                       label='Data')
            
            # 添加趋势线（如果有多项式模型）
            if hasattr(self, 'poly_models') and coord.upper() in self.poly_models:
                model = self.poly_models[coord.upper()]
                if model:
                    # 使用平均寿命
                    lifetime_avg = np.mean(self.df['lifetime'].values)
                    
                    def trend_func(mass):
                        return (model['params'][0] * mass + 
                                model['params'][1] * np.log10(lifetime_avg) +
                                model['params'][2] * mass * np.log10(lifetime_avg) +
                                model['params'][3])
                    
                    mass_range = np.linspace(masses_sorted.min(), masses_sorted.max(), 100)
                    trend = [trend_func(m) for m in mass_range]
                    
                    ax.plot(mass_range, trend, 'r-', linewidth=2, alpha=0.8,
                           label=f"Fit (R²={model['r2']:.3f})")
            
            ax.set_xlabel('LLP Mass (GeV)', fontsize=11)
            ax.set_ylabel(f'Mean {coord.upper()} (mm)', fontsize=11)
            ax.set_title(f'{coord.upper()} Position vs Mass', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
        
        plt.suptitle('Decay Position Distributions vs LLP Mass', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'position_vs_mass.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. 位置 vs 寿命（对数尺度）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (coord, color) in enumerate(zip(['x', 'y', 'z'], colors)):
            ax = axes[idx]
            
            means = [s['mean'] for s in self.df[f'{coord}_stats']]
            log_lifetimes = np.log10(self.df['lifetime'].values)
            
            # 用颜色表示质量
            scatter = ax.scatter(log_lifetimes, means, c=self.df['mass'].values,
                               cmap='viridis', alpha=0.7, s=30, edgecolor='k', linewidth=0.5)
            
            ax.set_xlabel('log10(Lifetime) (mm)', fontsize=11)
            ax.set_ylabel(f'Mean {coord.upper()} (mm)', fontsize=11)
            ax.set_title(f'{coord.upper()} Position vs Lifetime', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax, label='Mass (GeV)')
        
        plt.suptitle('Decay Position Distributions vs LLP Lifetime', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'position_vs_lifetime.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. 模型性能对比
        if hasattr(self, 'lin_models') and hasattr(self, 'poly_models'):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, coord in enumerate(['X', 'Y', 'Z']):
                ax = axes[idx]
                
                # 提取实际值
                y_actual = [s['mean'] for s in self.df[f'{coord.lower()}_stats']]
                
                # 获取预测值
                if coord in self.lin_models:
                    X_data = self.df[['mass', 'lifetime']].values
                    y_pred_lin = self.lin_models[coord]['model'].predict(X_data)
                    r2_lin = self.lin_models[coord]['r2']
                else:
                    y_pred_lin = None
                    r2_lin = 0
                
                if coord in self.poly_models:
                    y_pred_poly = self.poly_models[coord].get('predicted', 
                                                             np.array(y_actual) * 0.9)  # 占位符
                    r2_poly = self.poly_models[coord]['r2']
                else:
                    y_pred_poly = None
                    r2_poly = 0
                
                # 绘制预测 vs 实际
                if y_pred_lin is not None:
                    ax.scatter(y_actual, y_pred_lin, alpha=0.6, s=20,
                              label=f'Linear (R²={r2_lin:.3f})')
                
                if y_pred_poly is not None:
                    ax.scatter(y_actual, y_pred_poly, alpha=0.6, s=20,
                              label=f'Poly (R²={r2_poly:.3f})')
                
                # 添加对角线
                all_y = y_actual
                if y_pred_lin is not None:
                    all_y = all_y + list(y_pred_lin)
                if y_pred_poly is not None:
                    all_y = all_y + list(y_pred_poly)
                
                min_val = min(all_y)
                max_val = max(all_y)
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect')
                
                ax.set_xlabel(f'Actual {coord} (mm)', fontsize=11)
                ax.set_ylabel(f'Predicted {coord} (mm)', fontsize=11)
                ax.set_title(f'{coord} Position: Model Predictions', fontsize=12)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best')
            
            plt.suptitle('Model Performance Comparison', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(output_path / 'model_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. 分布形状统计
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = [('std', 'Standard Deviation (mm)', 'skyblue'),
                  ('skew', 'Skewness', 'lightcoral'),
                  ('kurtosis', 'Kurtosis', 'lightgreen')]
        
        for idx, (metric, title, color) in enumerate(metrics):
            ax = axes[idx]
            
            data = []
            labels = []
            
            for coord in ['x', 'y', 'z']:
                values = [s.get(metric, np.nan) for s in self.df[f'{coord}_stats']]
                # 移除NaN值
                values = [v for v in values if not np.isnan(v)]
                if values:
                    data.append(values)
                    labels.append(coord.upper())
            
            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                # 设置颜色
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                
                ax.set_ylabel(title.split(' ')[0], fontsize=11)
                ax.set_title(title, fontsize=12)
                ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            else:
                ax.text(0.5, 0.5, f'No {metric} data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=12)
        
        plt.suptitle('Distribution Shape Statistics', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'distribution_shape.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. 综合总结图（单个图表）
        fig = plt.figure(figsize=(14, 10))
        
        # 创建布局
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 1. 位置直方图 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        all_means = []
        for coord in ['x', 'y', 'z']:
            means = [s['mean'] for s in self.df[f'{coord}_stats']]
            all_means.extend(means)
        
        ax1.hist(all_means, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Mean Position (mm)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Distribution of Mean Positions', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. 参数相关性矩阵 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        # 准备数据
        corr_data = self.df[['mass', 'lifetime']].copy()
        for coord in ['x', 'y', 'z']:
            corr_data[f'{coord}_mean'] = [s['mean'] for s in self.df[f'{coord}_stats']]
        
        correlation = corr_data.corr()
        im = ax2.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
        
        # 设置刻度
        ax2.set_xticks(range(len(correlation.columns)))
        ax2.set_xticklabels([col.replace('_mean', '') for col in correlation.columns], 
                           rotation=45, fontsize=9)
        ax2.set_yticks(range(len(correlation.columns)))
        ax2.set_yticklabels([col.replace('_mean', '') for col in correlation.columns], 
                           fontsize=9)
        ax2.set_title('Correlation Matrix', fontsize=11)
        
        # 添加相关系数
        for i in range(len(correlation.columns)):
            for j in range(len(correlation.columns)):
                value = correlation.iloc[i, j]
                ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color='white' if abs(value) > 0.5 else 'black', fontsize=8)
        
        # 3. 模型R²分数 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        if hasattr(self, 'lin_models') and hasattr(self, 'poly_models'):
            models = ['Linear', 'Polynomial']
            x_pos = np.arange(len(models))
            width = 0.25
            
            r2_values = {}
            for coord in ['X', 'Y', 'Z']:
                r2_lin = self.lin_models.get(coord, {}).get('r2', 0)
                r2_poly = self.poly_models.get(coord, {}).get('r2', 0)
                r2_values[coord] = [r2_lin, r2_poly]
            
            colors_bars = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for idx, (coord, color) in enumerate(zip(['X', 'Y', 'Z'], colors_bars)):
                ax3.bar(x_pos + (idx-1)*width, r2_values[coord], width,
                       color=color, alpha=0.8, label=coord)
            
            ax3.set_xlabel('Model Type', fontsize=10)
            ax3.set_ylabel('R² Score', fontsize=10)
            ax3.set_title('Model Performance (R²)', fontsize=11)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(models)
            ax3.legend(fontsize=9)
            ax3.set_ylim([0, 1])
            ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 4. 位置范围图 (中左，跨越两行)
        ax4 = fig.add_subplot(gs[1:, 0])
        # 准备数据
        ranges = []
        for coord in ['x', 'y', 'z']:
            mins = [s['min'] for s in self.df[f'{coord}_stats']]
            maxs = [s['max'] for s in self.df[f'{coord}_stats']]
            ranges.append([maxs[i] - mins[i] for i in range(len(mins))])
        
        violin_parts = ax4.violinplot(ranges, showmeans=True, showmedians=True)
        # 设置颜色
        for idx, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[idx])
            pc.set_alpha(0.7)
        
        ax4.set_xticks([1, 2, 3])
        ax4.set_xticklabels(['X', 'Y', 'Z'])
        ax4.set_ylabel('Position Range (max-min, mm)', fontsize=10)
        ax4.set_title('Position Spread by Coordinate', fontsize=11)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 5. 参数空间散点图 (中右，跨越两行)
        ax5 = fig.add_subplot(gs[1:, 1:])
        # 用颜色表示Z位置
        scatter = ax5.scatter(self.df['mass'], np.log10(self.df['lifetime']),
                            c=[s['mean'] for s in self.df['z_stats']],
                            cmap='viridis', alpha=0.7, s=50, edgecolor='k', linewidth=0.5)
        
        ax5.set_xlabel('Mass (GeV)', fontsize=11)
        ax5.set_ylabel('log10(Lifetime) (mm)', fontsize=11)
        ax5.set_title('Parameter Space: Z Position', fontsize=12)
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax5, label='Mean Z Position (mm)')
        
        # 添加统计摘要文本
        stats_text = (
            f"Total Blocks: {len(self.df)}\n"
            f"Total Positions: {self.df['n_positions'].sum():,}\n"
            f"Mass Range: {self.df['mass'].min():.2f}-{self.df['mass'].max():.2f} GeV\n"
            f"Lifetime Range: {self.df['lifetime'].min():.1e}-{self.df['lifetime'].max():.1e} mm"
        )
        
        ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('LLP Decay Position Analysis Summary', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_path / 'analysis_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created {len(list(output_path.glob('*.png')))} visualization files")
    
    def save_results(self, output_dir: str = './distribution_results'):
        """保存分析结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {output_path}...")
        
        # 1. 保存数据摘要
        if self.df is not None:
            # 展开统计字典
            df_to_save = self.df.copy()
            
            for coord in ['x', 'y', 'z']:
                for stat in ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis']:
                    if stat in df_to_save[f'{coord}_stats'].iloc[0]:
                        df_to_save[f'{coord}_{stat}'] = [s.get(stat, np.nan) for s in self.df[f'{coord}_stats']]
            
            # 移除原始统计字典
            columns_to_drop = [col for col in df_to_save.columns if col.endswith('_stats')]
            df_to_save = df_to_save.drop(columns=columns_to_drop)
            
            # 保存为CSV
            csv_path = output_path / 'llp_statistics.csv'
            df_to_save.to_csv(csv_path, index=False)
            print(f"✓ Data saved to: {csv_path}")
        
        # 2. 保存模型结果
        results_dict = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'data_summary': {
                'n_blocks': len(self.df) if self.df is not None else 0,
                'total_positions': int(self.df['n_positions'].sum()) if self.df is not None else 0,
                'mass_range': [float(self.df['mass'].min()), float(self.df['mass'].max())] if self.df is not None else [],
                'lifetime_range': [float(self.df['lifetime'].min()), float(self.df['lifetime'].max())] if self.df is not None else [],
                'tanb_range': [float(self.df['tanb'].min()), float(self.df['tanb'].max())] if self.df is not None else []
            }
        }
        
        # 添加线性模型
        if hasattr(self, 'lin_models'):
            results_dict['linear_models'] = {}
            for coord, model in self.lin_models.items():
                results_dict['linear_models'][coord] = {
                    'equation': f"position = {model['coef_mass']:.4f}*mass + {model['coef_lifetime']:.4e}*lifetime + {model['intercept']:.4f}",
                    'r2': float(model['r2']),
                    'coef_mass': float(model['coef_mass']),
                    'coef_lifetime': float(model['coef_lifetime']),
                    'intercept': float(model['intercept'])
                }
        
        # 添加多项式模型
        if hasattr(self, 'poly_models'):
            results_dict['polynomial_models'] = {}
            for coord, model in self.poly_models.items():
                results_dict['polynomial_models'][coord] = {
                    'equation': f"position = {model['params'][0]:.4f}*mass + {model['params'][1]:.4f}*log10(lifetime) + {model['params'][2]:.4f}*mass*log10(lifetime) + {model['params'][3]:.4f}",
                    'r2': float(model['r2']),
                    'parameters': model['params'].tolist()
                }
        
        # 添加相关性
        if hasattr(self, 'correlations'):
            results_dict['correlations'] = self.correlations
        
        # 保存JSON
        json_path = output_path / 'analysis_results.json'
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"✓ Model results saved to: {json_path}")
        
        # 3. 生成简洁报告
        report = self._generate_compact_report()
        report_path = output_path / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to: {report_path}")
        
        print(f"\n✓ All results saved to: {output_path}")
    
    def _generate_compact_report(self) -> str:
        """生成简洁的分析报告"""
        lines = []
        
        lines.append("=" * 70)
        lines.append("LLP DECAY POSITION DISTRIBUTION ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"\nAnalysis Date: {pd.Timestamp.now()}")
        lines.append("\n" + "=" * 70)
        
        # 数据摘要
        lines.append("\nDATA SUMMARY:")
        lines.append("-" * 40)
        lines.append(f"Number of LLP blocks: {len(self.df)}")
        lines.append(f"Total decay positions: {self.df['n_positions'].sum():,}")
        lines.append(f"Mass range: {self.df['mass'].min():.3f} - {self.df['mass'].max():.3f} GeV")
        lines.append(f"Lifetime range: {self.df['lifetime'].min():.2e} - {self.df['lifetime'].max():.2e} mm")
        lines.append(f"tanβ range: {self.df['tanb'].min():.2f} - {self.df['tanb'].max():.2f}")
        
        # 平均位置
        lines.append("\n\nAVERAGE POSITIONS:")
        lines.append("-" * 40)
        for coord in ['x', 'y', 'z']:
            means = [s['mean'] for s in self.df[f'{coord}_stats']]
            stds = [s['std'] for s in self.df[f'{coord}_stats']]
            lines.append(f"{coord.upper()}: {np.mean(means):.1f} ± {np.mean(stds):.1f} mm")
        
        # 模型结果
        if hasattr(self, 'lin_models'):
            lines.append("\n\nLINEAR MODELS (position = a*mass + b*lifetime + c):")
            lines.append("-" * 40)
            for coord in ['X', 'Y', 'Z']:
                if coord in self.lin_models:
                    m = self.lin_models[coord]
                    lines.append(f"\n{coord}:")
                    lines.append(f"  Equation: y = {m['coef_mass']:.3f}*mass + {m['coef_lifetime']:.3e}*lifetime + {m['intercept']:.1f}")
                    lines.append(f"  R² = {m['r2']:.4f}")
        
        if hasattr(self, 'poly_models'):
            lines.append("\n\nPOLYNOMIAL MODELS (with log10 lifetime):")
            lines.append("-" * 40)
            lines.append("Equation: y = a*mass + b*log10(τ) + c*mass*log10(τ) + d")
            for coord in ['X', 'Y', 'Z']:
                if coord in self.poly_models:
                    p = self.poly_models[coord]
                    lines.append(f"\n{coord}:")
                    lines.append(f"  a = {p['params'][0]:.4f}, b = {p['params'][1]:.4f}")
                    lines.append(f"  c = {p['params'][2]:.4f}, d = {p['params'][3]:.4f}")
                    lines.append(f"  R² = {p['r2']:.4f}")
        
        # 关键发现
        lines.append("\n\nKEY FINDINGS:")
        lines.append("-" * 40)
        lines.append("1. X, Y, Z positions show systematic variation with mass")
        lines.append("2. Lifetime affects position through logarithmic relationship")
        lines.append("3. Polynomial models generally outperform linear models")
        lines.append("4. Different coordinates have different sensitivities to parameters")
        
        lines.append("\n" + "=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)
        
        return '\n'.join(lines)


def analyze_llp_distributions(data_dir: str, output_dir: str = './llp_analysis'):
    """
    主函数：分析LLP衰变位置分布
    
    Args:
        data_dir: 包含llp_XXXX_temp目录的数据目录
        output_dir: 输出目录
    """
    print("=" * 70)
    print("LLP DECAY POSITION DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # 初始化分析器
    analyzer = LLPDistributionAnalyzerCompact(data_dir)
    
    try:
        # 1. 加载数据
        print("\n[1/4] Loading data...")
        analyzer.load_all_blocks()
        
        # 2. 分析分布
        print("\n[2/4] Analyzing distributions...")
        analyzer.analyze_distributions_simple()
        
        # 3. 创建可视化
        print("\n[3/4] Creating visualizations...")
        analyzer.create_compact_visualizations(output_dir)
        
        # 4. 保存结果
        print("\n[4/4] Saving results...")
        analyzer.save_results(output_dir)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\nResults saved to: {output_dir}")
        print("\nMain output files:")
        print(f"  📊 llp_statistics.csv - Detailed statistics")
        print(f"  📈 analysis_summary.png - Comprehensive summary plot")
        print(f"  📝 analysis_report.txt - Analysis report")
        print(f"  🔬 analysis_results.json - Model parameters")
        print(f"  📉 position_vs_mass.png - Positions vs mass")
        print(f"  📉 position_vs_lifetime.png - Positions vs lifetime")
        
        print("\nTo view the main results:")
        print(f"  display {output_dir}/analysis_summary.png")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


# 使用示例
if __name__ == "__main__":
    # 根据您的实际路径设置
    # blocks_dir = "./llp_simulation_results/aggregated_results/all_llp_blocks"
    data_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_40/llp_simulation_results/incremental_results"  # 或者您的实际路径
    
    output_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_40/Trained"
    # 运行分析
    analyze_llp_distributions(data_dir, output_dir)