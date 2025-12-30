import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pickle

class LLPDistributionAnalyzer:
    """
    LLP衰变位置分布分析器
    专门处理您的数据结构
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.llp_data = {}
        self.distribution_models = {}
        self.summary_df = None
        
    def load_all_data(self):
        """加载所有LLP数据"""
        print("=" * 70)
        print("LOADING LLP DATA")
        print("=" * 70)
        
        # 查找所有llp目录
        llp_dirs = sorted(list(self.data_dir.glob("llp_*_temp")))
        print(f"Found {len(llp_dirs)} LLP directories")
        
        for llp_dir in tqdm(llp_dirs, desc="Loading data"):
            llp_id = llp_dir.stem.replace('_temp', '')
            
            try:
                # 从blocks目录加载数据
                data = self._load_llp_data(llp_dir, llp_id)
                if data:
                    self.llp_data[llp_id] = data
                    
            except Exception as e:
                print(f"\nWarning: Failed to load {llp_id}: {e}")
        
        if not self.llp_data:
            raise ValueError("No data loaded!")
        
        print(f"\nSuccessfully loaded {len(self.llp_data)} LLP datasets")
        self._create_summary_dataframe()
    
    def _load_llp_data(self, llp_dir: Path, llp_id: str) -> Optional[Dict]:
        """加载单个LLP的数据"""
        blocks_dir = llp_dir / "blocks"
        if not blocks_dir.exists():
            return None
        
        # 查找block子目录
        block_dirs = [d for d in blocks_dir.iterdir() if d.is_dir()]
        if not block_dirs:
            return None
        
        # 加载第一个block（通常只有一个）
        block_dir = block_dirs[0]
        h5_file = block_dir / "data.h5"
        
        if not h5_file.exists():
            return None
        
        # 读取HDF5文件
        with h5py.File(h5_file, 'r') as f:
            # 读取位置数据
            positions = f['positions'][:]
            
            # 读取权重
            weights = f['weights'][:]
            
            # 读取参数
            params_group = f['parameters']
            params = dict(params_group.attrs)
            
            # 确保参数是Python原生类型
            for key, value in params.items():
                if hasattr(value, 'item'):  # numpy类型
                    params[key] = value.item()
        
        return {
            'llp_id': llp_id,
            'positions': positions,
            'weights': weights,
            'params': params,
            'n_samples': len(positions),
            'total_weight': float(np.sum(weights))
        }
    
    def _create_summary_dataframe(self):
        """创建摘要DataFrame"""
        summary_data = []
        
        for llp_id, data in self.llp_data.items():
            params = data['params']
            
            summary = {
                'llp_id': llp_id,
                'mass': float(params.get('mass', np.nan)),
                'lifetime': float(params.get('lifetime', np.nan)),
                'tanb': float(params.get('tanb', np.nan)),
                'vis_br': float(params.get('vis_br', np.nan)),
                'n_samples': data['n_samples'],
                'total_weight': data['total_weight']
            }
            
            # 添加位置统计
            positions = data['positions']
            for idx, coord in enumerate(['x', 'y', 'z']):
                coord_data = positions[:, idx]
                weights = data['weights']
                
                weighted_mean = np.average(coord_data, weights=weights)
                weighted_std = np.sqrt(np.average((coord_data - weighted_mean)**2, weights=weights))
                
                summary[f'{coord}_mean'] = float(weighted_mean)
                summary[f'{coord}_std'] = float(weighted_std)
                summary[f'{coord}_min'] = float(np.min(coord_data))
                summary[f'{coord}_max'] = float(np.max(coord_data))
            
            summary_data.append(summary)
        
        self.summary_df = pd.DataFrame(summary_data)
        
        print(f"\nDATA SUMMARY:")
        print("-" * 40)
        print(f"Total LLP parameter sets: {len(self.summary_df)}")
        print(f"Total positions: {self.summary_df['n_samples'].sum():,}")
        print(f"Mass range: {self.summary_df['mass'].min():.3f} - {self.summary_df['mass'].max():.3f} GeV")
        print(f"Lifetime range: {self.summary_df['lifetime'].min():.2e} - {self.summary_df['lifetime'].max():.2e} mm")
        print(f"tanβ range: {self.summary_df['tanb'].min():.2f} - {self.summary_df['tanb'].max():.2f}")
        
        print(f"\nFirst 3 LLPs:")
        for i, row in self.summary_df.head(3).iterrows():
            print(f"  {row['llp_id']}: m={row['mass']:.3f}GeV, τ={row['lifetime']:.2e}mm, tanβ={row['tanb']:.1f}")
    
    def analyze_distributions(self):
        """为每个LLP分析衰变位置分布"""
        print(f"\n{'='*70}")
        print("ANALYZING DISTRIBUTIONS")
        print('='*70)
        
        for llp_id, data in tqdm(self.llp_data.items(), desc="Analyzing distributions"):
            try:
                distribution_model = self._analyze_llp_distribution(llp_id, data)
                if distribution_model:
                    self.distribution_models[llp_id] = distribution_model
            except Exception as e:
                print(f"\nWarning: Failed to analyze {llp_id}: {e}")
        
        print(f"\nSuccessfully analyzed distributions for {len(self.distribution_models)} LLPs")
    
    def _analyze_llp_distribution(self, llp_id: str, data: Dict) -> Dict:
        """分析单个LLP的分布"""
        positions = data['positions']
        weights = data['weights']
        params = data['params']
        
        # 分析每个坐标的分布
        coord_models = {}
        
        for idx, coord in enumerate(['x', 'y', 'z']):
            coord_data = positions[:, idx]
            
            # 计算加权统计
            weighted_mean = np.average(coord_data, weights=weights)
            weighted_std = np.sqrt(np.average((coord_data - weighted_mean)**2, weights=weights))
            
            # 创建核密度估计
            from scipy.stats import gaussian_kde
            weights_norm = weights / np.sum(weights)
            kde = gaussian_kde(coord_data, weights=weights_norm)
            
            # 计算百分位数
            percentiles = {
                'p5': float(np.percentile(coord_data, 5)),
                'p25': float(np.percentile(coord_data, 25)),
                'p50': float(np.percentile(coord_data, 50)),
                'p75': float(np.percentile(coord_data, 75)),
                'p95': float(np.percentile(coord_data, 95))
            }
            
            # 创建分布函数
            def pdf_func(x, use_kde=True):
                if use_kde:
                    return kde(x)
                else:
                    # 高斯近似
                    return stats.norm.pdf(x, loc=weighted_mean, scale=weighted_std)
            
            def cdf_func(x):
                # 经验CDF
                sorted_data = np.sort(coord_data)
                ecdf = np.searchsorted(sorted_data, x, side='right') / len(coord_data)
                return ecdf
            
            def rvs_func(size=1):
                # 从经验分布采样
                indices = np.random.choice(len(coord_data), size=size, p=weights_norm)
                return coord_data[indices]
            
            coord_models[coord] = {
                'mean': float(weighted_mean),
                'std': float(weighted_std),
                'min': float(np.min(coord_data)),
                'max': float(np.max(coord_data)),
                'percentiles': percentiles,
                'kde': kde,
                'pdf': pdf_func,
                'cdf': cdf_func,
                'rvs': rvs_func
            }
        
        return {
            'llp_id': llp_id,
            'params': params,
            'models': coord_models,
            'n_samples': data['n_samples'],
            'total_weight': data['total_weight']
        }
    
    def create_distribution_plots(self, output_dir: str = './llp_distributions'):
        """创建分布图"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating distribution plots in {output_path}...")
        
        # 为每个LLP创建图
        for llp_id, dist_model in tqdm(self.distribution_models.items(), 
                                      desc="Creating plots"):
            try:
                self._create_llp_distribution_plot(llp_id, dist_model, output_path)
            except Exception as e:
                print(f"\nWarning: Failed to create plot for {llp_id}: {e}")
        
        # 创建比较图
        self._create_comparison_plots(output_path)
        
        print(f"\n✓ All plots saved to {output_path}")
    
    def _create_llp_distribution_plot(self, llp_id: str, dist_model: Dict, output_path: Path):
        """创建单个LLP的分布图"""
        params = dist_model['params']
        
        # 创建标题
        title = f"LLP: {llp_id}\n"
        title += f"Mass: {params.get('mass', 'N/A'):.3f} GeV, "
        title += f"τ: {params.get('lifetime', 'N/A'):.2e} mm, "
        title += f"tanβ: {params.get('tanb', 'N/A'):.1f}"
        
        # 创建2x3的子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=14, y=1.02)
        
        # 第一行：PDF图
        for idx, coord in enumerate(['x', 'y', 'z']):
            ax = axes[0, idx]
            model = dist_model['models'][coord]
            
            # 生成x范围
            x_min, x_max = model['min'], model['max']
            x_range = np.linspace(x_min, x_max, 1000)
            
            # 绘制KDE PDF
            pdf_kde = model['pdf'](x_range, use_kde=True)
            ax.plot(x_range, pdf_kde, 'r-', linewidth=2, label='KDE PDF')
            
            # 绘制高斯近似
            pdf_gauss = model['pdf'](x_range, use_kde=False)
            ax.plot(x_range, pdf_gauss, 'b--', linewidth=1.5, alpha=0.7, label='Gaussian')
            
            # 添加均值和标准差线
            mean = model['mean']
            std = model['std']
            
            ax.axvline(mean, color='g', linestyle='-', alpha=0.5, label=f'Mean: {mean:.1f}')
            ax.axvline(mean - std, color='g', linestyle=':', alpha=0.5)
            ax.axvline(mean + std, color='g', linestyle=':', alpha=0.5, label=f'Std: ±{std:.1f}')
            
            ax.set_xlabel(f'{coord.upper()} Position (mm)', fontsize=11)
            ax.set_ylabel('Probability Density', fontsize=11)
            ax.set_title(f'{coord.upper()} Distribution', fontsize=12)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # 第二行：CDF和箱线图
        for idx, coord in enumerate(['x', 'y', 'z']):
            ax = axes[1, idx]
            model = dist_model['models'][coord]
            
            # 绘制CDF
            x_min, x_max = model['min'], model['max']
            x_range = np.linspace(x_min, x_max, 1000)
            cdf_vals = model['cdf'](x_range)
            
            ax.plot(x_range, cdf_vals, 'b-', linewidth=2, label='Empirical CDF')
            
            # 添加百分位数标记
            percentiles = model['percentiles']
            p_labels = ['5%', '25%', '50%', '75%', '95%']
            colors = ['r', 'orange', 'g', 'orange', 'r']
            
            for (key, value), label, color in zip(percentiles.items(), p_labels, colors):
                ax.axvline(value, color=color, linestyle='--', alpha=0.5)
                cdf_value = model['cdf'](np.array([value]))[0]
                ax.plot(value, cdf_value, 'o', color=color, markersize=5)
                ax.text(value, cdf_value + 0.05, label, 
                       ha='center', fontsize=9, color=color)
            
            ax.set_xlabel(f'{coord.upper()} Position (mm)', fontsize=11)
            ax.set_ylabel('Cumulative Probability', fontsize=11)
            ax.set_title(f'{coord.upper()} CDF with Percentiles', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_path / f'{llp_id}_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建3D散点图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取原始数据
        positions = self.llp_data[llp_id]['positions']
        
        # 抽样显示（避免太多点）
        if len(positions) > 5000:
            sample_idx = np.random.choice(len(positions), 5000, replace=False)
            plot_pos = positions[sample_idx]
        else:
            plot_pos = positions
        
        # 使用颜色表示权重
        weights = self.llp_data[llp_id]['weights']
        if len(weights) > 5000:
            plot_weights = weights[sample_idx]
        else:
            plot_weights = weights
        
        scatter = ax.scatter(plot_pos[:, 0], plot_pos[:, 1], plot_pos[:, 2],
                           c=plot_weights, cmap='viridis', alpha=0.3, s=1)
        
        # 添加均值点
        x_mean = dist_model['models']['x']['mean']
        y_mean = dist_model['models']['y']['mean']
        z_mean = dist_model['models']['z']['mean']
        
        ax.scatter([x_mean], [y_mean], [z_mean], c='red', s=100, marker='*', label='Mean')
        
        ax.set_xlabel('X (mm)', fontsize=11)
        ax.set_ylabel('Y (mm)', fontsize=11)
        ax.set_zlabel('Z (mm)', fontsize=11)
        ax.set_title(f'{llp_id}: 3D Decay Positions', fontsize=12)
        ax.legend()
        
        plt.colorbar(scatter, ax=ax, label='Weight')
        plt.tight_layout()
        plt.savefig(output_path / f'{llp_id}_3d_positions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_comparison_plots(self, output_path: Path):
        """创建LLP之间的比较图"""
        if len(self.distribution_models) < 2:
            return
        
        print("\nCreating comparison plots...")
        
        # 1. 分布统计随参数的变化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for row_idx, param in enumerate(['mass', 'lifetime']):
            for col_idx, coord in enumerate(['x', 'y', 'z']):
                ax = axes[row_idx, col_idx]
                
                # 收集数据
                x_vals = []
                mean_vals = []
                std_vals = []
                
                for llp_id, dist_model in self.distribution_models.items():
                    if param in dist_model['params']:
                        x_vals.append(float(dist_model['params'][param]))
                        mean_vals.append(dist_model['models'][coord]['mean'])
                        std_vals.append(dist_model['models'][coord]['std'])
                
                if len(x_vals) > 0:
                    # 排序
                    sort_idx = np.argsort(x_vals)
                    x_sorted = np.array(x_vals)[sort_idx]
                    mean_sorted = np.array(mean_vals)[sort_idx]
                    std_sorted = np.array(std_vals)[sort_idx]
                    
                    ax.errorbar(x_sorted, mean_sorted, yerr=std_sorted,
                               fmt='o-', alpha=0.7, capsize=3, markersize=4)
                    
                    x_label = 'Mass (GeV)' if param == 'mass' else 'Lifetime (mm)'
                    if param == 'lifetime':
                        ax.set_xscale('log')
                    
                    ax.set_xlabel(x_label, fontsize=11)
                    ax.set_ylabel(f'{coord.upper()} Mean ± Std (mm)', fontsize=11)
                    ax.set_title(f'{coord.upper()} vs {param.capitalize()}', fontsize=12)
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle('Distribution Statistics vs LLP Parameters', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'parameter_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. 参数空间热图：分布宽度
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (coord, cmap) in enumerate(zip(['x', 'y', 'z'], ['viridis', 'plasma', 'inferno'])):
            ax = axes[idx]
            
            x_vals = []
            y_vals = []
            z_vals = []
            
            for llp_id, dist_model in self.distribution_models.items():
                params = dist_model['params']
                if 'mass' in params and 'lifetime' in params:
                    x_vals.append(float(params['mass']))
                    y_vals.append(float(params['lifetime']))
                    z_vals.append(dist_model['models'][coord]['std'])
            
            if len(x_vals) > 0:
                scatter = ax.scatter(x_vals, np.log10(y_vals), c=z_vals,
                                   cmap=cmap, alpha=0.7, s=50)
                
                ax.set_xlabel('Mass (GeV)', fontsize=11)
                ax.set_ylabel('log10(Lifetime) (mm)', fontsize=11)
                ax.set_title(f'{coord.upper()} Std Dev in Parameter Space', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                plt.colorbar(scatter, ax=ax, label=f'{coord.upper()} Std Dev (mm)')
        
        plt.suptitle('Distribution Width Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'std_dev_heatmaps.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. 综合总结图
        fig = plt.figure(figsize=(16, 12))
        
        # 创建布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 子图1: 参数空间
        ax1 = fig.add_subplot(gs[0, 0])
        x_vals = []
        y_vals = []
        colors = []
        
        for llp_id, dist_model in self.distribution_models.items():
            params = dist_model['params']
            if 'mass' in params and 'lifetime' in params:
                x_vals.append(float(params['mass']))
                y_vals.append(float(params['lifetime']))
                # 使用总权重作为颜色
                colors.append(dist_model['total_weight'])
        
        if x_vals:
            scatter1 = ax1.scatter(x_vals, np.log10(y_vals), c=colors, 
                                 cmap='viridis', alpha=0.7, s=50)
            ax1.set_xlabel('Mass (GeV)', fontsize=11)
            ax1.set_ylabel('log10(Lifetime) (mm)', fontsize=11)
            ax1.set_title('Parameter Space (colored by total weight)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter1, ax=ax1, label='Total Weight')
        
        # 子图2: 样本数量分布
        ax2 = fig.add_subplot(gs[0, 1])
        n_samples = [d['n_samples'] for d in self.llp_data.values()]
        ax2.hist(n_samples, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Positions', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution of Sample Sizes', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 平均位置分布
        ax3 = fig.add_subplot(gs[0, 2])
        mean_positions = []
        for dist_model in self.distribution_models.values():
            for coord in ['x', 'y', 'z']:
                mean_positions.append(dist_model['models'][coord]['mean'])
        
        ax3.hist(mean_positions, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Mean Position (mm)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Distribution of Mean Positions', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 子图4-6: 各坐标的标准差分布
        for idx, coord in enumerate(['x', 'y', 'z']):
            ax = fig.add_subplot(gs[1, idx])
            std_vals = [dist_model['models'][coord]['std'] 
                       for dist_model in self.distribution_models.values()]
            
            ax.hist(std_vals, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(f'{coord.upper()} Std Dev (mm)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{coord.upper()} Spread Distribution', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # 子图7-9: 相关系数
        ax7 = fig.add_subplot(gs[2, 0])
        # 准备相关数据
        corr_data = []
        for dist_model in self.distribution_models.values():
            params = dist_model['params']
            models = dist_model['models']
            
            if 'mass' in params and 'lifetime' in params:
                row = {
                    'mass': float(params['mass']),
                    'log_lifetime': np.log10(float(params['lifetime'])),
                    'x_mean': models['x']['mean'],
                    'y_mean': models['y']['mean'],
                    'z_mean': models['z']['mean']
                }
                corr_data.append(row)
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            correlation = corr_df.corr()
            
            im = ax7.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
            ax7.set_xticks(range(len(correlation.columns)))
            ax7.set_xticklabels([col[:10] for col in correlation.columns], 
                              rotation=45, fontsize=9)
            ax7.set_yticks(range(len(correlation.columns)))
            ax7.set_yticklabels([col[:10] for col in correlation.columns], 
                              fontsize=9)
            ax7.set_title('Correlation Matrix', fontsize=12)
            plt.colorbar(im, ax=ax7)
        
        # 子图8: 关键统计
        ax8 = fig.add_subplot(gs[2, 1])
        stats_text = [
            f"Total LLPs: {len(self.llp_data)}",
            f"Total positions: {self.summary_df['n_samples'].sum():,}",
            f"Mass range: {self.summary_df['mass'].min():.3f}-{self.summary_df['mass'].max():.3f} GeV",
            f"Lifetime range: {self.summary_df['lifetime'].min():.2e}-{self.summary_df['lifetime'].max():.2e} mm",
            f"tanβ range: {self.summary_df['tanb'].min():.2f}-{self.summary_df['tanb'].max():.2f}"
        ]
        
        ax8.text(0.1, 0.9, '\n'.join(stats_text), transform=ax8.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax8.axis('off')
        ax8.set_title('Key Statistics', fontsize=12)
        
        # 子图9: 参数分布
        ax9 = fig.add_subplot(gs[2, 2])
        if 'mass' in self.summary_df.columns and 'tanb' in self.summary_df.columns:
            scatter9 = ax9.scatter(self.summary_df['mass'], self.summary_df['tanb'],
                                 c=np.log10(self.summary_df['lifetime']), 
                                 cmap='viridis', alpha=0.7, s=50)
            ax9.set_xlabel('Mass (GeV)', fontsize=11)
            ax9.set_ylabel('tanβ', fontsize=11)
            ax9.set_title('Mass vs tanβ (colored by log10(τ))', fontsize=12)
            ax9.grid(True, alpha=0.3)
            plt.colorbar(scatter9, ax=ax9, label='log10(Lifetime)')
        
        plt.suptitle('LLP Decay Position Analysis Summary', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'analysis_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir: str = './llp_distributions'):
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {output_path}...")
        
        # 1. 保存摘要数据
        if self.summary_df is not None:
            csv_path = output_path / 'llp_summary.csv'
            self.summary_df.to_csv(csv_path, index=False)
            print(f"✓ Summary saved to: {csv_path}")
        
        # 2. 保存分布模型（轻量级版本）
        if self.distribution_models:
            models_dir = output_path / 'distribution_models'
            models_dir.mkdir(exist_ok=True)
            
            for llp_id, dist_model in self.distribution_models.items():
                # 创建轻量级版本（移除函数和KDE对象）
                light_model = {
                    'llp_id': dist_model['llp_id'],
                    'params': dist_model['params'],
                    'n_samples': dist_model['n_samples'],
                    'total_weight': dist_model['total_weight'],
                    'model_stats': {}
                }
                
                for coord in ['x', 'y', 'z']:
                    model = dist_model['models'][coord]
                    light_model['model_stats'][coord] = {
                        'mean': model['mean'],
                        'std': model['std'],
                        'min': model['min'],
                        'max': model['max'],
                        'percentiles': model['percentiles']
                    }
                
                # 保存为JSON
                import json
                model_file = models_dir / f'{llp_id}_model.json'
                with open(model_file, 'w') as f:
                    json.dump(light_model, f, indent=2, default=str)
            
            print(f"✓ Distribution models saved to: {models_dir}/")
        
        # 3. 生成报告
        self._generate_report(output_path)
        
        print(f"\n✓ All results saved to: {output_path}")
    
    def _generate_report(self, output_path: Path):
        """生成分析报告"""
        report = []
        
        report.append("=" * 70)
        report.append("LLP DECAY POSITION DISTRIBUTION ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"\nAnalysis Date: {pd.Timestamp.now()}")
        report.append(f"Total LLP datasets analyzed: {len(self.llp_data)}")
        
        if self.summary_df is not None:
            report.append(f"\nDATA SUMMARY:")
            report.append("-" * 40)
            report.append(f"Total positions: {self.summary_df['n_samples'].sum():,}")
            report.append(f"Total weighted events: {self.summary_df['total_weight'].sum():.0f}")
            report.append(f"Mass range: {self.summary_df['mass'].min():.3f} - {self.summary_df['mass'].max():.3f} GeV")
            report.append(f"Lifetime range: {self.summary_df['lifetime'].min():.2e} - {self.summary_df['lifetime'].max():.2e} mm")
            report.append(f"tanβ range: {self.summary_df['tanb'].min():.2f} - {self.summary_df['tanb'].max():.2f}")
            report.append(f"Visible BR range: {self.summary_df['vis_br'].min():.2e} - {self.summary_df['vis_br'].max():.2e}")
        
        # 分布统计
        report.append(f"\n\nDISTRIBUTION STATISTICS:")
        report.append("-" * 40)
        
        if self.distribution_models:
            for coord in ['x', 'y', 'z']:
                means = [model['models'][coord]['mean'] for model in self.distribution_models.values()]
                stds = [model['models'][coord]['std'] for model in self.distribution_models.values()]
                
                report.append(f"\n{coord.upper()} coordinate:")
                report.append(f"  Mean position: {np.mean(means):.1f} ± {np.std(means):.1f} mm")
                report.append(f"  Average spread: {np.mean(stds):.1f} ± {np.std(stds):.1f} mm")
                report.append(f"  Position range: [{np.min(means):.1f}, {np.max(means):.1f}] mm")
        
        # 关键发现
        report.append(f"\n\nKEY FINDINGS:")
        report.append("-" * 40)
        
        if self.distribution_models and len(self.distribution_models) > 1:
            # 检查参数相关性
            mass_vals = []
            x_means = []
            
            for model in self.distribution_models.values():
                if 'mass' in model['params']:
                    mass_vals.append(float(model['params']['mass']))
                    x_means.append(model['models']['x']['mean'])
            
            if len(mass_vals) > 1:
                corr = np.corrcoef(mass_vals, x_means)[0, 1]
                report.append(f"1. Correlation between mass and X position: {corr:.3f}")
            
            # 检查寿命对分布宽度的影响
            lifetime_vals = []
            std_vals = []
            
            for model in self.distribution_models.values():
                if 'lifetime' in model['params']:
                    lifetime_vals.append(float(model['params']['lifetime']))
                    std_vals.append(model['models']['z']['std'])
            
            if len(lifetime_vals) > 1:
                corr = np.corrcoef(np.log10(lifetime_vals), std_vals)[0, 1]
                report.append(f"2. Correlation between log10(lifetime) and Z spread: {corr:.3f}")
        
        report.append(f"\n\nOUTPUT FILES:")
        report.append("-" * 40)
        report.append("1. llp_summary.csv - Summary statistics for all LLPs")
        report.append("2. distribution_models/ - JSON files with distribution statistics")
        report.append("3. Individual PNG files - Distribution plots for each LLP")
        report.append("4. Comparison PNG files - Parameter space and statistical plots")
        report.append("5. analysis_summary.png - Comprehensive summary plot")
        
        report.append(f"\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        report_text = '\n'.join(report)
        
        with open(output_path / 'analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"✓ Report saved to: {output_path}/analysis_report.txt")


def main():
    """主函数"""
    # 设置路径
    data_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_F/llp_simulation_results/incremental_results"
    output_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_F/distributution_density"
    
    print("=" * 70)
    print("LLP DECAY POSITION DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # 创建分析器
    analyzer = LLPDistributionAnalyzer(data_dir)
    
    try:
        # 1. 加载数据
        print("\n[1/3] Loading data...")
        analyzer.load_all_data()
        
        # 2. 分析分布
        print("\n[2/3] Analyzing distributions...")
        analyzer.analyze_distributions()
        
        # 3. 创建可视化
        print("\n[3/3] Creating visualizations...")
        # analyzer.create_distribution_plots(output_dir)
        
        # 4. 保存结果
        analyzer.save_results(output_dir)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\n✅ Results saved to: {output_dir}")
        print(f"\n📊 Key output files:")
        print(f"  {output_dir}/llp_summary.csv - Complete summary")
        print(f"  {output_dir}/analysis_summary.png - Comprehensive summary plot")
        print(f"  {output_dir}/analysis_report.txt - Detailed report")
        print(f"  {output_dir}/distribution_models/ - Individual distribution models")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()