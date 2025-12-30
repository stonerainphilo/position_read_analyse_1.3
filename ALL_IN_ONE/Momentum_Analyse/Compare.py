import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings
import json
import h5py
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

class KDE_MC_ComparisonAnalyzer:
    """
    KDE分布模拟与MC模拟结果对比分析器
    读取KDE拟合文件和原始MC数据进行比较
    """
    
    def __init__(self, kde_results_dir: str, mc_data_dir: str):
        """
        初始化对比分析器
        
        Parameters:
        -----------
        kde_results_dir : str
            存放KDE拟合结果的目录（包含distribution_models/文件夹）
        mc_data_dir : str
            存放原始MC模拟数据的目录
        """
        self.kde_dir = Path(kde_results_dir)
        self.mc_dir = Path(mc_data_dir)
        
        # 存储数据
        self.kde_models = {}
        self.mc_data = {}
        self.comparison_results = {}
        self.summary_stats = pd.DataFrame()
        
        print("=" * 70)
        print("KDE-MC COMPARISON ANALYZER")
        print("=" * 70)
    
    def load_kde_models(self):
        """加载KDE拟合模型"""
        print("\n[1/3] Loading KDE models...")
        
        models_dir = self.kde_dir / "distribution_models"
        if not models_dir.exists():
            raise FileNotFoundError(f"KDE models directory not found: {models_dir}")
        
        model_files = list(models_dir.glob("*_model.json"))
        print(f"Found {len(model_files)} KDE model files")
        
        for model_file in tqdm(model_files, desc="Loading KDE models"):
            try:
                with open(model_file, 'r') as f:
                    model_data = json.load(f)
                
                llp_id = model_data['llp_id']
                self.kde_models[llp_id] = model_data
                
            except Exception as e:
                print(f"\nWarning: Failed to load {model_file.name}: {e}")
        
        print(f"✓ Successfully loaded {len(self.kde_models)} KDE models")
    
    def load_mc_data(self):
        """加载原始MC模拟数据"""
        print("\n[2/3] Loading MC simulation data...")
        
        # 查找所有llp目录（原始数据）
        llp_dirs = sorted(list(self.mc_dir.glob("llp_*_temp")))
        print(f"Found {len(llp_dirs)} MC data directories")
        
        for llp_dir in tqdm(llp_dirs, desc="Loading MC data"):
            llp_id = llp_dir.stem.replace('_temp', '')
            
            # 检查对应的KDE模型是否存在
            if llp_id not in self.kde_models:
                continue
            
            try:
                mc_data = self._load_mc_data_single(llp_dir, llp_id)
                if mc_data:
                    self.mc_data[llp_id] = mc_data
                    
            except Exception as e:
                print(f"\nWarning: Failed to load MC data for {llp_id}: {e}")
        
        print(f"✓ Successfully loaded {len(self.mc_data)} MC datasets")
        print(f"  Matching KDE-MC pairs: {len(set(self.kde_models.keys()) & set(self.mc_data.keys()))}")
    
    def _load_mc_data_single(self, llp_dir: Path, llp_id: str) -> Optional[Dict]:
        """加载单个LLP的MC数据"""
        blocks_dir = llp_dir / "blocks"
        if not blocks_dir.exists():
            return None
        
        # 查找block子目录
        block_dirs = [d for d in blocks_dir.iterdir() if d.is_dir()]
        if not block_dirs:
            return None
        
        # 加载第一个block
        block_dir = block_dirs[0]
        h5_file = block_dir / "data.h5"
        
        if not h5_file.exists():
            return None
        
        # 读取HDF5文件
        with h5py.File(h5_file, 'r') as f:
            positions = f['positions'][:]
            weights = f['weights'][:]
            
            # 读取参数
            params_group = f['parameters']
            params = dict(params_group.attrs)
            
            # 确保参数是Python原生类型
            for key, value in params.items():
                if hasattr(value, 'item'):
                    params[key] = value.item()
        
        return {
            'llp_id': llp_id,
            'positions': positions,
            'weights': weights,
            'params': params,
            'n_samples': len(positions),
            'total_weight': float(np.sum(weights))
        }
    
    def compare_distributions(self):
        """对比KDE模型和MC数据"""
        print("\n[3/3] Comparing distributions...")
        
        comparison_data = []
        
        # 获取共同的LLP ID
        common_ids = set(self.kde_models.keys()) & set(self.mc_data.keys())
        print(f"Comparing {len(common_ids)} common LLPs")
        
        for llp_id in tqdm(common_ids, desc="Comparing"):
            try:
                kde_model = self.kde_models[llp_id]
                mc_data = self.mc_data[llp_id]
                
                comparison = self._compare_single_llp(llp_id, kde_model, mc_data)
                if comparison:
                    self.comparison_results[llp_id] = comparison
                    comparison_data.append(comparison['summary'])
                    
            except Exception as e:
                print(f"\nWarning: Failed to compare {llp_id}: {e}")
        
        # 创建汇总DataFrame
        if comparison_data:
            self.summary_stats = pd.DataFrame(comparison_data)
            print(f"\n✓ Comparison completed for {len(self.comparison_results)} LLPs")
            
            # 打印总体统计
            self._print_summary_statistics()
    
    def _compare_single_llp(self, llp_id: str, kde_model: Dict, mc_data: Dict) -> Dict:
        """对比单个LLP的KDE和MC分布"""
        mc_positions = mc_data['positions']
        mc_weights = mc_data['weights']
        mc_params = mc_data['params']
        
        # 验证参数一致性
        param_checks = {}
        for key in ['mass', 'lifetime', 'tanb', 'vis_br']:
            if key in kde_model['params'] and key in mc_params:
                kde_val = float(kde_model['params'][key])
                mc_val = float(mc_params[key])
                param_checks[key] = {
                    'kde': kde_val,
                    'mc': mc_val,
                    'relative_diff': abs(kde_val - mc_val) / mc_val if mc_val != 0 else np.nan
                }
        
        # 对比每个坐标的分布
        coord_comparisons = {}
        
        for coord in ['x', 'y', 'z']:
            if coord not in kde_model['model_stats']:
                continue
            
            # 获取MC数据
            idx = ['x', 'y', 'z'].index(coord)
            mc_values = mc_positions[:, idx]
            
            # 获取KDE统计
            kde_stats = kde_model['model_stats'][coord]
            
            # 1. 统计量对比
            mc_mean = np.average(mc_values, weights=mc_weights)
            mc_std = np.sqrt(np.average((mc_values - mc_mean)**2, weights=mc_weights))
            mc_median = np.median(mc_values)
            
            stats_comparison = {
                'kde_mean': kde_stats['mean'],
                'mc_mean': float(mc_mean),
                'mean_diff': kde_stats['mean'] - mc_mean,
                'mean_rel_diff': (kde_stats['mean'] - mc_mean) / mc_mean if mc_mean != 0 else np.nan,
                
                'kde_std': kde_stats['std'],
                'mc_std': float(mc_std),
                'std_diff': kde_stats['std'] - mc_std,
                'std_rel_diff': (kde_stats['std'] - mc_std) / mc_std if mc_std != 0 else np.nan,
                
                'kde_median': kde_stats['percentiles'].get('p50', np.nan),
                'mc_median': float(mc_median),
                'median_diff': kde_stats['percentiles'].get('p50', np.nan) - mc_median,
            }
            
            # 2. KS检验（Kolmogorov-Smirnov test）
            # 从MC数据中采样（带权重）
            weights_norm = mc_weights / np.sum(mc_weights)
            mc_sample = np.random.choice(mc_values, size=min(10000, len(mc_values)), 
                                        p=weights_norm, replace=True)
            
            # 从KDE分布中采样（模拟）
            # 注意：这里我们实际上没有KDE的采样函数，只能使用原始数据
            # 所以使用百分位数对比作为替代
            
            # 3. 分位数对比
            percentiles = [5, 25, 50, 75, 95]
            percentile_comparison = {}
            
            for p in percentiles:
                kde_p = kde_stats['percentiles'].get(f'p{p}', np.nan)
                mc_p = float(np.percentile(mc_values, p))
                
                percentile_comparison[f'p{p}'] = {
                    'kde': kde_p,
                    'mc': mc_p,
                    'diff': kde_p - mc_p,
                    'rel_diff': (kde_p - mc_p) / mc_p if mc_p != 0 else np.nan
                }
            
            # 4. 分布形状对比（使用直方图）
            n_bins = 50
            mc_hist, bin_edges = np.histogram(mc_values, bins=n_bins, weights=mc_weights, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 计算KDE在bin中心的值（估计值）
            # 由于我们没有KDE函数，使用高斯近似
            kde_pdf = stats.norm.pdf(bin_centers, loc=kde_stats['mean'], scale=kde_stats['std'])
            
            # 计算直方图与KDE的差异
            mse = np.mean((mc_hist - kde_pdf) ** 2)
            mae = np.mean(np.abs(mc_hist - kde_pdf))
            
            shape_comparison = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'max_diff': np.max(np.abs(mc_hist - kde_pdf))
            }
            
            coord_comparisons[coord] = {
                'stats': stats_comparison,
                'percentiles': percentile_comparison,
                'shape': shape_comparison,
                'hist_data': {
                    'mc_hist': mc_hist,
                    'kde_pdf': kde_pdf,
                    'bin_centers': bin_centers,
                    'bin_edges': bin_edges
                }
            }
        
        # 汇总对比结果
        comparison_summary = {
            'llp_id': llp_id,
            'n_samples_mc': mc_data['n_samples'],
            'n_samples_kde': kde_model['n_samples'],
            'total_weight_mc': mc_data['total_weight'],
            'total_weight_kde': kde_model['total_weight'],
            'params_check': param_checks,
            
            # 平均统计差异
            'mean_abs_mean_diff': np.mean([abs(c['stats']['mean_diff']) for c in coord_comparisons.values()]),
            'mean_abs_std_diff': np.mean([abs(c['stats']['std_diff']) for c in coord_comparisons.values()]),
            'mean_rmse': np.mean([c['shape']['rmse'] for c in coord_comparisons.values()]),
            'max_rmse': np.max([c['shape']['rmse'] for c in coord_comparisons.values()]),
        }
        
        return {
            'llp_id': llp_id,
            'kde_model': kde_model,
            'mc_data': mc_data,
            'comparisons': coord_comparisons,
            'summary': comparison_summary
        }
    
    def _print_summary_statistics(self):
        """打印对比统计摘要"""
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY STATISTICS")
        print("=" * 70)
        
        if self.summary_stats.empty:
            print("No comparison results available.")
            return
        
        print(f"\nTotal LLPs compared: {len(self.summary_stats)}")
        
        # 位置差异统计
        print(f"\nPosition Differences:")
        print(f"  Mean absolute mean diff: {self.summary_stats['mean_abs_mean_diff'].mean():.4f} ± {self.summary_stats['mean_abs_mean_diff'].std():.4f} mm")
        print(f"  Mean absolute std diff:  {self.summary_stats['mean_abs_std_diff'].mean():.4f} ± {self.summary_stats['mean_abs_std_diff'].std():.4f} mm")
        
        # 形状差异统计
        print(f"\nDistribution Shape Differences:")
        print(f"  Mean RMSE: {self.summary_stats['mean_rmse'].mean():.6f} ± {self.summary_stats['mean_rmse'].std():.6f}")
        print(f"  Max RMSE:  {self.summary_stats['max_rmse'].max():.6f}")
        
        # 按参数分组统计
        if 'mass' in self.kde_models[next(iter(self.kde_models.keys()))]['params']:
            masses = [self.kde_models[llp_id]['params']['mass'] for llp_id in self.summary_stats['llp_id']]
            print(f"\nCorrelation with mass:")
            if len(masses) > 1:
                corr_mean = np.corrcoef(masses, self.summary_stats['mean_abs_mean_diff'])[0, 1]
                corr_rmse = np.corrcoef(masses, self.summary_stats['mean_rmse'])[0, 1]
                print(f"  Correlation (mass vs mean diff): {corr_mean:.3f}")
                print(f"  Correlation (mass vs RMSE):      {corr_rmse:.3f}")
    
    def create_comparison_plots(self, output_dir: str = './kde_mc_comparison'):
        """创建对比可视化"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating comparison plots in {output_path}...")
        
        # 1. 为每个LLP创建详细对比图
        for llp_id, comparison in tqdm(self.comparison_results.items(), 
                                      desc="Creating detailed plots"):
            try:
                self._create_single_comparison_plot(llp_id, comparison, output_path)
            except Exception as e:
                print(f"\nWarning: Failed to create plot for {llp_id}: {e}")
        
        # 2. 创建汇总对比图
        self._create_summary_comparison_plots(output_path)
        
        # 3. 创建性能评估图
        self._create_performance_evaluation_plots(output_path)
        
        print(f"\n✓ All comparison plots saved to {output_path}")
    
    def _create_single_comparison_plot(self, llp_id: str, comparison: Dict, output_path: Path):
        """创建单个LLP的详细对比图"""
        kde_model = comparison['kde_model']
        mc_data = comparison['mc_data']
        coord_comparisons = comparison['comparisons']
        
        # 创建标题
        params = kde_model['params']
        title = f"KDE vs MC Comparison: {llp_id}\n"
        title += f"Mass: {params.get('mass', 'N/A'):.3f} GeV, "
        title += f"τ: {params.get('lifetime', 'N/A'):.2e} mm, "
        title += f"tanβ: {params.get('tanb', 'N/A'):.1f}"
        
        # 创建2x3的子图
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(title, fontsize=14, y=1.02)
        
        gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.3)
        
        # 第一行：PDF对比
        for idx, coord in enumerate(['x', 'y', 'z']):
            ax = plt.subplot(gs[0, idx])
            comp = coord_comparisons.get(coord)
            
            if comp is None:
                ax.text(0.5, 0.5, f"No data for {coord}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{coord.upper()} Distribution", fontsize=12)
                continue
            
            hist_data = comp['hist_data']
            
            # 绘制MC直方图
            ax.bar(hist_data['bin_centers'], hist_data['mc_hist'], 
                  width=hist_data['bin_edges'][1] - hist_data['bin_edges'][0],
                  alpha=0.6, label='MC (histogram)', color='blue')
            
            # 绘制KDE PDF
            ax.plot(hist_data['bin_centers'], hist_data['kde_pdf'], 
                   'r-', linewidth=2, label='KDE (Gaussian approx)', alpha=0.8)
            
            # 添加统计信息
            stats_text = f"Δmean = {comp['stats']['mean_diff']:.2f} mm\n"
            stats_text += f"Δstd = {comp['stats']['std_diff']:.2f} mm\n"
            stats_text += f"RMSE = {comp['shape']['rmse']:.4f}"
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(f'{coord.upper()} Position (mm)', fontsize=11)
            ax.set_ylabel('Probability Density', fontsize=11)
            ax.set_title(f'{coord.upper()} Distribution Comparison', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # 第二行：分位数对比（箱线图）
        for idx, coord in enumerate(['x', 'y', 'z']):
            ax = plt.subplot(gs[1, idx])
            comp = coord_comparisons.get(coord)
            
            if comp is None:
                continue
            
            percentiles = comp['percentiles']
            p_values = [5, 25, 50, 75, 95]
            
            kde_percentiles = [percentiles[f'p{p}']['kde'] for p in p_values]
            mc_percentiles = [percentiles[f'p{p}']['mc'] for p in p_values]
            diffs = [percentiles[f'p{p}']['diff'] for p in p_values]
            
            # 创建箱线图对比
            positions = [1, 2]
            labels = ['KDE', 'MC']
            
            # 绘制箱线图
            bp1 = ax.boxplot([kde_percentiles], positions=[positions[0]], 
                           widths=0.3, patch_artist=True)
            bp2 = ax.boxplot([mc_percentiles], positions=[positions[1]], 
                           widths=0.3, patch_artist=True)
            
            bp1['boxes'][0].set_facecolor('red')
            bp1['boxes'][0].set_alpha(0.6)
            bp2['boxes'][0].set_facecolor('blue')
            bp2['boxes'][0].set_alpha(0.6)
            
            # 添加差异值
            for i, p in enumerate(p_values):
                ax.plot(positions, [kde_percentiles[i], mc_percentiles[i]], 
                       'k--', alpha=0.5, linewidth=0.5)
                ax.text(1.5, (kde_percentiles[i] + mc_percentiles[i])/2,
                       f'Δ={diffs[i]:.1f}', ha='center', fontsize=8)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.set_xlabel('Method', fontsize=11)
            ax.set_ylabel('Percentile Values (mm)', fontsize=11)
            ax.set_title(f'{coord.upper()} Percentile Comparison', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 第三行：误差分析
        # 子图1：统计量差异
        ax1 = plt.subplot(gs[2, 0])
        metrics = ['Mean Diff', 'Std Diff']
        colors = ['skyblue', 'lightcoral']
        
        for idx, coord in enumerate(['x', 'y', 'z']):
            comp = coord_comparisons.get(coord)
            if comp is None:
                continue
            
            mean_diff = comp['stats']['mean_diff']
            std_diff = comp['stats']['std_diff']
            
            ax1.bar(idx - 0.2, mean_diff, width=0.4, color=colors[0], 
                   alpha=0.7, label='Mean Diff' if idx == 0 else "")
            ax1.bar(idx + 0.2, std_diff, width=0.4, color=colors[1], 
                   alpha=0.7, label='Std Diff' if idx == 0 else "")
        
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(['X', 'Y', 'Z'])
        ax1.set_xlabel('Coordinate', fontsize=11)
        ax1.set_ylabel('Difference (mm)', fontsize=11)
        ax1.set_title('Statistical Differences', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 子图2：形状误差
        ax2 = plt.subplot(gs[2, 1])
        errors = []
        coords = []
        
        for idx, coord in enumerate(['x', 'y', 'z']):
            comp = coord_comparisons.get(coord)
            if comp is None:
                continue
            
            errors.append(comp['shape']['rmse'])
            coords.append(coord)
        
        bars = ax2.bar(coords, errors, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Coordinate', fontsize=11)
        ax2.set_ylabel('RMSE', fontsize=11)
        ax2.set_title('Distribution Shape Errors', fontsize=12)
        
        # 在柱子上添加数值
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{error:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 子图3：总体评估
        ax3 = plt.subplot(gs[2, 2])
        
        summary = comparison['summary']
        metrics_names = ['Mean Abs Mean Diff', 'Mean Abs Std Diff', 'Mean RMSE']
        metrics_values = [summary['mean_abs_mean_diff'], 
                         summary['mean_abs_std_diff'], 
                         summary['mean_rmse']]
        colors_eval = ['gold', 'orange', 'coral']
        
        bars = ax3.bar(metrics_names, metrics_values, color=colors_eval, alpha=0.7)
        ax3.set_ylabel('Error Value', fontsize=11)
        ax3.set_title('Overall Performance Metrics', fontsize=12)
        ax3.tick_params(axis='x', rotation=15)
        
        # 在柱子上添加数值
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{llp_id}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_summary_comparison_plots(self, output_path: Path):
        """创建汇总对比图"""
        if len(self.comparison_results) < 2:
            return
        
        print("\nCreating summary comparison plots...")
        
        # 1. 所有LLP的误差分布
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('KDE vs MC Comparison - Overall Statistics', fontsize=16, y=1.02)
        
        # 子图1：平均误差分布
        ax1 = axes[0, 0]
        mean_diffs = []
        std_diffs = []
        
        for llp_id, comp in self.comparison_results.items():
            for coord in ['x', 'y', 'z']:
                coord_comp = comp['comparisons'].get(coord)
                if coord_comp:
                    mean_diffs.append(coord_comp['stats']['mean_diff'])
                    std_diffs.append(coord_comp['stats']['std_diff'])
        
        ax1.hist(mean_diffs, bins=30, alpha=0.7, color='skyblue', edgecolor='black', 
                label=f'Mean Diff\nμ={np.mean(mean_diffs):.3f}±{np.std(mean_diffs):.3f}')
        ax1.hist(std_diffs, bins=30, alpha=0.7, color='lightcoral', edgecolor='black',
                label=f'Std Diff\nμ={np.mean(std_diffs):.3f}±{np.std(std_diffs):.3f}')
        
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Difference (mm)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Distribution of Statistical Differences', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：RMSE分布
        ax2 = axes[0, 1]
        rmse_values = []
        
        for llp_id, comp in self.comparison_results.items():
            for coord in ['x', 'y', 'z']:
                coord_comp = comp['comparisons'].get(coord)
                if coord_comp:
                    rmse_values.append(coord_comp['shape']['rmse'])
        
        ax2.hist(rmse_values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('RMSE', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution of Shape Errors (RMSE)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        ax2.text(0.7, 0.9, f'Mean: {np.mean(rmse_values):.5f}\nStd: {np.std(rmse_values):.5f}',
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 子图3：误差与样本量的关系
        ax3 = axes[1, 0]
        sample_sizes = []
        mean_errors = []
        
        for llp_id, comp in self.comparison_results.items():
            sample_sizes.append(comp['mc_data']['n_samples'])
            mean_errors.append(comp['summary']['mean_rmse'])
        
        ax3.scatter(sample_sizes, mean_errors, alpha=0.6, s=50, color='purple')
        ax3.set_xlabel('MC Sample Size', fontsize=11)
        ax3.set_ylabel('Mean RMSE', fontsize=11)
        ax3.set_title('Error vs Sample Size', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 添加趋势线（如果足够多数据点）
        if len(sample_sizes) > 1:
            try:
                z = np.polyfit(sample_sizes, mean_errors, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(sample_sizes), max(sample_sizes), 100)
                ax3.plot(x_range, p(x_range), "r--", alpha=0.7, 
                        label=f'Slope: {z[0]:.2e}')
                ax3.legend(fontsize=10)
            except:
                pass
        
        # 子图4：误差随参数的变化
        ax4 = axes[1, 1]
        
        if 'mass' in self.kde_models[next(iter(self.kde_models.keys()))]['params']:
            masses = []
            lifetime_errors = []
            
            for llp_id, comp in self.comparison_results.items():
                if llp_id in self.kde_models:
                    params = self.kde_models[llp_id]['params']
                    if 'mass' in params and 'lifetime' in params:
                        masses.append(float(params['mass']))
                        # 使用Z坐标的RMSE作为代表性误差
                        z_comp = comp['comparisons'].get('z')
                        if z_comp:
                            lifetime_errors.append(z_comp['shape']['rmse'])
            
            if masses and lifetime_errors and len(masses) == len(lifetime_errors):
                scatter = ax4.scatter(masses, lifetime_errors, alpha=0.6, s=50, 
                                     c=masses, cmap='viridis')
                ax4.set_xlabel('Mass (GeV)', fontsize=11)
                ax4.set_ylabel('Z-coordinate RMSE', fontsize=11)
                ax4.set_title('Error vs Mass', fontsize=12)
                ax4.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax4, label='Mass (GeV)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'summary_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. 热力图：不同参数的误差
        if len(self.comparison_results) > 5:
            self._create_error_heatmaps(output_path)
    
    def _create_error_heatmaps(self, output_path: Path):
        """创建误差热力图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Error Analysis in Parameter Space', fontsize=14, y=1.02)
        
        # 准备数据
        data = []
        for llp_id, comp in self.comparison_results.items():
            if llp_id in self.kde_models:
                params = self.kde_models[llp_id]['params']
                if 'mass' in params and 'lifetime' in params:
                    row = {
                        'mass': float(params['mass']),
                        'log_lifetime': np.log10(float(params['lifetime'])),
                        'tanb': float(params.get('tanb', 1.0)),
                        'mean_error': comp['summary']['mean_abs_mean_diff'],
                        'rmse': comp['summary']['mean_rmse']
                    }
                    data.append(row)
        
        if not data:
            return
        
        error_df = pd.DataFrame(data)
        
        # 子图1：平均误差热图
        ax1 = axes[0]
        if len(error_df) > 4:
            # 创建2D直方图
            heatmap, xedges, yedges = np.histogram2d(
                error_df['mass'], 
                error_df['log_lifetime'], 
                bins=10,
                weights=error_df['mean_error']
            )
            
            im1 = ax1.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                           origin='lower', aspect='auto', cmap='hot')
            ax1.set_xlabel('Mass (GeV)', fontsize=11)
            ax1.set_ylabel('log10(Lifetime) (mm)', fontsize=11)
            ax1.set_title('Mean Position Error', fontsize=12)
            plt.colorbar(im1, ax=ax1, label='Mean Error (mm)')
        
        # 子图2：RMSE热图
        ax2 = axes[1]
        if len(error_df) > 4:
            heatmap, xedges, yedges = np.histogram2d(
                error_df['mass'], 
                error_df['log_lifetime'], 
                bins=10,
                weights=error_df['rmse']
            )
            
            im2 = ax2.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                           origin='lower', aspect='auto', cmap='plasma')
            ax2.set_xlabel('Mass (GeV)', fontsize=11)
            ax2.set_ylabel('log10(Lifetime) (mm)', fontsize=11)
            ax2.set_title('RMSE Distribution', fontsize=12)
            plt.colorbar(im2, ax=ax2, label='RMSE')
        
        # 子图3：tanβ vs 误差
        ax3 = axes[2]
        if 'tanb' in error_df.columns:
            scatter = ax3.scatter(error_df['tanb'], error_df['rmse'], 
                                 c=error_df['mass'], cmap='viridis', 
                                 alpha=0.7, s=50)
            ax3.set_xlabel('tanβ', fontsize=11)
            ax3.set_ylabel('Mean RMSE', fontsize=11)
            ax3.set_title('Error vs tanβ (colored by mass)', fontsize=12)
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='Mass (GeV)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'error_heatmaps.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_performance_evaluation_plots(self, output_path: Path):
        """创建性能评估图"""
        if self.summary_stats.empty:
            return
        
        print("\nCreating performance evaluation plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('KDE Performance Evaluation', fontsize=16, y=1.02)
        
        # 1. 误差分布累积图
        ax1 = axes[0, 0]
        all_rmse = []
        for comp in self.comparison_results.values():
            for coord in ['x', 'y', 'z']:
                coord_comp = comp['comparisons'].get(coord)
                if coord_comp:
                    all_rmse.append(coord_comp['shape']['rmse'])
        
        sorted_rmse = np.sort(all_rmse)
        cdf = np.arange(1, len(sorted_rmse) + 1) / len(sorted_rmse)
        
        ax1.plot(sorted_rmse, cdf, 'b-', linewidth=2)
        ax1.axvline(x=np.median(sorted_rmse), color='r', linestyle='--', 
                   label=f'Median: {np.median(sorted_rmse):.5f}')
        ax1.axvline(x=np.mean(sorted_rmse), color='g', linestyle='--',
                   label=f'Mean: {np.mean(sorted_rmse):.5f}')
        
        ax1.set_xlabel('RMSE', fontsize=11)
        ax1.set_ylabel('Cumulative Probability', fontsize=11)
        ax1.set_title('CDF of RMSE Values', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 箱线图：各坐标误差比较
        ax2 = axes[0, 1]
        rmse_by_coord = {'X': [], 'Y': [], 'Z': []}
        
        for comp in self.comparison_results.values():
            for coord, label in zip(['x', 'y', 'z'], ['X', 'Y', 'Z']):
                coord_comp = comp['comparisons'].get(coord)
                if coord_comp:
                    rmse_by_coord[label].append(coord_comp['shape']['rmse'])
        
        box_data = [rmse_by_coord[label] for label in ['X', 'Y', 'Z']]
        bp = ax2.boxplot(box_data, labels=['X', 'Y', 'Z'], patch_artist=True)
        
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax2.set_ylabel('RMSE', fontsize=11)
        ax2.set_title('Error Distribution by Coordinate', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 按质量分组的误差
        ax3 = axes[0, 2]
        if 'mass' in self.kde_models[next(iter(self.kde_models.keys()))]['params']:
            # 按质量分组
            mass_groups = {}
            for llp_id, comp in self.comparison_results.items():
                if llp_id in self.kde_models:
                    mass = self.kde_models[llp_id]['params']['mass']
                    mass_key = f"{float(mass):.1f}"
                    if mass_key not in mass_groups:
                        mass_groups[mass_key] = []
                    mass_groups[mass_key].append(comp['summary']['mean_rmse'])
            
            # 绘制箱线图
            labels = sorted(mass_groups.keys())
            box_data = [mass_groups[label] for label in labels]
            
            if box_data:
                bp = ax3.boxplot(box_data, labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('gold')
                    patch.set_alpha(0.6)
                
                ax3.set_xlabel('Mass (GeV)', fontsize=11)
                ax3.set_ylabel('Mean RMSE', fontsize=11)
                ax3.set_title('Error Distribution by Mass', fontsize=12)
                ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 收敛性分析（如果有样本量变化）
        ax4 = axes[1, 0]
        sample_sizes = []
        errors = []
        
        for comp in self.comparison_results.values():
            sample_sizes.append(comp['mc_data']['n_samples'])
            errors.append(comp['summary']['mean_rmse'])
        
        if len(sample_sizes) > 1:
            # 排序
            sort_idx = np.argsort(sample_sizes)
            sorted_sizes = np.array(sample_sizes)[sort_idx]
            sorted_errors = np.array(errors)[sort_idx]
            
            # 移动平均
            window_size = max(3, len(sorted_sizes) // 10)
            if window_size < len(sorted_sizes):
                ma_errors = np.convolve(sorted_errors, np.ones(window_size)/window_size, mode='valid')
                ma_sizes = sorted_sizes[window_size-1:]
                
                ax4.plot(ma_sizes, ma_errors, 'r-', linewidth=2, label='Moving average')
            
            ax4.scatter(sorted_sizes, sorted_errors, alpha=0.5, s=20)
            ax4.set_xlabel('Sample Size', fontsize=11)
            ax4.set_ylabel('Mean RMSE', fontsize=11)
            ax4.set_title('Convergence Analysis', fontsize=12)
            ax4.set_xscale('log')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        # 5. 误差矩阵（坐标间相关性）
        ax5 = axes[1, 1]
        # 收集各坐标的误差
        error_matrix = np.zeros((3, 3))
        coord_labels = ['X', 'Y', 'Z']
        
        error_data = {'X': [], 'Y': [], 'Z': []}
        for comp in self.comparison_results.values():
            for coord, label in zip(['x', 'y', 'z'], ['X', 'Y', 'Z']):
                coord_comp = comp['comparisons'].get(coord)
                if coord_comp:
                    error_data[label].append(coord_comp['shape']['rmse'])
        
        # 计算相关性
        for i, label1 in enumerate(coord_labels):
            for j, label2 in enumerate(coord_labels):
                if i == j:
                    error_matrix[i, j] = 1.0
                else:
                    if error_data[label1] and error_data[label2]:
                        corr = np.corrcoef(error_data[label1], error_data[label2])[0, 1]
                        error_matrix[i, j] = corr
        
        im = ax5.imshow(error_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax5.set_xticks(range(3))
        ax5.set_xticklabels(coord_labels)
        ax5.set_yticks(range(3))
        ax5.set_yticklabels(coord_labels)
        
        # 在单元格中添加数值
        for i in range(3):
            for j in range(3):
                ax5.text(j, i, f'{error_matrix[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontsize=12)
        
        ax5.set_title('Error Correlation Between Coordinates', fontsize=12)
        plt.colorbar(im, ax=ax5, label='Correlation Coefficient')
        
        # 6. 性能评分
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 计算综合评分
        performance_metrics = []
        
        # 1. 平均RMSE（0-1评分，越小越好）
        mean_rmse = np.mean(self.summary_stats['mean_rmse'])
        rmse_score = max(0, 1 - mean_rmse * 100)  # 假设RMSE在0.01以下
        performance_metrics.append(('Mean RMSE', mean_rmse, rmse_score))
        
        # 2. 平均位置误差
        mean_pos_error = np.mean(self.summary_stats['mean_abs_mean_diff'])
        pos_score = max(0, 1 - mean_pos_error / 10)  # 假设误差在10mm以下
        performance_metrics.append(('Mean Pos Error', mean_pos_error, pos_score))
        
        # 3. 一致性（低标准差）
        std_rmse = np.std(self.summary_stats['mean_rmse'])
        consistency_score = max(0, 1 - std_rmse * 100)
        performance_metrics.append(('Consistency', std_rmse, consistency_score))
        
        # 4. 总体评分
        overall_score = np.mean([s for _, _, s in performance_metrics])
        
        # 创建评分表
        score_text = "PERFORMANCE SCORES\n"
        score_text += "=" * 30 + "\n\n"
        
        for name, value, score in performance_metrics:
            score_text += f"{name}:\n"
            score_text += f"  Value: {value:.4f}\n"
            score_text += f"  Score: {score:.2%}\n\n"
        
        score_text += f"OVERALL SCORE: {overall_score:.2%}\n\n"
        
        if overall_score > 0.9:
            score_text += "✓ EXCELLENT FIT"
        elif overall_score > 0.7:
            score_text += "✓ GOOD FIT"
        elif overall_score > 0.5:
            score_text += "✓ ACCEPTABLE FIT"
        else:
            score_text += "⚠ NEEDS IMPROVEMENT"
        
        ax6.text(0.1, 0.5, score_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax6.set_title('KDE Performance Assessment', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_comparison_results(self, output_dir: str = './kde_mc_comparison'):
        """保存对比结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving comparison results to {output_path}...")
        
        # 1. 保存详细对比结果
        if self.comparison_results:
            results_dir = output_path / 'detailed_results'
            results_dir.mkdir(exist_ok=True)
            
            for llp_id, comparison in self.comparison_results.items():
                # 创建可序列化的版本
                serializable = {
                    'llp_id': comparison['llp_id'],
                    'params': comparison['kde_model']['params'],
                    'summary': comparison['summary'],
                    'comparisons': {}
                }
                
                for coord, comp in comparison['comparisons'].items():
                    serializable['comparisons'][coord] = {
                        'stats': comp['stats'],
                        'percentiles': comp['percentiles'],
                        'shape': comp['shape']
                    }
                
                # 保存为JSON
                result_file = results_dir / f'{llp_id}_comparison.json'
                with open(result_file, 'w') as f:
                    json.dump(serializable, f, indent=2, default=str)
            
            print(f"✓ Detailed results saved to: {results_dir}/")
        
        # 2. 保存汇总统计
        if not self.summary_stats.empty:
            csv_path = output_path / 'comparison_summary.csv'
            self.summary_stats.to_csv(csv_path, index=False)
            print(f"✓ Summary statistics saved to: {csv_path}")
        
        # 3. 生成对比报告
        self._generate_comparison_report(output_path)
        
        print(f"\n✓ All comparison results saved to: {output_path}")
    
    def _generate_comparison_report(self, output_path: Path):
        """生成对比分析报告"""
        report = []
        
        report.append("=" * 80)
        report.append("KDE DISTRIBUTION MODEL vs MC SIMULATION COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"\nAnalysis Date: {pd.Timestamp.now()}")
        report.append(f"KDE models loaded: {len(self.kde_models)}")
        report.append(f"MC datasets loaded: {len(self.mc_data)}")
        report.append(f"LLPs compared: {len(self.comparison_results)}")
        
        if not self.summary_stats.empty:
            report.append(f"\nOVERALL COMPARISON STATISTICS:")
            report.append("-" * 50)
            
            # 位置误差
            mean_pos_error = self.summary_stats['mean_abs_mean_diff'].mean()
            std_pos_error = self.summary_stats['mean_abs_mean_diff'].std()
            report.append(f"Position accuracy (mean diff): {mean_pos_error:.3f} ± {std_pos_error:.3f} mm")
            
            # 分布宽度误差
            mean_width_error = self.summary_stats['mean_abs_std_diff'].mean()
            std_width_error = self.summary_stats['mean_abs_std_diff'].std()
            report.append(f"Width accuracy (std diff): {mean_width_error:.3f} ± {std_width_error:.3f} mm")
            
            # 形状匹配
            mean_rmse = self.summary_stats['mean_rmse'].mean()
            std_rmse = self.summary_stats['mean_rmse'].std()
            report.append(f"Distribution shape (RMSE): {mean_rmse:.5f} ± {std_rmse:.5f}")
            
            # 最佳和最差表现
            best_idx = self.summary_stats['mean_rmse'].idxmin()
            worst_idx = self.summary_stats['mean_rmse'].idxmax()
            
            best_llp = self.summary_stats.loc[best_idx, 'llp_id']
            worst_llp = self.summary_stats.loc[worst_idx, 'llp_id']
            
            report.append(f"\nBEST PERFORMING LLP: {best_llp}")
            report.append(f"  RMSE: {self.summary_stats.loc[best_idx, 'mean_rmse']:.5f}")
            report.append(f"  Mean position error: {self.summary_stats.loc[best_idx, 'mean_abs_mean_diff']:.3f} mm")
            
            report.append(f"\nWORST PERFORMING LLP: {worst_llp}")
            report.append(f"  RMSE: {self.summary_stats.loc[worst_idx, 'mean_rmse']:.5f}")
            report.append(f"  Mean position error: {self.summary_stats.loc[worst_idx, 'mean_abs_mean_diff']:.3f} mm")
        
        # 按坐标分析
        report.append(f"\n\nCOORDINATE-WISE ANALYSIS:")
        report.append("-" * 50)
        
        if self.comparison_results:
            coord_stats = {'X': [], 'Y': [], 'Z': []}
            for comp in self.comparison_results.values():
                for coord, label in zip(['x', 'y', 'z'], ['X', 'Y', 'Z']):
                    coord_comp = comp['comparisons'].get(coord)
                    if coord_comp:
                        coord_stats[label].append(coord_comp['shape']['rmse'])
            
            for coord, errors in coord_stats.items():
                if errors:
                    report.append(f"\n{coord} coordinate:")
                    report.append(f"  Average RMSE: {np.mean(errors):.5f} ± {np.std(errors):.5f}")
                    report.append(f"  Min RMSE: {np.min(errors):.5f}")
                    report.append(f"  Max RMSE: {np.max(errors):.5f}")
        
        # 参数相关性分析
        report.append(f"\n\nPARAMETER CORRELATIONS:")
        report.append("-" * 50)
        
        if not self.summary_stats.empty and 'mass' in self.kde_models[next(iter(self.kde_models.keys()))]['params']:
            # 收集参数和误差数据
            param_data = []
            for llp_id in self.summary_stats['llp_id']:
                if llp_id in self.kde_models:
                    params = self.kde_models[llp_id]['params']
                    param_data.append({
                        'mass': float(params.get('mass', np.nan)),
                        'lifetime': float(params.get('lifetime', np.nan)),
                        'tanb': float(params.get('tanb', np.nan)),
                    })
            
            if param_data:
                param_df = pd.DataFrame(param_data)
                error_series = self.summary_stats['mean_rmse'].values
                
                if len(param_df) == len(error_series):
                    for param in ['mass', 'lifetime', 'tanb']:
                        if param in param_df.columns:
                            valid_idx = ~np.isnan(param_df[param]) & ~np.isnan(error_series)
                            if np.sum(valid_idx) > 1:
                                if param == 'lifetime':
                                    # 对寿命取对数
                                    corr = np.corrcoef(np.log10(param_df[param][valid_idx]), 
                                                      error_series[valid_idx])[0, 1]
                                    report.append(f"  log10({param}) vs RMSE: {corr:.3f}")
                                else:
                                    corr = np.corrcoef(param_df[param][valid_idx], 
                                                      error_series[valid_idx])[0, 1]
                                    report.append(f"  {param} vs RMSE: {corr:.3f}")
        
        # 结论和建议
        report.append(f"\n\nCONCLUSIONS AND RECOMMENDATIONS:")
        report.append("-" * 50)
        
        if not self.summary_stats.empty:
            mean_rmse = self.summary_stats['mean_rmse'].mean()
            
            if mean_rmse < 0.001:
                report.append("✓ EXCELLENT AGREEMENT: KDE model accurately represents MC data.")
                report.append("  Recommendation: KDE model can be reliably used for further analysis.")
            elif mean_rmse < 0.01:
                report.append("✓ GOOD AGREEMENT: Minor discrepancies observed.")
                report.append("  Recommendation: KDE model is suitable for most applications.")
            elif mean_rmse < 0.05:
                report.append("⚠ MODERATE AGREEMENT: Noticeable differences in distribution shapes.")
                report.append("  Recommendation: Consider refining KDE bandwidth or using additional features.")
            else:
                report.append("⚠ POOR AGREEMENT: Significant differences between KDE and MC.")
                report.append("  Recommendation: Re-evaluate KDE modeling approach. Consider:")
                report.append("    - Using adaptive bandwidth")
                report.append("    - Adding tail corrections")
                report.append("    - Using mixture models for multi-modal distributions")
        
        report.append(f"\n\nOUTPUT FILES:")
        report.append("-" * 50)
        report.append(f"1. {output_path}/comparison_summary.csv - Summary statistics")
        report.append(f"2. {output_path}/detailed_results/ - Individual comparison JSON files")
        report.append(f"3. {output_path}/*_comparison.png - Individual comparison plots")
        report.append(f"4. {output_path}/summary_comparison.png - Overall comparison")
        report.append(f"5. {output_path}/performance_evaluation.png - Performance assessment")
        
        report.append(f"\n" + "=" * 80)
        report.append("END OF COMPARISON REPORT")
        report.append("=" * 80)
        
        report_text = '\n'.join(report)
        
        with open(output_path / 'comparison_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"✓ Comparison report saved to: {output_path}/comparison_report.txt")


def main_comparison():
    """对比分析主函数"""
    # 设置路径
    kde_results_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_F/distributution_density"
    mc_data_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_F/llp_simulation_results/incremental_results"
    output_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_F/kde_mc_comparison"
    
    print("\n" + "=" * 80)
    print("KDE-MC DISTRIBUTION COMPARISON ANALYSIS")
    print("=" * 80)
    
    # 创建对比分析器
    analyzer = KDE_MC_ComparisonAnalyzer(kde_results_dir, mc_data_dir)
    
    try:
        # 1. 加载KDE模型
        print("\n[1/4] Loading KDE distribution models...")
        analyzer.load_kde_models()
        
        # 2. 加载MC数据
        print("\n[2/4] Loading MC simulation data...")
        analyzer.load_mc_data()
        
        # 3. 执行对比分析
        print("\n[3/4] Comparing distributions...")
        analyzer.compare_distributions()
        
        # 4. 创建可视化
        print("\n[4/4] Creating comparison visualizations...")
        analyzer.create_comparison_plots(output_dir)
        
        # 5. 保存结果
        analyzer.save_comparison_results(output_dir)
        
        print("\n" + "=" * 80)
        print("COMPARISON ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"\n✅ Comparison results saved to: {output_dir}")
        print(f"\n📊 Key output files:")
        print(f"  {output_dir}/comparison_summary.csv - Summary statistics")
        print(f"  {output_dir}/comparison_report.txt - Detailed report")
        print(f"  {output_dir}/summary_comparison.png - Overall comparison plot")
        print(f"  {output_dir}/performance_evaluation.png - Performance assessment")
        
    except Exception as e:
        print(f"\n❌ Error during comparison analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_comparison()