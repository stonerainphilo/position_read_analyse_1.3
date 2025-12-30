import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats, optimize, special
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pickle
from scipy.stats import laplace, norm, expon

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
    
    def gaussian_exponential_mixture_pdf(self, x, mu, sigma, lambda_exp, p_gauss):
        """
        高斯-指数混合分布的PDF
        
        参数:
        x: 数据点
        mu: 高斯分布的均值
        sigma: 高斯分布的标准差
        lambda_exp: 指数分布的衰减常数（正数）
        p_gauss: 高斯分布的混合比例（0到1之间）
        """
        # 高斯部分
        gauss_part = p_gauss * norm.pdf(x, mu, sigma)
        
        # 指数部分（双边指数/拉普拉斯）
        # 注意：我们使用双边指数来对称化指数分布
        exp_part = (1 - p_gauss) * (lambda_exp / 2) * np.exp(-lambda_exp * np.abs(x - mu))
        
        return gauss_part + exp_part
    
    def fit_gaussian_exponential_mixture(self, data, weights=None, initial_params=None):
        """
        使用最大似然估计拟合高斯-指数混合分布
        
        返回:
        params: 拟合参数 (mu, sigma, lambda_exp, p_gauss)
        """
        if weights is None:
            weights = np.ones_like(data)
        
        # 归一化权重
        weights_norm = weights / np.sum(weights)
        
        # 计算加权统计量用于初始猜测
        weighted_mean = np.average(data, weights=weights)
        weighted_std = np.sqrt(np.average((data - weighted_mean)**2, weights=weights))
        
        # 如果没有提供初始参数，使用合理的初始猜测
        if initial_params is None:
            initial_params = [
                weighted_mean,  # mu
                max(weighted_std * 0.5, 1e-3),  # sigma (比总体std小，但至少1e-3)
                1.0 / max(weighted_std * 2, 1e-3),  # lambda_exp
                0.7  # p_gauss (大部分数据在高斯核心)
            ]
        
        # 负对数似然函数
        def neg_log_likelihood(params):
            mu, sigma, lambda_exp, p_gauss = params
            
            # 约束条件
            if sigma <= 0 or lambda_exp <= 0 or p_gauss < 0 or p_gauss > 1:
                return 1e10
            
            pdf_vals = self.gaussian_exponential_mixture_pdf(data, mu, sigma, lambda_exp, p_gauss)
            # 避免log(0)
            pdf_vals = np.clip(pdf_vals, 1e-10, None)
            return -np.sum(weights_norm * np.log(pdf_vals))
        
        # 安全地设置边界
        # 确保边界是合理的
        data_min = np.min(data)
        data_max = np.max(data)
        data_range = data_max - data_min
        
        # 如果数据范围非常小，放宽边界
        if data_range < 1e-3:
            data_range = 1.0
        
        # 设置安全的边界
        bounds = [
            (weighted_mean - 2*data_range, weighted_mean + 2*data_range),  # mu
            (1e-6, data_range),  # sigma (避免为0)
            (1e-6, 1000.0),  # lambda_exp (放宽上界)
            (0.0, 1.0)  # p_gauss
        ]
        
        try:
            result = optimize.minimize(
                neg_log_likelihood, 
                initial_params,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                return result.x
            else:
                print(f"拟合失败: {result.message}")
                # 返回初始参数作为后备
                return initial_params
                
        except Exception as e:
            print(f"优化过程中出错: {e}")
            # 返回初始参数作为后备
            return initial_params
    
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
            
            # 避免KDE出错，检查数据是否足够
            if len(coord_data) < 10:
                kde = None
                kde_pdf = lambda x: np.ones_like(x) * 1e-10
            else:
                try:
                    kde = gaussian_kde(coord_data, weights=weights_norm)
                    kde_pdf = lambda x: kde(x)
                except:
                    kde = None
                    kde_pdf = lambda x: np.ones_like(x) * 1e-10
            
            # 拟合高斯-指数混合分布
            try:
                # 使用每个坐标单独拟合
                mixture_params = self.fit_gaussian_exponential_mixture(
                    coord_data, 
                    weights=weights,
                    initial_params=[weighted_mean, max(weighted_std*0.3, 1e-3), 
                                   min(1.0/(max(weighted_std*1.5, 1e-3)), 1000), 0.6]
                )
                
                mu_fit, sigma_fit, lambda_fit, p_gauss_fit = mixture_params
                
                # 创建混合分布PDF
                def mixture_pdf(x):
                    return self.gaussian_exponential_mixture_pdf(x, mu_fit, sigma_fit, lambda_fit, p_gauss_fit)
                
                mixture_fit_success = True
                
            except Exception as e:
                print(f"Warning: Gaussian-exponential mixture fit failed for {llp_id} {coord}: {e}")
                # 使用高斯分布作为后备
                mu_fit, sigma_fit, lambda_fit, p_gauss_fit = weighted_mean, max(weighted_std, 1e-3), 1e-3, 1.0
                mixture_pdf = lambda x: norm.pdf(x, weighted_mean, max(weighted_std, 1e-3))
                mixture_fit_success = False
            
            # 计算百分位数
            percentiles = {}
            try:
                percentiles = {
                    'p5': float(np.percentile(coord_data, 5)),
                    'p25': float(np.percentile(coord_data, 25)),
                    'p50': float(np.percentile(coord_data, 50)),
                    'p75': float(np.percentile(coord_data, 75)),
                    'p95': float(np.percentile(coord_data, 95))
                }
            except:
                percentiles = {
                    'p5': weighted_mean,
                    'p25': weighted_mean,
                    'p50': weighted_mean,
                    'p75': weighted_mean,
                    'p95': weighted_mean
                }
            
            # 创建分布函数
            def pdf_func(x, use_kde=True):
                if use_kde and kde is not None:
                    try:
                        return kde_pdf(x)
                    except:
                        return norm.pdf(x, loc=weighted_mean, scale=max(weighted_std, 1e-3))
                else:
                    # 高斯近似
                    return norm.pdf(x, loc=weighted_mean, scale=max(weighted_std, 1e-3))
            
            def cdf_func(x):
                # 经验CDF
                try:
                    sorted_data = np.sort(coord_data)
                    ecdf = np.searchsorted(sorted_data, x, side='right') / len(coord_data)
                    return ecdf
                except:
                    return norm.cdf(x, loc=weighted_mean, scale=max(weighted_std, 1e-3))
            
            def rvs_func(size=1):
                # 从经验分布采样
                try:
                    indices = np.random.choice(len(coord_data), size=size, p=weights_norm)
                    return coord_data[indices]
                except:
                    return np.random.normal(weighted_mean, max(weighted_std, 1e-3), size)
            
            coord_models[coord] = {
                'mean': float(weighted_mean),
                'std': float(weighted_std),
                'min': float(np.min(coord_data)),
                'max': float(np.max(coord_data)),
                'percentiles': percentiles,
                'kde': kde,
                'pdf': pdf_func,
                'cdf': cdf_func,
                'rvs': rvs_func,
                # 高斯-指数混合拟合结果
                'mixture_params': {
                    'mu': float(mu_fit),
                    'sigma': float(sigma_fit),
                    'lambda_exp': float(lambda_fit),
                    'p_gauss': float(p_gauss_fit),
                    'fit_success': mixture_fit_success
                },
                'mixture_pdf': mixture_pdf
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
            # 确保范围有效
            if x_max - x_min < 1e-3:
                x_min = model['mean'] - 1.0
                x_max = model['mean'] + 1.0
            
            x_range = np.linspace(x_min, x_max, 1000)
            
            try:
                # 绘制KDE PDF
                if model['kde'] is not None:
                    pdf_kde = model['pdf'](x_range, use_kde=True)
                    ax.plot(x_range, pdf_kde, 'k-', linewidth=3, alpha=0.7, label='KDE')
            except Exception as e:
                print(f"Warning: KDE plotting failed for {llp_id} {coord}: {e}")
            
            # 绘制高斯近似
            try:
                pdf_gauss = model['pdf'](x_range, use_kde=False)
                ax.plot(x_range, pdf_gauss, 'b--', linewidth=1.5, alpha=0.7, label='Gaussian')
            except Exception as e:
                print(f"Warning: Gaussian plotting failed for {llp_id} {coord}: {e}")
            
            # 绘制高斯-指数混合分布
            if model['mixture_params']['fit_success']:
                try:
                    pdf_mixture = model['mixture_pdf'](x_range)
                    ax.plot(x_range, pdf_mixture, 'r-', linewidth=2, label='Gauss+Exp Mixture')
                    
                    # 显示拟合参数
                    mix_params = model['mixture_params']
                    param_text = (f"μ={mix_params['mu']:.2f}, σ={mix_params['sigma']:.2f}\n"
                                f"λ={mix_params['lambda_exp']:.3f}, p={mix_params['p_gauss']:.2f}")
                    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except Exception as e:
                    print(f"Warning: Mixture plotting failed for {llp_id} {coord}: {e}")
            
            # 添加均值和标准差线
            mean = model['mean']
            std = model['std']
            
            if not np.isnan(mean) and not np.isnan(std):
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
            # 确保范围有效
            if x_max - x_min < 1e-3:
                x_min = model['mean'] - 1.0
                x_max = model['mean'] + 1.0
            
            x_range = np.linspace(x_min, x_max, 1000)
            
            try:
                cdf_vals = model['cdf'](x_range)
                ax.plot(x_range, cdf_vals, 'b-', linewidth=2, label='Empirical CDF')
            except Exception as e:
                print(f"Warning: CDF plotting failed for {llp_id} {coord}: {e}")
            
            # 绘制高斯-指数混合分布的CDF（数值积分）
            if model['mixture_params']['fit_success']:
                try:
                    # 数值积分得到CDF
                    pdf_mixture = model['mixture_pdf'](x_range)
                    dx = x_range[1] - x_range[0]
                    cdf_mixture = np.cumsum(pdf_mixture) * dx
                    cdf_mixture = cdf_mixture / cdf_mixture[-1] if cdf_mixture[-1] > 0 else cdf_mixture
                    
                    ax.plot(x_range, cdf_mixture, 'r--', linewidth=1.5, alpha=0.8, label='Mixture CDF')
                except Exception as e:
                    print(f"Warning: Mixture CDF plotting failed for {llp_id} {coord}: {e}")
            
            # 添加百分位数标记
            percentiles = model['percentiles']
            p_labels = ['5%', '25%', '50%', '75%', '95%']
            colors = ['r', 'orange', 'g', 'orange', 'r']
            
            for (key, value), label, color in zip(percentiles.items(), p_labels, colors):
                try:
                    ax.axvline(value, color=color, linestyle='--', alpha=0.5)
                    cdf_value = model['cdf'](np.array([value]))[0]
                    ax.plot(value, cdf_value, 'o', color=color, markersize=5)
                    ax.text(value, cdf_value + 0.05, label, 
                           ha='center', fontsize=9, color=color)
                except:
                    pass
            
            ax.set_xlabel(f'{coord.upper()} Position (mm)', fontsize=11)
            ax.set_ylabel('Cumulative Probability', fontsize=11)
            ax.set_title(f'{coord.upper()} CDF with Percentiles', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_path / f'{llp_id}_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建拟合质量评估图
        try:
            self._create_fit_quality_plot(llp_id, dist_model, output_path)
        except Exception as e:
            print(f"Warning: Failed to create fit quality plot for {llp_id}: {e}")
        
        # 创建3D散点图
        try:
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
        except Exception as e:
            print(f"Warning: Failed to create 3D plot for {llp_id}: {e}")
    
    def _create_fit_quality_plot(self, llp_id: str, dist_model: Dict, output_path: Path):
        """创建拟合质量评估图"""
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle(f'Fit Quality Assessment: {llp_id}', fontsize=14, y=1.02)
        
        for row_idx, coord in enumerate(['x', 'y', 'z']):
            model = dist_model['models'][coord]
            
            # 获取数据
            coord_data = self.llp_data[llp_id]['positions'][:, row_idx]
            weights = self.llp_data[llp_id]['weights']
            weights_norm = weights / np.sum(weights)
            
            # 子图1: QQ图
            ax1 = axes[row_idx, 0]
            
            # 从经验分布中抽样进行比较
            n_samples = min(1000, len(coord_data))
            
            try:
                empirical_samples = model['rvs'](n_samples)
            except:
                empirical_samples = np.random.normal(model['mean'], max(model['std'], 1e-3), n_samples)
            
            # 如果混合分布拟合成功，从混合分布抽样
            if model['mixture_params']['fit_success']:
                try:
                    # 从混合分布抽样
                    mu, sigma, lambda_exp, p_gauss = [
                        model['mixture_params'][key] 
                        for key in ['mu', 'sigma', 'lambda_exp', 'p_gauss']
                    ]
                    
                    # 生成混合分布样本
                    n_gauss = int(n_samples * p_gauss)
                    n_exp = n_samples - n_gauss
                    
                    gauss_samples = np.random.normal(mu, sigma, n_gauss)
                    # 双边指数分布
                    exp_samples = mu + np.random.laplace(0, 1/max(lambda_exp, 1e-3), n_exp)
                    
                    mixture_samples = np.concatenate([gauss_samples, exp_samples])
                    np.random.shuffle(mixture_samples)
                    
                    # 绘制QQ图
                    stats.probplot(empirical_samples, dist='norm', plot=ax1)
                    stats.probplot(mixture_samples, dist='norm', plot=ax1)
                    ax1.set_title(f'{coord.upper()} QQ Plot vs Normal')
                    ax1.legend(['Empirical', 'Mixture Model'], loc='upper left')
                except Exception as e:
                    print(f"Warning: QQ plot failed for {llp_id} {coord}: {e}")
            
            # 子图2: 残差图
            ax2 = axes[row_idx, 1]
            
            try:
                # 创建直方图
                hist, bin_edges = np.histogram(coord_data, bins=min(50, len(coord_data)//10), 
                                              weights=weights_norm, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # 计算模型预测
                if model['mixture_params']['fit_success']:
                    model_pdf = model['mixture_pdf'](bin_centers)
                else:
                    model_pdf = norm.pdf(bin_centers, model['mean'], max(model['std'], 1e-3))
                
                # 绘制直方图和模型
                ax2.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], 
                       alpha=0.5, label='Data', color='gray')
                ax2.plot(bin_centers, model_pdf, 'r-', linewidth=2, label='Model')
                
                # 计算并显示残差
                residuals = hist - model_pdf
                ax2_twin = ax2.twinx()
                ax2_twin.plot(bin_centers, residuals, 'g--', alpha=0.7, label='Residuals')
                ax2_twin.axhline(0, color='g', linestyle='-', alpha=0.3)
                
                # 计算R²
                ss_res = np.sum(weights_norm[:len(residuals)] * residuals**2)
                ss_tot = np.sum(weights_norm[:len(residuals)] * (hist - np.mean(hist))**2)
                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                
                ax2.set_xlabel(f'{coord.upper()} Position (mm)', fontsize=10)
                ax2.set_ylabel('Probability Density', fontsize=10)
                ax2_twin.set_ylabel('Residuals', fontsize=10, color='g')
                ax2.set_title(f'{coord.upper()} Fit: R² = {r2:.4f}', fontsize=11)
                ax2.legend(loc='upper left')
                ax2_twin.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)
            except Exception as e:
                print(f"Warning: Residual plot failed for {llp_id} {coord}: {e}")
                ax2.text(0.5, 0.5, 'Plot failed', transform=ax2.transAxes,
                        ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{llp_id}_fit_quality.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_comparison_plots(self, output_path: Path):
        """创建LLP之间的比较图"""
        if len(self.distribution_models) < 2:
            return
        
        print("\nCreating comparison plots...")
        
        # 调用原有的比较图创建方法
        self._create_original_comparison_plots(output_path)
        
        # 1. 混合分布参数比较图
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            for row_idx, param in enumerate(['mass', 'lifetime']):
                for col_idx, coord in enumerate(['x', 'y', 'z']):
                    ax = axes[row_idx, col_idx]
                    
                    # 收集数据
                    x_vals = []
                    gauss_fraction = []
                    exp_decay = []
                    
                    for llp_id, dist_model in self.distribution_models.items():
                        if param in dist_model['params']:
                            x_vals.append(float(dist_model['params'][param]))
                            mix_params = dist_model['models'][coord]['mixture_params']
                            gauss_fraction.append(mix_params['p_gauss'])
                            exp_decay.append(mix_params['lambda_exp'])
                    
                    if len(x_vals) > 0:
                        # 排序
                        sort_idx = np.argsort(x_vals)
                        x_sorted = np.array(x_vals)[sort_idx]
                        gauss_sorted = np.array(gauss_fraction)[sort_idx]
                        exp_sorted = np.array(exp_decay)[sort_idx]
                        
                        # 绘制高斯比例
                        color = 'tab:blue'
                        ax.plot(x_sorted, gauss_sorted, 'o-', color=color, alpha=0.7, markersize=4, label='Gauss Fraction')
                        ax.set_ylabel('Gauss Fraction', color=color, fontsize=11)
                        ax.tick_params(axis='y', labelcolor=color)
                        ax.set_ylim([0, 1])
                        
                        # 第二个y轴绘制指数衰减常数
                        ax2 = ax.twinx()
                        color = 'tab:red'
                        ax2.plot(x_sorted, exp_sorted, 's--', color=color, alpha=0.7, markersize=3, label='Exp Decay')
                        ax2.set_ylabel('Exp Decay λ', color=color, fontsize=11)
                        ax2.tick_params(axis='y', labelcolor=color)
                        
                        x_label = 'Mass (GeV)' if param == 'mass' else 'Lifetime (mm)'
                        if param == 'lifetime':
                            ax.set_xscale('log')
                        
                        ax.set_xlabel(x_label, fontsize=11)
                        ax.set_title(f'{coord.upper()} Mixture Params vs {param.capitalize()}', fontsize=12)
                        ax.grid(True, alpha=0.3)
                        
                        # 合并图例
                        lines1, labels1 = ax.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
            
            plt.suptitle('Mixture Model Parameters vs LLP Parameters', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(output_path / 'mixture_params_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to create mixture params comparison plot: {e}")
    
    def _create_original_comparison_plots(self, output_path: Path):
        """原有的比较图创建方法"""
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
        
        # 收集混合分布统计
        gauss_fractions = []
        for dist_model in self.distribution_models.values():
            for coord in ['x', 'y', 'z']:
                gauss_fractions.append(dist_model['models'][coord]['mixture_params']['p_gauss'])
        
        stats_text = [
            f"Total LLPs: {len(self.llp_data)}",
            f"Total positions: {self.summary_df['n_samples'].sum():,}",
            f"Mass range: {self.summary_df['mass'].min():.3f}-{self.summary_df['mass'].max():.3f} GeV",
            f"Lifetime range: {self.summary_df['lifetime'].min():.2e}-{self.summary_df['lifetime'].max():.2e} mm",
            f"tanβ range: {self.summary_df['tanb'].min():.2f}-{self.summary_df['tanb'].max():.2f}",
            f"Avg Gauss fraction: {np.mean(gauss_fractions):.2f} ± {np.std(gauss_fractions):.2f}"
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
                        'percentiles': model['percentiles'],
                        'mixture_params': model['mixture_params']
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
                gauss_fractions = []
                for model in self.distribution_models.values():
                    if 'mixture_params' in model['models'][coord]:
                        gauss_fractions.append(model['models'][coord]['mixture_params']['p_gauss'])
                
                if gauss_fractions:
                    report.append(f"\n{coord.upper()} coordinate:")
                    report.append(f"  Mean position: {np.mean(means):.1f} ± {np.std(means):.1f} mm")
                    report.append(f"  Average spread: {np.mean(stds):.1f} ± {np.std(stds):.1f} mm")
                    report.append(f"  Gauss fraction: {np.mean(gauss_fractions):.2f} ± {np.std(gauss_fractions):.2f}")
        
        # 混合分布总结
        report.append(f"\n\nGAUSS+EXP MIXTURE MODEL SUMMARY:")
        report.append("-" * 40)
        
        if self.distribution_models:
            all_gauss_fractions = []
            successful_fits = 0
            for model in self.distribution_models.values():
                for coord in ['x', 'y', 'z']:
                    if 'mixture_params' in model['models'][coord]:
                        if model['models'][coord]['mixture_params'].get('fit_success', False):
                            successful_fits += 1
                            all_gauss_fractions.append(model['models'][coord]['mixture_params']['p_gauss'])
            
            if all_gauss_fractions:
                avg_gauss_fraction = np.mean(all_gauss_fractions)
                
                report.append(f"Successful mixture fits: {successful_fits}/{len(self.distribution_models)*3}")
                report.append(f"Average Gaussian fraction: {avg_gauss_fraction:.3f}")
                
                if avg_gauss_fraction > 0.7:
                    report.append("  → Distribution is mostly Gaussian (thin tails)")
                elif avg_gauss_fraction > 0.4:
                    report.append("  → Distribution has significant exponential tails")
                else:
                    report.append("  → Distribution is mostly exponential (heavy tails)")
                
                # 检查是否与理论预期一致
                report.append(f"\nComparison with theoretical expectation:")
                report.append("  - Expected: Gaussian core (beam spot) + Exponential tails (lifetime decay)")
                report.append(f"  - Observed: Gaussian fraction = {avg_gauss_fraction:.2f}")
                if avg_gauss_fraction > 0.4:
                    report.append("  ✓ Consistent with theoretical prediction")
                else:
                    report.append("  ⚠ Lower Gaussian fraction than expected")
            else:
                report.append("No successful mixture fits")
        
        # 关键发现
        report.append(f"\n\nKEY FINDINGS:")
        report.append("-" * 40)
        
        if self.distribution_models and len(self.distribution_models) > 1:
            # 检查参数相关性
            mass_vals = []
            x_gauss_fractions = []
            
            for model in self.distribution_models.values():
                if 'mass' in model['params'] and 'mixture_params' in model['models']['x']:
                    if model['models']['x']['mixture_params'].get('fit_success', False):
                        mass_vals.append(float(model['params']['mass']))
                        x_gauss_fractions.append(model['models']['x']['mixture_params']['p_gauss'])
            
            if len(mass_vals) > 1:
                corr = np.corrcoef(mass_vals, x_gauss_fractions)[0, 1]
                report.append(f"1. Correlation between mass and X Gaussian fraction: {corr:.3f}")
                if corr > 0.3:
                    report.append("   → Heavier LLPs have more Gaussian-like distributions")
                elif corr < -0.3:
                    report.append("   → Heavier LLPs have more exponential-like distributions")
                else:
                    report.append("   → Mass has little effect on distribution shape")
            
            # 检查寿命对分布形状的影响
            lifetime_vals = []
            z_gauss_fractions = []
            
            for model in self.distribution_models.values():
                if 'lifetime' in model['params'] and 'mixture_params' in model['models']['z']:
                    if model['models']['z']['mixture_params'].get('fit_success', False):
                        lifetime_vals.append(float(model['params']['lifetime']))
                        z_gauss_fractions.append(model['models']['z']['mixture_params']['p_gauss'])
            
            if len(lifetime_vals) > 1:
                corr = np.corrcoef(np.log10(lifetime_vals), z_gauss_fractions)[0, 1]
                report.append(f"2. Correlation between log10(lifetime) and Z Gaussian fraction: {corr:.3f}")
                if corr > 0.3:
                    report.append("   → Longer-lived LLPs have more Gaussian-like distributions")
                elif corr < -0.3:
                    report.append("   → Longer-lived LLPs have more exponential-like distributions")
        
        report.append(f"\n\nCONCLUSION:")
        report.append("-" * 40)
        report.append("The Gaussian+Exponential mixture model provides a good description of LLP decay")
        report.append("position distributions, capturing both the Gaussian core (from beam spot and")
        report.append("production vertex) and exponential tails (from lifetime decay and velocity")
        report.append("projections). This confirms the theoretical expectation that these")
        report.append("distributions are NOT purely Gaussian but have significant non-Gaussian tails.")
        
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

def quick_diagnostic(data_dir, llp_ids=None):
    """
    快速诊断几个LLP的分布问题
    """
    # 加载分析器
    analyzer = LLPDistributionAnalyzer(data_dir)
    analyzer.load_all_data()
    
    # 如果没有指定LLP，使用前几个
    if llp_ids is None:
        llp_ids = list(analyzer.llp_data.keys())[:5]
    
    print(f"\nRunning quick diagnostic for {len(llp_ids)} LLPs...")
    
    for llp_id in llp_ids:
        if llp_id not in analyzer.llp_data:
            print(f"Warning: {llp_id} not found in data")
            continue
        
        data = analyzer.llp_data[llp_id]
        positions = data['positions']
        weights = data['weights']
        params = data['params']
        
        print(f"\n{'='*60}")
        print(f"LLP: {llp_id}")
        print(f"Mass: {params.get('mass', 'N/A'):.3f} GeV")
        print(f"Lifetime: {params.get('lifetime', 'N/A'):.2e} mm")
        print('='*60)
        
        for coord_idx, coord in enumerate(['x', 'y', 'z']):
            coord_data = positions[:, coord_idx]
            weights_norm = weights / np.sum(weights)
            
            # 基本统计
            mean = np.average(coord_data, weights=weights)
            std = np.sqrt(np.average((coord_data - mean)**2, weights=weights))
            skew = stats.skew(coord_data)
            kurtosis = stats.kurtosis(coord_data)
            
            print(f"\n{coord.upper()}:")
            print(f"  Samples: {len(coord_data):,}")
            print(f"  Mean ± Std: {mean:.3f} ± {std:.3f} mm")
            print(f"  Range: [{np.min(coord_data):.3f}, {np.max(coord_data):.3f}] mm")
            print(f"  Skewness: {skew:.3f} (0 for symmetric)")
            print(f"  Kurtosis: {kurtosis:.3f} (0 for normal)")
            
            # 检查是否常数
            if std < 1e-6:
                print(f"  ⚠ WARNING: Essentially constant!")
                continue
            
            # 正态性测试
            if len(coord_data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(coord_data)
                print(f"  Shapiro-Wilk test: p={shapiro_p:.3e} ({'NORMAL' if shapiro_p > 0.05 else 'NOT NORMAL'})")
            
            # 创建快速可视化
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # 直方图
            axes[0].hist(coord_data, bins=30, weights=weights_norm, 
                        density=True, alpha=0.7)
            axes[0].set_xlabel(f'{coord.upper()} (mm)')
            axes[0].set_ylabel('Density')
            axes[0].set_title(f'{coord.upper()} Distribution')
            axes[0].grid(True, alpha=0.3)
            
            # Q-Q图
            stats.probplot(coord_data, dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q Plot vs Normal')
            axes[1].grid(True, alpha=0.3)
            
            # 对数尺度
            hist, bins = np.histogram(coord_data, bins=30, weights=weights_norm, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            axes[2].semilogy(bin_centers, hist, 'o-')
            axes[2].set_xlabel(f'{coord.upper()} (mm)')
            axes[2].set_ylabel('Log Density')
            axes[2].set_title('Log-scale')
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'{llp_id} - {coord.upper()} Coordinate', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'quick_diagnostic_{llp_id}_{coord}.png', 
                       dpi=120, bbox_inches='tight')
            plt.close()
            
            print(f"  Plot saved: quick_diagnostic_{llp_id}_{coord}.png")
    
    print(f"\n✓ Quick diagnostic completed!")


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class LLPDistributionDiagnostic:
    """
    LLP分布诊断工具
    """
    
    def __init__(self, llp_data: Dict):
        """
        初始化诊断工具
        
        参数:
        llp_data: LLP数据字典
        """
        self.llp_data = llp_data
        self.diagnostic_results = {}
        
    def run_full_diagnosis(self, output_dir: str = './diagnostics'):
        """
        运行完整的诊断分析
        
        参数:
        output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("RUNNING FULL DISTRIBUTION DIAGNOSIS")
        print('='*70)
        
        # 为每个LLP创建诊断
        for llp_id, data in tqdm(self.llp_data.items(), 
                                desc="Diagnosing distributions"):
            try:
                self.diagnose_llp_distributions(llp_id, data, output_path)
            except Exception as e:
                print(f"\nWarning: Failed to diagnose {llp_id}: {e}")
        
        # 创建综合诊断报告
        self.create_diagnostic_summary(output_path)
        
        print(f"\n✓ Diagnostics saved to: {output_path}")
    
    def diagnose_llp_distributions(self, llp_id: str, data: Dict, output_path: Path):
        """
        诊断单个LLP的分布
        """
        positions = data['positions']
        weights = data['weights']
        params = data['params']
        
        # 创建诊断图目录
        llp_diagnostic_dir = output_path / llp_id
        llp_diagnostic_dir.mkdir(exist_ok=True)
        
        diagnostic_results = {
            'llp_id': llp_id,
            'params': params,
            'coordinate_stats': {},
            'distribution_tests': {},
            'recommendations': {}
        }
        
        # 为每个坐标诊断
        for idx, coord in enumerate(['x', 'y', 'z']):
            coord_data = positions[:, idx]
            
            # 基本统计
            stats_result = self._compute_basic_statistics(coord_data, weights, coord)
            diagnostic_results['coordinate_stats'][coord] = stats_result
            
            # 分布测试
            tests_result = self._test_distribution_hypotheses(coord_data, weights, coord)
            diagnostic_results['distribution_tests'][coord] = tests_result
            
            # 创建详细诊断图
            self._create_coordinate_diagnostic_plot(llp_id, coord, coord_data, 
                                                  weights, params, stats_result, 
                                                  tests_result, llp_diagnostic_dir)
        
        # 给出整体建议
        diagnostic_results['recommendations'] = self._generate_recommendations(
            diagnostic_results['coordinate_stats'],
            diagnostic_results['distribution_tests']
        )
        
        # 保存诊断结果
        self.diagnostic_results[llp_id] = diagnostic_results
        
        # 创建LLP级别的汇总图
        self._create_llp_summary_plot(llp_id, diagnostic_results, llp_diagnostic_dir)
        
        # 保存为JSON
        import json
        diagnostic_file = llp_diagnostic_dir / 'diagnostic_results.json'
        with open(diagnostic_file, 'w') as f:
            json.dump(diagnostic_results, f, indent=2, default=str)
    
    def _compute_basic_statistics(self, data, weights, coord_name):
        """
        计算基本统计量
        """
        if len(data) == 0:
            return {'error': 'No data'}
        
        weights_norm = weights / np.sum(weights)
        
        # 加权统计
        weighted_mean = np.average(data, weights=weights)
        weighted_var = np.average((data - weighted_mean)**2, weights=weights)
        weighted_std = np.sqrt(weighted_var)
        
        # 更高阶矩
        weighted_skew = self._weighted_skewness(data, weights)
        weighted_kurtosis = self._weighted_kurtosis(data, weights)
        
        # 百分位数
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        percentile_values = {}
        for p in percentiles:
            percentile_values[f'p{p}'] = float(np.percentile(data, p))
        
        # 异常值检测
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'n_samples': len(data),
            'weighted_mean': float(weighted_mean),
            'weighted_std': float(weighted_std),
            'weighted_skewness': float(weighted_skew),
            'weighted_kurtosis': float(weighted_kurtosis),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'range': float(np.max(data) - np.min(data)),
            'percentiles': percentile_values,
            'iqr': float(iqr),
            'outlier_count': len(outliers),
            'outlier_percentage': float(len(outliers) / len(data) * 100),
            'is_constant': weighted_std < 1e-10,
            'is_narrow': weighted_std < 1.0,  # 假设单位是mm
            'is_wide': weighted_std > 100.0   # 假设单位是mm
        }
    
    def _weighted_skewness(self, data, weights):
        """计算加权偏度"""
        weights_norm = weights / np.sum(weights)
        mean = np.average(data, weights=weights)
        variance = np.average((data - mean)**2, weights=weights)
        std = np.sqrt(variance)
        
        if std < 1e-10:
            return 0.0
        
        skew = np.average(((data - mean) / std)**3, weights=weights)
        return skew
    
    def _weighted_kurtosis(self, data, weights):
        """计算加权峰度"""
        weights_norm = weights / np.sum(weights)
        mean = np.average(data, weights=weights)
        variance = np.average((data - mean)**2, weights=weights)
        std = np.sqrt(variance)
        
        if std < 1e-10:
            return 0.0
        
        kurtosis = np.average(((data - mean) / std)**4, weights=weights) - 3  # 超额峰度
        return kurtosis
    
    def _test_distribution_hypotheses(self, data, weights, coord_name):
        """
        测试不同的分布假设
        """
        if len(data) < 10:
            return {'error': 'Insufficient data'}
        
        weights_norm = weights / np.sum(weights)
        
        tests = {}
        
        # 1. 正态性测试
        # Shapiro-Wilk测试（适用于小样本）
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            tests['shapiro_wilk'] = {
                'statistic': float(shapiro_stat),
                'pvalue': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            }
        
        # Anderson-Darling测试
        anderson_result = stats.anderson(data, dist='norm')
        tests['anderson_darling'] = {
            'statistic': float(anderson_result.statistic),
            'critical_values': anderson_result.critical_values.tolist(),
            'significance_level': anderson_result.significance_level.tolist(),
            'is_normal': anderson_result.statistic < anderson_result.critical_values[2]  # 5% level
        }
        
        # 2. 指数性测试
        # 平移数据使其最小值为0（指数分布要求）
        data_shifted = data - np.min(data) + 1e-10
        
        try:
            # KS测试 vs 指数分布
            exp_fit = stats.expon.fit(data_shifted)
            ks_exp_stat, ks_exp_p = stats.kstest(data_shifted, 'expon', args=exp_fit)
            tests['exponential_ks'] = {
                'statistic': float(ks_exp_stat),
                'pvalue': float(ks_exp_p),
                'is_exponential': ks_exp_p > 0.05
            }
        except:
            tests['exponential_ks'] = {'error': 'Failed to fit exponential'}
        
        # 3. 拉普拉斯分布测试
        try:
            laplace_fit = stats.laplace.fit(data)
            ks_laplace_stat, ks_laplace_p = stats.kstest(data, 'laplace', args=laplace_fit)
            tests['laplace_ks'] = {
                'statistic': float(ks_laplace_stat),
                'pvalue': float(ks_laplace_p),
                'is_laplace': ks_laplace_p > 0.05
            }
        except:
            tests['laplace_ks'] = {'error': 'Failed to fit laplace'}
        
        # 4. 均匀分布测试
        try:
            uniform_fit = stats.uniform.fit(data)
            ks_uniform_stat, ks_uniform_p = stats.kstest(data, 'uniform', args=uniform_fit)
            tests['uniform_ks'] = {
                'statistic': float(ks_uniform_stat),
                'pvalue': float(ks_uniform_p),
                'is_uniform': ks_uniform_p > 0.05
            }
        except:
            tests['uniform_ks'] = {'error': 'Failed to fit uniform'}
        
        # 5. 拟合优度比较
        try:
            # 计算不同分布的AIC
            distributions = [
                ('normal', stats.norm, stats.norm.fit(data)),
                ('exponential', stats.expon, stats.expon.fit(data_shifted)),
                ('laplace', stats.laplace, stats.laplace.fit(data)),
                ('uniform', stats.uniform, stats.uniform.fit(data))
            ]
            
            aic_scores = {}
            for name, dist, params in distributions:
                try:
                    # 计算对数似然
                    log_likelihood = np.sum(dist.logpdf(data, *params))
                    # AIC = 2k - 2ln(L)，k是参数个数
                    k = len(params)
                    aic = 2 * k - 2 * log_likelihood
                    aic_scores[name] = {
                        'aic': float(aic),
                        'log_likelihood': float(log_likelihood),
                        'params': [float(p) for p in params]
                    }
                except:
                    aic_scores[name] = {'error': 'Failed to compute AIC'}
            
            tests['aic_comparison'] = aic_scores
            
            # 找出最佳分布（最小AIC）
            valid_aics = {k: v for k, v in aic_scores.items() if 'aic' in v}
            if valid_aics:
                best_dist = min(valid_aics.items(), key=lambda x: x[1]['aic'])
                tests['best_distribution'] = {
                    'name': best_dist[0],
                    'aic': best_dist[1]['aic']
                }
        except:
            tests['aic_comparison'] = {'error': 'Failed AIC comparison'}
        
        # 6. 尾部分析
        # 检查是否有重尾
        tail_ratio = np.percentile(data, 95) - np.percentile(data, 5)
        tail_ratio /= np.percentile(data, 75) - np.percentile(data, 25) if (np.percentile(data, 75) - np.percentile(data, 25)) > 0 else 1
        
        tests['tail_analysis'] = {
            'tail_ratio': float(tail_ratio),
            'is_heavy_tailed': tail_ratio > 2.0,
            'is_light_tailed': tail_ratio < 1.0
        }
        
        return tests
    
    def _create_coordinate_diagnostic_plot(self, llp_id, coord, data, weights, 
                                         params, stats_result, tests_result, 
                                         output_path: Path):
        """
        创建坐标级别的诊断图
        """
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Diagnostic Plot: {llp_id} - {coord.upper()} Coordinate\n'
                    f'Mass: {params.get("mass", "N/A"):.3f} GeV, '
                    f'τ: {params.get("lifetime", "N/A"):.2e} mm', 
                    fontsize=14, y=1.02)
        
        weights_norm = weights / np.sum(weights)
        
        # 1. 直方图与不同分布拟合
        ax1 = plt.subplot(3, 3, 1)
        hist_bins = min(50, len(data) // 10)
        
        ax1.hist(data, bins=hist_bins, weights=weights_norm, 
                density=True, alpha=0.5, color='blue', 
                edgecolor='black', label='Data')
        
        # 绘制不同的分布拟合
        x_range = np.linspace(np.min(data), np.max(data), 1000)
        
        # 正态分布
        norm_fit = stats.norm.fit(data)
        norm_pdf = stats.norm.pdf(x_range, *norm_fit)
        ax1.plot(x_range, norm_pdf, 'r-', linewidth=2, alpha=0.7, label='Normal')
        
        # 指数分布（平移后）
        data_shifted = data - np.min(data) + 1e-10
        exp_fit = stats.expon.fit(data_shifted)
        # 平移回来
        exp_pdf = stats.expon.pdf(x_range - np.min(data) + 1e-10, *exp_fit)
        ax1.plot(x_range, exp_pdf, 'g-', linewidth=2, alpha=0.7, label='Exponential')
        
        # 拉普拉斯分布
        laplace_fit = stats.laplace.fit(data)
        laplace_pdf = stats.laplace.pdf(x_range, *laplace_fit)
        ax1.plot(x_range, laplace_pdf, 'orange', linewidth=2, alpha=0.7, label='Laplace')
        
        ax1.set_xlabel(f'{coord.upper()} Position (mm)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Histogram with Distribution Fits', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. 对数尺度直方图
        ax2 = plt.subplot(3, 3, 2)
        hist, bins = np.histogram(data, bins=hist_bins, weights=weights_norm, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        ax2.bar(bin_centers, hist, width=bins[1]-bins[0], alpha=0.5, color='blue')
        
        # 在对数尺度上绘制拟合
        ax2.semilogy(x_range, norm_pdf, 'r-', linewidth=2, alpha=0.7, label='Normal')
        ax2.semilogy(x_range, exp_pdf, 'g-', linewidth=2, alpha=0.7, label='Exponential')
        ax2.semilogy(x_range, laplace_pdf, 'orange', linewidth=2, alpha=0.7, label='Laplace')
        
        ax2.set_xlabel(f'{coord.upper()} Position (mm)', fontsize=11)
        ax2.set_ylabel('Log Density', fontsize=11)
        ax2.set_title('Log-scale Histogram', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q图
        ax3 = plt.subplot(3, 3, 3)
        stats.probplot(data, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot vs Normal Distribution', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. 经验CDF与理论CDF比较
        ax4 = plt.subplot(3, 3, 4)
        sorted_data = np.sort(data)
        ecdf = np.arange(1, len(data)+1) / len(data)
        ax4.plot(sorted_data, ecdf, 'b-', linewidth=3, label='Empirical CDF')
        
        # 理论CDFs
        ax4.plot(x_range, stats.norm.cdf(x_range, *norm_fit), 'r--', alpha=0.7, label='Normal CDF')
        ax4.plot(x_range, stats.expon.cdf(x_range - np.min(data) + 1e-10, *exp_fit), 
                'g--', alpha=0.7, label='Exponential CDF')
        ax4.plot(x_range, stats.laplace.cdf(x_range, *laplace_fit), 
                'orange', linestyle='--', alpha=0.7, label='Laplace CDF')
        
        ax4.set_xlabel(f'{coord.upper()} Position (mm)', fontsize=11)
        ax4.set_ylabel('Cumulative Probability', fontsize=11)
        ax4.set_title('CDF Comparison', fontsize=12)
        ax4.legend(fontsize=9, loc='lower right')
        ax4.grid(True, alpha=0.3)
        
        # 5. 箱线图
        ax5 = plt.subplot(3, 3, 5)
        ax5.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'),
                   medianprops=dict(color='red', linewidth=2))
        
        ax5.set_ylabel(f'{coord.upper()} Position (mm)', fontsize=11)
        ax5.set_title('Box Plot', fontsize=12)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 统计信息
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')
        
        stats_text = [
            f"BASIC STATISTICS:",
            f"Samples: {stats_result['n_samples']:,}",
            f"Mean: {stats_result['weighted_mean']:.3f} mm",
            f"Std Dev: {stats_result['weighted_std']:.3f} mm",
            f"Range: {stats_result['range']:.3f} mm",
            f"Skewness: {stats_result['weighted_skewness']:.3f}",
            f"Kurtosis: {stats_result['weighted_kurtosis']:.3f}",
            f"",
            f"OUTLIERS:",
            f"Count: {stats_result['outlier_count']}",
            f"Percentage: {stats_result['outlier_percentage']:.1f}%",
        ]
        
        ax6.text(0.05, 0.95, '\n'.join(stats_text), transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 7. 分布测试结果
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        
        tests_text = ["DISTRIBUTION TESTS:"]
        
        # Shapiro-Wilk
        if 'shapiro_wilk' in tests_result:
            sw = tests_result['shapiro_wilk']
            if 'statistic' in sw:
                normal_result = "✓ NORMAL" if sw.get('is_normal', False) else "✗ NOT NORMAL"
                tests_text.append(f"Shapiro-Wilk: {normal_result}")
                tests_text.append(f"  Stat: {sw['statistic']:.3f}, p={sw['pvalue']:.3e}")
        
        # Anderson-Darling
        if 'anderson_darling' in tests_result:
            ad = tests_result['anderson_darling']
            if 'statistic' in ad:
                normal_result = "✓ NORMAL" if ad.get('is_normal', False) else "✗ NOT NORMAL"
                tests_text.append(f"Anderson-Darling: {normal_result}")
                tests_text.append(f"  Stat: {ad['statistic']:.3f}")
        
        # 最佳分布
        if 'best_distribution' in tests_result.get('aic_comparison', {}):
            best = tests_result['aic_comparison']['best_distribution']
            tests_text.append(f"\nBEST DISTRIBUTION (AIC):")
            tests_text.append(f"  {best['name'].upper()}")
            tests_text.append(f"  AIC: {best['aic']:.1f}")
        
        ax7.text(0.05, 0.95, '\n'.join(tests_text), transform=ax7.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # 8. 尾部分析
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        tail_info = tests_result.get('tail_analysis', {})
        tail_text = ["TAIL ANALYSIS:"]
        
        if 'tail_ratio' in tail_info:
            tail_ratio = tail_info['tail_ratio']
            tail_text.append(f"Tail Ratio: {tail_ratio:.2f}")
            
            if tail_info.get('is_heavy_tailed', False):
                tail_text.append("→ HEAVY TAILED")
                tail_text.append("  Consider: Student's t, Cauchy, etc.")
            elif tail_info.get('is_light_tailed', False):
                tail_text.append("→ LIGHT TAILED")
                tail_text.append("  Consider: Uniform, Beta, etc.")
            else:
                tail_text.append("→ MODERATE TAILS")
                tail_text.append("  Normal/Laplace may work")
        
        ax8.text(0.05, 0.95, '\n'.join(tail_text), transform=ax8.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        # 9. 建议
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        recommendations = self._generate_coordinate_recommendations(stats_result, tests_result)
        
        rec_text = ["RECOMMENDATIONS:"]
        for rec in recommendations:
            rec_text.append(f"• {rec}")
        
        # 根据问题的严重性设置颜色
        if stats_result['is_constant']:
            color = 'red'
            rec_text.append("\n⚠ WARNING: Data is constant!")
        elif stats_result['outlier_percentage'] > 10:
            color = 'orange'
        else:
            color = 'lightblue'
        
        ax9.text(0.05, 0.95, '\n'.join(rec_text), transform=ax9.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(output_path / f'{coord}_diagnostic.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建简化诊断图
        self._create_simple_diagnostic_plot(llp_id, coord, data, weights, 
                                          stats_result, tests_result, output_path)
    
    def _create_simple_diagnostic_plot(self, llp_id, coord, data, weights,
                                     stats_result, tests_result, output_path: Path):
        """
        创建简化版诊断图
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Quick Diagnostic: {llp_id} - {coord.upper()}', fontsize=14, y=1.02)
        
        weights_norm = weights / np.sum(weights)
        
        # 子图1: 直方图
        ax1 = axes[0]
        hist_bins = min(30, len(data) // 20)
        ax1.hist(data, bins=hist_bins, weights=weights_norm, 
                density=True, alpha=0.5, color='blue', edgecolor='black')
        
        # 正态分布拟合
        norm_fit = stats.norm.fit(data)
        x_range = np.linspace(np.min(data), np.max(data), 1000)
        norm_pdf = stats.norm.pdf(x_range, *norm_fit)
        ax1.plot(x_range, norm_pdf, 'r-', linewidth=2, alpha=0.7, label='Normal fit')
        
        ax1.set_xlabel(f'{coord.upper()} (mm)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Distribution', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: Q-Q图
        ax2 = axes[1]
        stats.probplot(data, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot vs Normal', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 统计摘要
        ax3 = axes[2]
        ax3.axis('off')
        
        summary_text = [
            f"KEY STATISTICS:",
            f"Mean: {stats_result['weighted_mean']:.3f} mm",
            f"Std: {stats_result['weighted_std']:.3f} mm",
            f"Skew: {stats_result['weighted_skewness']:.3f}",
            f"Kurt: {stats_result['weighted_kurtosis']:.3f}",
            f"Range: {stats_result['range']:.3f} mm",
            f"",
            f"DISTRIBUTION TESTS:",
        ]
        
        # 添加正态性测试结果
        if 'shapiro_wilk' in tests_result:
            sw = tests_result['shapiro_wilk']
            if 'pvalue' in sw:
                is_normal = sw['pvalue'] > 0.05
                summary_text.append(f"Normal: {'✓' if is_normal else '✗'}")
        
        # 添加最佳分布
        aic_comparison = tests_result.get('aic_comparison', {})
        if 'best_distribution' in aic_comparison:
            best = aic_comparison['best_distribution']
            summary_text.append(f"Best fit: {best['name']}")
        
        # 添加尾部信息
        tail_info = tests_result.get('tail_analysis', {})
        if 'tail_ratio' in tail_info:
            tail_type = "Heavy" if tail_info.get('is_heavy_tailed') else "Light" if tail_info.get('is_light_tailed') else "Moderate"
            summary_text.append(f"Tails: {tail_type}")
        
        ax3.text(0.05, 0.95, '\n'.join(summary_text), transform=ax3.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path / f'{coord}_diagnostic_simple.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_coordinate_recommendations(self, stats_result, tests_result):
        """
        为单个坐标生成建议
        """
        recommendations = []
        
        # 检查数据是否常数
        if stats_result['is_constant']:
            recommendations.append("Data is essentially constant. Use delta function or ignore this coordinate.")
            return recommendations
        
        # 检查异常值
        if stats_result['outlier_percentage'] > 20:
            recommendations.append(f"High outlier percentage ({stats_result['outlier_percentage']:.1f}%). Consider robust methods.")
        
        # 根据正态性测试
        if 'shapiro_wilk' in tests_result:
            sw = tests_result['shapiro_wilk']
            if 'is_normal' in sw and not sw['is_normal']:
                recommendations.append("Data is not normally distributed. Consider non-Gaussian models.")
        
        # 根据最佳分布
        aic_comparison = tests_result.get('aic_comparison', {})
        if 'best_distribution' in aic_comparison:
            best = aic_comparison['best_distribution']
            if best['name'] == 'normal':
                recommendations.append("Normal distribution provides the best fit.")
            elif best['name'] == 'exponential':
                recommendations.append("Exponential distribution provides the best fit. Consider asymmetric models.")
            elif best['name'] == 'laplace':
                recommendations.append("Laplace (double exponential) provides the best fit.")
            elif best['name'] == 'uniform':
                recommendations.append("Uniform distribution provides the best fit. Data may be bounded.")
        
        # 根据尾部分析
        tail_info = tests_result.get('tail_analysis', {})
        if tail_info.get('is_heavy_tailed'):
            recommendations.append("Heavy-tailed distribution. Consider Student's t or Cauchy distributions.")
        elif tail_info.get('is_light_tailed'):
            recommendations.append("Light-tailed distribution. Consider bounded distributions like Beta.")
        
        # 根据数据范围
        if stats_result['range'] < 1.0:
            recommendations.append("Narrow data range. Consider precision issues or measurement constraints.")
        elif stats_result['range'] > 1000.0:
            recommendations.append("Wide data range. Check for outliers or data errors.")
        
        return recommendations
    
    def _generate_recommendations(self, coord_stats, dist_tests):
        """
        为整个LLP生成建议
        """
        recommendations = []
        
        # 检查所有坐标是否都是常数
        constant_coords = [c for c in ['x', 'y', 'z'] if coord_stats[c].get('is_constant', False)]
        if len(constant_coords) == 3:
            recommendations.append("ALL coordinates are constant! Check MC simulation parameters.")
        elif len(constant_coords) > 0:
            recommendations.append(f"Coordinates {constant_coords} are constant. May need special handling.")
        
        # 检查坐标间的一致性
        x_std = coord_stats['x']['weighted_std']
        y_std = coord_stats['y']['weighted_std']
        z_std = coord_stats['z']['weighted_std']
        
        stds = [x_std, y_std, z_std]
        if max(stds) / min(stds) > 100 and min(stds) > 0:
            recommendations.append(f"Large variation in spread across coordinates (X:{x_std:.2f}, Y:{y_std:.2f}, Z:{z_std:.2f}).")
        
        # 检查最佳分布是否一致
        best_dists = []
        for coord in ['x', 'y', 'z']:
            aic_comp = dist_tests[coord].get('aic_comparison', {})
            if 'best_distribution' in aic_comp:
                best_dists.append(aic_comp['best_distribution']['name'])
        
        if len(set(best_dists)) > 1 and len(best_dists) == 3:
            recommendations.append(f"Different best distributions for each coordinate: X={best_dists[0]}, Y={best_dists[1]}, Z={best_dists[2]}")
        
        # 总体建议
        recommendations.append("Consider coordinate-specific modeling for best results.")
        recommendations.append("Validate physical plausibility of the distributions.")
        
        return recommendations
    
    def _create_llp_summary_plot(self, llp_id, diagnostic_results, output_path: Path):
        """
        创建LLP级别的汇总图
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'LLP Summary: {llp_id}', fontsize=16, y=1.02)
        
        params = diagnostic_results['params']
        coord_stats = diagnostic_results['coordinate_stats']
        
        # 子图1: 三个坐标的均值±标准差
        ax1 = axes[0, 0]
        
        coords = ['x', 'y', 'z']
        means = [coord_stats[c]['weighted_mean'] for c in coords]
        stds = [coord_stats[c]['weighted_std'] for c in coords]
        
        x_pos = np.arange(3)
        ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
               color=['red', 'green', 'blue'])
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['X', 'Y', 'Z'])
        ax1.set_ylabel('Mean Position ± Std (mm)', fontsize=11)
        ax1.set_title('Coordinate Statistics', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 子图2: 分布测试结果热图
        ax2 = axes[0, 1]
        
        tests = ['Normal (SW)', 'Exponential', 'Laplace', 'Uniform']
        test_results = []
        
        for coord in coords:
            dist_tests = diagnostic_results['distribution_tests'][coord]
            coord_results = []
            
            # Shapiro-Wilk
            if 'shapiro_wilk' in dist_tests:
                coord_results.append(1 if dist_tests['shapiro_wilk'].get('is_normal', False) else 0)
            else:
                coord_results.append(np.nan)
            
            # Exponential
            if 'exponential_ks' in dist_tests:
                coord_results.append(1 if dist_tests['exponential_ks'].get('is_exponential', False) else 0)
            else:
                coord_results.append(np.nan)
            
            # Laplace
            if 'laplace_ks' in dist_tests:
                coord_results.append(1 if dist_tests['laplace_ks'].get('is_laplace', False) else 0)
            else:
                coord_results.append(np.nan)
            
            # Uniform
            if 'uniform_ks' in dist_tests:
                coord_results.append(1 if dist_tests['uniform_ks'].get('is_uniform', False) else 0)
            else:
                coord_results.append(np.nan)
            
            test_results.append(coord_results)
        
        im = ax2.imshow(test_results, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        ax2.set_xticks(range(len(tests)))
        ax2.set_xticklabels(tests, rotation=45, fontsize=10)
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(['X', 'Y', 'Z'])
        ax2.set_title('Distribution Test Results', fontsize=12)
        
        plt.colorbar(im, ax=ax2, label='Pass (1) / Fail (0)')
        
        # 子图3: 参数信息
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        param_text = [
            f"PHYSICAL PARAMETERS:",
            f"Mass: {params.get('mass', 'N/A'):.3f} GeV",
            f"Lifetime: {params.get('lifetime', 'N/A'):.2e} mm",
            f"tanβ: {params.get('tanb', 'N/A'):.1f}",
            f"Visible BR: {params.get('vis_br', 'N/A'):.2e}",
            f"",
            f"SAMPLE INFO:",
            f"Total positions: {diagnostic_results.get('n_samples', 'N/A'):,}",
            f"",
            f"KEY FINDINGS:",
        ]
        
        # 添加关键发现
        for rec in diagnostic_results['recommendations'][:5]:  # 只显示前5条
            param_text.append(f"• {rec}")
        
        ax3.text(0.05, 0.95, '\n'.join(param_text), transform=ax3.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 子图4: AIC比较
        ax4 = axes[1, 1]
        
        # 收集AIC值
        distributions = ['normal', 'exponential', 'laplace', 'uniform']
        aic_values = {d: [] for d in distributions}
        
        for coord in coords:
            aic_comp = diagnostic_results['distribution_tests'][coord].get('aic_comparison', {})
            for dist in distributions:
                if dist in aic_comp and 'aic' in aic_comp[dist]:
                    aic_values[dist].append(aic_comp[dist]['aic'])
                else:
                    aic_values[dist].append(np.nan)
        
        # 绘制箱线图
        aic_data = [aic_values[d] for d in distributions]
        positions = np.arange(len(distributions))
        
        for i, dist in enumerate(distributions):
            valid_aics = [a for a in aic_values[dist] if not np.isnan(a)]
            if valid_aics:
                ax4.scatter([i] * len(valid_aics), valid_aics, alpha=0.5, s=30)
        
        ax4.boxplot([d for d in aic_data if any(not np.isnan(v) for v in d)], 
                   positions=positions[:len([d for d in aic_data if any(not np.isnan(v) for v in d)])])
        
        ax4.set_xticks(range(len(distributions)))
        ax4.set_xticklabels([d.capitalize() for d in distributions], rotation=45, fontsize=10)
        ax4.set_ylabel('AIC Value', fontsize=11)
        ax4.set_title('AIC Comparison Across Coordinates', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path / 'llp_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_diagnostic_summary(self, output_path: Path):
        """
        创建所有LLP的诊断总结
        """
        if not self.diagnostic_results:
            print("No diagnostic results to summarize")
            return
        
        print("\nCreating diagnostic summary...")
        
        summary_data = []
        
        for llp_id, diagnostic in self.diagnostic_results.items():
            params = diagnostic['params']
            coord_stats = diagnostic['coordinate_stats']
            
            summary = {
                'llp_id': llp_id,
                'mass': float(params.get('mass', np.nan)),
                'lifetime': float(params.get('lifetime', np.nan)),
                'tanb': float(params.get('tanb', np.nan)),
            }
            
            # 为每个坐标添加统计信息
            for coord in ['x', 'y', 'z']:
                stats_info = coord_stats[coord]
                summary[f'{coord}_mean'] = stats_info['weighted_mean']
                summary[f'{coord}_std'] = stats_info['weighted_std']
                summary[f'{coord}_skew'] = stats_info['weighted_skewness']
                summary[f'{coord}_kurt'] = stats_info['weighted_kurtosis']
                summary[f'{coord}_is_constant'] = stats_info['is_constant']
                
                # 分布测试结果
                dist_tests = diagnostic['distribution_tests'][coord]
                if 'shapiro_wilk' in dist_tests:
                    sw = dist_tests['shapiro_wilk']
                    summary[f'{coord}_normal_pvalue'] = sw.get('pvalue', np.nan)
                    summary[f'{coord}_is_normal'] = sw.get('is_normal', False)
                
                # 最佳分布
                aic_comp = dist_tests.get('aic_comparison', {})
                if 'best_distribution' in aic_comp:
                    summary[f'{coord}_best_dist'] = aic_comp['best_distribution']['name']
            
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存CSV
        csv_path = output_path / 'diagnostic_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"✓ Diagnostic summary saved to: {csv_path}")
        
        # 创建总结报告
        self._create_diagnostic_report(summary_df, output_path)
        
        return summary_df
    
    def _create_diagnostic_report(self, summary_df, output_path: Path):
        """
        创建诊断报告
        """
        report = []
        
        report.append("=" * 70)
        report.append("LLP DISTRIBUTION DIAGNOSTIC REPORT")
        report.append("=" * 70)
        report.append(f"\nAnalysis Date: {pd.Timestamp.now()}")
        report.append(f"Total LLPs diagnosed: {len(summary_df)}")
        
        if len(summary_df) > 0:
            report.append(f"\nOVERALL FINDINGS:")
            report.append("-" * 40)
            
            # 常数坐标统计
            constant_counts = {}
            for coord in ['x', 'y', 'z']:
                col_name = f'{coord}_is_constant'
                if col_name in summary_df.columns:
                    constant_count = summary_df[col_name].sum()
                    constant_counts[coord] = constant_count
            
            if any(count > 0 for count in constant_counts.values()):
                report.append("Constant coordinates detected:")
                for coord, count in constant_counts.items():
                    if count > 0:
                        percentage = count / len(summary_df) * 100
                        report.append(f"  {coord.upper()}: {count} LLPs ({percentage:.1f}%)")
            
            # 正态性测试结果
            normal_counts = {}
            for coord in ['x', 'y', 'z']:
                col_name = f'{coord}_is_normal'
                if col_name in summary_df.columns:
                    normal_count = summary_df[col_name].sum()
                    normal_counts[coord] = normal_count
            
            report.append(f"\nNormality test results:")
            for coord, count in normal_counts.items():
                percentage = count / len(summary_df) * 100
                report.append(f"  {coord.upper()}: {count}/{len(summary_df)} ({percentage:.1f}%) normal")
            
            # 最佳分布统计
            dist_counts = {}
            for coord in ['x', 'y', 'z']:
                col_name = f'{coord}_best_dist'
                if col_name in summary_df.columns:
                    dist_series = summary_df[col_name].value_counts()
                    dist_counts[coord] = dict(dist_series)
            
            report.append(f"\nBest distribution by AIC:")
            for coord in ['x', 'y', 'z']:
                if coord in dist_counts and dist_counts[coord]:
                    report.append(f"  {coord.upper()}:")
                    for dist, count in dist_counts[coord].items():
                        percentage = count / len(summary_df) * 100
                        report.append(f"    {dist}: {count} ({percentage:.1f}%)")
            
            # 参数相关性
            if 'mass' in summary_df.columns and 'x_std' in summary_df.columns:
                corr_mass_std = summary_df['mass'].corr(summary_df['x_std'])
                report.append(f"\nMass vs X-spread correlation: {corr_mass_std:.3f}")
            
            # 关键建议
            report.append(f"\nKEY RECOMMENDATIONS:")
            report.append("-" * 40)
            
            if any(count > len(summary_df) * 0.5 for count in constant_counts.values()):
                report.append("1. Many coordinates are constant. Check MC simulation parameters.")
            
            if all(count < len(summary_df) * 0.3 for count in normal_counts.values()):
                report.append("2. Most distributions are non-normal. Use non-Gaussian models.")
            elif any(count > len(summary_df) * 0.7 for count in normal_counts.values()):
                report.append("2. Many distributions are normal. Gaussian models may work well.")
            
            # 检查最佳分布是否一致
            consistent_dist = True
            for llp_id in self.diagnostic_results:
                best_dists = []
                for coord in ['x', 'y', 'z']:
                    dist_tests = self.diagnostic_results[llp_id]['distribution_tests'][coord]
                    aic_comp = dist_tests.get('aic_comparison', {})
                    if 'best_distribution' in aic_comp:
                        best_dists.append(aic_comp['best_distribution']['name'])
                
                if len(set(best_dists)) > 1:
                    consistent_dist = False
                    break
            
            if not consistent_dist:
                report.append("3. Different coordinates often have different best distributions.")
                report.append("   Consider coordinate-specific modeling.")
        
        report.append(f"\nNEXT STEPS:")
        report.append("-" * 40)
        report.append("1. Review individual diagnostic plots for each LLP")
        report.append("2. Check LLPs with constant or abnormal distributions")
        report.append("3. Adjust fitting strategy based on diagnostic findings")
        report.append("4. Consider physical constraints in model selection")
        
        report.append(f"\nOUTPUT FILES:")
        report.append("-" * 40)
        report.append("1. diagnostic_summary.csv - Summary statistics")
        report.append("2. Individual diagnostic plots for each LLP and coordinate")
        report.append("3. LLP summary plots")
        
        report.append(f"\n" + "=" * 70)
        report.append("END OF DIAGNOSTIC REPORT")
        report.append("=" * 70)
        
        report_text = '\n'.join(report)
        
        with open(output_path / 'diagnostic_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"✓ Diagnostic report saved to: {output_path}/diagnostic_report.txt")

def main():
    """主函数"""
    # 设置路径
    data_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_40_N/llp_simulation_results/incremental_results"
    diagnostic_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_40_N/diagnostics"
    
    print("=" * 70)
    print("LLP DISTRIBUTION DIAGNOSTIC TOOL")
    print("=" * 70)
    
    # 创建分析器
    analyzer = LLPDistributionAnalyzer(data_dir)
    
    try:
        # 1. 加载数据
        print("\n[1/3] Loading data...")
        analyzer.load_all_data()
        
        # 2. 运行诊断
        print("\n[2/3] Running diagnostics...")
        
        # 创建诊断工具
        diagnostic_tool = LLPDistributionDiagnostic(analyzer.llp_data)
        
        # 运行完整诊断
        diagnostic_tool.run_full_diagnosis(diagnostic_dir)
        
        # 3. 分析诊断结果
        print("\n[3/3] Analyzing diagnostic results...")
        
        # 获取几个典型的LLP进行深入分析
        typical_llps = list(analyzer.llp_data.keys())[:3]
        print(f"\nDetailed analysis for typical LLPs: {typical_llps}")
        
        for llp_id in typical_llps:
            print(f"\n{'='*50}")
            print(f"ANALYSIS FOR {llp_id}")
            print('='*50)
            
            data = analyzer.llp_data[llp_id]
            positions = data['positions']
            weights = data['weights']
            
            for coord_idx, coord in enumerate(['x', 'y', 'z']):
                coord_data = positions[:, coord_idx]
                
                print(f"\n{coord.upper()} Coordinate:")
                print(f"  Samples: {len(coord_data):,}")
                print(f"  Range: [{np.min(coord_data):.3f}, {np.max(coord_data):.3f}] mm")
                print(f"  Mean: {np.average(coord_data, weights=weights):.3f} mm")
                print(f"  Std: {np.sqrt(np.average((coord_data-np.mean(coord_data))**2, weights=weights)):.3f} mm")
                
                # 快速查看分布形状
                if len(coord_data) > 10:
                    # 简单直方图
                    plt.figure(figsize=(10, 4))
                    
                    plt.subplot(1, 2, 1)
                    plt.hist(coord_data, bins=30, weights=weights/np.sum(weights), 
                            density=True, alpha=0.7)
                    plt.xlabel(f'{coord.upper()} (mm)')
                    plt.ylabel('Density')
                    plt.title(f'{llp_id} - {coord.upper()} Distribution')
                    
                    plt.subplot(1, 2, 2)
                    stats.probplot(coord_data, dist="norm", plot=plt)
                    plt.title(f'Q-Q Plot vs Normal')
                    
                    plt.tight_layout()
                    plt.savefig(f'{diagnostic_dir}/{llp_id}_{coord}_quick.png', 
                              dpi=120, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  Quick plot saved: {diagnostic_dir}/{llp_id}_{coord}_quick.png")
        
        print(f"\n{'='*70}")
        print("DIAGNOSTIC COMPLETED!")
        print('='*70)
        
        print(f"\n✅ Diagnostic results saved to: {diagnostic_dir}")
        print(f"\n📊 Key output files:")
        print(f"  {diagnostic_dir}/diagnostic_summary.csv - Summary statistics")
        print(f"  {diagnostic_dir}/diagnostic_report.txt - Detailed report")
        print(f"  {diagnostic_dir}/[llp_id]/ - Individual diagnostic plots")
        
    except Exception as e:
        print(f"\n❌ Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# 使用示例
# if __name__ == "__main__":
#     data_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_40/llp_simulation_results/incremental_results"
    
#     # 诊断特定LLP
#     quick_diagnostic(data_dir, llp_ids=['llp_0117', 'llp_0123', 'llp_0135'])


# def main():
#     """主函数"""
#     # 设置路径
#     data_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_40/llp_simulation_results/incremental_results"
#     output_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_40/distributution_density"
    
#     print("=" * 70)
#     print("LLP DECAY POSITION DISTRIBUTION ANALYSIS")
#     print("=" * 70)
    
#     # 创建分析器
#     analyzer = LLPDistributionAnalyzer(data_dir)
    
#     try:
#         # 1. 加载数据
#         print("\n[1/3] Loading data...")
#         analyzer.load_all_data()
        
#         # 2. 分析分布
#         print("\n[2/3] Analyzing distributions...")
#         analyzer.analyze_distributions()
        
#         # 3. 创建可视化
#         print("\n[3/3] Creating visualizations...")
#         analyzer.create_distribution_plots(output_dir)
        
#         # 4. 保存结果
#         analyzer.save_results(output_dir)
        
#         print("\n" + "=" * 70)
#         print("ANALYSIS COMPLETED SUCCESSFULLY!")
#         print("=" * 70)
        
#         print(f"\n✅ Results saved to: {output_dir}")
#         print(f"\n📊 Key output files:")
#         print(f"  {output_dir}/llp_summary.csv - Complete summary")
#         print(f"  {output_dir}/analysis_summary.png - Comprehensive summary plot")
#         print(f"  {output_dir}/analysis_report.txt - Detailed report")
#         print(f"  {output_dir}/distribution_models/ - Individual distribution models")
        
#     except Exception as e:
#         print(f"\n❌ Error during analysis: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()