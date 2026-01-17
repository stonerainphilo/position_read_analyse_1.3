import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import json
import os
import pickle

def analyze_distributions_with_comparison(input_csv, output_dir='distribution_analysis'):
    """
    分析衰变位置分布：拟合、绘图、保存函数，并对比拟合与原始结果
    
    Args:
        input_csv: 输入CSV文件路径
        output_dir: 输出目录
    """
    # 1. 读取数据
    df = pd.read_csv(input_csv)
    required_columns = ['decay_pos_x', 'decay_pos_y', 'decay_pos_z']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    
    # 存储拟合结果
    fit_results = {
        'filename': base_name,
        'fits': {},
        'statistics': {}
    }
    
    # 3. 对每个坐标轴进行分析
    for coord in required_columns:
        print(f"\n=== Analyzing {coord} ===")
        data = df[coord].dropna()
        if len(data) < 2:
            print(f"Warning: Insufficient data for {coord}, skipping.")
            continue
        
        # 3.1 计算基本统计量
        stats_dict = {
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'n_points': int(len(data)),
            'skewness': float(data.skew()),
            'kurtosis': float(data.kurtosis())
        }
        fit_results['statistics'][coord] = stats_dict
        
        # 3.2 拟合KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data, bw_method='scott')
        
        # 创建拟合函数
        x_fit = np.linspace(data.min(), data.max(), 1000)
        y_fit = kde(x_fit)
        
        # 存储KDE拟合
        fit_results['fits'][coord] = {
            'type': 'kde',
            'bandwidth': kde.factor,  # KDE带宽参数
            'data_range': [float(data.min()), float(data.max())],
            'n_points_fit': 1000
        }
        
        # 3.3 可选：参数分布拟合（尝试多种分布）
        parametric_fits = try_parametric_fits(data)
        if parametric_fits:
            fit_results['fits'][coord]['parametric'] = parametric_fits
        
        # 3.4 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Distribution Analysis: {coord}\nFile: {base_name}', fontsize=14, fontweight='bold')
        
        # 子图1: 直方图与KDE拟合对比
        axes[0, 0].hist(data, bins='auto', density=True, alpha=0.6, 
                       color='skyblue', edgecolor='black', label='Histogram')
        axes[0, 0].plot(x_fit, y_fit, 'r-', linewidth=2, label='KDE Fit')
        axes[0, 0].axvline(stats_dict['mean'], color='green', linestyle='--', 
                          label=f'Mean: {stats_dict["mean"]:.2f}')
        axes[0, 0].axvline(stats_dict['median'], color='orange', linestyle='--', 
                          label=f'Median: {stats_dict["median"]:.2f}')
        axes[0, 0].set_xlabel(coord, fontsize=11)
        axes[0, 0].set_ylabel('Probability Density', fontsize=11)
        axes[0, 0].set_title('Histogram with KDE Fit', fontsize=12)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2: 累积分布函数对比
        data_sorted = np.sort(data)
        cdf_empirical = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        cdf_fitted = np.cumsum(y_fit) * (x_fit[1] - x_fit[0])
        
        axes[0, 1].plot(data_sorted, cdf_empirical, 'b-', linewidth=2, label='Empirical CDF')
        axes[0, 1].plot(x_fit, cdf_fitted, 'r--', linewidth=2, label='Fitted CDF (from KDE)')
        axes[0, 1].set_xlabel(coord, fontsize=11)
        axes[0, 1].set_ylabel('Cumulative Probability', fontsize=11)
        axes[0, 1].set_title('Cumulative Distribution Comparison', fontsize=12)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3: 分位数-分位数图 (Q-Q Plot)
        if parametric_fits and 'normal' in parametric_fits:
            from scipy.stats import norm
            theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, len(data)))
            axes[1, 0].scatter(theoretical_quantiles, np.sort(data), alpha=0.6)
            axes[1, 0].plot([theoretical_quantiles.min(), theoretical_quantiles.max()],
                           [theoretical_quantiles.min(), theoretical_quantiles.max()], 
                           'r--', label='Perfect Fit')
            axes[1, 0].set_xlabel('Theoretical Normal Quantiles', fontsize=11)
            axes[1, 0].set_ylabel('Sample Quantiles', fontsize=11)
            axes[1, 0].set_title('Q-Q Plot vs Normal Distribution', fontsize=12)
            axes[1, 0].legend(fontsize=9)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: 残差分析（拟合误差）
        # 使用直方图分箱计算残差
        hist_counts, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        kde_at_bins = kde(bin_centers)
        residuals = hist_counts - kde_at_bins
        
        axes[1, 1].bar(bin_centers, residuals, width=bin_edges[1]-bin_edges[0], 
                      alpha=0.6, color='gray', edgecolor='black')
        axes[1, 1].axhline(y=0, color='r', linestyle='-', linewidth=1)
        axes[1, 1].set_xlabel(coord, fontsize=11)
        axes[1, 1].set_ylabel('Residual (Data - Fit)', fontsize=11)
        axes[1, 1].set_title('Residual Analysis', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 在残差图上添加统计信息
        res_stats = f'Residual Stats:\nMean: {np.mean(residuals):.2e}\nStd: {np.std(residuals):.2e}'
        axes[1, 1].text(0.02, 0.98, res_stats, transform=axes[1, 1].transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        # 保存图像
        plot_filename = os.path.join(output_dir, f'{base_name}_{coord}_analysis.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_filename}")
        plt.close()
    
    # 4. 保存拟合函数和结果
    save_fit_results(fit_results, output_dir, base_name)
    
    # 5. 生成总结报告
    generate_summary_report(fit_results, output_dir, base_name)
    
    return fit_results

def try_parametric_fits(data):
    """尝试多种参数分布拟合"""
    parametric_results = {}
    
    # 尝试正态分布
    try:
        mu, sigma = stats.norm.fit(data)
        parametric_results['normal'] = {
            'params': {'mu': float(mu), 'sigma': float(sigma)},
            'loglikelihood': float(np.sum(stats.norm.logpdf(data, mu, sigma)))
        }
    except:
        pass
    
    # 尝试指数分布
    try:
        loc, scale = stats.expon.fit(data)
        parametric_results['exponential'] = {
            'params': {'loc': float(loc), 'scale': float(scale)},
            'loglikelihood': float(np.sum(stats.expon.logpdf(data, loc, scale)))
        }
    except:
        pass
    
    # 尝试均匀分布
    try:
        loc, scale = stats.uniform.fit(data)
        parametric_results['uniform'] = {
            'params': {'loc': float(loc), 'scale': float(scale)},
            'loglikelihood': float(np.sum(stats.uniform.logpdf(data, loc, scale)))
        }
    except:
        pass
    
    return parametric_results

def save_fit_results(fit_results, output_dir, base_name):
    """保存拟合结果到多种格式"""
    
    # 保存为JSON（可读性好）
    json_file = os.path.join(output_dir, f'{base_name}_fit_results.json')
    with open(json_file, 'w') as f:
        json.dump(fit_results, f, indent=2)
    print(f"Saved fit results (JSON): {json_file}")
    
    # 保存为Pickle（保留完整Python对象）
    pkl_file = os.path.join(output_dir, f'{base_name}_fit_results.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(fit_results, f)
    print(f"Saved fit results (Pickle): {pkl_file}")
    
    # 创建可调用的拟合函数文件
    create_callable_functions(fit_results, output_dir, base_name)

def create_callable_functions(fit_results, output_dir, base_name):
    """创建可调用的拟合函数Python文件"""
    func_file = os.path.join(output_dir, f'{base_name}_fit_functions.py')
    
    with open(func_file, 'w') as f:
        f.write(f'''"""
Fitted functions for {fit_results['filename']}
Generated automatically - DO NOT EDIT MANUALLY
"""

import numpy as np
from scipy.stats import gaussian_kde

class FittedDistributions:
    """Container for all fitted distribution functions"""
    
    def __init__(self):
        self.filename = "{fit_results['filename']}"
        
    def get_statistics(self, coord):
        """Get statistics for a coordinate"""
        stats = {k: {sk: sv for sk, sv in v.items() if k == coord} 
                for k, v in {fit_results['statistics']}.items()}
        return stats.get(coord, {{}})
    
    # 注意：实际KDE函数需要在运行时重新创建
    # 这里提供使用说明
    def create_kde_function(self, data, coord):
        """Create a KDE function for given data and coordinate"""
        from scipy.stats import gaussian_kde
        return gaussian_kde(data, bw_method='scott')

# Example usage:
# from {base_name}_fit_functions import FittedDistributions
# fits = FittedDistributions()
# stats_x = fits.get_statistics('decay_pos_x')
# print(f"Mean of X: {{stats_x.get('mean', 'N/A')}}")
''')
    print(f"Created callable functions: {func_file}")

def generate_summary_report(fit_results, output_dir, base_name):
    """生成文本总结报告"""
    report_file = os.path.join(output_dir, f'{base_name}_summary_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"DISTRIBUTION ANALYSIS SUMMARY REPORT\n")
        f.write(f"File: {fit_results['filename']}\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n")
        f.write("=" * 70 + "\n\n")
        
        for coord, stats in fit_results['statistics'].items():
            f.write(f"\n{'='*60}\n")
            f.write(f"ANALYSIS FOR {coord.upper()}\n")
            f.write(f"{'='*60}\n")
            
            f.write(f"\nBasic Statistics:\n")
            f.write(f"  Data points: {stats['n_points']:,}\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Median: {stats['median']:.4f}\n")
            f.write(f"  Std Dev: {stats['std']:.4f}\n")
            f.write(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n")
            f.write(f"  Skewness: {stats['skewness']:.4f}\n")
            f.write(f"  Kurtosis: {stats['kurtosis']:.4f}\n")
            
            if coord in fit_results['fits']:
                fit_info = fit_results['fits'][coord]
                f.write(f"\nFitting Information:\n")
                f.write(f"  Fit type: {fit_info['type']}\n")
                f.write(f"  KDE bandwidth: {fit_info.get('bandwidth', 'N/A'):.4f}\n")
                
                if 'parametric' in fit_info:
                    f.write(f"\nParametric Fits (Log-Likelihood):\n")
                    for dist_name, dist_info in fit_info['parametric'].items():
                        f.write(f"  {dist_name}: {dist_info['loglikelihood']:.2f}\n")
        
        f.write(f"\n{'='*70}\n")
        f.write("FILES GENERATED:\n")
        f.write(f"  1. PNG plots for each coordinate\n")
        f.write(f"  2. JSON file with all fit results: {base_name}_fit_results.json\n")
        f.write(f"  3. Pickle file with Python objects: {base_name}_fit_results.pkl\n")
        f.write(f"  4. Callable functions: {base_name}_fit_functions.py\n")
        f.write(f"  5. This summary report: {base_name}_summary_report.txt\n")
        f.write(f"{'='*70}\n")
    
    print(f"Generated summary report: {report_file}")

def load_and_use_fitted_functions(results_file):
    """加载并使用保存的拟合函数"""
    import pickle
    import json
    
    if results_file.endswith('.pkl'):
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    elif results_file.endswith('.json'):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .pkl or .json")
    
    print(f"Loaded results from: {results_file}")
    print(f"Filename: {results['filename']}")
    
    # 示例：使用统计信息
    for coord, stats in results['statistics'].items():
        print(f"\n{coord}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    return results

# 批量处理函数
def batch_process_files(file_list, output_base_dir='distribution_analysis'):
    """批量处理多个文件"""
    all_results = {}
    
    for i, file_path in enumerate(file_list):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(file_list)}: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        # 为每个文件创建单独的输出目录
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(output_base_dir, file_base)
        
        try:
            results = analyze_distributions_with_comparison(file_path, output_dir)
            all_results[file_path] = results
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return all_results

# 使用示例
if __name__ == "__main__":
    # 示例1: 处理单个文件
    print("Starting distribution analysis...")
    
    # 替换为您的实际文件路径
    input_file = "/media/ubuntu/SRPPS/CODEX-b/CODEX_b-1/mass_1.00e-01_ctau_1.03e+03_br_1.00e+00_seed_1.csv"  # 或您的数据文件
    
    if os.path.exists(input_file):
        results = analyze_distributions_with_comparison(
            input_file, 
            output_dir='/media/ubuntu/SRPPS/CODEX-b/CODEX_b-1_distribution_plots'
        )
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
    else:
        print(f"File not found: {input_file}")
    #     print("Creating sample data for demonstration...")
        
    #     # 创建示例数据
    #     np.random.seed(42)
    #     sample_data = {
    #         'decay_pos_x': np.random.normal(0, 100, 1000),
    #         'decay_pos_y': np.random.exponential(50, 1000),
    #         'decay_pos_z': np.random.uniform(-200, 200, 1000)
    #     }
    #     df_sample = pd.DataFrame(sample_data)
    #     df_sample.to_csv('sample_decay_data.csv', index=False)
        
    #     # 分析示例数据
    #     results = analyze_distributions_with_comparison(
    #         'sample_decay_data.csv',
    #         output_dir='sample_analysis'
    #     )