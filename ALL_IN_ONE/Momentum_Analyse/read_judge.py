import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class LLPProbabilityCalculator:
    """
    计算LLP在指定立方体区域内衰变的概率
    读取之前生成的分布模型文件
    """
    
    def __init__(self, models_dir: str):
        """
        初始化概率计算器
        
        参数:
        - models_dir: 包含KDE模型JSON文件的目录
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.probabilities = []
        
    def load_all_models(self):
        """加载所有分布模型"""
        print("=" * 70)
        print("LOADING LLP DISTRIBUTION MODELS")
        print("=" * 70)
        
        # 查找所有JSON模型文件
        json_files = list(self.models_dir.glob("*.json"))
        print(f"Found {len(json_files)} model files")
        
        if not json_files:
            # 尝试在子目录中查找
            models_subdir = self.models_dir / "distribution_models"
            if models_subdir.exists():
                json_files = list(models_subdir.glob("*.json"))
                print(f"Found {len(json_files)} model files in subdirectory")
        
        for json_file in tqdm(json_files, desc="Loading models"):
            try:
                model = self._load_model_file(json_file)
                if model:
                    model_id = model.get('formatted_name', json_file.stem)
                    self.models[model_id] = model
            except Exception as e:
                print(f"\nWarning: Failed to load {json_file.name}: {e}")
        
        if not self.models:
            raise ValueError("No models loaded!")
        
        print(f"\nSuccessfully loaded {len(self.models)} distribution models")
        
        # 显示一些示例模型
        print(f"\nFirst 3 models:")
        for model_id, model in list(self.models.items())[:3]:
            params = model['params']
            print(f"  {model_id}: m={params.get('mass', 'N/A'):.3f}GeV, "
                  f"τ={params.get('lifetime', 'N/A'):.2e}mm")
    
    def _load_model_file(self, json_file: Path) -> Optional[Dict]:
        """加载单个模型文件"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 检查数据完整性
        if 'model_stats' not in data or 'params' not in data:
            print(f"Warning: Incomplete model file {json_file.name}")
            return None
        
        # 从文件名获取格式化名称（如果数据中没有）
        if 'formatted_name' not in data:
            data['formatted_name'] = json_file.stem
        
        return data
    
    def calculate_probabilities_in_region(self, 
                                         x_range: Tuple[float, float],
                                         y_range: Tuple[float, float], 
                                         z_range: Tuple[float, float],
                                         method: str = 'monte_carlo',
                                         n_samples: int = 100000):
        """
        计算所有模型在指定立方体区域内的衰变概率
        
        参数:
        - x_range: X轴范围 (min, max)
        - y_range: Y轴范围 (min, max)
        - z_range: Z轴范围 (min, max)
        - method: 计算方法 ('monte_carlo' 或 'integral')
        - n_samples: 蒙特卡洛方法采样数
        """
        print(f"\n{'='*70}")
        print("CALCULATING DECAY PROBABILITIES")
        print('='*70)
        print(f"Region: X={x_range}, Y={y_range}, Z={z_range}")
        print(f"Method: {method}")
        print(f"Number of models: {len(self.models)}")
        print('='*70)
        
        self.probabilities = []
        
        for model_id, model in tqdm(self.models.items(), desc="Calculating probabilities"):
            try:
                probability = self._calculate_single_probability(
                    model, x_range, y_range, z_range, method, n_samples
                )
                
                params = model['params']
                
                result = {
                    'model_id': model_id,
                    'formatted_name': model.get('formatted_name', model_id),
                    'mass': float(params.get('mass', np.nan)),
                    'lifetime': float(params.get('lifetime', np.nan)),
                    'tanb': float(params.get('tanb', np.nan)),
                    'vis_br': float(params.get('vis_br', np.nan)),
                    'probability': probability,
                    'x_min': x_range[0],
                    'x_max': x_range[1],
                    'y_min': y_range[0],
                    'y_max': y_range[1],
                    'z_min': z_range[0],
                    'z_max': z_range[1],
                    'calculation_method': method,
                    'possibility': probability,  # 添加这一行，保持兼容性
                }
                
                self.probabilities.append(result)
                
            except Exception as e:
                print(f"\nWarning: Failed to calculate probability for {model_id}: {e}")
        
        print(f"\n✓ Successfully calculated probabilities for {len(self.probabilities)} models")
    
    def _calculate_single_probability(self, 
                                     model: Dict,
                                     x_range: Tuple[float, float],
                                     y_range: Tuple[float, float],
                                     z_range: Tuple[float, float],
                                     method: str,
                                     n_samples: int) -> float:
        """
        计算单个模型在指定区域内的概率
        
        注意：由于原始KDE对象没有保存，我们只能使用保存的统计信息进行近似计算
        这里假设分布是独立的高斯分布
        """
        model_stats = model['model_stats']
        
        if method == 'integral':
            # 使用积分方法（假设各坐标独立）
            prob_x = self._gaussian_probability_in_range(
                model_stats['x']['mean'], 
                model_stats['x']['std'], 
                x_range
            )
            
            prob_y = self._gaussian_probability_in_range(
                model_stats['y']['mean'], 
                model_stats['y']['std'], 
                y_range
            )
            
            prob_z = self._gaussian_probability_in_range(
                model_stats['z']['mean'], 
                model_stats['z']['std'], 
                z_range
            )
            
            # 假设独立，联合概率 = P(x) * P(y) * P(z)
            probability = prob_x * prob_y * prob_z
            
        elif method == 'monte_carlo':
            # 使用蒙特卡洛方法
            probability = self._monte_carlo_probability(
                model_stats, x_range, y_range, z_range, n_samples
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return probability
    
    def _gaussian_probability_in_range(self, 
                                      mean: float, 
                                      std: float, 
                                      value_range: Tuple[float, float]) -> float:
        """计算高斯分布在指定范围内的概率"""
        if std <= 0:
            # 如果标准差为0或负，检查均值是否在范围内
            return 1.0 if value_range[0] <= mean <= value_range[1] else 0.0
        
        # 使用scipy的norm.cdf计算累积概率
        prob_lower = stats.norm.cdf(value_range[0], loc=mean, scale=std)
        prob_upper = stats.norm.cdf(value_range[1], loc=mean, scale=std)
        
        return max(0.0, prob_upper - prob_lower)
    
    def _monte_carlo_probability(self,
                                model_stats: Dict,
                                x_range: Tuple[float, float],
                                y_range: Tuple[float, float],
                                z_range: Tuple[float, float],
                                n_samples: int) -> float:
        """使用蒙特卡洛方法估计概率"""
        # 从高斯分布中采样
        x_samples = np.random.normal(
            model_stats['x']['mean'],
            model_stats['x']['std'],
            n_samples
        )
        
        y_samples = np.random.normal(
            model_stats['y']['mean'],
            model_stats['y']['std'],
            n_samples
        )
        
        z_samples = np.random.normal(
            model_stats['z']['mean'],
            model_stats['z']['std'],
            n_samples
        )
        
        # 检查哪些点在区域内
        in_region = (
            (x_samples >= x_range[0]) & (x_samples <= x_range[1]) &
            (y_samples >= y_range[0]) & (y_samples <= y_range[1]) &
            (z_samples >= z_range[0]) & (z_samples <= z_range[1])
        )
        
        # 计算概率
        probability = np.mean(in_region)
        
        return float(probability)
    
    def save_probabilities(self, 
                          output_file: str = 'llp_probabilities.csv',
                          simple_format: bool = True):
        """
        保存概率计算结果
        
        参数:
        - output_file: 输出CSV文件名
        - simple_format: 是否使用简化格式 (m, ltime, possibility)
        """
        if not self.probabilities:
            print("No probabilities calculated yet!")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if simple_format:
            # 简化格式: m, ltime, possibility
            simple_data = []
            for prob in self.probabilities:
                simple_data.append({
                    'm': prob['mass'],
                    'ltime': prob['lifetime'],
                    'possibility': prob['probability']
                })
            
            df = pd.DataFrame(simple_data)
            # 按质量排序
            df = df.sort_values(['m', 'ltime'])
            
            print(f"\nSaving simplified CSV: m, ltime, possibility")
            print(f"Total entries: {len(df)}")
            
        else:
            # 完整格式
            df = pd.DataFrame(self.probabilities)
            df = df.sort_values(['mass', 'lifetime'])
            
            print(f"\nSaving detailed CSV with {len(df.columns)} columns")
        
        # 保存到CSV
        df.to_csv(output_path, index=False)
        
        print(f"✓ Probabilities saved to: {output_path}")
        
        # 显示统计信息
        print(f"\nProbability Statistics:")
        print("-" * 40)
        print(f"Mean probability: {df['possibility'].mean():.6f}")
        print(f"Min probability:  {df['possibility'].min():.6f}")
        print(f"Max probability:  {df['possibility'].max():.6f}")
        print(f"Median probability: {df['possibility'].median():.6f}")
        
        return df
    
    def create_probability_plots(self, 
                                output_dir: str = './probability_plots'):
        """创建概率可视化图"""
        if not self.probabilities:
            print("No probabilities calculated yet!")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating probability plots in {output_path}...")
        
        # 转换为DataFrame以便绘图
        df = pd.DataFrame(self.probabilities)
        
        # 1. 概率随参数变化的热图
        self._create_probability_heatmap(df, output_path)
        
        # 2. 概率分布直方图
        self._create_probability_histogram(df, output_path)
        
        # 3. 概率与参数的散点图
        self._create_probability_scatter_plots(df, output_path)
        
        print(f"\n✓ All probability plots saved to {output_path}")
    
    def _create_probability_heatmap(self, df: pd.DataFrame, output_path: Path):
        """创建概率热图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 创建透视表
        try:
            # 尝试创建质量-寿命网格
            pivot_table = df.pivot_table(
                values='probability',
                index=pd.cut(df['mass'], bins=10),
                columns=pd.cut(np.log10(df['lifetime']), bins=10),
                aggfunc='mean'
            )
            
            # 热图1: 原始概率
            im1 = axes[0].imshow(pivot_table.values, cmap='viridis', aspect='auto')
            axes[0].set_xlabel('log10(Lifetime) bins', fontsize=11)
            axes[0].set_ylabel('Mass bins', fontsize=11)
            axes[0].set_title('Probability Heatmap', fontsize=12)
            plt.colorbar(im1, ax=axes[0], label='Probability')
            
            # 热图2: 对数概率
            pivot_table_log = np.log10(pivot_table.values + 1e-10)
            im2 = axes[1].imshow(pivot_table_log, cmap='plasma', aspect='auto')
            axes[1].set_xlabel('log10(Lifetime) bins', fontsize=11)
            axes[1].set_ylabel('Mass bins', fontsize=11)
            axes[1].set_title('log10(Probability) Heatmap', fontsize=12)
            plt.colorbar(im2, ax=axes[1], label='log10(Probability)')
            
        except Exception as e:
            print(f"Warning: Could not create heatmap: {e}")
            axes[0].text(0.5, 0.5, "Insufficient data\nfor heatmap",
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, "Insufficient data\nfor heatmap",
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path / 'probability_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_probability_histogram(self, df: pd.DataFrame, output_path: Path):
        """创建概率分布直方图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 整体概率分布
        axes[0, 0].hist(df['probability'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Probability', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Probability Distribution', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 对数概率分布
        log_prob = np.log10(df['probability'] + 1e-10)
        axes[0, 1].hist(log_prob, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('log10(Probability)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Log Probability Distribution', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 概率与质量的关系
        axes[1, 0].scatter(df['mass'], df['probability'], alpha=0.6, s=20)
        axes[1, 0].set_xlabel('Mass (GeV)', fontsize=11)
        axes[1, 0].set_ylabel('Probability', fontsize=11)
        axes[1, 0].set_title('Probability vs Mass', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 概率与寿命的关系
        axes[1, 1].scatter(np.log10(df['lifetime']), df['probability'], alpha=0.6, s=20)
        axes[1, 1].set_xlabel('log10(Lifetime) (mm)', fontsize=11)
        axes[1, 1].set_ylabel('Probability', fontsize=11)
        axes[1, 1].set_title('Probability vs Lifetime', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Probability Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path / 'probability_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_probability_scatter_plots(self, df: pd.DataFrame, output_path: Path):
        """创建概率散点图"""
        fig = plt.figure(figsize=(14, 10))
        
        # 创建3D散点图
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(df['mass'], 
                            np.log10(df['lifetime']), 
                            df['probability'],
                            c=df['probability'], 
                            cmap='viridis',
                            s=50,
                            alpha=0.7)
        
        ax.set_xlabel('Mass (GeV)', fontsize=11, labelpad=10)
        ax.set_ylabel('log10(Lifetime) (mm)', fontsize=11, labelpad=10)
        ax.set_zlabel('Probability', fontsize=11, labelpad=10)
        ax.set_title('3D Probability Scatter Plot', fontsize=12, pad=20)
        
        plt.colorbar(scatter, ax=ax, label='Probability', pad=0.1)
        
        plt.tight_layout()
        plt.savefig(output_path / 'probability_3d_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, 
                       x_range: Tuple[float, float],
                       y_range: Tuple[float, float],
                       z_range: Tuple[float, float],
                       output_dir: str = './probability_results'):
        """生成概率分析报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.probabilities:
            print("No probabilities calculated yet!")
            return
        
        df = pd.DataFrame(self.probabilities)
        
        report = []
        report.append("=" * 70)
        report.append("LLP DECAY PROBABILITY ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"\nAnalysis Date: {pd.Timestamp.now()}")
        report.append(f"Number of models analyzed: {len(self.probabilities)}")
        
        report.append(f"\nREGION DEFINITION:")
        report.append("-" * 40)
        report.append(f"X range: [{x_range[0]}, {x_range[1]}] mm")
        report.append(f"Y range: [{y_range[0]}, {y_range[1]}] mm")
        report.append(f"Z range: [{z_range[0]}, {z_range[1]}] mm")
        report.append(f"Volume: {(x_range[1]-x_range[0]) * (y_range[1]-y_range[0]) * (z_range[1]-z_range[0]):.2f} mm³")
        
        report.append(f"\nPROBABILITY STATISTICS:")
        report.append("-" * 40)
        report.append(f"Mean probability: {df['probability'].mean():.6f}")
        report.append(f"Minimum probability: {df['probability'].min():.6f}")
        report.append(f"Maximum probability: {df['probability'].max():.6f}")
        report.append(f"Median probability: {df['probability'].median():.6f}")
        report.append(f"Standard deviation: {df['probability'].std():.6f}")
        
        # 高概率模型
        high_prob_threshold = df['probability'].quantile(0.9)
        high_prob_models = df[df['probability'] >= high_prob_threshold]
        
        report.append(f"\nHIGH PROBABILITY MODELS (top 10%):")
        report.append("-" * 40)
        if len(high_prob_models) > 0:
            for i, row in high_prob_models.nlargest(10, 'probability').iterrows():
                report.append(f"{row['formatted_name']}: "
                            f"m={row['mass']:.3f}GeV, "
                            f"τ={row['lifetime']:.2e}mm, "
                            f"P={row['probability']:.6f}")
        else:
            report.append("No models with high probability found.")
        
        # 低概率模型
        low_prob_threshold = df['probability'].quantile(0.1)
        low_prob_models = df[df['probability'] <= low_prob_threshold]
        
        report.append(f"\nLOW PROBABILITY MODELS (bottom 10%):")
        report.append("-" * 40)
        if len(low_prob_models) > 0:
            for i, row in low_prob_models.nsmallest(10, 'probability').iterrows():
                report.append(f"{row['formatted_name']}: "
                            f"m={row['mass']:.3f}GeV, "
                            f"τ={row['lifetime']:.2e}mm, "
                            f"P={row['probability']:.6f}")
        else:
            report.append("No models with low probability found.")
        
        # 参数相关性
        report.append(f"\nPARAMETER CORRELATIONS WITH PROBABILITY:")
        report.append("-" * 40)
        
        try:
            # 计算相关系数
            corr_with_mass = df['mass'].corr(df['probability'])
            corr_with_log_lifetime = np.log10(df['lifetime']).corr(df['probability'])
            corr_with_tanb = df['tanb'].corr(df['probability']) if 'tanb' in df.columns else np.nan
            
            report.append(f"Correlation with mass: {corr_with_mass:.3f}")
            report.append(f"Correlation with log10(lifetime): {corr_with_log_lifetime:.3f}")
            if not np.isnan(corr_with_tanb):
                report.append(f"Correlation with tanβ: {corr_with_tanb:.3f}")
        except:
            report.append("Could not calculate correlations.")
        
        report.append(f"\nOUTPUT FILES:")
        report.append("-" * 40)
        report.append("1. llp_probabilities.csv - Main results (m, ltime, possibility)")
        report.append("2. llp_probabilities_detailed.csv - Detailed results with all columns")
        report.append("3. Probability plots in ./probability_plots/")
        report.append("4. This report file")
        
        report.append(f"\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        report_text = '\n'.join(report)
        
        report_file = output_path / 'probability_analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"✓ Report saved to: {report_file}")


def main(Detector = 'CODEX-b'):
    """主函数 - 计算LLP在指定区域内的衰变概率"""
    
    # 设置路径
    models_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/14TeV_LLP_Distribution/distribution_models"
    output_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/14TeV_LLP_Distribution/"
    
    # 定义感兴趣的区域（单位：mm）
    # 可以根据你的探测器几何或感兴趣的区域来设置
    x_range = (-100000, 100000)    # X方向范围
    y_range = (100000, 125000)    # Y方向范围
    z_range = (100000, 300000)       # Z方向范围
    
    print("=" * 70)
    print("LLP DECAY PROBABILITY CALCULATOR")
    print("=" * 70)
    print(f"Target Region: X={x_range}, Y={y_range}, Z={z_range}")
    print("=" * 70)
    
    # 创建概率计算器
    calculator = LLPProbabilityCalculator(models_dir)
    
    try:
        # 1. 加载模型
        print("\n[1/3] Loading distribution models...")
        calculator.load_all_models()
        
        # 2. 计算概率
        print("\n[2/3] Calculating decay probabilities...")
        calculator.calculate_probabilities_in_region(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            method='monte_carlo',  # 使用蒙特卡洛方法
            n_samples=1000000       # 采样数
        )
        
        # 3. 保存结果（简化格式）
        print("\n[3/3] Saving results...")
        
        # 保存简化格式CSV
        simple_csv = Path(output_dir) / f"{Detector}.csv"
        df_simple = calculator.save_probabilities(
            output_file=str(simple_csv),
            simple_format=True
        )
        
        # 保存详细格式CSV
        detailed_csv = Path(output_dir) / f"{Detector}_detailed.csv"
        calculator.save_probabilities(
            output_file=str(detailed_csv),
            simple_format=False
        )
        
        # 4. 创建可视化
        print("\n[4/3] Creating visualizations...")
        calculator.create_probability_plots(output_dir)
        
        # 5. 生成报告
        calculator.generate_report(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            output_dir=output_dir
        )
        
        print("\n" + "=" * 70)
        print("PROBABILITY CALCULATION COMPLETED!")
        print("=" * 70)
        
        print(f"\n✅ Results saved to: {output_dir}")
        print(f"\n📊 Main output files:")
        print(f"  {simple_csv} - Simplified CSV (m, ltime, possibility)")
        print(f"  {detailed_csv} - Detailed CSV with all information")
        print(f"  {output_dir}/probability_plots/ - Visualization plots")
        print(f"  {output_dir}/probability_analysis_report.txt - Analysis report")
        
        # 显示一些示例结果
        if df_simple is not None and not df_simple.empty:
            print(f"\n📈 Example results (first 5):")
            print("-" * 40)
            for i, row in df_simple.head().iterrows():
                print(f"m={row['m']:.3f}GeV, "
                      f"τ={row['ltime']:.2e}mm, "
                      f"P={row['possibility']:.6f}")
        
    except Exception as e:
        print(f"\n❌ Error during probability calculation: {e}")
        import traceback
        traceback.print_exc()


def batch_calculate_probabilities():
    """
    批量计算多个不同区域的概率
    可以用于系统研究不同几何区域
    """
    models_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_F/distributution_density"
    base_output_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_F"
    
    # 定义多个感兴趣的区域
    regions = {
        'region_small': {
            'x_range': (-50, 50),
            'y_range': (-50, 50),
            'z_range': (0, 100),
            'description': 'Small central region'
        },
        'region_medium': {
            'x_range': (-100, 100),
            'y_range': (-100, 100),
            'z_range': (0, 200),
            'description': 'Medium central region'
        },
        'region_large': {
            'x_range': (-200, 200),
            'y_range': (-200, 200),
            'z_range': (0, 400),
            'description': 'Large central region'
        },
        'region_forward': {
            'x_range': (-50, 50),
            'y_range': (-50, 50),
            'z_range': (200, 400),
            'description': 'Forward region'
        }
    }
    
    print("=" * 70)
    print("BATCH PROBABILITY CALCULATION")
    print("=" * 70)
    
    # 加载模型一次，重复使用
    calculator = LLPProbabilityCalculator(models_dir)
    calculator.load_all_models()
    
    all_results = []
    
    for region_name, region_config in regions.items():
        print(f"\n📐 Calculating for region: {region_config['description']}")
        
        output_dir = Path(base_output_dir) / "probability_results" / region_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算概率
        calculator.calculate_probabilities_in_region(
            x_range=region_config['x_range'],
            y_range=region_config['y_range'],
            z_range=region_config['z_range'],
            method='monte_carlo',
            n_samples=100000
        )
        
        # 保存结果
        csv_file = output_dir / "llp_probabilities.csv"
        df = calculator.save_probabilities(
            output_file=str(csv_file),
            simple_format=True
        )
        
        # 添加区域信息
        df['region'] = region_name
        df['region_description'] = region_config['description']
        
        all_results.append(df)
        
        print(f"✓ Results saved to: {csv_file}")
    
    # 合并所有结果
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = Path(base_output_dir) / "probability_results" / "all_regions_combined.csv"
        combined_df.to_csv(combined_file, index=False)
        
        print(f"\n✅ All results combined and saved to: {combined_file}")
        print(f"Total entries: {len(combined_df)}")
    
    print("\n" + "=" * 70)
    print("BATCH CALCULATION COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    # 运行单个区域计算
    main(Detector='MATHUSLA_test')
    
    # 如果需要批量计算多个区域，取消下面行的注释
    # batch_calculate_probabilities()