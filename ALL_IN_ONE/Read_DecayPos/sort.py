import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import mplhep as hep
hep.style.use("ALICE")

# 方法1：使用凸包算法（适用于寻找数据的外边界）
def plot_envelope_convexhull(filename):
    """
    使用凸包算法绘制数据的外包络线
    """
    # 读取CSV文件
    df = pd.read_csv(filename)
    
    # 提取m和theta数据
    m = df['m'].values
    theta = df['theta'].values
    
    # 将数据组合成二维点集
    points = np.column_stack((m, theta))
    
    # 计算凸包
    hull = ConvexHull(points)
    
    # 获取凸包顶点并按m值排序
    hull_points = points[hull.vertices]
    hull_points = hull_points[hull_points[:, 0].argsort()]  # 按m排序
    
    # 分离上下包络线
    # 找到中间点，将凸包分为上下两部分
    mid_m = (m.min() + m.max()) / 2
    mid_idx = np.argmin(np.abs(hull_points[:, 0] - mid_m))
    
    # 顺时针遍历凸包点，分为上下两部分
    upper_points = []
    lower_points = []
    
    # 从中间点开始，分别向两个方向遍历
    n = len(hull_points)
    for i in range(n):
        idx = (mid_idx + i) % n
        point = hull_points[idx]
        # 根据位置判断是上包络还是下包络
        # 这里使用简单判断：theta值较大的点归为上包络
        # 实际应用中可能需要根据具体情况调整
        if i < n/2:
            upper_points.append(point)
        else:
            lower_points.append(point)
    
    upper_points = np.array(upper_points)
    lower_points = np.array(lower_points)
    
    # 分别按m值排序
    upper_points = upper_points[upper_points[:, 0].argsort()]
    lower_points = lower_points[lower_points[:, 0].argsort()]
    
    # 绘制图形
    plt.figure(figsize=(12, 8))
    
    # 绘制原始数据点
    plt.scatter(m, theta, alpha=0.5, s=20, label='原始数据', color='lightblue')
    
    # 绘制凸包
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'r-', alpha=0.3, linewidth=1)
    
    # 绘制上下包络线
    plt.plot(upper_points[:, 0], upper_points[:, 1], 'r-', linewidth=2, label='上包络线')
    plt.plot(lower_points[:, 0], lower_points[:, 1], 'b-', linewidth=2, label='下包络线')
    
    plt.xlabel('m', fontsize=12)
    plt.ylabel('theta', fontsize=12)
    plt.title('数据包络线（凸包算法）', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 方法2：使用滑动窗口方法（适用于寻找局部极值包络）
def plot_envelope_sliding_window(filename, filename2, filename3,window_size=0.05, detector_name='CODEX-b'):
    """
    使用滑动窗口方法绘制包络线
    """
    # 读取数据
    df = pd.read_csv(filename)
    df2 = pd.read_csv(filename2)
    df3 = pd.read_csv(filename3)
    # 按m值排序
    df_sorted = df.sort_values('m').reset_index(drop=True)
    df2_sorted = df2.sort_values('m').reset_index(drop=True)
    df3_sorted = df3.sort_values('m').reset_index(drop=True)
    m = df_sorted['m'].values
    m2 = df2_sorted['m'].values
    m3 = df3_sorted['m'].values
    theta = df_sorted['theta'].values
    theta2 = df2_sorted['theta'].values
    theta3 = df3_sorted['theta'].values
    # 确定滑动窗口的步长
    m_range = m.max() - m.min()
    m2_range = m2.max() - m2.min()
    m3_range = m3.max() - m3.min()
    window_width = m_range * window_size
    window_width2 = m2_range * window_size
    window_width3 = m3_range * window_size
    # 生成一系列m值点
    m_grid = np.linspace(m.min(), m.max(), 200)
    m2_grid = np.linspace(m2.min(), m2.max(), 200)
    m3_grid = np.linspace(m3.min(), m3.max(), 200)
    # 计算每个网格点的上下包络值
    upper_envelope = []
    upper_envelope2 = []
    upper_envelope3 = []
    lower_envelope = []
    lower_envelope2 = []
    lower_envelope3 = []

    for m_val in m_grid:
        # 找到窗口内的点
        mask = (m >= m_val - window_width/2) & (m <= m_val + window_width/2)
        if np.any(mask):
            window_theta = theta[mask]
            upper_envelope.append(np.max(window_theta))
            lower_envelope.append(np.min(window_theta))
        else:
            upper_envelope.append(np.nan)
            lower_envelope.append(np.nan)

    for m_val2 in m2_grid:
        # 找到窗口内的点
        mask2 = (m2 >= m_val2 - window_width2/2) & (m2 <= m_val2 + window_width2/2)
        if np.any(mask2):
            window_theta2 = theta2[mask2]
            upper_envelope2.append(np.max(window_theta2))
            lower_envelope2.append(np.min(window_theta2))
        else:
            upper_envelope2.append(np.nan)
            lower_envelope2.append(np.nan)

    for m_val3 in m3_grid:
        # 找到窗口内的点
        mask3 = (m3 >= m_val3 - window_width3/2) & (m3 <= m_val3 + window_width3/2)
        if np.any(mask3):
            window_theta3 = theta3[mask3]
            upper_envelope3.append(np.max(window_theta3))
            lower_envelope3.append(np.min(window_theta3))
        else:
            upper_envelope3.append(np.nan)
            lower_envelope3.append(np.nan)
    
    # 插值去除NaN值
    valid_mask = ~np.isnan(upper_envelope)
    valid_mask2 = ~np.isnan(upper_envelope2)
    valid_mask3 = ~np.isnan(upper_envelope3)
    if np.any(valid_mask):
        # 创建插值函数
        interp_upper = interp1d(
            m_grid[valid_mask], 
            np.array(upper_envelope)[valid_mask],
            kind='quadratic',
            fill_value='extrapolate'
        )
        interp_lower = interp1d(
            m_grid[valid_mask], 
            np.array(lower_envelope)[valid_mask],
            kind='quadratic',
            fill_value='extrapolate'
        )
        
        # 插值得到完整的包络线
        upper_smooth = interp_upper(m_grid)
        lower_smooth = interp_lower(m_grid)
    else:
        upper_smooth = np.array(upper_envelope)
        lower_smooth = np.array(lower_envelope)
    if np.any(valid_mask2):
        # 创建插值函数
        interp_upper2 = interp1d(
            m2_grid[valid_mask2], 
            np.array(upper_envelope2)[valid_mask2],
            kind='quadratic',
            fill_value='extrapolate'
        )
        interp_lower2 = interp1d(
            m2_grid[valid_mask2], 
            np.array(lower_envelope2)[valid_mask2],
            kind='quadratic',
            fill_value='extrapolate'
        )
        
        # 插值得到完整的包络线
        upper_smooth2 = interp_upper2(m2_grid)
        lower_smooth2 = interp_lower2(m2_grid)
    else:
        upper_smooth2 = np.array(upper_envelope2)
        lower_smooth2 = np.array(lower_envelope2)
    if np.any(valid_mask3):
        # 创建插值函数
        interp_upper3 = interp1d(
            m3_grid[valid_mask3], 
            np.array(upper_envelope3)[valid_mask3],
            kind='quadratic',
            fill_value='extrapolate'
        )
        interp_lower3 = interp1d(
            m3_grid[valid_mask3], 
            np.array(lower_envelope3)[valid_mask3],
            kind='quadratic',
            fill_value='extrapolate'
        )
        
        # 插值得到完整的包络线
        upper_smooth3 = interp_upper3(m3_grid)
        lower_smooth3 = interp_lower3(m3_grid)
    else:
        upper_smooth3 = np.array(upper_envelope3)
        lower_smooth3 = np.array(lower_envelope3)

    # 绘制图形
    plt.figure(figsize=(12, 8))
    
    # 绘制原始数据点
    # plt.scatter(df['m'], np.square(np.sin(df['theta'])), alpha=0.5, s=20, label='original', color='lightblue')
    
    # 绘制包络线
    plt.plot(m_grid, np.square(np.sin(upper_smooth)), 'r-', linewidth=2, label='MATHUSLA')
    plt.plot(m_grid, np.square(np.sin(lower_smooth)), 'r-', linewidth=2, label='MATHUSLA')
    plt.plot(m2_grid, np.square(np.sin(upper_smooth2)), 'g-', linewidth=2, label='CODEX-b')
    plt.plot(m2_grid, np.square(np.sin(lower_smooth2)), 'g-', linewidth=2, label='CODEX-b')
    plt.plot(m3_grid, np.square(np.sin(upper_smooth3)), 'b-', linewidth=2, label='FASER2')
    plt.plot(m3_grid, np.square(np.sin(lower_smooth3)), 'b-', linewidth=2, label='FASER2')
    # 填充包络线之间的区域
    plt.fill_between(m_grid, np.square(np.sin(lower_smooth)), np.square(np.sin(upper_smooth)), alpha=0.2, color='gray')
    plt.fill_between(m2_grid, np.square(np.sin(lower_smooth2)), np.square(np.sin(upper_smooth2)), alpha=0.2, color='gray')
    plt.fill_between(m3_grid, np.square(np.sin(lower_smooth3)), np.square(np.sin(upper_smooth3)), alpha=0.2, color='gray')
    plt.xlabel('m', fontsize=12)
    plt.ylabel('theta', fontsize=12)
    plt.title(f'95%', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'/media/ubuntu/SRPPS/Results/{detector_name}.png')
    
    return m_grid, upper_smooth, lower_smooth

# 方法3：使用分箱统计方法（简单有效）
def plot_envelope_binning(filename, num_bins=50):
    """
    使用分箱统计方法绘制包络线
    """
    # 读取数据
    df = pd.read_csv(filename)
    
    # 创建m值的分箱
    bins = np.linspace(df['m'].min(), df['m'].max(), num_bins + 1)
    
    # 计算每个分箱的统计量
    bin_centers = (bins[:-1] + bins[1:]) / 2
    max_theta = []
    min_theta = []
    mean_theta = []
    
    for i in range(num_bins):
        mask = (df['m'] >= bins[i]) & (df['m'] < bins[i+1])
        if i == num_bins - 1:  # 最后一个分箱包含最大值
            mask = (df['m'] >= bins[i]) & (df['m'] <= bins[i+1])
        
        if np.any(mask):
            bin_data = df.loc[mask, 'theta']
            max_theta.append(bin_data.max())
            min_theta.append(bin_data.min())
            mean_theta.append(bin_data.mean())
        else:
            max_theta.append(np.nan)
            min_theta.append(np.nan)
            mean_theta.append(np.nan)
    
    # 插值去除NaN值
    valid_mask = ~np.isnan(max_theta)
    if np.any(valid_mask):
        interp_max = interp1d(
            bin_centers[valid_mask], 
            np.array(max_theta)[valid_mask],
            kind='cubic',
            fill_value='extrapolate'
        )
        interp_min = interp1d(
            bin_centers[valid_mask], 
            np.array(min_theta)[valid_mask],
            kind='cubic',
            fill_value='extrapolate'
        )
        
        # 生成平滑曲线
        m_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 200)
        max_smooth = interp_max(m_smooth)
        min_smooth = interp_min(m_smooth)
    else:
        m_smooth = bin_centers
        max_smooth = max_theta
        min_smooth = min_theta
    
    # 绘制图形
    plt.figure(figsize=(12, 8))
    
    # 绘制原始数据点
    # plt.scatter(df['m'], df['theta'], alpha=0.3, s=15, label='原始数据', color='lightblue')
    
    # 绘制包络线和均值线
    plt.plot(m_smooth, np.square(np.sin(max_smooth)), 'r-', linewidth=2, label='上包络线（最大值）')
    plt.plot(m_smooth, np.square(np.sin(min_smooth)), 'b-', linewidth=2, label='下包络线（最小值）')
    
    # 填充包络区域
    plt.fill_between(m_smooth, np.square(np.sin(min_smooth)), np.square(np.sin(max_smooth)), alpha=0.1, color='gray')
    
    # 绘制分箱中心点（可选）
    # plt.scatter(bin_centers[valid_mask], np.array(max_theta)[valid_mask], 
    #            s=30, color='red', alpha=0.7, zorder=5)
    # plt.scatter(bin_centers[valid_mask], np.array(min_theta)[valid_mask], 
    #            s=30, color='blue', alpha=0.7, zorder=5)
    
    plt.xlabel('m', fontsize=12)
    plt.ylabel('theta', fontsize=12)
    plt.title(f'数据包络线（分箱方法，{num_bins}个分箱）', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return m_smooth, max_smooth, min_smooth

# 主函数：根据数据特性选择合适的方法
def main():
    # 替换为您的CSV文件路径
    filename = 'your_data.csv'  # 修改为您的文件路径
    
    try:
        # 先预览数据
        df = pd.read_csv(filename)
        print("数据预览:")
        print(df.head())
        print(f"\n数据形状: {df.shape}")
        print(f"m范围: [{df['m'].min():.4f}, {df['m'].max():.4f}]")
        print(f"theta范围: [{df['theta'].min():.4f}, {df['theta'].max():.4f}]")
        print(f"数据点数: {len(df)}")
        
        # 根据数据特性选择方法
        if len(df) < 100:
            print("\n数据点较少，推荐使用凸包算法...")
            plot_envelope_convexhull(filename)
        else:
            print("\n数据点较多，推荐使用分箱方法...")
            plot_envelope_binning(filename, num_bins=min(50, len(df)//10))
            
            print("\n也可以尝试滑动窗口方法...")
            plot_envelope_sliding_window(filename, window_size=0.05)
            
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filename}'")
        print("请确保文件路径正确，并修改main()函数中的filename变量")
    except KeyError as e:
        print(f"错误：CSV文件中缺少必要的列 {e}")
        print("请确保CSV文件包含'm'和'theta'列")
    except Exception as e:
        print(f"发生错误: {e}")

# 简单使用的函数
def quick_plot_envelope(filename, method='binning', **kwargs):
    """
    快速绘制包络线的函数
    
    参数:
    filename: CSV文件路径
    method: 方法类型，可选 'convexhull', 'sliding', 'binning'
    **kwargs: 方法特定参数
    """
    if method == 'convexhull':
        plot_envelope_convexhull(filename)
    elif method == 'sliding':
        window_size = kwargs.get('window_size', 0.05)
        plot_envelope_sliding_window(filename, window_size)
    elif method == 'binning':
        num_bins = kwargs.get('num_bins', 50)
        plot_envelope_binning(filename, num_bins)
    else:
        print(f"未知方法: {method}")

if __name__ == "__main__":
    # 使用示例
    # 1. 直接运行主函数（自动选择方法）
    # main()
    
    # 2. 或使用快速函数指定方法
    # quick_plot_envelope('your_data.csv', method='binning', num_bins=40)
    
    # 3. 或直接调用特定方法
    plot_envelope_sliding_window('/media/ubuntu/SRPPS/Results/Higgs_portal/CODEX_HP_exclusion.csv', filename2='/media/ubuntu/SRPPS/Results/Higgs_portal/MATHUSLA_HP_exclusion.csv', filename3='/media/ubuntu/SRPPS/Results/Higgs_portal/FASER2_HP_exclusion.csv', window_size=0.05, detector_name='higg_portal',)
    
    print("请先修改代码中的文件路径，然后取消注释相应的函数调用")