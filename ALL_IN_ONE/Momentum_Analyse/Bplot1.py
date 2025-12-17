import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# 读取数据
Bdf = pd.read_csv('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18/block_summary.csv')

# 计算每个块的动量大小和角度
p = np.sqrt(Bdf['momentum_mean_px']**2 + Bdf['momentum_mean_py']**2 + Bdf['momentum_mean_pz']**2)
theta_rad = np.arccos(Bdf['momentum_mean_pz'] / p)

# 使用与绘图1相同的对数分箱函数
def create_log_bins(data_min, data_max, n_bins, log_offset=1e-10):
    data_min_log = np.log10(max(data_min, log_offset))
    data_max_log = np.log10(data_max)
    log_bins = np.logspace(data_min_log, data_max_log, n_bins + 1)
    return log_bins

# 设置与绘图1相同的分箱数
n_bins_theta = 90
n_bins_p = 90

# 创建对数分箱
theta_bins = create_log_bins(theta_rad.min(), theta_rad.max(), n_bins_theta)
p_bins = create_log_bins(p.min(), p.max(), n_bins_p)

# 使用加权统计（考虑每个块的粒子数）
hist, theta_edges, p_edges = np.histogram2d(
    theta_rad, p,
    bins=[theta_bins, p_bins],
    weights=Bdf['particle_count'].values
)

# 创建绘图
plt.figure(figsize=(10, 8))
plt.pcolormesh(theta_edges, p_edges, hist.T,  # 注意转置
              cmap='viridis', norm=LogNorm(), shading='auto')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Angle with Z-axis (rad)')
plt.ylabel('Momentum Magnitude (GeV/c)')
plt.colorbar(label='Weighted Counts')
plt.grid(True, alpha=0.3)

# 设置与绘图1相同的坐标范围
plt.xlim(0, 1.07)

plt.tight_layout()
plt.savefig('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18/consistent_plot.png', dpi=300, bbox_inches='tight')
# plt.show()

# 保存分箱数据
# theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
# p_centers = (p_edges[:-1] + p_edges[1:]) / 2

# 保存为矩阵格式
# matrix_df = pd.DataFrame(hist.T, index=p_centers, columns=theta_centers)
# matrix_df.index.name = 'p_center_GeV'
# matrix_df.columns.name = 'theta_center_rad'
# matrix_df.to_csv('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18/consistent_hist_matrix.csv')