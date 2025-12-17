import pandas as pd
import numpy as np
Bdf = pd.read_csv('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/Log_p/block_summary.csv')
p = np.sqrt(Bdf['momentum_mean_px']**2 + Bdf['momentum_mean_py']**2 + Bdf['momentum_mean_pz']**2)
theta_rad = np.arccos(Bdf['momentum_mean_pz'] / p)

# ---------- 对数坐标分箱设置（log-spaced bins） ----------
# 分箱数量（可调整）
n_theta_bins = 500
n_p_bins = 500

# theta（弧度）必须为正值才能做对数分箱；处理零或极小值
theta_positive = theta_rad[theta_rad > 0]
if theta_positive.size == 0:
    raise ValueError('No positive theta values available for log binning')
theta_min = max(theta_positive.min(), 1e-12)
theta_max = theta_rad.max()
if theta_max <= theta_min:
    theta_max = theta_min * 10.0

theta_edges = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta_bins + 1)

# momentum p 也必须为正
p_positive = p[p > 0]
if p_positive.size == 0:
    raise ValueError('No positive momentum values available for log binning')
p_min = max(p_positive.min(), 1e-12)
p_max = p.max()
if p_max <= p_min:
    p_max = p_min * 10.0

p_edges = np.logspace(np.log10(p_min), np.log10(p_max), n_p_bins + 1)

# 使用 edges 数组作为 bins 参数（对数分箱）并加权 particle_count
hist, theta_edges, p_edges = np.histogram2d(theta_rad, p,
                                            bins=[theta_edges, p_edges],
                                            weights=Bdf['particle_count'].values)

# 计算几何中心（对数坐标的中心使用几何平均更合理）
theta_centers = np.sqrt(theta_edges[:-1] * theta_edges[1:])
p_centers = np.sqrt(p_edges[:-1] * p_edges[1:])

theta_grid = np.repeat(theta_centers, len(p_centers))
p_grid = np.tile(p_centers, len(theta_centers))
counts = hist.flatten()

df_hist = pd.DataFrame({
    'theta_center_rad': theta_grid,
    'p_center_GeV': p_grid,
    'count': counts
})

# （可选）只保存非空箱
df_hist_nonzero = df_hist[df_hist['count'] > 0].reset_index(drop=True)
df_hist_nonzero.to_csv('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/Log_p/hist.csv', index=False)

# 也保存矩阵格式（行=p_center，列=theta_center）
matrix_df = pd.DataFrame(hist.T, index=p_centers, columns=theta_centers)
matrix_df.index.name = 'p_center_GeV'
matrix_df.columns.name = 'theta_center_rad'
matrix_df.to_csv('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/Log_p/hist_matrix.csv')

# print(hist.shape, theta_edges.shape, p_edges.shape)
# print(p_edges)
# plt.figure(figsize=(10, 8))
# plt.pcolormesh(theta_edges, p_edges, hist.T, cmap='viridis', norm=LogNorm())
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Angle with Z-axis (rad)', fontsize=12)
# plt.ylabel('Momentum Magnitude (GeV/c)', fontsize=12)
# plt.title(f'B_511 Block Momentum Distribution\n(Total events: {Bdf["particle_count"].sum():,})', fontsize=14)
# plt.xlim(1e-10, 1.07)
# plt.colorbar(label='Counts (log scale)')
# plt.grid(True, alpha=0.3)
# plt.show()