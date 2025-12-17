import pandas as pd
import numpy as np
Bdf = pd.read_csv('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18/block_summary.csv')
p = np.sqrt(Bdf['momentum_mean_px']**2 + Bdf['momentum_mean_py']**2 + Bdf['momentum_mean_pz']**2)
theta_rad = np.arccos(Bdf['momentum_mean_pz'] / p)
# 使用 histogram2d 按 theta/p 分箱并加权 particle_count
bins_theta = np.unique(theta_rad).size
bins_p = np.unique(p).size
hist, theta_edges, p_edges = np.histogram2d(theta_rad, p,
                                            bins=[bins_theta, bins_p],
                                            weights=Bdf['particle_count'].values)
# 计算 bin 中心并展开为表格（每个二元箱一行）
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])

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
df_hist_nonzero.to_csv('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18/hist.csv', index=False)

# 也保存矩阵格式（行=p_center，列=theta_center）
matrix_df = pd.DataFrame(hist.T, index=p_centers, columns=theta_centers)
matrix_df.index.name = 'p_center_GeV'
matrix_df.columns.name = 'theta_center_rad'
matrix_df.to_csv('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18/hist_matrix.csv')

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