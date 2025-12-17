import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np

# prefer matrix format (rows = p_centers, cols = theta_centers)
matrix_path = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18/hist_matrix.csv'
if os.path.exists(matrix_path):
    matrix = pd.read_csv(matrix_path, index_col=0)
    p_centers = matrix.index.astype(float).values
    theta_centers = matrix.columns.astype(float).values
    hist2d = matrix.values  # shape (n_p, n_theta)
else:
    df = pd.read_csv('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18/hist.csv')
    hist_pivot = df.pivot_table(index='p_center_GeV', columns='theta_center_rad', values='count', fill_value=0)
    p_centers = hist_pivot.index.astype(float).values
    theta_centers = hist_pivot.columns.astype(float).values
    hist2d = hist_pivot.values

def centers_to_edges(centers):
    if len(centers) == 1:
        return np.array([centers[0] - 0.5 * centers[0], centers[0] + 0.5 * centers[0]])
    diffs = np.diff(centers)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = centers[:-1] + diffs / 2
    edges[0] = centers[0] - diffs[0] / 2
    edges[-1] = centers[-1] + diffs[-1] / 2
    return edges

x_edges = centers_to_edges(theta_centers)
y_edges = centers_to_edges(p_centers)

# ensure positive edges for log scale
if np.any(x_edges <= 0):
    pos_min = np.min(x_edges[x_edges > 0]) if np.any(x_edges > 0) else 1e-12
    x_edges = np.where(x_edges <= 0, pos_min * 1e-3, x_edges)
if np.any(y_edges <= 0):
    pos_min = np.min(y_edges[y_edges > 0]) if np.any(y_edges > 0) else 1e-12
    y_edges = np.where(y_edges <= 0, pos_min * 1e-3, y_edges)

# choose vmin for LogNorm from smallest positive count
positive_counts = hist2d[hist2d > 0]
vmin = positive_counts.min() if positive_counts.size > 0 else 1e-10

# --- New: make cells visually larger by rebinning and increasing figure size ---
# Set these to >1 to merge adjacent bins (coarser grid -> larger cells)
rebin_p = 3        # merge every 4 p bins (rows)
rebin_theta = 3     # merge every 4 theta bins (cols)

def rebin_and_group(hist, p_centers, theta_centers, rp, rt):
    if rp == 1 and rt == 1:
        return hist, p_centers, theta_centers
    n_p, n_t = hist.shape
    n_p_eff = (n_p // rp) * rp
    n_t_eff = (n_t // rt) * rt
    h = hist[:n_p_eff, :n_t_eff]
    h = h.reshape(n_p_eff//rp, rp, n_t_eff//rt, rt).sum(axis=(1,3))
    p_grp = p_centers[:n_p_eff].reshape(-1, rp).mean(axis=1)
    theta_grp = theta_centers[:n_t_eff].reshape(-1, rt).mean(axis=1)
    return h, p_grp, theta_grp

hist_plot, p_centers_plot, theta_centers_plot = rebin_and_group(hist2d, p_centers, theta_centers, rebin_p, rebin_theta)
x_edges = centers_to_edges(theta_centers_plot)
y_edges = centers_to_edges(p_centers_plot)

# # recompute vmin for plotted data
positive_counts = hist_plot[hist_plot > 0]
vmin = positive_counts.min() if positive_counts.size > 0 else vmin

# larger figure + lower dpi when saving so cells appear bigger
plt.figure(figsize=(14, 10))
plt.pcolormesh(x_edges, y_edges, hist_plot, cmap='viridis', norm=LogNorm(vmin=vmin),
               shading='auto', rasterized=True)
# im1 = plt.pcolormesh(x_edges, y_edges, hist.T,  # 转置以适应pcolormesh
#                     cmap='viridis', norm=LogNorm())
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Angle with Z-axis (rad)'); plt.ylabel('Momentum (GeV/c)')
plt.xlim(0, 1.07)
plt.colorbar(label='Counts (weighted)')
plt.grid(True, alpha=0.3)
plt.savefig('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18/B_block_p_x_log.png', dpi=200)