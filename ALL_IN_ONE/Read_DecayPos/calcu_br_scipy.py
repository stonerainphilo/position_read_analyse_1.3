import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def fast_interpolation():
    # 读取文件
    df_a = pd.read_csv('/media/ubuntu/SRPPS/Results/ALL/merged.csv')
    # df_b = pd.read_csv('/media/ubuntu/SRPPS/Results/Through_higgs_portal_all.csv')
    df_b = pd.read_csv('/media/ubuntu/SRPPS/Results/2HDM_H_ALL.csv')
    
    # 对tau取对数
    df_b['log_tau'] = np.log10(df_b['tau'] + 1e-100)
    df_a['log_tau'] = np.log10(df_a['tau'] + 1e-100)
    
    # 构建KD树（在对数空间）
    b_points = df_b[['m', 'log_tau']].values
    tree = cKDTree(b_points)
    
    # 查询最近邻
    print("使用KD树搜索最近邻...")
    distances, indices = tree.query(df_a[['m', 'log_tau']].values, k=3)
    
    # 加权平均
    weights = 1.0 / (distances + 1e-100)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # 计算插值
    br_values = df_b['Br_visible'].values
    tanb_values = df_b['tanb'].values
    
    df_a['Br_visible'] = np.sum(br_values[indices] * weights, axis=1)
    df_a['tanb'] = np.sum(tanb_values[indices] * weights, axis=1)
    
    # 删除临时列
    df_a.drop('log_tau', axis=1, inplace=True)
    
    # 保存结果
    df_a.to_csv('/media/ubuntu/SRPPS/Results/C_2HDM_H.csv', index=False)
    print("结果已保存到 C_fast2.csv")

    return df_a

# 运行快速版本
fast_interpolation()