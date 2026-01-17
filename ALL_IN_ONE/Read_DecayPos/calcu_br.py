import pandas as pd
import numpy as np

def nearest_neighbor_interpolation(df_a, df_b):
    """
    使用最近邻插值方法
    """
    print("使用最近邻插值...")
    
    # 创建结果列
    df_a['Br_visible'] = np.nan
    df_a['tanb'] = np.nan
    
    # 对A中的每个点，在B中寻找最近的(m, tau)点
    for idx, row_a in df_a.iterrows():
        # 计算距离（欧氏距离）
        distances = np.sqrt(
            (df_b['m'] - row_a['m'])**2 + 
            (df_b['tau'] - row_a['tau'])**2
        )
        
        # 找到最近的点
        nearest_idx = distances.idxmin()
        
        df_a.at[idx, 'Br_visible'] = df_b.loc[nearest_idx, 'Br_visible']
        df_a.at[idx, 'tanb'] = df_b.loc[nearest_idx, 'tanb']
    
    return df_a

def bilinear_interpolation_simple(df_a, df_b):
    """
    简化的双线性插值（假设数据网格化）
    """
    print("使用双线性插值...")
    
    # 假设B文件中的数据是网格化的
    # 获取唯一的m和tau值
    m_values = sorted(df_b['m'].unique())
    tau_values = sorted(df_b['tau'].unique())
    
    # 创建插值网格
    br_grid = df_b.pivot(index='m', columns='tau', values='Br_visible')
    tanb_grid = df_b.pivot(index='m', columns='tau', values='tanb')
    
    # 插值函数
    def interpolate_value(m, tau, grid):
        # 找到四个最近的点
        m_low = max([x for x in m_values if x <= m], default=min(m_values))
        m_high = min([x for x in m_values if x >= m], default=max(m_values))
        tau_low = max([x for x in tau_values if x <= tau], default=min(tau_values))
        tau_high = min([x for x in tau_values if x >= tau], default=max(tau_values))
        
        # 如果正好在网格点上
        if m_low == m_high and tau_low == tau_high:
            return grid.loc[m_low, tau_low]
        
        # 双线性插值
        f00 = grid.loc[m_low, tau_low]
        f01 = grid.loc[m_low, tau_high]
        f10 = grid.loc[m_high, tau_low]
        f11 = grid.loc[m_high, tau_high]
        
        # 插值计算
        if m_low != m_high:
            m_ratio = (m - m_low) / (m_high - m_low)
        else:
            m_ratio = 0
            
        if tau_low != tau_high:
            tau_ratio = (tau - tau_low) / (tau_high - tau_low)
        else:
            tau_ratio = 0
        
        value = (f00 * (1 - m_ratio) * (1 - tau_ratio) +
                f10 * m_ratio * (1 - tau_ratio) +
                f01 * (1 - m_ratio) * tau_ratio +
                f11 * m_ratio * tau_ratio)
        
        return value
    
    # 应用插值
    df_a['Br_visible'] = df_a.apply(
        lambda row: interpolate_value(row['m'], row['tau'], br_grid), 
        axis=1
    )
    
    df_a['tanb'] = df_a.apply(
        lambda row: interpolate_value(row['m'], row['tau'], tanb_grid), 
        axis=1
    )
    
    return df_a

# 使用示例
df_a = pd.read_csv('/media/ubuntu/SRPPS/Results/ALL/merged.csv')
df_b = pd.read_csv('/media/ubuntu/SRPPS/Results/2HDM_H_ALL.csv')

# 选择插值方法
method = input("请选择插值方法 (1: 最近邻, 2: 双线性): ")

if method == '1':
    result = nearest_neighbor_interpolation(df_a.copy(), df_b)
else:
    result = bilinear_interpolation_simple(df_a.copy(), df_b)

# 保存结果
result.to_csv('/media/ubuntu/SRPPS/Results/C2.csv', index=False)
print("结果已保存到 C.csv")

