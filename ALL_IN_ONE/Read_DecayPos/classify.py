import pandas as pd
import numpy as np
from collections import defaultdict

def process_llp_data(input_file, output_file):
    """
    处理LLP数据，按[m, tau_input]分组输出位置坐标
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    """
    
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查必要的列是否存在
    required_columns = ['m', 'tau_input', 'decay_pos_x', 'decay_pos_y', 'decay_pos_z']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件中缺少必要的列: {col}")
    
    # 创建一个字典来存储分组数据
    grouped_data = defaultdict(list)
    
    # 遍历每一行数据
    for _, row in df.iterrows():
        # 创建唯一的分组键
        group_key = (row['m'], row['tau_input'])
        
        # 添加位置坐标到对应的分组
        grouped_data[group_key].append({
            'decay_pos_x': row['decay_pos_x'],
            'decay_pos_y': row['decay_pos_y'],
            'decay_pos_z': row['decay_pos_z'],
            'id': row.get('id', '')  # 保留原始ID（如果存在）
        })
    
    # 准备输出数据
    output_rows = []
    
    for (m_value, tau_value), positions in grouped_data.items():
        # 为每个[m, tau_input]组合创建一个记录
        output_row = {
            'm': m_value,
            'tau_input': tau_value,
            'num_positions': len(positions),
            'positions': positions
        }
        output_rows.append(output_row)
    
    # 创建输出DataFrame（扁平化格式）
    flat_output = []
    
    for row in output_rows:
        m_value = row['m']
        tau_value = row['tau_input']
        
        for i, pos in enumerate(row['positions']):
            flat_output.append({
                'm': m_value,
                'tau_input': tau_value,
                'position_index': i + 1,
                'decay_pos_x': pos['decay_pos_x'],
                'decay_pos_y': pos['decay_pos_y'],
                'decay_pos_z': pos['decay_pos_z'],
                'id': pos.get('id', '')
            })
    
    # 转换为DataFrame并保存
    output_df = pd.DataFrame(flat_output)
    
    # 重新排列列顺序
    column_order = ['m', 'tau_input', 'position_index', 'decay_pos_x', 'decay_pos_y', 'decay_pos_z']
    if 'id' in output_df.columns:
        column_order.append('id')
    
    output_df = output_df[column_order]
    output_df.to_csv(output_file, index=False)
    
    print(f"处理完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"处理了 {len(df)} 行数据")
    print(f"得到 {len(grouped_data)} 个唯一的 [m, tau_input] 组合")
    print(f"总共 {len(output_df)} 个位置记录")
    
    return output_df


# 使用示例
if __name__ == "__main__":
    # 假设输入文件名为 'llp_data.csv'
    input_filename = '/media/ubuntu/SRPPS/CODEX-b/CODEX_b-1/mass_1.00e-01_ctau_1.03e+03_br_1.00e+00_seed_1.csv'
    output_filename = '/media/ubuntu/SRPPS/CODEX-b/CODEX_b-1_test.csv'
    

    
    # 处理方法1：保留所有位置
    print("=== 方法1：保留所有位置 ===")
    df_all_positions = process_llp_data(input_filename, 'output_all_positions.csv')
    
    # 显示结果
    print("\n=== 输出示例 ===")
    print("所有位置版本的前5行:")
    print(df_all_positions.head())