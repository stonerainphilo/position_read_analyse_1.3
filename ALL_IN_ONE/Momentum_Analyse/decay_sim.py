import numpy as np
import pandas as pd
import os

def lorentz_boost(p, beta, gamma, beta_sq):
    """Boost four-momentum using beta and gamma"""
    if beta_sq < 1e-12:  # No boost needed for nearly stationary particles
        return p.copy()
    
    p_space = p[1:]
    p_dot_beta = np.dot(p_space, beta)
    
    # Using lorentz transform
    E_lab = gamma * (p[0] + p_dot_beta)
    p_lab = p_space + ((gamma - 1) * p_dot_beta / beta_sq + gamma * p[0]) * beta
    
    return np.array([E_lab, p_lab[0], p_lab[1], p_lab[2]])

def two_body_decay_lab(parent_four_momentum, parent_mass, mass1, mass2):
    """
    Simulate a two-body decay in the laboratory frame.
    """
    if parent_mass < mass1 + mass2:
        raise ValueError(f"Mother_mass {parent_mass} isn't big enough for {mass1} and {mass2}")

    # Calculate beta and gamma for the boost from CM to lab frame
    beta_x = parent_four_momentum[1] / parent_four_momentum[0]
    beta_y = parent_four_momentum[2] / parent_four_momentum[0]
    beta_z = parent_four_momentum[3] / parent_four_momentum[0]
    beta = np.array([beta_x, beta_y, beta_z])
    beta_sq = np.dot(beta, beta)
    gamma = parent_four_momentum[0] / parent_mass
    
    # Calculate energies and momentum in CM frame
    E1_cm = (parent_mass**2 + mass1**2 - mass2**2) / (2 * parent_mass)
    E2_cm = (parent_mass**2 + mass2**2 - mass1**2) / (2 * parent_mass)
    
    p_cm = np.sqrt(E1_cm**2 - mass1**2)

    # Random decay direction in CM frame
    theta = np.arccos(2 * np.random.random() - 1)
    phi = 2 * np.pi * np.random.random()
    px_cm = p_cm * np.sin(theta) * np.cos(phi)
    py_cm = p_cm * np.sin(theta) * np.sin(phi)
    pz_cm = p_cm * np.cos(theta)
    
    # Four-momenta in CM frame
    p1_cm = np.array([E1_cm, px_cm, py_cm, pz_cm])
    p2_cm = np.array([E2_cm, -px_cm, -py_cm, -pz_cm])
    
    # Boost to lab frame
    four_momentum1 = lorentz_boost(p1_cm, beta, gamma, beta_sq)
    four_momentum2 = lorentz_boost(p2_cm, beta, gamma, beta_sq)
    
    return four_momentum1, four_momentum2

def calculate_abs(x, y, z):
    return np.sqrt((x**2+y**2+z**2))

def calculate_decay_position(px, py, pz, m, ctau, prod_x, prod_y, prod_z):
    p = calculate_abs(px, py, pz)
    k = p/m
    v = np.sqrt(k**2/(1+(np.square(k)))) # V's Unit is C
    gamma = 1/np.sqrt(1-v**2)
    r = v * ctau * gamma # The ctau's Unit is mm/C
    x = r * (px/p) + prod_x
    y = r * (py/p) + prod_y
    z = r * (pz/p) + prod_z
    return x, y, z

def generate_decay_position(ltime, momentum, birth_position):
    """
    Generate decay position considering proper lifetime and time dilation
    """
    # Generate random decay time following exponential distribution
    decay_time = np.random.exponential(ltime)
    
    # Calculate gamma factor for time dilation
    p_vec = momentum[1:]
    energy = momentum[0]
    NAN = [np.nan, np.nan, np.nan]
    energy_sq_minus_p_sq = energy**2 - np.dot(p_vec, p_vec)
    
    if energy_sq_minus_p_sq < 0:
        return NAN, decay_time
    
    m = np.sqrt(energy_sq_minus_p_sq)
    decay_position = calculate_decay_position(p_vec[0], p_vec[1], p_vec[2], m, decay_time, 
                                             birth_position[0], birth_position[1], birth_position[2])
    
    return decay_position, decay_time

def LLP_decay_sim_521(llp_data, B521_data, output_dir):
    """
    为LLP数据的每一行生成单独的CSV文件
    
    参数:
    llp_data: LLP数据文件路径
    B521_data: B521数据文件路径
    output_dir: 输出目录路径
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取LLP数据
    df_LLP = pd.read_csv(llp_data)
    if df_LLP.empty:
        raise ValueError("LLP data is empty.")
    
    # 读取B521数据
    df_B521 = pd.read_csv(B521_data)
    if df_B521.empty:
        raise ValueError("B521 data is empty.")
    
    # 处理LLP的每一行
    for llp_idx, llp_row in df_LLP.iterrows():
        # 获取当前LLP的参数
        tau_llp = llp_row['ltime']
        mass_llp = llp_row['mH']
        tanb = llp_row['tanb']
        
        # 创建输出文件名，包含LLP Mass和Lifetime
        # 使用科学计数法简化文件名
        mass_str = f"{mass_llp:.6f}".rstrip('0').rstrip('.')
        tau_str = f"{tau_llp:.6e}".replace('.', 'p').replace('+', '').replace('-', 'm')
        filename = f"LLP_mass_{mass_str}_tanb_{tanb}.csv"
        output_path = os.path.join(output_dir, filename)
        
        # print(f"\n处理LLP #{llp_idx}: mass={mass_llp}, lifetime={tau_llp}")
        # print(f"输出文件: {filename}")
        
        # 初始化存储当前LLP的所有结果
        all_decay_positions = []
        all_birth_positions = []
        all_momenta = []
        all_B521_indices = []
        all_tau_real = []
        
        # 对当前LLP参数，处理所有B521事件
        for i, B521_row in df_B521.iterrows():
            # 获取B521的出生位置和四动量
            llp_birth_pos_521 = B521_row[['decay_pos_x_521', 'decay_pos_y_521', 'decay_pos_z_521']].values
            momentum_B521 = B521_row[['e_521', 'px_521', 'py_521', 'pz_521']].values
            
            # 从B521生成LLP
            try:
                llp_birth_momentum_521, _ = two_body_decay_lab(momentum_B521, 5.279, mass_llp, 0.494)
            except ValueError as e:
                print(f"警告: 事件 {i} 跳过 - {e}")
                continue
            
            # 生成衰变位置
            llp_decay_pos_521, tau = generate_decay_position(tau_llp, llp_birth_momentum_521, llp_birth_pos_521)
            
            # 保存结果
            all_decay_positions.append(llp_decay_pos_521)
            all_birth_positions.append(llp_birth_pos_521)
            all_momenta.append(llp_birth_momentum_521)
            all_B521_indices.append(i)
            all_tau_real.append(tau)
        
        # 转换为numpy数组
        all_decay_positions = np.array(all_decay_positions)
        all_birth_positions = np.array(all_birth_positions)
        all_momenta = np.array(all_momenta)
        
        # 创建DataFrame保存结果
        results_df = pd.DataFrame({
            'decay_pos_x': all_decay_positions[:, 0],
            'decay_pos_y': all_decay_positions[:, 1],
            'decay_pos_z': all_decay_positions[:, 2],
            # 'birth_pos_x': all_birth_positions[:, 0],
            # 'birth_pos_y': all_birth_positions[:, 1],
            # 'birth_pos_z': all_birth_positions[:, 2],
            # 'energy': all_momenta[:, 0],
            # 'px': all_momenta[:, 1],
            # 'py': all_momenta[:, 2],
            # 'pz': all_momenta[:, 3],
            # 'B521_index': all_B521_indices,
            # 'tau_real': all_tau_real,
            # 'llp_mass': mass_llp,
            # 'llp_lifetime': tau_llp,
            # 'tanb': tanb
        })
        
        # 保存到CSV文件
        results_df.to_csv(output_path, index=False)
        # print(f"已保存 {len(results_df)} 个事件到 {output_path}")
    
    print(f"\n完成! 共处理了 {len(df_LLP)} 个LLP参数集")
    return len(df_LLP)

# 示例用法
if __name__ == "__main__":
    # 输入文件路径
    LLP_data = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/test/2HDM_H_B_decay_1.csv'
    # B_521 = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Decay_B2025-12-03_2HDM_B_test/B_521.csv'
    B_521 = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Decay_B2025-12-03_2HDM_B_test/LLP_data/B_521_seed438768.csv'
    
    # 输出目录
    output_directory = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Decay_B2025-12-03_2HDM_B_test/LLP_decay_results/'
    
    try:
        num_processed = LLP_decay_sim_521(LLP_data, B_521, output_directory)
        print(f"\n成功处理了 {num_processed} 个LLP参数集")
        
        # 列出生成的文件
        print("\n生成的文件:")
        for filename in os.listdir(output_directory):
            if filename.endswith('.csv'):
                print(f"  {filename}")
                
    except Exception as e:
        print(f"错误: {e}")