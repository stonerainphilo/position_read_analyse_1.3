import numpy as np
import pandas as pd

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
    # kx = px/m
    # ky = py/m
    # kz = pz/m
    # vx = np.sqrt(1/(1+(np.square(kx))))*kx
    # vy = np.sqrt(1/(1+(np.square(ky))))*ky
    # vz = np.sqrt(1/(1+(np.square(kz))))*kz
    v = np.sqrt(k**2/(1+(np.square(k)))) # V's Unit is C
    # gamma = k/v
    gamma = 1/np.sqrt(1-v**2)
    r = v * ctau * gamma # The ctau's Unit is mm/C
    x = r * (px/p) + prod_x
    y = r * (py/p) + prod_y
    z = r * (pz/p) + prod_z
    # print(kx, ky, kz)
    # print(gamma)
    # print(gamma2)
    # return gamma, gamma2
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
    m = np.sqrt(energy**2 - np.dot(p_vec, p_vec))
    decay_position = calculate_decay_position(p_vec[0], p_vec[1], p_vec[2], m, decay_time, 
                                             birth_position[0], birth_position[1], birth_position[2])
    
    return decay_position, decay_time

def LLP_decay_sim_521(llp_data, B521_data):
    # 读取B521数据（多行）
    df_B521 = pd.read_csv(B521_data)
    if df_B521.empty:
        raise ValueError("B521 data is empty.")
    
    # 读取LLP数据（只有一行）
    df_LLP = pd.read_csv(llp_data)
    if df_LLP.empty:
        raise ValueError("LLP data is empty.")
    
    # 从LLP数据中获取质量和寿命（只有一行）
    tau_llp = df_LLP['ltime'].iloc[0]
    mass_llp = df_LLP['mH'].iloc[0]
    
    # 存储所有事件的结果
    all_decay_positions = []
    all_birth_positions = []
    all_momenta = []
    all_B521_indices = []
    all_tau_real = []
    
    # 对每个B521粒子进行处理
    for i, B521_row in df_B521.iterrows():
        # 获取B521的出生位置和动量
        llp_birth_pos_521 = B521_row[['decay_pos_x_521', 'decay_pos_y_521', 'decay_pos_z_521']].values
        momentum_B521 = B521_row[['e_521', 'px_521', 'py_521', 'pz_521']].values
        
        # 生成LLP动量从B521衰变
        llp_birth_momentum_521, _ = two_body_decay_lab(momentum_B521, 5.279, mass_llp, 0.494)
        
        # 生成衰变位置
        llp_decay_pos_521, tau = generate_decay_position(tau_llp, llp_birth_momentum_521, llp_birth_pos_521)
        
        # 存储结果
        all_decay_positions.append(llp_decay_pos_521)
        all_birth_positions.append(llp_birth_pos_521)
        all_momenta.append(llp_birth_momentum_521)
        all_B521_indices.append(i)
        all_tau_real.append(tau)
    
    # 转换为numpy数组
    all_decay_positions = np.array(all_decay_positions)
    all_birth_positions = np.array(all_birth_positions)
    all_momenta = np.array(all_momenta)

    
    return {
        'decay_positions': all_decay_positions,  # 形状: (n_events, 3)
        'birth_positions': all_birth_positions,   # 形状: (n_events, 3)
        'momenta': all_momenta,                   # 形状: (n_events, 4)
        'B521_indices': all_B521_indices,
        'llp_mass': mass_llp,
        'tau_input': tau_llp,
        'tau_real': all_tau_real,
        'n_events': len(df_B521)
    }

# 使用示例
# LLP_data = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/test/2HDM_H_test1.csv'
# B_521 = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/test/test/2025-11-30_2HDM_B_test/LLP_data/B_521_1.csv'

# try:
#     result = LLP_decay_sim_521(LLP_data, B_521)
    
#     print(f"处理了 {result['n_events']} 个事件")
#     print(f"LLP质量: {result['llp_mass']} GeV")
#     print(f"LLP寿命: {result['tau_input']} s")
#     print("\n前5个事件的衰变位置:")
#     for i in range(min(5, result['n_events'])):
#         print(f"事件 {i}: {result['decay_positions'][i]}")
#     df = pd.DataFrame(result['decay_positions'], columns=['decay_pos_x', 'decay_pos_y', 'decay_pos_z'])
#     df['tau'] = result['tau_real']
#     df.to_csv('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/test/test_decay.csv')
# except Exception as e:
#     print(f"Error: {e}")