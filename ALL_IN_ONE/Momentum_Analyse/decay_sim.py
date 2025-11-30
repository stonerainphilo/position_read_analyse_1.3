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

def two_body_decay_lab(parent_four_momentum, parent_mass, mass1, mass2): # all in lab frame
    """
    Simulate a two-body decay in the laboratory frame.
    """
    if parent_mass < mass1 + mass2:
        raise ValueError(f"<<ERROR>>Mother_mass {parent_mass} isn't big enough for {mass1} and {mass2}")

    # Calcu late beta and gamma for the boost from CM to lab frame
    beta_x = parent_four_momentum[1] / parent_four_momentum[0]
    beta_y = parent_four_momentum[2] / parent_four_momentum[0]
    beta_z = parent_four_momentum[3] / parent_four_momentum[0]
    beta = np.array([beta_x, beta_y, beta_z])
    beta_sq = np.dot(beta, beta)
    gamma = parent_four_momentum[0] / parent_mass
    

    
    # Calcu m and p in lab frame
    E1_cm = (parent_mass**2 + mass1**2 - mass2**2) / (2 * parent_mass)
    E2_cm = (parent_mass**2 + mass2**2 - mass1**2) / (2 * parent_mass)
    
    p_cm = np.sqrt(E1_cm**2 - mass1**2)

    # decay to every direction
    theta = np.arccos(2 * np.random.random() - 1)
    phi = 2 * np.pi * np.random.random()
    px_cm = p_cm * np.sin(theta) * np.cos(phi)
    py_cm = p_cm * np.sin(theta) * np.sin(phi)
    pz_cm = p_cm * np.cos(theta)
    
    # momentum in CM frame
    
    p1_cm = np.array([E1_cm, px_cm, py_cm, pz_cm])
    p2_cm = np.array([E2_cm, -px_cm, -py_cm, -pz_cm])
    
    # boost to lab frame
    four_momentum1 = lorentz_boost(p1_cm, beta, gamma, beta_sq)
    four_momentum2 = lorentz_boost(p2_cm, beta, gamma, beta_sq)
    
    return four_momentum1, four_momentum2

def decay_distribution(ltime, momentum):
    t = np.linspace(0, 10*ltime, 1000)
    posibility = np.exp(- (t/ltime))
    decay_pos_x = posibility * t * momentum[1]
    decay_pos_y = posibility * t * momentum[2]
    decay_pos_z = posibility * t * momentum[3]
    decay_pos = np.array([decay_pos_x, decay_pos_y, decay_pos_z])
    gamma = momentum[0] / np.sqrt(momentum[0]**2 - np.dot(momentum[1:], momentum[1:]))
    decay_pos = decay_pos * gamma  # time dilation
    # decay_r = np.abs(decay_pos)
    return decay_pos

def LLP_decay_sim_521(llp_data, B521_data):
    df_B521 = pd.read_csv(B521_data)
    if df_B521.empty:
        raise ValueError("B521 data is empty.")
        
    llp_birth_pos_521 = df_B521[['decay_pos_x_521', 'decay_pos_y_521', 'decay_pos_z_521']].to_numpy().T
    momentum_B521 = df_B521[['e_521', 'px_521', 'py_521', 'pz_521']].to_numpy()
    
    df_LLP = pd.read_csv(llp_data)
    if df_LLP.empty:
        raise ValueError("LLP data is empty.")
        
    tau_llp = df_LLP['ltime'].to_numpy()
    mass_llp = df_LLP['mH'].to_numpy()
    
    llp_birth_momentum_521 = two_body_decay_lab(momentum_B521, 5.279, mass_llp, 0.494)[0]
    llp_decay_pos_521 = decay_distribution(tau_llp, llp_birth_momentum_521) + llp_birth_pos_521
    
    return {
        'decay_positions': llp_decay_pos_521,
        'birth_positions': llp_birth_pos_521,
        'momentum': llp_birth_momentum_521
    }
LLP_data = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/test/2HDM_H_test.csv'
B_521 = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/test/test/2025-11-30_2HDM_B_test/LLP_data/B_521_a.csv'
decay_pos = LLP_decay_sim_521(LLP_data, B_521)
# def LLP_decay_sim_521(llp_data, B521_data):
#     df_B521 = pd.read_csv(B521_data)
#     llp_birth_pos_521 = np.array([df_B521['decay_pos_x_521'], df_B521['decay_pos_y_521'], df_B521['decay_pos_z_521']]) 
#     momentum_B521 = np.array([df_B521['e_521'], df_B521['px_521'], df_B521['py_521'], df_B521['pz_521']])
#     # df_B511 = pd.read_csv(B511_data)
#     # decay_B511 = np.array([df_B511['decay_pos_x_511'], df_B511['decay_pos_y_511'], df_B511['decay_pos_z_511']]) 
#     # momentum_B511 = np.array([df_B511['e_511'], df_B511['px_511'], df_B511['py_511'], df_B511['pz_511']])
#     df_LLP = pd.read_csv(llp_data)
#     tau_llp = df_LLP['ltime']
#     mass_llp = df_LLP['mH']
#     llp_birth_momentum_521 = two_body_decay_lab(momentum_B521, 5.279, mass_llp, 0.494)[0]
#     llp_decay_pos_521 = decay_distribution(tau_llp, llp_birth_momentum_521) + llp_birth_pos_521
#     # llp_birth_pos_511 = two_body_decay_lab(momentum_B511, 5.279, mass_llp, 0.494)[0]
#     return llp_decay_pos_521

# if __name__ == "__main__":
#     # Example：B+->J/psi and K+
#     # B+: 5279 MeV, J/psi: 3097 MeV, K+: 494 MeV
#     parent_mass = 5279.0
#     mass1 = 3097.0  # J/psi
#     mass2 = 494.0   # K+
    
#     # In lab frame, B+'s four-momentum [E, px, py, pz] in GeV

#     parent_four_momentum = np.array([10000, 0, 0, np.sqrt(10000**2 - parent_mass**2)])
    
#     # Simulate 2body decay
#     p1, p2 = two_body_decay_lab(parent_four_momentum, parent_mass, mass1, mass2)
    
#     print("4-Momentum of (J/psi):")
#     print(f"E = {p1[0]:.2f} MeV, px = {p1[1]:.2f} MeV, py = {p1[2]:.2f} MeV, pz = {p1[3]:.2f} MeV")
#     print(f"Mass Check: {np.sqrt(p1[0]**2 - np.dot(p1[1:], p1[1:])):.2f} MeV")
    
#     print('--------------------------------')

#     print("\n4-Momentum of (K+):")
#     print(f"E = {p2[0]:.2f} MeV, px = {p2[1]:.2f} MeV, py = {p2[2]:.2f} MeV, pz = {p2[3]:.2f} MeV")
#     print(f"Mass Check: {np.sqrt(p2[0]**2 - np.dot(p2[1:], p2[1:])):.2f} MeV")
    
#     # Chech conservation of four-momentum
#     total_energy = p1[0] + p2[0]
#     total_px = p1[1] + p2[1]
#     total_py = p1[2] + p2[2]
#     total_pz = p1[3] + p2[3]
    
#     print(f"\nCheck 4-momentum Conservation:")
#     print(f"Energy: {total_energy:.2f} vs {parent_four_momentum[0]:.2f}")
#     print(f"px: {total_px:.2f} vs {parent_four_momentum[1]:.2f}")
#     print(f"py: {total_py:.2f} vs {parent_four_momentum[2]:.2f}")
#     print(f"pz: {total_pz:.2f} vs {parent_four_momentum[3]:.2f}")


