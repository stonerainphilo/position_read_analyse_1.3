import numpy as np
import pandas as pd
import sys
# sys.path.append("../LSD")
sys.path.append("/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/ALL_IN_ONE/FIT_FUNC")
sys.path.append("/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/ALL_IN_ONE/Detector")  # Replace with the actual path to the Detector directory
import judge as judge

L1 = np.sqrt(np.square(5)+np.square(26))
L2 = np.sqrt(15*15+36*36)
L2_precise = np.sqrt(15*15+36*36+7*7)

# Constants
G_F = 1.1663787e-5  # Fermi coupling constant in GeV^-2
v = 246.0            # Vacuum expectation value in GeV
pi = np.pi
m_W = 80.379        # W boson mass in GeV
m_h = 125.0         # Higgs boson mass in GeV
# CKM Martics
V_cb = 0.04
V_cs = 0.974
V_tb = 1.0
V_ub = 3.5e-3
V_us = 0.2248
V_ts = -3.9e-2

# upper quark masses in GeV
m_u = 2.2e-3
m_c = 1.28
m_t = 172.57
m_b = 4.18
# m_d = 


ctau = 1
beta = 1
gamma = 1
weight_approx = np.exp(-L1/(ctau*beta*gamma))-np.exp(-L2/(ctau*beta*gamma))

vertices = [
    [750, 2150, 45000],  # SHiP Bottom Vertices
    [750, -2150, 45000],
    [-750, -2150, 45000],
    [-750, 2150, 45000],
    [2500, 5000, 95000],   # SHiP Top Vertices
    [2500, -5000, 95000],
    [-2500, -5000, 95000],
    [-2500, 5000, 95000]
    
]

# Similar to the cube, faces are defined by the indices of the vertices
faces = [
    [0, 1, 2, 3],  # Bottom face
    [4, 5, 6, 7],  # Top face
    [0, 1, 5, 4],  # Side faces
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7]
]

def f_function(mc, mb): # The real f_function is not this one
    mc = 1.27
    mb = 4.18
    return mc/mb


def calculate_Sin(mphi, Br1 = 6*10**(-8), Br2 = 0.9, g = 2):
    default_Br1 = 6*10**(-8)
    default_Br2 = 0.9
    default_g = 2
    mt = 172.76 
    mb = 4.78
    mw = 80.379
    mc = 1.67
    Vts = -0.0405
    Vtb = 0.9991
    Vcb = 0.041
    para1 = (256 * np.square(np.pi))/(27 * np.square(g))
    para2 = (np.square(mt)*np.square(mt))/(np.square(mb)*np.square(mw))
    para3 = np.square(1-(mphi/mb))/0.51 # 0.51 is The phase space Factor. CITE: Limits on a light Higgs boson
    para4 = np.square(Vts*Vtb/Vcb)
    Sin_square = (Br1/Br2) * para1 / (para2 * para3 * para4)
    return Sin_square


def calculate_Br(mphi, sin_square_theta = 6*10**(-8), Br2 = 0.1, g = 0.65):
    default_Br1 = 6*10**(-8)
    default_Br2 = 0.9
    default_g = 2
    mt = 172.76 
    mb = 4.18
    mw = 80.379
    mc = 1.27
    Vts = -0.0405
    Vtb = 0.9991
    Vcb = 0.041
    para1 = (27 * np.square(g))/(256 * np.square(np.pi))
    para2 = (np.square(mt)*np.square(mt))/(np.square(mb)*np.square(mw))
    para3 = np.square(1-np.square(mphi/mb))/0.511365146826185 # 0.51 is The phase space Factor. CITE: Limits on a light Higgs boson
    #Or f(x) = (1-8x+x^2)(1- x^2) - 12x^2 lnx CITE: Light Scalar at FASER
    # x = np.square(mc/mb)
    # f = (1-8*x+x**2)*(1-x**2) - 12*x**2*np.log(x)
    # print(f = 0.511365146826185) 
    para4 = np.square(Vts*Vtb/Vcb)
    # para4 = 1
    Br = (sin_square_theta*Br2) * para1 * (para2 * para3 * para4)
    # print((Br2) * para1 * (para2 * para4))
    return Br



# def calculate_Br_2HDM_I(mphi, sin_square_theta = 6*10**(-8), Br2 = 0.104, g = 0.65): # CITE: Light Scalar at FASER
#     default_Br1 = 6*10**(-8)
#     default_Br2 = 0.9
#     default_g = 0.65
#     mt = 172.76 
#     mb = 4.18
#     mw = 80.379
#     mc = 1.27
#     v = 246
#     Vts = -0.0405
#     Vtb = 0.9991
#     Vcb = 0.041
#     Xi_bs = 0.041
#     para1 = 12 * np.square(np.pi) * np.square(v) / np.square(mb)
#     # para2 = (np.square(mt)*np.square(mt))/(np.square(mb)*np.square(mw))
#     para3 = np.square(1-np.square(mphi/mb))/0.511365146826185 # 0.51 is The phase space Factor.
#     #Or f(x) = (1-8x+x^2)(1- x^2) - 12x^2 lnx CITE: Light Scalar at FASER
#     # x = np.square(mc/mb)
#     # f = (1-8*x+x**2)*(1-x**2) - 12*x**2*np.log(x)
#     # print(f = 0.511365146826185) 
#     para4 = np.square(Xi_bs/Vcb)
#     # para4 = 1
#     Br = (sin_square_theta*Br2) * para1 * (para3 * para4)
#     # print((Br2) * para1 * (para2 * para4))
#     return Br




# g0, g1, g2 
def g0(xk, xHpm, tanb):
    cotb = 1/tanb
    numerator = -cotb**2 * (3* xHpm**2 - 4 * xHpm*xk + xk**2 - 2*xk*(2*xHpm - xk)*np.log(xHpm/xk)) 
    denominator = 16 * (xHpm - xk)**3
    return numerator / denominator

def g1(xk, xHpm, tanb):
    cotb = 1/tanb
    return - 3/4 + ((cotb**2 * xk) * (
        5*xHpm**2 - 8*xHpm*xk + 3*xk**2 - 2*xHpm*(2*xHpm - xk)*np.log(xHpm/xk)
    )) / (4 *(xHpm - xk)**3)

def X1(xk, xHpm):
    t1 = (xHpm/(xHpm - xk)) - (6/(xk-1)**2) + 3
    t2 = -1 * xHpm * ((3 * xHpm - 2 * xk)/((xHpm - xk)**2)) * np.log(xHpm)
    t3 = ((xHpm* (3* xHpm - 2* xk)/((xHpm - xk)**2)) + (3*(xk + 1)/((xk -1)**3))) * np.log(xk)
    return (-1/4) * (t1 + t2 + t3)  

def X2(xk, xHpm):
    t1 = xk* (5 *xHpm -3*xk) / (4*(xHpm - xk)**2)
    t2 = (xHpm * xk *(2*xHpm -xk)/(2*((xHpm-xk)**3)))*np.log(xHpm/xk)
    return t1 - t2


def g2(xk, xHpm, tanb):
    cotb = 1/tanb
    term1 = cotb * X1(xk, xHpm)
    term2 = cotb**3 * X2(xk, xHpm)
    return (term1 + term2)

# 相空间因子 f(x)
def f(x):
        # sqrt_term = np.sqrt(1 - 1/x)
    log_term = np.log(x)
    return (1 - 8*x + x**2) * (1-x**2) - 12 * x**2 * log_term



def calcu_lambda_HHpHm(m_H, m_HC, tanb, cosba):
    lambda_HHpHm = (-1 / v) * (2 * m_HC**2 - m_H**2) * cosba
    return lambda_HHpHm

def calcu_lambda_hHpHm(m_H, m_HC, tanb, cosba):
    sinba = np.sqrt(1 - cosba**2)
    lambda_hHpHm = (-1/v) * ((2*m_HC**2 - m_H**2 - m_h**2) * sinba + 2 * (m_H**2-m_h**2) * cosba * ((1 -tanb**2)/2 * tanb))
    return lambda_hHpHm

# 分支比计算
def calcu_Br_B_to_H(m_phi, tanb, cosba, m_HC = 600.0):
    # Plot with different tanb to see the effect
    # take m_Hpm = 600 GeV

    # take m_b = 4.18 GeV 
    m_b = 4.18
    # x = (m_c**2) / (m_b**2)
    f_x = 0.506 # Approximate value for f(m_c^2 / m_b^2)

    # upper quarks are: k = u, c, t
    quarks = ['u', 'c', 't']
    masses = {'u': m_u, 'c': m_c, 't': m_t}
    m_Hpm = m_HC
    x_Hpm = (m_Hpm**2) / (m_W**2)
    
    # tan(beta) and cos(beta - alpha) 
    # tan_beta = tanb
    # cos_beta_alpha = cosba
    # beta = np.arctan(tan_beta)

    cotb = 1 / tanb

    # tri Higgs Coupling: lambda_hH+H-
    # if lambda_v2 = 0, then from Light Scalar at FASER (C.3)
    # lambda_hHpHm = (-1/v) * ((2*m_Hpm**2 - m_H**2 - m_h**2) * sinba + 2 * (m_H**2-m_h**2) * cosba * ((1 -tanb**2)/2 * tanb))
    # take m_H
    m_H = m_phi  # GeV
    # lambda_HHpHm = -1 / v * (2 * m_Hpm**2 - m_H**2) * cosba

    # 计算 xi_bs_phi
    xi_bs_phi = 0.0
    for k in quarks:
        if k == 'u':
            Vkb = V_ub
            Vks = V_us
        elif k == 'c':
            Vkb = V_cb
            Vks = V_cs
        elif k == 't':
            Vkb = V_tb
            Vks = V_ts
        m_k = masses[k]
        xk = (m_k**2) / (m_W**2)
        
        # calcu g0, g1, g2
        try:
            g_1 = g1(xk, x_Hpm, tanb)
            g_2 = g2(xk, x_Hpm, tanb)
            g_0 = g0(xk, x_Hpm, tanb)
        except Exception as e:
            print(f"Error computing g functions for k={k}: {e}")
            g_1 = g_2 = g_0 = 0
        
        sinba = np.sqrt(1 - cosba**2)
        # compute xi_bs_phi
        term = g_1 * (cosba) - \
               g_2 * (sinba) - \
               g_0 * (calcu_lambda_HHpHm(m_H, m_Hpm, tanb, cosba)) * (2 * v / (m_W**2))
        # print(f"g0: {g_0}, g1: {g_1}, g2: {g_2}")
        xi_bs_phi += Vkb.conjugate() * Vks * (m_k**2) * term
        # print(f"xi_bs_phi for k={k}: {Vkb.conjugate() * Vks * (m_k**2) * term}")

    # 计算 xi_bs_phi 的模方
    xi_bs_phi_sq = np.abs(xi_bs_phi * (-4 * G_F * 1.414/(16 * 3.14**2)))**2

    # 计算 Γ(b -> s phi) / Γ(b -> c e nu)
    t1 = 12 * np.pi**2 * v **2 / (m_b**2)
    t2 = (1 - (m_phi**2) / (m_b**2))**2
    t3 = 1/0.506
    t4 = np.square(xi_bs_phi * (-4 * G_F * np.sqrt(2)/(16 * 3.14**2))/V_cb)
    # print(f"t1: {t1}, t2: {t2}, t3: {t3}, t4: {t4}")
    return t1 * t2 * t3 * t4 * 0.104

# # 示例计算
# if __name__ == "__main__":
#     m_phi = 1.0  # GeV，假设CP-even标量的质量为1 GeV
#     br = Br_B_to_H(m_phi)
#     print(f"Br(B -> H) = {br:.2e}")



def calculate_xi_bs_simple(tanb, m_Hpm, m_t=173, m_W=80.4, V_tb=1.0, V_ts=0.04):
    v = 246  # GeV
    G_F = 1.1663787e-5  # GeV^-2
    m_W2 = m_W**2
    x_t = (m_t**2) / (m_W2**2)
    x_Hpm = (m_Hpm**2) / (m_W2**2)
    cot_beta = 1 / tanb
    cos_ba = 1/tanb
    sin_ba = np.sqrt(1 - cos_ba**2)
    # only consider the t quark contribution
    g1_val = g1(x_t, x_Hpm, tanb)  
    g2_val = g2(x_t, x_Hpm, tanb)   
    
    # ignore g0
    xi_bs = - ((4 * G_F * np.sqrt(2)) / (16 * np.pi**2)) * V_tb * (m_t**2) * (g1_val * cos_ba - g2_val * sin_ba) * V_ts
    
    return xi_bs

def calcu_Br_B_H_2HDM_I_simple(m_phi, tanb, m_Hpm=600.0):
    t1 = 12 * np.pi**2 * (v**2) / (m_b**2)
    t2 = (1 - (m_phi**2) / (m_b**2))**2
    t3 = 1/0.506
    t4 = np.square(calculate_xi_bs_simple(tanb, m_Hpm)/V_cb)
    return t1 * t2 * t3 * t4 * 0.104


# calculate_Br(0.1, 6*10**(-8), 0.1, 0.65)
# The calculation formula is in "Searching for Long-lived Particles: A Compact Detector for Exotics at LHCb
#By Vladimir V. Gligorov,1 Simon Knapen,2, 3 Michele Papucci,2, 3 and Dean J. Robinson4"


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
    return pd.DataFrame({'r': r, 'x': x, 'y': y, 'z': z})
    # return pd.DataFrame({'r': [r], 'x': [x], 'y': [y], 'z': [z]}, index=[0])


# print(calculate_decay_position(-1123, -2123, -314, 141, 2124, 0, 0, 0))

def whether_in_the_detector_by_position(x, y, z, detector_xmin=26000, detector_xmax=36000,
                                        detector_ymin=-7000, detector_ymax=3000,
                                        detector_zmin=5000, detector_zmax=15000):
    in_detector = (
        (x >= detector_xmin) & (x <= detector_xmax) &
        (y >= detector_ymin) & (y <= detector_ymax) &
        (z >= detector_zmin) & (z <= detector_zmax)
    )
    return in_detector.astype(int)#0 is not in detector, 1 is in the detector

def whether_in_the_detector_by_r(x, y, z, detector_xmin=26000, detector_xmax=36000,
                                        detector_ymin=-7000, detector_ymax=3000,
                                        detector_zmin=5000, detector_zmax=15000):
    r_min = np.sqrt(0 + np.square(detector_zmin) + np.square(detector_xmin))
    r_max = np.sqrt(np.square(detector_ymin) + np.square(detector_xmax) + np.square(detector_zmax))
    r = np.sqrt(x**2 + y**2 + z**2)
    in_detector = (
        (r <= r_max) & (r >= r_min)
    )
    return in_detector.astype(int)#0 is not in detector, 1 is in the detector

def SHiP(x, y, z):
    point = [x, y, z]
    result = judge.is_point_in_polyhedron(point, vertices, faces)
    return result[1]