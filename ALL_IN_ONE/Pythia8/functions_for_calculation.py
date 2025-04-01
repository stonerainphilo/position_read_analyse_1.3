import numpy as np
import pandas as pd
L1 = np.sqrt(np.square(5)+np.square(26))
L2 = np.sqrt(15*15+36*36)
L2_precise = np.sqrt(15*15+36*36+7*7)

ctau = 1
beta = 1
gamma = 1
weight_approx = np.exp(-L1/(ctau*beta*gamma))-np.exp(-L2/(ctau*beta*gamma))

def f_function(mc, mb): # The real f_function is not this one
    mc = 1.27
    mb = 4.18
    return mc/mb


def calculate_Sin(mphi, Br1 = 6*10**(-8), Br2 = 0.9, g = 2):
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
    para1 = (256 * np.square(np.pi))/(27 * np.square(g))
    para2 = (np.square(mt)*np.square(mt))/(np.square(mb)*np.square(mw))
    para3 = np.square(1-(mphi/mb))/0.51 # 0.51 is The phase space Factor. CITE: Limits on a light Higgs boson
    para4 = np.square(Vts*Vtb/Vcb)
    Sin_square = (Br1/Br2) * para1 / (para2 * para3 * para4)
    return Sin_square


def calculate_Br(mphi, sin_square_theta = 6*10**(-8), Br2 = 0.9, g = 2):
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
    para1 = (256 * np.square(np.pi))/(27 * np.square(g))
    para2 = (np.square(mt)*np.square(mt))/(np.square(mb)*np.square(mw))
    para3 = np.square(1-(mphi/mb))/0.51 # 0.51 is The phase space Factor. CITE: Limits on a light Higgs boson
    para4 = np.square(Vts*Vtb/Vcb)
    Sin_square = (sin_square_theta*Br2) * para1 * (para2 * para3 * para4)
    return Sin_square

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