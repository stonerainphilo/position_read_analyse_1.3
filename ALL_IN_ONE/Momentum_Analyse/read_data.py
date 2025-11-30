import pandas as pd
import numpy as np

def read_meson_data(file_path_521, file_path_511):
    df_521 = pd.read_csv(file_path_521)
    df_511 = pd.read_csv(file_path_511)

    meson_data = pd.DataFrame({
        'px_521': df_521['px_521'],
        'py_521': df_521['py_521'], 
        'pz_521': df_521['pz_521'],
        'e_521': df_521['e_521'],
        'px_511': df_511['px_511'],
        'py_511': df_511['py_511'],
        'pz_511': df_511['pz_511'],
        'e_511': df_511['e_511'],
        'm_521': df_521['m'],
        'm_511': df_511['m'],
        'dpx_521': df_521['decay_pos_x_521'],
        'dpx_511': df_511['decay_pos_x_511'],
        'dpy_521': df_521['decay_pos_y_521'],
        'dpy_511': df_511['decay_pos_y_511'],
        'dpz_521': df_521['decay_pos_z_521'],
        'dpz_511': df_511['decay_pos_z_511'],
    })
    
    return meson_data

def read_521(file_path):
    df = pd.read_csv(file_path)
    p = np.array([df['e_521'], df['px_521'], df['py_521'], df['pz_521']])
    decay_pos = np.array([df['decay_pos_x_521'], df['decay_pos_y_521'], df['decay_pos_z_521']])
    m = df['m']
    return p, decay_pos, m

def read_511(file_path):
    df = pd.read_csv(file_path)
    p = np.array([df['e_511'], df['px_511'], df['py_511'], df['pz_511']])
    decay_pos = np.array([df['decay_pos_x_511'], df['decay_pos_y_511'], df['decay_pos_z_511']])
    m = df['m']
    return p, decay_pos, m

# def read_LLP(file_path)

# print(read_meson_data('ALL_IN_ONE/Momentum_Analyse/B_521.csv','ALL_IN_ONE/Momentum_Analyse/B_511.csv')[:5])

def read_2HDM_data(file_path_2HDM):
    df_2HDM = pd.read_csv(file_path_2HDM)
    ctau = df_2HDM['ltime'] # lifetime in mm/c
    tanb = df_2HDM['tanb']
    mA = df_2HDM['mA']  # CP-Odd mass in GeV
    mH = df_2HDM['mH']  # CP-Even mass in GeV
    mHC = df_2HDM['mHC']  # Charged-Higgs in GeV
    decay_width = df_2HDM['Decay_width_total']  # decay width in GeV
    return ctau, mA, mH, mHC, decay_width, tanb

def read_2HDM_data_simple(file_path_2HDM):
    df_2HDM = pd.read_csv(file_path_2HDM)
    tanb = df_2HDM['tanb']
    ctau = df_2HDM['ltime'] # lifetime in mm/c
    mA = df_2HDM['mA']  # CP-Odd mass in GeV
    # mH = df_2HDM['mH']  # CP-Even mass in GeV
    # mHC = df_2HDM['mHC']  # Charged-Higgs in GeV
    decay_width = df_2HDM['Decay_width_total']  # decay width in GeV
    return ctau, mA, decay_width, tanb

