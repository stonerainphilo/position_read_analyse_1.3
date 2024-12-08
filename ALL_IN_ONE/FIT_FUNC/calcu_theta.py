import numpy as np
import pandas as pd
from Basic_calcu import calc_weight
def closest_two_mass_list(mass = 0.0, df = []):
    
    abs_min = (df['mH'] - mass).abs()
    sorted_abs_min = abs_min.sort_values()
    two_closest_indices = sorted_abs_min.index[:2]
    closest_two_mass = df['mH'][two_closest_indices]
    flag = 1 # sign for whether mass is in the data
    if closest_two_mass.values[0] == closest_two_mass.values[1]:
        for i in range(2, len(sorted_abs_min)):
            next_closest_index = sorted_abs_min.index[i]
            if df['mH'][next_closest_index] != mass:
                closest_two_mass = [closest_two_mass.values[0], df['mH'][next_closest_index]]
                break
        flag = 0
    # print(type(closest_two_mass))
    closest_two_mass.insert(2, flag)
    # closest_two_mass = pd.concat([closest_two_mass, pd.Series([flag])])
    # print(closest_two_mass)
    return closest_two_mass

def closest_two_mass_Series(mass = 0.0, df = []):
    
    abs_min = (df['mH'] - mass).abs()
    sorted_abs_min = abs_min.sort_values()
    two_closest_indices = sorted_abs_min.index[:2]
    closest_two_mass = df['mH'][two_closest_indices]
    flag = 1 # sign for whether mass is in the data
    if closest_two_mass.values[0] == closest_two_mass.values[1]:
        for i in range(2, len(sorted_abs_min)):
            next_closest_index = sorted_abs_min.index[i]
            if df['mH'][next_closest_index] != mass:
                closest_two_mass = [closest_two_mass.values[0], df['mH'][next_closest_index]]
                break
        flag = 0
    # print(type(closest_two_mass))
    closest_two_mass = pd.concat([closest_two_mass, pd.Series([flag])])
    # closest_two_mass = pd.concat([closest_two_mass, pd.Series([flag])])
    # print(closest_two_mass)
    return closest_two_mass

def closest_two_tau(tau, mass, df): #df is DATAFRAME of LLP DATA
    if mass not in df['mH'].values:
        print("----------ERROR---------")
        print("ERROR FROM `closest_two_tau` FUNCTION: ")
        print("``ERROR: NO SUCH MASS VALUE IN EXAMPLE DATA``")
        print("----------END---------")
        return (np.nan, np.nan)
    df = df.loc[df['mH'] == mass]
    abs_min = (df['ltime'] - tau).abs()
    sorted_abs_min = abs_min.sort_values()
    two_closest_indices = sorted_abs_min.index[:2]
    closest_two_tau = df.loc[two_closest_indices, 'ltime']
    return closest_two_tau.values[0], closest_two_tau.values[1]

# df = pd.read_csv('All_mass_k_b.csv')
# massss = closest_two(1.025, df)
# print(massss)
# print(massss[0])

def calcu_log_10_theta_by_mass_ln_tau_by_csv(mass, tau_ln, k_b_filename):
    df = pd.read_csv(k_b_filename)
    # print('fine')
    massssss = closest_two_mass_Series(mass, df).values
    # print(type(massssss))
    if massssss[2] == 1:
        df = df.loc[(df['mH'] == massssss[0]) | (df['mH'] == massssss[1])]
        k = df['k']
        b = df['b']
    if massssss[2] == 0:
        df = df.loc[(df['mH'] == massssss[0])]
        k = df['k']
        b = df['b']
    theta_log10 = k * tau_ln + b
    
    return np.average(theta_log10)



def calcu_average_log10_theta(mass, tau, DATA_file):
    df3 = pd.read_csv(DATA_file)
    massssss = closest_two_mass_list(mass, df3)
    closest_two_taus_values_mass_1 = closest_two_tau(tau, massssss[0], df3)
    closest_two_taus_values_mass_2 = closest_two_tau(tau, massssss[1], df3)
    if [(mass == massssss[0]) | (mass == massssss[1])]:
        a = (np.log(df3['theta'].loc[(df3['mH'] == massssss[0]) & (df3['ltime'] == closest_two_taus_values_mass_1[0])]))/np.log(10)
        a_ = (np.log(df3['theta'].loc[(df3['mH'] == massssss[0]) & (df3['ltime'] == closest_two_taus_values_mass_1[1])]))/np.log(10)
        # print('mass is listed')
        weights = calc_weight(tau, closest_two_taus_values_mass_1[0], closest_two_taus_values_mass_1[1])
        ave = a.values[0]*weights[0] + a_.values[0]*weights[1]
        return ave
    if [(mass != massssss[0]) & (mass != massssss[1])]:
    # print(type(massssss[0]))
        a = (np.log(df3['theta'].loc[(df3['mH'] == massssss[0]) & (df3['ltime'] == closest_two_taus_values_mass_1[0])]))/np.log(10)
        a_ = (np.log(df3['theta'].loc[(df3['mH'] == massssss[0]) & (df3['ltime'] == closest_two_taus_values_mass_1[1])]))/np.log(10)
        b = (np.log(df3['theta'].loc[(df3['mH'] == massssss[1]) & (df3['ltime'] == closest_two_taus_values_mass_2[0])]))/np.log(10)
        b_ = (np.log(df3['theta'].loc[(df3['mH'] == massssss[1]) & (df3['ltime'] == closest_two_taus_values_mass_2[1])]))/np.log(10)
        # print(a)
        # print(b)
        c = a.values[0]
        d = b.values[0]
        c_ = a_.values[0]
        d_ = b_.values[0]
        weights_a = calc_weight(tau, closest_two_taus_values_mass_1[0], closest_two_taus_values_mass_1[1])
        ave_a = a.values[0]*weights_a[0] + a_.values[0]*weights_a[1]
        weights_b = calc_weight(tau, closest_two_taus_values_mass_2[0], closest_two_taus_values_mass_2[1])
        ave_b = b.values[0]*weights_b[0] + b_.values[0]*weights_b[1]
        weights_mass_a_b = calc_weight(mass, massssss[0], massssss[1])
        ave = ave_a*weights_mass_a_b[0] + ave_b*weights_mass_a_b[1]
        return ave
# df = pd.read_csv('all_DATA_H_rough.csv')
# df2 = pd.read_csv('All_mass_k_b_rough.csv')
# df3 = pd.read_csv('all_DATA_H.csv')
# tau = 1e5
# mass = 2.05
# print(calcu_log_10_theta_by_mass_ln_tau_by_csv(mass, np.log(tau), 'All_mass_k_b.csv'))
# # print((np.log(df['theta'].loc[(df['mH'] == mass) & (df['ltime'] == tau)]))/np.log(10))
# # print(closest_two_tau(tau, mass, df))
# # print(closest_two_tau(tau, 1.05, df))
# # print((np.log(df3['theta'].loc[(df3['mH'] == 1.05) & (df3['ltime'] == closest_two_tau(tau, 1.05, df3)[0])]))/np.log(10))
# # print((np.log(df3['theta'].loc[(df3['mH'] == 1.05) & (df3['ltime'] == closest_two_tau(tau, 1.05, df3)[1])]))/np.log(10))
# print(calcu_average_log10_theta(mass, tau, 'all_DATA_H.csv'))