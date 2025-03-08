import pandas as pd
import numpy as np
def see_ctau_2(csv_file):
    df = pd.read_csv(csv_file)
    i = 0
    j = 0
    for tau_in, tau in zip(df['tau_input'], df['tau']):
        if(tau_in < tau):
            print("input is smaller")
            i = i + 1
            
        else:
            j = j + 1    
    return i, j

def print_max_min_log10_sin_theta_2(filename):
    df = pd.read_csv(filename)
    theta = df['theta_input']
    sin_square_theta = np.sin(theta)**2
    max_ = np.log(max(sin_square_theta))/np.log(10)
    min_ = np.log(min(sin_square_theta))/np.log(10)
    return max_, min_
    