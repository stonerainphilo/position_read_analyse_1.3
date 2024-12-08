import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_dataframe_single_mass(dataframe):
    df = dataframe
    mH = df['mH'] 
    tau_ln = np.log(df['ltime'])
    theta_log10 = np.log(df['theta'])/np.log(10)
    return mH, tau_ln, theta_log10
