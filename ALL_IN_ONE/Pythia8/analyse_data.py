import pandas as pd

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
    