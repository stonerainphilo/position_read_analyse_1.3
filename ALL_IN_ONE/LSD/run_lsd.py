from datetime import datetime
from Pythia8.functions_for_run import mkdir_1
import subprocess
import os
def run_T1Aa_csv(m, seed, Br, tau, out_path, T1Aa_path):
    
    # today = str(datetime.now().date())
    out_dir = out_path +'/' + 'LLP_Theory_data/'

    # change path
    os.chdir(T1Aa_path)
    # command0 = f'make main41 -j4'
    # process0 = subprocess.Popen(command0, stdout = subprocess.PIPE, shell = True)
    # out_0, err_0 = process0.communicate()
    # print(out_0, err_0)
    # print('command 0 complated')
    mkdir_1(out_dir)
    command1 = f'./T1Aa.x {out_path}' # Yet to add out path to T1Aa.x
    process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    # get error if any
    output, error = process1.communicate()
    
    # save to txt
    out_path = out_dir+"all_DATA_H.csv"
    
    return out_path, out_dir