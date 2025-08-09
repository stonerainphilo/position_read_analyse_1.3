import subprocess
from datetime import datetime
import os
import subprocess
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import random
from tqdm import tqdm
import glob
from mpl_toolkits.mplot3d import Axes3D
from functions_for_run import mkdir_1
import re
from functions_for_read import get_llp
from functions_for_calculation import calculate_decay_position, whether_in_the_detector_by_position, whether_in_the_detector_by_r
import sys
from cross_section import calculate_cross_section, counting_total_LLP
import DETECTOR as dt


now = datetime.now()

def run_save_main41_txt(m, seed, Br, tau, out_path, main_41_path):
    
    mass = str(m)
    Br_str = str(Br)
    today = str(datetime.now().date())
    tau_str = str(tau)
    out_dir = out_path + today +'/' + 'LLP_data/'
    random_seed = str(seed)
    mass_line = 'the mass = '
    seed_line = 'the seed is: '
    Br_line= 'the Br = '
    ctau_line = 'the ctau = '
    # change path
    os.chdir(main_41_path)
    # command0 = f'make main41 -j4'
    # process0 = subprocess.Popen(command0, stdout = subprocess.PIPE, shell = True)
    # out_0, err_0 = process0.communicate()
    # print(out_0, err_0)
    # print('command 0 complated')
    mkdir_1(out_dir)
    command1 = f'./main41 {mass} {tau_str} {Br_str} {random_seed} {out_dir}'
    process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    filename = "filtered_mass_" + mass + "_ctau_" + tau_str + "_br_" + Br_str + "_seed_" + random_seed + ".csv"
    # get error if any
    output, error = process1.communicate()
    
    # save to txt
    out_path = out_dir+filename
    # with open(out_path, 'w') as file:
    #     file.write(seed_line + random_seed + '\n')
    #     file.write(mass_line + mass + '\n')
    #     file.write(Br_line + str(Br)+'\n')
    #     file.write(ctau_line + str(tau))
    #     file.write(output.decode('utf-8'))

    
        

    # Save error if any
    # if error:
    #     with open('error_'+filename, 'w') as file:
    #         file.write(error.decode('utf-8'))
    
    return out_path, mass, seed, out_dir, tau_str, Br_str

def run_save_main41_csv(m, seed, Br, tau, out_path, main_41_path):
    
    mass = str(m)
    Br_str = str(Br)
    today = str(datetime.now().date())
    tau_str = str(tau)
    out_dir = out_path + today +'/' + 'LLP_data/'
    random_seed = str(seed)
    mass_line = 'the mass = '
    seed_line = 'the seed is: '
    Br_line= 'the Br = '
    ctau_line = 'the ctau = '
    # change path
    os.chdir(main_41_path)
    # command0 = f'make main41 -j4'
    # process0 = subprocess.Popen(command0, stdout = subprocess.PIPE, shell = True)
    # out_0, err_0 = process0.communicate()
    # print(out_0, err_0)
    # print('command 0 complated')
    mkdir_1(out_dir)
    command1 = f'./main41 {mass} {tau_str} {Br_str} {random_seed} {out_dir}'
    process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    filename = "filtered_mass_" + mass + "_ctau_" + tau_str + "_br_" + Br_str + "_seed_" + random_seed + ".csv"
    # get error if any
    output, error = process1.communicate()
    
    # save to txt
    out_path = out_dir+filename
    # with open(out_path, 'w') as file:
    #     file.write(seed_line + random_seed + '\n')
    #     file.write(mass_line + mass + '\n')
    #     file.write(Br_line + str(Br)+'\n')
    #     file.write(ctau_line + str(tau))
    #     file.write(output.decode('utf-8'))

    
        

    # Save error if any
    # if error:
    #     with open('error_'+filename, 'w') as file:
    #         file.write(error.decode('utf-8'))
    
    return out_path, out_dir, mass, seed, tau_str, Br_str

def run_save_main131_csv(m, seed, Br, tau, out_path, main_41_path):
    
    mass = str(m)
    Br_str = str(Br)
    today = str(datetime.now().date())
    tau_str = str(tau)
    out_dir = out_path + today +'/' + 'LLP_data/'
    random_seed = str(seed)
    mass_line = 'the mass = '
    seed_line = 'the seed is: '
    Br_line= 'the Br = '
    ctau_line = 'the ctau = '
    # change path
    os.chdir(main_41_path)
    # command0 = f'make main41 -j4'
    # process0 = subprocess.Popen(command0, stdout = subprocess.PIPE, shell = True)
    # out_0, err_0 = process0.communicate()
    # print(out_0, err_0)
    # print('command 0 complated')
    mkdir_1(out_dir)
    command1 = f'./main131 {mass} {tau_str} {Br_str} {random_seed} {out_dir}'
    process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    filename = "filtered_mass_" + mass + "_ctau_" + tau_str + "_br_" + Br_str + "_seed_" + random_seed + ".csv"
    # get error if any
    output, error = process1.communicate()
    
    # save to txt
    out_path = out_dir+filename
    # with open(out_path, 'w') as file:
    #     file.write(seed_line + random_seed + '\n')
    #     file.write(mass_line + mass + '\n')
    #     file.write(Br_line + str(Br)+'\n')
    #     file.write(ctau_line + str(tau))
    #     file.write(output.decode('utf-8'))

    
        

    # Save error if any
    # if error:
    #     with open('error_'+filename, 'w') as file:
    #         file.write(error.decode('utf-8'))
    
    return out_path, out_dir, mass, seed, tau_str, Br_str
    
def run_save_main41_csv_all_br(m, seed, Br, tau, out_path, main_41_path, 
                               Br_Hee, Br_HKK, Br_HPIPI, Br_Htautau, Br_HGluon,
                               Br_Hmumu, Br_Hgaga, Br_H4Pi, Br_Hss, Br_Hcc, theta, Decay_width_total):
    
    mass = str(m)
    Br_str = str(Br)
    today = str(datetime.now().date())
    tau_str = str(tau)
    out_dir = out_path + today +'/' + 'LLP_data/'
    random_seed = str(seed)
    
    mass_line = 'the mass = '
    seed_line = 'the seed is: '
    Br_line= 'the Br = '
    ctau_line = 'the ctau = '
    # change path
    os.chdir(main_41_path)
    # command0 = f'make main41 -j4'
    # process0 = subprocess.Popen(command0, stdout = subprocess.PIPE, shell = True)
    # out_0, err_0 = process0.communicate()
    # print(out_0, err_0)
    # print('command 0 complated')
    mkdir_1(out_dir)
    command1 = f'./main41 {mass} {tau_str} {Br_str} {random_seed} {out_dir} {Br_Hee} {Br_HKK} {Br_HPIPI} {Br_Htautau} {Br_HGluon} {Br_Hmumu} {Br_Hgaga} {Br_H4Pi} {Br_Hss} {Br_Hcc} {theta} {Decay_width_total}'
                               
    process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    filename = "filtered_mass_" + mass + "_ctau_" + tau_str + "_br_" + Br_str + "_seed_" + random_seed + ".csv"
    # get error if any
    output, error = process1.communicate()
    
    # save to txt
    out_path = out_dir+filename
    # with open(out_path, 'w') as file:
    #     file.write(seed_line + random_seed + '\n')
    #     file.write(mass_line + mass + '\n')
    #     file.write(Br_line + str(Br)+'\n')
    #     file.write(ctau_line + str(tau))
    #     file.write(output.decode('utf-8'))

    
        

    # Save error if any
    # if error:
    #     with open('error_'+filename, 'w') as file:
    #         file.write(error.decode('utf-8'))
    
    return out_path, out_dir, mass, seed, tau_str, Br_str    

def run_save_main131_csv_all_br_main131(m, seed, Br, tau, out_path, main_41_path, 
                               Br_Hee, Br_HKK, Br_HPIPI, Br_Htautau, Br_HGluon,
                               Br_Hmumu, Br_Hgaga, Br_H4Pi, Br_Hss, Br_Hcc, 
                               theta, Decay_width_total, today):
    
    mass = str(m)
    Br_str = str(Br)
    tau_str = str(tau)
    out_dir = out_path + today +'/' + 'LLP_data/'
    random_seed = str(seed)
    
    mass_line = 'the mass = '
    seed_line = 'the seed is: '
    Br_line= 'the Br = '
    ctau_line = 'the ctau = '
    # change path
    os.chdir(main_41_path)
    # command0 = f'make main131 -j4'
    # process0 = subprocess.Popen(command0, stdout = subprocess.PIPE, shell = True)
    # out_0, err_0 = process0.communicate()
    # print(out_0, err_0)
    # print('command 0 complated')
    mkdir_1(out_dir)
    command1 = f'./main131 {mass} {tau_str} {Br_str} {random_seed} {out_dir} {Br_Hee} {Br_HKK} {Br_HPIPI} {Br_Htautau} {Br_HGluon} {Br_Hmumu} {Br_Hgaga} {Br_H4Pi} {Br_Hss} {Br_Hcc} {theta} {Decay_width_total}'
                               
    process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    filename = "filtered_mass_" + mass + "_ctau_" + tau_str + "_br_" + Br_str + "_seed_" + random_seed + ".csv"
    # get error if any
    output, error = process1.communicate()
    
    # save to txt
    out_path = out_dir+filename
    # with open(out_path, 'w') as file:
    #     file.write(seed_line + random_seed + '\n')
    #     file.write(mass_line + mass + '\n')
    #     file.write(Br_line + str(Br)+'\n')
    #     file.write(ctau_line + str(tau))
    #     file.write(output.decode('utf-8'))

    
        

    # Save error if any
    # if error:
    #     with open('error_'+filename, 'w') as file:
    #         file.write(error.decode('utf-8'))
    
    return out_path, out_dir, mass, seed, tau_str, Br_str   

def run_save_main131_simple(m, seed, Br, tau, out_path, main_41_path, 
                               theta, today):
    
    mass = str(m)
    Br_str = str(Br)
    tau_str = str(tau)
    out_dir = out_path + today +'/' + 'LLP_data/'
    random_seed = str(seed)
    
    mass_line = 'the mass = '
    seed_line = 'the seed is: '
    Br_line= 'the Br = '
    ctau_line = 'the ctau = '
    # change path
    os.chdir(main_41_path)
    # command0 = f'make main131 -j4'
    # process0 = subprocess.Popen(command0, stdout = subprocess.PIPE, shell = True)
    # out_0, err_0 = process0.communicate()
    # print(out_0, err_0)
    # print('command 0 complated')
    mkdir_1(out_dir)
    command1 = f'./main131 {mass} {tau_str} {Br_str} {random_seed} {out_dir} {theta}'
                               
    process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    filename = "filtered_mass_" + mass + "_ctau_" + tau_str + "_br_" + Br_str + "_seed_" + random_seed + ".csv"
    # get error if any
    output, error = process1.communicate()
    
    # save to txt
    out_path = out_dir+filename
    # with open(out_path, 'w') as file:
    #     file.write(seed_line + random_seed + '\n')
    #     file.write(mass_line + mass + '\n')
    #     file.write(Br_line + str(Br)+'\n')
    #     file.write(ctau_line + str(tau))
    #     file.write(output.decode('utf-8'))

    
        

    # Save error if any
    # if error:
    #     with open('error_'+filename, 'w') as file:
    #         file.write(error.decode('utf-8'))
    
    return out_path, out_dir, mass, seed, tau_str, Br_str   


def run_save_main131_simple_2HDM(m, seed, Br, tau, out_path, main_41_path, 
                               theta, today):
    
    mass = str(m)
    Br_str = str(Br)
    tau_str = str(tau)
    out_dir = out_path + today +'/' + 'LLP_data/'
    random_seed = str(seed)
    
    mass_line = 'the mass = '
    seed_line = 'the seed is: '
    Br_line= 'the Br = '
    ctau_line = 'the ctau = '
    # change path
    os.chdir(main_41_path)

    mkdir_1(out_dir)
    command1 = f'./main131_B_2HDM {mass} {tau_str} {Br_str} {random_seed} {out_dir} {theta}'
                               
    process1 = subprocess.Popen(command1, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    filename = "filtered_mass_" + mass + "_ctau_" + tau_str + "_br_" + Br_str + "_seed_" + random_seed + ".csv"
    # get error if any
    output, error = process1.communicate()
    
    # save to txt
    out_path = out_dir+filename

    return out_path, out_dir, mass, seed, tau_str, Br_str   


def run_save_main131_csv_all_br_2HDM_BKD(m, seed, Br, tau, out_path, main_41_path, 
                               Br_Hee, Br_HKK, Br_HPIPI, Br_Htautau, Br_HGluon,
                               Br_Hmumu, Br_Hgaga, Br_H4Pi, Br_Hss, Br_Hcc, 
                               tanb, Decay_width_total, today):
    
    mass = str(m)
    Br_str = str(Br)
    tau_str = str(tau)
    out_dir = out_path + today +'/' + 'LLP_data/'
    random_seed = str(seed)
    
    mass_line = 'the mass = '
    seed_line = 'the seed is: '
    Br_line= 'the Br = '
    ctau_line = 'the ctau = '
    # change path
    os.chdir(main_41_path)
    # command0 = f'make main131 -j4'
    # process0 = subprocess.Popen(command0, stdout = subprocess.PIPE, shell = True)
    # out_0, err_0 = process0.communicate()
    # print(out_0, err_0)
    # print('command 0 complated')

    commandB = f'./main131_B_2HDM {mass} {tau_str} {Br_str} {random_seed} {out_dir} {Br_Hee} {Br_HKK} {Br_HPIPI} {Br_Htautau} {Br_HGluon} {Br_Hmumu} {Br_Hgaga} {Br_H4Pi} {Br_Hss} {Br_Hcc} {tanb} {Decay_width_total}'
    commandK = f'./main131_K_2HDM {mass} {tau_str} {Br_str} {random_seed} {out_dir} {Br_Hee} {Br_HKK} {Br_HPIPI} {Br_Htautau} {Br_HGluon} {Br_Hmumu} {Br_Hgaga} {Br_H4Pi} {Br_Hss} {Br_Hcc} {tanb} {Decay_width_total}'
    commandD = f'./main131_D_2HDM {mass} {tau_str} {Br_str} {random_seed} {out_dir} {Br_Hee} {Br_HKK} {Br_HPIPI} {Br_Htautau} {Br_HGluon} {Br_Hmumu} {Br_Hgaga} {Br_H4Pi} {Br_Hss} {Br_Hcc} {tanb} {Decay_width_total}'
    
    processB = subprocess.Popen(commandB, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    processK = subprocess.Popen(commandK, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    processD = subprocess.Popen(commandD, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    filename = "filtered_mass_" + mass + "_ctau_" + tau_str + "_br_" + Br_str + "_seed_" + random_seed + ".csv"
    # get error if any
    output, error = processB.communicate()
    output, error = processK.communicate()
    output, error = processD.communicate()
    # save to txt
    out_path = out_dir+filename
    # with open(out_path, 'w') as file:
    #     file.write(seed_line + random_seed + '\n')
    #     file.write(mass_line + mass + '\n')
    #     file.write(Br_line + str(Br)+'\n')
    #     file.write(ctau_line + str(tau))
    #     file.write(output.decode('utf-8'))

    
        

    # Save error if any
    # if error:
    #     with open('error_'+filename, 'w') as file:
    #         file.write(error.decode('utf-8'))
    
    return out_path, out_dir, mass, seed, tau_str, Br_str


def run_save_main131_csv_all_br_2HDM_B(m, seed, Br, tau, out_path, main_41_path, 
                               Br_Hee, Br_HKK, Br_HPIPI, Br_Htautau, Br_HGluon,
                               Br_Hmumu, Br_Hgaga, Br_H4Pi, Br_Hss, Br_Hcc, 
                               tanb, Decay_width_total, today):
    
    mass = str(m)
    Br_str = str(Br)
    tau_str = str(tau)
    out_dir = out_path + today +'/' + 'LLP_data/'
    random_seed = str(seed)
    
    mass_line = 'the mass = '
    seed_line = 'the seed is: '
    Br_line= 'the Br = '
    ctau_line = 'the ctau = '
    # change path
    os.chdir(main_41_path)
    # command0 = f'make main131 -j4'
    # process0 = subprocess.Popen(command0, stdout = subprocess.PIPE, shell = True)
    # out_0, err_0 = process0.communicate()
    # print(out_0, err_0)
    # print('command 0 complated')

    commandB = f'./main131_B_2HDM {mass} {tau_str} {Br_str} {random_seed} {out_dir} {Br_Hee} {Br_HKK} {Br_HPIPI} {Br_Htautau} {Br_HGluon} {Br_Hmumu} {Br_Hgaga} {Br_H4Pi} {Br_Hss} {Br_Hcc} {tanb} {Decay_width_total}'
    # commandK = f'./main131_K_2HDM {mass} {tau_str} {Br_str} {random_seed} {out_dir} {Br_Hee} {Br_HKK} {Br_HPIPI} {Br_Htautau} {Br_HGluon} {Br_Hmumu} {Br_Hgaga} {Br_H4Pi} {Br_Hss} {Br_Hcc} {tanb} {Decay_width_total}'
    # commandD = f'./main131_D_2HDM {mass} {tau_str} {Br_str} {random_seed} {out_dir} {Br_Hee} {Br_HKK} {Br_HPIPI} {Br_Htautau} {Br_HGluon} {Br_Hmumu} {Br_Hgaga} {Br_H4Pi} {Br_Hss} {Br_Hcc} {tanb} {Decay_width_total}'
    
    processB = subprocess.Popen(commandB, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # processK = subprocess.Popen(commandK, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # processD = subprocess.Popen(commandD, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    filename = "filtered_mass_" + mass + "_ctau_" + tau_str + "_br_" + Br_str + "_seed_" + random_seed + ".csv"
    # get error if any
    output, error = processB.communicate()
    # output, error = processK.communicate()
    # output, error = processD.communicate()
    # save to txt
    out_path = out_dir+filename
    
    return out_path, out_dir, mass, seed, tau_str, Br_str


def main131_2HDM_A(m, seed, Br, tau, out_path, main_41_path, 
                               tanb, Decay_width_total, today):
    
    mass = str(m)
    Br_str = str(Br)
    tau_str = str(tau)
    out_dir = out_path + today +'/' + 'LLP_data/'
    random_seed = str(seed)
    
    mass_line = 'the mass = '
    seed_line = 'the seed is: '
    Br_line= 'the Br = '
    ctau_line = 'the ctau = '
    # change path
    os.chdir(main_41_path)
    # command0 = f'make main131 -j4'
    # process0 = subprocess.Popen(command0, stdout = subprocess.PIPE, shell = True)
    # out_0, err_0 = process0.communicate()
    # print(out_0, err_0)
    # print('command 0 complated')

    commandB = f'./main131_B_2HDM_A {mass} {tau_str} {Br_str} {random_seed} {out_dir} {tanb} {Decay_width_total}'
    # commandK = f'./main131_K_2HDM {mass} {tau_str} {Br_str} {random_seed} {out_dir} {Br_Hee} {Br_HKK} {Br_HPIPI} {Br_Htautau} {Br_HGluon} {Br_Hmumu} {Br_Hgaga} {Br_H4Pi} {Br_Hss} {Br_Hcc} {tanb} {Decay_width_total}'
    # commandD = f'./main131_D_2HDM {mass} {tau_str} {Br_str} {random_seed} {out_dir} {Br_Hee} {Br_HKK} {Br_HPIPI} {Br_Htautau} {Br_HGluon} {Br_Hmumu} {Br_Hgaga} {Br_H4Pi} {Br_Hss} {Br_Hcc} {tanb} {Decay_width_total}'
    # print(commandB)
    processB = subprocess.Popen(commandB, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # processK = subprocess.Popen(commandK, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # processD = subprocess.Popen(commandD, stdout=subprocess.PIPE, stderr = subprocess.DEVNULL, shell=True)
    # print('command 1 complated')
    filename = "filtered_mass_" + mass + "_ctau_" + tau_str + "_br_" + Br_str + "_seed_" + random_seed + ".csv"
    # get error if any
    output, error = processB.communicate()
    # output, error = processK.communicate()
    # output, error = processD.communicate()
    # save to txt
    out_path = out_dir+filename
    
    return out_path, out_dir, mass, seed, tau_str, Br_str




def add_typed_in_data(filename, input_file_folder_path =  '/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/BtoKa/auto_data/test_files/'):
    pattern = r'm_([0-9\.e-]+)_seed_\d+_br_([0-9\.e-]+)_tau_([0-9\.e-]+)_trimed_events([0-9\.e-]+).csv'
    match = re.match(pattern, filename)
    # print(match)
    if match:
        m_value = float(match.group(1))
        br_value = float(match.group(2))
        ctau_type_in_value = float(match.group(3))


        file_path = os.path.join(input_file_folder_path, filename)
        df = pd.read_csv(file_path)


        df['m_typed_in'] = m_value
        df['br'] = br_value
        df['ctau_typed_in'] = ctau_type_in_value
        parent_dir = os.path.dirname(input_file_folder_path)
        processed_folder = os.path.join(parent_dir, 'processed')

        mkdir_1(processed_folder)
        new_file_path = os.path.join(parent_dir+'/processed/', f'processed_{filename}')
        df.to_csv(new_file_path, index=False)
        # print(new_file_path)
        return new_file_path
    else:
        print('The file name format is incorrect for add_typed_in_data function for file:'+'\n' +filename)
        return False


# for files in os.listdir('/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/BtoKa/auto_data/test_files/'):
#     # print(files)
#     add_typed_in_data(files, '/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/BtoKa/auto_data/test_files/')

    
def add_whether_in_the_detector(filename, out_folder_path):
    mkdir_1(out_folder_path)
    file_path_only, file_name_only = os.path.split(filename)
    file_parent_path_only = os.path.dirname(file_path_only)
    # print(file_path_only)
    # add_typed_data = add_typed_in_data(file_name_only, file_path_only)
    # df = pd.read_csv(add_typed_data)
    llp_data = pd.read_csv(filename)
    llp_decay_position = calculate_decay_position(llp_data['p_x'], llp_data['p_y'], llp_data['p_z'], llp_data['m'], llp_data['tau'], llp_data['xProd'], llp_data['yProd'], llp_data['zProd'])
    llp_whether_in_detector = whether_in_the_detector_by_position(llp_decay_position['x'], llp_decay_position['y'], llp_decay_position['z'])
    llp_data['decay_pos_x'] = llp_decay_position['x']
    llp_data['decay_pos_y'] = llp_decay_position['y']
    llp_data['decay_pos_z'] = llp_decay_position['z']
    llp_data['detected'] = llp_whether_in_detector
    final_data_folder = file_parent_path_only + '/Completed_llp_data_precise'
    mkdir_1(final_data_folder)
    final_data_path = os.path.join(final_data_folder + f'/final_data_{file_name_only}')
    llp_data.to_csv(final_data_path, index = False)
    return final_data_folder
        
def add_whether_in_the_detector_without_Decay_calcu(filename, out_folder_path):
    mkdir_1(out_folder_path)
    file_path_only, file_name_only = os.path.split(filename)
    file_parent_path_only = os.path.dirname(file_path_only)
    # print(file_path_only)
    # add_typed_data = add_typed_in_data(file_name_only, file_path_only)
    # df = pd.read_csv(add_typed_data)
    llp_data = pd.read_csv(filename)
    # llp_decay_position = calculate_decay_position(llp_data['p_x'], llp_data['p_y'], llp_data['p_z'], llp_data['m'], llp_data['tau'], llp_data['xProd'], llp_data['yProd'], llp_data['zProd'])
    llp_whether_in_detector = whether_in_the_detector_by_position(llp_data['decay_pos_x'], llp_data['decay_pos_y'], llp_data['decay_pos_z'])
    llp_data['detected'] = llp_whether_in_detector
    final_data_folder = file_parent_path_only + '/Completed_llp_data_precise'
    mkdir_1(final_data_folder)
    final_data_path = os.path.join(final_data_folder + f'/final_data_{file_name_only}')
    llp_data.to_csv(final_data_path, index = False)
    return final_data_folder

def add_whether_in_the_detector_without_angle(filename, out_folder_path_for_final, detector_xmin=26000, detector_xmax=36000,
                                        detector_ymin=-7000, detector_ymax=3000,
                                        detector_zmin=5000, detector_zmax=15000):
    mkdir_1(out_folder_path_for_final)
    file_path_only, file_name_only = os.path.split(filename)
    # print(file_path_only)
    file_parent_path_only = os.path.dirname(file_path_only)
    # add_typed_data = add_typed_in_data(file_name_only, file_path_only)

    # df = pd.read_csv(add_typed_data)
    llp_data = pd.read_csv(filename)
    llp_decay_position = calculate_decay_position(llp_data['p_x'], llp_data['p_y'], llp_data['p_z'], llp_data['m'], llp_data['tau'], llp_data['xProd'], llp_data['yProd'], llp_data['zProd'])
    llp_whether_in_detector = whether_in_the_detector_by_r(llp_decay_position['x'], llp_decay_position['y'], llp_decay_position['z'])
    llp_data['decay_pos_x'] = llp_decay_position['x']
    llp_data['decay_pos_y'] = llp_decay_position['y']
    llp_data['decay_pos_z'] = llp_decay_position['z']
    llp_data['detected'] = llp_whether_in_detector
    final_data_folder = file_parent_path_only + '/Completed_llp_data'
    mkdir_1(final_data_folder)
    final_data_path = os.path.join(final_data_folder + f'/final_data_r_{file_name_only}')
    # print(final_data_path)
    llp_data.to_csv(final_data_path, index = False)
    return final_data_folder


def add_whether_in_the_detector_without_angle_without_Decay_calcu(filename, out_folder_path_for_final, detector_xmin=26000, detector_xmax=36000,
                                        detector_ymin=-7000, detector_ymax=3000,
                                        detector_zmin=5000, detector_zmax=15000):
    mkdir_1(out_folder_path_for_final)
    file_path_only, file_name_only = os.path.split(filename)
    # print(file_path_only)
    file_parent_path_only = os.path.dirname(file_path_only)
    # add_typed_data = add_typed_in_data(file_name_only, file_path_only)

    # df = pd.read_csv(add_typed_data)
    llp_data = pd.read_csv(filename)
    # llp_decay_position = calculate_decay_position(llp_data['p_x'], llp_data['p_y'], llp_data['p_z'], llp_data['m'], llp_data['tau'], llp_data['xProd'], llp_data['yProd'], llp_data['zProd'])
    llp_whether_in_detector = whether_in_the_detector_by_r(llp_data['decay_pos_x'], llp_data['decay_pos_y'], llp_data['decay_pos_z'])
    llp_data['detected'] = llp_whether_in_detector
    final_data_folder = file_parent_path_only + '/Completed_llp_data'
    mkdir_1(final_data_folder)
    final_data_path = os.path.join(final_data_folder + f'/final_data_r_{file_name_only}')
    # print(final_data_path)
    llp_data.to_csv(final_data_path, index = False)
    return final_data_folder



def add_whether_in_the_detector_without_Decay_calcu_add_cross_section(filename, out_folder_path):
    mkdir_1(out_folder_path)
    file_path_only, file_name_only = os.path.split(filename)
    file_parent_path_only = os.path.dirname(file_path_only)
    llp_data = pd.read_csv(filename)
    llp_whether_in_detector = whether_in_the_detector_by_position(llp_data['decay_pos_x'], llp_data['decay_pos_y'], llp_data['decay_pos_z'])
    cross_section = calculate_cross_section(llp_data)
    llp_data['detected'] = llp_whether_in_detector
    llp_data['cross_section'] = cross_section
    llp_data['detector_acceptance'] = sum(llp_data['detected']) / counting_total_LLP(llp_data)
    final_data_folder = file_parent_path_only + '/Completed_llp_data_precise_cross_section'
    mkdir_1(final_data_folder)
    final_data_path = os.path.join(final_data_folder + f'/final_data_cross_section_{file_name_only}')
    llp_data.to_csv(final_data_path, index = False)
    return final_data_folder



def add_whether_in_the_detector_without_Decay_calcu_add_cross_section_CODEX_MATHUSLA(filename, out_folder_path):
    mkdir_1(out_folder_path)
    file_path_only, file_name_only = os.path.split(filename)
    file_parent_path_only = os.path.dirname(file_path_only)
    llp_data = pd.read_csv(filename)
    llp_whether_in_detector = whether_in_the_detector_by_position(llp_data['decay_pos_x'], llp_data['decay_pos_y'], llp_data['decay_pos_z'])
    llp_whether_in_detector_MATHUSLA = whether_in_the_detector_by_position(llp_data['decay_pos_x'], llp_data['decay_pos_y'], llp_data['decay_pos_z'], -100000, 100000, 100000, 125000, 100000, 300000)
    cross_section = calculate_cross_section(llp_data)
    llp_data['detected'] = llp_whether_in_detector
    llp_data['detected_MATHUSLA'] = llp_whether_in_detector_MATHUSLA
    llp_data['cross_section'] = cross_section
    llp_data['detector_acceptance'] = sum(llp_data['detected']) / counting_total_LLP(llp_data)
    llp_data['detector_acceptance_MATHUSLA'] = sum(llp_data['detected_MATHUSLA']) / counting_total_LLP(llp_data)
    final_data_folder = file_parent_path_only + '/Completed_llp_data_precise_cross_section'
    mkdir_1(final_data_folder)
    final_data_path = os.path.join(final_data_folder + f'/final_data_cross_section_{file_name_only}')
    llp_data.to_csv(final_data_path, index = False)
    return final_data_folder

def add_whether_in_the_detector_without_Decay_calcu_add_cross_section_CODEX_MATHUSLA_SHiP(filename, out_folder_path):
    mkdir_1(out_folder_path)
    file_path_only, file_name_only = os.path.split(filename)
    file_parent_path_only = os.path.dirname(file_path_only)
    llp_data = pd.read_csv(filename)
    llp_whether_in_detector_SHiP = llp_data.apply(
        lambda row: dt.SHiP([row['decay_pos_x'], row['decay_pos_y'], row['decay_pos_z']])[0],
        axis=1
    )
    llp_whether_in_detector = whether_in_the_detector_by_position(llp_data['decay_pos_x'], llp_data['decay_pos_y'], llp_data['decay_pos_z'])

    llp_whether_in_detector_MATHUSLA = whether_in_the_detector_by_position(llp_data['decay_pos_x'], llp_data['decay_pos_y'], llp_data['decay_pos_z'], -100000, 100000, 100000, 125000, 100000, 300000)
    cross_section = calculate_cross_section(llp_data)
    llp_data['detected'] = llp_whether_in_detector
    llp_data['detected_MATHUSLA'] = llp_whether_in_detector_MATHUSLA
    llp_data['detected_SHiP'] = llp_whether_in_detector_SHiP
    llp_data['cross_section'] = cross_section
    llp_data['detector_acceptance'] = sum(llp_data['detected']) / counting_total_LLP(llp_data)
    llp_data['detector_acceptance_MATHUSLA'] = sum(llp_data['detected_MATHUSLA']) / counting_total_LLP(llp_data)
    llp_data['detector_acceptance_SHiP'] = sum(llp_data['detected_SHiP']) / counting_total_LLP(llp_data)
    final_data_folder = file_parent_path_only + '/Completed_llp_data_precise_cross_section'
    mkdir_1(final_data_folder)
    final_data_path = os.path.join(final_data_folder + f'/final_data_cross_section_{file_name_only}')
    llp_data.to_csv(final_data_path, index = False)
    return final_data_folder