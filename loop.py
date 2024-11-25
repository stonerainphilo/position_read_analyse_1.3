from functions_for_run import generate_randomseed
import numpy as np
from tqdm import tqdm
from run_save import run_save_main41_csv, run_save_main41_csv_all_br
import pandas as pd
import os

def loop_ctau_br(br_lower_lim, br_upper_lim, br_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, mass, seed_array_length, out_put_path, main41_path):

    total_iterations = seed_array_length * ctau_array_length * br_array_length

    with tqdm(total=total_iterations) as pbar:
        for seed in generate_randomseed(seed_array_length):
            for taus in np.logspace(ctau_lower_lim, ctau_upper_lim, ctau_array_length):
                for br in np.logspace(br_lower_lim, br_upper_lim, br_array_length):
                    out_path_name_LLP_data = run_save_main41_csv(mass, seed, br, taus, out_put_path, main41_path)[0]
                    
                    pbar.update(1)
    out_dir_name = os.path.dirname(out_path_name_LLP_data)
    return out_dir_name


def loop_ctau_br_certain_seed(br_lower_lim, br_upper_lim, br_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, mass, seed_array, out_put_path, main41_path):

    total_iterations = len(seed_array) * ctau_array_length * br_array_length

    with tqdm(total=total_iterations) as pbar:
        for seed in seed_array:
            for taus in np.logspace(ctau_lower_lim, ctau_upper_lim, ctau_array_length):
                for br in np.logspace(br_lower_lim, br_upper_lim, br_array_length):
                    out_path_name_LLP_data = run_save_main41_csv(mass, seed, br, taus, out_put_path, main41_path)[0]
                    
                    pbar.update(1)
    out_dir_name = os.path.dirname(out_path_name_LLP_data)
    return out_dir_name

def loop_mass_ctau(mass_lower_lim, mass_upper_lim, mass_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, br, seed_array_length, out_put_path, main41_path):
    
    total_iterations = seed_array_length * ctau_array_length * mass_array_length

    with tqdm(total=total_iterations) as pbar:
        for seed in generate_randomseed(seed_array_length):
            for taus in np.logspace(ctau_lower_lim, ctau_upper_lim, ctau_array_length):
                for mass in np.linspace(mass_lower_lim, mass_upper_lim, mass_array_length):
                    out_path_name_LLP_data = run_save_main41_csv(mass, seed, br, taus, out_put_path, main41_path)[0]
                    
                    pbar.update(1)
    out_dir_name = os.path.dirname(out_path_name_LLP_data)
    return out_dir_name

def loop_mass_ctau_certain_seed(mass_lower_lim, mass_upper_lim, mass_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, br, seed_array, out_put_path, main41_path):
    total_iterations = len(seed_array) * ctau_array_length * mass_array_length

    with tqdm(total=total_iterations) as pbar:
        for seed in seed_array:
            for taus in np.logspace(ctau_lower_lim, ctau_upper_lim, ctau_array_length):
                for mass in np.linspace(mass_lower_lim, mass_upper_lim, mass_array_length):
                    out_path_name_LLP_data = run_save_main41_csv(mass, seed, br, taus, out_put_path, main41_path)[0]
                    
                    pbar.update(1)
    out_dir_name = os.path.dirname(out_path_name_LLP_data)
    return out_dir_name

def loop_mass_ctau_given_by_csv(csv_file, br, seed_amount, out_put_path, main41_path):
    df = pd.read_csv(csv_file)
    total_iterations = seed_amount * len(df['mH'])
    with tqdm(total=total_iterations) as pbar:
        for seed in generate_randomseed(seed_amount):
            for mH, taus in zip(df['mH'], df['ltime']):
                out_put_name_LLP_data = run_save_main41_csv(mH, seed, br, taus, out_put_path, main41_path)[0]
                
                pbar.update(1)
    out_dir_name = os.path.dirname(out_put_name_LLP_data)
    
    return out_dir_name

def loop_mass_ctau_br_given_by_csv(csv_file, br, seed_amount, out_put_path, main41_path):
    df = pd.read_csv(csv_file)
    total_iterations = seed_amount * len(df['mH'])
    with tqdm(total=total_iterations) as pbar:
        for seed in generate_randomseed(seed_amount):
            for mH, taus, Br_Hee, Br_HKK, Br_HPIPI, Br_Htautau, Br_HGluon,Br_Hmumu, Br_Hgaga, Br_H4Pi, Br_Hss, Br_Hcc, theta in zip(df['mH'], df['ltime'],
                                                                                                                             df['Br_Hee'], df['Br_HKK'], df['Br_HPiPi'], df['Br_Htautau'], df['Br_HGluon'],
                                                                                                                             df['Br_Hmumu'], df['Br_Hgaga'], df['Br_H4Pi'], df['Br_Hss'], df['Br_Hcc'], df['theta']):
                out_put_name_LLP_data = run_save_main41_csv_all_br(mH, seed, br, taus, out_put_path, main41_path, Br_Hee, Br_HKK, Br_HPIPI, Br_Htautau, Br_HGluon,Br_Hmumu, Br_Hgaga, Br_H4Pi, Br_Hss, Br_Hcc, theta)[0]
                
                pbar.update(1)
    out_dir_name = os.path.dirname(out_put_name_LLP_data)
    
    return out_dir_name