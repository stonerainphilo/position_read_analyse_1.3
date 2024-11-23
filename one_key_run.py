import os
from tqdm import tqdm

from run_save import run_save_main41_csv, add_whether_in_the_detector, add_whether_in_the_detector_without_angle
from combine import combine_files_precise, combine_files_precise_r
from loop import loop_ctau_br, loop_ctau_br_certain_seed, loop_mass_ctau, loop_mass_ctau_certain_seed, loop_mass_ctau_given_by_csv
from functions_for_run import mkdir_1
from multiprocessing import Pool

from run_save import add_whether_in_the_detector_without_Decay_calcu, add_whether_in_the_detector_without_angle_without_Decay_calcu


def detect_folder_files(LLP_data_folder_dir):
    # out_put_path = os.path.dirname(LLP_data_folder_dir) + '/detected_llp_data'
    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        if os.path.isfile(file_path_all):
            try:
                completed_data_folder = add_whether_in_the_detector(file_path_all, LLP_data_folder_dir)
            except Exception as e:
                print(f"Error with file: {file_path_all}")
                print(f"Error message: {str(e)}")
                continue
                file_path_all = ''

    return 'Detection Completed', completed_data_folder

def detect_folder_files_no_calcu(LLP_data_folder_dir):
    # out_put_path = os.path.dirname(LLP_data_folder_dir) + '/detected_llp_data'
    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        if os.path.isfile(file_path_all):
            try:
                completed_data_folder = add_whether_in_the_detector_without_Decay_calcu(file_path_all, LLP_data_folder_dir)
            except Exception as e:
                print(f"Error with file: {file_path_all}")
                print(f"Error message: {str(e)}")
                continue
                file_path_all = ''

    return 'Detection Completed', completed_data_folder

def detect_folder_files_r(LLP_data_folder_dir):
    # out_put_path = os.path.dirname(LLP_data_folder_dir) + '/detected_llp_data'
    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        # print(file_path_all)
        if os.path.isfile(file_path_all):
            
            completed_data_folder = add_whether_in_the_detector_without_angle(file_path_all, LLP_data_folder_dir)
                        
    return 'Detection Completed', completed_data_folder

def detect_folder_files_r_no_calcu(LLP_data_folder_dir):
    # out_put_path = os.path.dirname(LLP_data_folder_dir) + '/detected_llp_data'
    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        # print(file_path_all)
        if os.path.isfile(file_path_all):
            
            completed_data_folder = add_whether_in_the_detector_without_angle_without_Decay_calcu(file_path_all, LLP_data_folder_dir)
                        
    return 'Detection Completed', completed_data_folder


def one_key_run_br_ctau(br_lower_lim, br_upper_lim, br_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 mass, seed_array_length, out_put_path, main41_path):
    LLP_data_path = loop_ctau_br(br_lower_lim, br_upper_lim, br_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 mass, seed_array_length, out_put_path, main41_path)
    # run main41.cc with ctau, br, mass seed
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files(LLP_data_path)[1]
    # Find the LLPs can be detected or not
    print('The LLPs are Judged whether they are Detected or not')
    final_files = combine_files_precise(os.path.dirname(completed_data_dir))
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files


def one_key_run_br_ctau_certain_seed(br_lower_lim, br_upper_lim, br_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 mass, seed_array_length, out_put_path, main41_path):
    LLP_data_path = loop_ctau_br_certain_seed(br_lower_lim, br_upper_lim, br_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 mass, seed_array_length, out_put_path, main41_path)
    # run main41.cc with ctau, br, mass seed
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files(LLP_data_path)[1]
    # Find the LLPs can be detected or not
    print('The LLPs are Judged whether they are Detected or not')
    final_files = combine_files_precise(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_mass_ctau(mass_lower_lim, mass_upper_lim, mass_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 br, seed_array_length, out_put_path, main41_path):
    LLP_data_path = loop_mass_ctau(mass_lower_lim, mass_upper_lim, mass_array_length, 
                ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                br, seed_array_length, out_put_path, main41_path)
    completed_data_dir = detect_folder_files(LLP_data_path)[1]
    final_files = combine_files_precise(completed_data_dir)
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_mass_ctau_simple_ver(mass_lower_lim, mass_upper_lim, mass_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 br, seed_array_length, out_put_path, main41_path):
    LLP_data_path = loop_mass_ctau(mass_lower_lim, mass_upper_lim, mass_array_length, 
                ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                br, seed_array_length, out_put_path, main41_path)
    completed_data_dir = detect_folder_files_no_calcu(LLP_data_path)[1]
    final_files = combine_files_precise(completed_data_dir)
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_mass_ctau_certain_seed(mass_lower_lim, mass_upper_lim, mass_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 br, seed_array, out_put_path, main41_path):
    LLP_data_path = loop_mass_ctau_certain_seed(mass_lower_lim, mass_upper_lim, mass_array_length, 
                ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                br, seed_array, out_put_path, main41_path)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not')
    final_files = combine_files_precise(os.path.dirname(completed_data_dir))
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files[0]

def one_key_run_mass_ctau_given_by_csv(csv_file, br, seed_array, out_put_path, main41_path):
    LLP_data_path = loop_mass_ctau_given_by_csv(csv_file, br, seed_array, out_put_path, main41_path)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_no_calcu(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not')
    final_files = combine_files_precise(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

