import os
from tqdm import tqdm
import run_save as rs
import loop as lp
import combine as cb
import functions_for_run as ffr
from multiprocessing import Pool
import loop as lp

def detect_folder_files(LLP_data_folder_dir):
    # out_put_path = os.path.dirname(LLP_data_folder_dir) + '/detected_llp_data'
    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        if os.path.isfile(file_path_all):
            try:
                completed_data_folder = rs.add_whether_in_the_detector(file_path_all, LLP_data_folder_dir)
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
                completed_data_folder = rs.add_whether_in_the_detector_without_Decay_calcu(file_path_all, LLP_data_folder_dir)
            except Exception as e:
                print(f"Error with file: {file_path_all}")
                print(f"Error message: {str(e)}")
                continue
                file_path_all = ''

    return 'Detection Completed', completed_data_folder

def detect_folder_files_cross_section(LLP_data_folder_dir):
    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        if os.path.isfile(file_path_all):
            try:
                completed_data_folder = rs.add_whether_in_the_detector_without_Decay_calcu_add_cross_section(file_path_all, LLP_data_folder_dir)
            except Exception as e:
                print(f"Error with file: {file_path_all}")
                print(f"Error message: {str(e)}")
                continue
                file_path_all = ''

    return 'Detection and Calcu Cross-Section Completed', completed_data_folder


def detect_folder_files_cross_section_CODEX_MATHUSLA(LLP_data_folder_dir):
    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        if os.path.isfile(file_path_all):
            try:
                completed_data_folder = rs.add_whether_in_the_detector_without_Decay_calcu_add_cross_section_CODEX_MATHUSLA(file_path_all, LLP_data_folder_dir)
            except Exception as e:
                print(f"Error with file: {file_path_all}")
                print(f"Error message: {str(e)}")
                continue
                file_path_all = ''

    return 'Detection and Calcu Cross-Section Completed', completed_data_folder


def SHiP_CODEX_MATHUSLA(LLP_data_folder_dir):
    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        if os.path.isfile(file_path_all):
            try:
                completed_data_folder = rs.add_whether_in_the_detector_without_Decay_calcu_add_cross_section_CODEX_MATHUSLA_SHiP(file_path_all, LLP_data_folder_dir)
            except Exception as e:
                print(f"Error with file: {file_path_all}")
                print(f"Error message: {str(e)}")
                continue
                file_path_all = ''

    return 'Detection and Calcu Cross-Section Completed', completed_data_folder

def detect_folder_files_r(LLP_data_folder_dir):
    # out_put_path = os.path.dirname(LLP_data_folder_dir) + '/detected_llp_data'
    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        # print(file_path_all)
        if os.path.isfile(file_path_all):
            
            completed_data_folder = rs.add_whether_in_the_detector_without_angle(file_path_all, LLP_data_folder_dir)
                        
    return 'Detection Completed', completed_data_folder

def detect_folder_files_r_no_calcu(LLP_data_folder_dir):
    # out_put_path = os.path.dirname(LLP_data_folder_dir) + '/detected_llp_data'
    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        # print(file_path_all)
        if os.path.isfile(file_path_all):
            
            completed_data_folder = rs.add_whether_in_the_detector_without_angle_without_Decay_calcu(file_path_all, LLP_data_folder_dir)
                        
    return 'Detection Completed', completed_data_folder


def one_key_run_br_ctau(br_lower_lim, br_upper_lim, br_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 mass, seed_array_length, out_put_path, main41_path):
    LLP_data_path = lp.loop_ctau_br(br_lower_lim, br_upper_lim, br_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 mass, seed_array_length, out_put_path, main41_path)
    # run main41.cc with ctau, br, mass seed
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files(LLP_data_path)[1]
    # Find the LLPs can be detected or not
    print('The LLPs are Judged whether they are Detected or not')
    final_files = cb.combine_files_precise(os.path.dirname(completed_data_dir))
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files


def one_key_run_br_ctau_certain_seed(br_lower_lim, br_upper_lim, br_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 mass, seed_array_length, out_put_path, main41_path):
    LLP_data_path = lp.loop_ctau_br_certain_seed(br_lower_lim, br_upper_lim, br_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 mass, seed_array_length, out_put_path, main41_path)
    # run main41.cc with ctau, br, mass seed
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files(LLP_data_path)[1]
    # Find the LLPs can be detected or not
    print('The LLPs are Judged whether they are Detected or not')
    final_files = cb.combine_files_precise(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_mass_ctau(mass_lower_lim, mass_upper_lim, mass_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 br, seed_array_length, out_put_path, main41_path):
    LLP_data_path = lp.loop_mass_ctau(mass_lower_lim, mass_upper_lim, mass_array_length, 
                ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                br, seed_array_length, out_put_path, main41_path)
    completed_data_dir = detect_folder_files(LLP_data_path)[1]
    final_files = cb.combine_files_precise(completed_data_dir)
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_mass_ctau_simple_ver(mass_lower_lim, mass_upper_lim, mass_array_length, 
                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                 br, seed_array_length, out_put_path, main41_path):
    LLP_data_path = lp.loop_mass_ctau(mass_lower_lim, mass_upper_lim, mass_array_length, 
                ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
                br, seed_array_length, out_put_path, main41_path)
    completed_data_dir = detect_folder_files_no_calcu(LLP_data_path)[1]
    final_files = cb.combine_files_precise(completed_data_dir)
    return LLP_data_path, completed_data_dir, final_files

# def one_key_run_mass_ctau_certain_seed(mass_lower_lim, mass_upper_lim, mass_array_length, 
#                  ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
#                  br, seed_array, out_put_path, main41_path):
#     LLP_data_path = loop_mass_ctau_certain_seed(mass_lower_lim, mass_upper_lim, mass_array_length, 
#                 ctau_lower_lim, ctau_upper_lim, ctau_array_length, 
#                 br, seed_array, out_put_path, main41_path)
#     print('The Generation of LLPs is Completed')
#     completed_data_dir = detect_folder_files(LLP_data_path)[1]
#     print('The LLPs are Judged whether they are Detected or not')
#     final_files = combine_files_precise(os.path.dirname(completed_data_dir))
#     print('The Final Step is Over, See the .csv files for LLPs Completed Data')
#     return LLP_data_path, completed_data_dir, final_files[0]

def one_key_run_mass_ctau_given_by_csv(csv_file, br, seed_array, out_put_path, main41_path):
    LLP_data_path = lp.loop_mass_ctau_given_by_csv(csv_file, br, seed_array, out_put_path, main41_path)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_no_calcu(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not')
    final_files = cb.combine_files_precise(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_mass_ctau_br_given_by_csv(csv_file, br, seed_array, out_put_path, main41_path):
    LLP_data_path = lp.loop_mass_ctau_br_given_by_csv(csv_file, br, seed_array, out_put_path, main41_path)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_no_calcu(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not')
    final_files = cb.combine_files_precise(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_mass_ctau_br_given_by_csv_main131(csv_file, br, seed_array, out_put_path, main131_path):
    print("Running Simulation...")
    LLP_data_path = lp.loop_mass_ctau_br_given_by_csv_main131(csv_file, br, seed_array, out_put_path, main131_path)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_no_calcu(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not')
    final_files = cb.combine_files_precise(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files


def one_key_run_by_csv_cross_section_main131(csv_file, br, seed_array, out_put_path, main131_path):
    print("Running Simulation...")
    LLP_data_path = lp.loop_mass_ctau_br_given_by_csv_main131(csv_file, br, seed_array, out_put_path, main131_path)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_cross_section(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files


def one_key_run_by_csv_cross_section_main131_lower_eff(csv_file, br, seed_array, out_put_path, main131_path, today, sleep_time = 10): 
    print("Running Simulation...")
    LLP_data_path = lp.loop_mass_ctau_br_given_by_csv_main131_sleep_time(csv_file, br, seed_array, out_put_path, main131_path, sleep_time, today)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_cross_section(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_by_csv_cross_section_main131_lower_eff_all_detectors(csv_file, br, seed_array, out_put_path, main131_path, today, sleep_time = 10): 
    print("Running Simulation...")
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/')
    LLP_data_path = lp.loop_mass_ctau_br_given_by_csv_main131_sleep_time(csv_file, br, seed_array, out_put_path, main131_path, sleep_time, today)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_cross_section_CODEX_MATHUSLA(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise_CODEX_MATHUSLA(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_by_csv_cross_section_main131_simple(csv_file, br, seed_array, out_put_path, main131_path, today, sleep_time = 10): 
    print("Running Simulation...")
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/')
    LLP_data_path = lp.loop_mass_simple(csv_file, br, seed_array, out_put_path, main131_path, sleep_time, today)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_cross_section_CODEX_MATHUSLA(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise_CODEX_MATHUSLA(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_by_csv_cross_section_main131_simple_2HDM(csv_file, br, seed_array, out_put_path, main131_path, today, sleep_time = 10): 
    print("Running Simulation...")
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/')
    LLP_data_path = lp.loop_mass_simple_2HDM(csv_file, br, seed_array, out_put_path, main131_path, sleep_time, today)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_cross_section_CODEX_MATHUSLA(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise_CODEX_MATHUSLA(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files


def one_key_run_2HDM_cross_section_main131_lower_eff_all_detectors_B(csv_file, br, seed_array, out_put_path, main131_path, today, sleep_time = 10): 
    print("Running Simulation...")
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/')
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/B_2HDM/')
    # mkdir_1(out_put_path + today +'/' + 'LLP_data/D_2HDM/')
    # mkdir_1(out_put_path + today +'/' + 'LLP_data/K_2HDM/')
    LLP_data_path = lp.loop_mass_ctau_br_given_by_csv_main131_sleep_time_B(csv_file, br, seed_array, out_put_path, main131_path, sleep_time, today)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_cross_section_CODEX_MATHUSLA(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise_CODEX_MATHUSLA(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files


def one_key_run_2HDMA_cross_section_main131_lower_eff_all_detectors_B(csv_file, br, seed_array, out_put_path, main131_path, today, sleep_time = 10): 
    print("Running Simulation...")
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/')
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/B_2HDM_A/')
    # mkdir_1(out_put_path + today +'/' + 'LLP_data/D_2HDM/')
    # mkdir_1(out_put_path + today +'/' + 'LLP_data/K_2HDM/')
    LLP_data_path = lp.loop_2HDM_A(csv_file, br, seed_array, out_put_path, main131_path, sleep_time, today)
    print('The Generation of LLPs is Completed')
    completed_data_dir = SHiP_CODEX_MATHUSLA(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise_CODEX_MATHUSLA(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_2HDMA_cross_section_main131_lower_eff_MA_CO_B(csv_file, br, seed_array, out_put_path, main131_path, today, sleep_time = 10): 
    print("Running Simulation...")
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/')
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/B_2HDM_A/')
    # mkdir_1(out_put_path + today +'/' + 'LLP_data/D_2HDM/')
    # mkdir_1(out_put_path + today +'/' + 'LLP_data/K_2HDM/')
    LLP_data_path = lp.loop_2HDM_A(csv_file, br, seed_array, out_put_path, main131_path, sleep_time, today)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_cross_section_CODEX_MATHUSLA(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise_CODEX_MATHUSLA(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_2HDM_cross_section_main131_lower_eff_all_detectors_BKD(csv_file, br, seed_array, out_put_path, main131_path, today, sleep_time = 10): 
    print("Running Simulation...")
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/')
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/B_2HDM/')
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/D_2HDM/')
    ffr.mkdir_1(out_put_path + today +'/' + 'LLP_data/K_2HDM/')
    LLP_data_path = lp.loop_mass_ctau_br_given_by_csv_main131_sleep_time_BKD(csv_file, br, seed_array, out_put_path, main131_path, sleep_time, today)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_cross_section_CODEX_MATHUSLA(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise_CODEX_MATHUSLA(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def one_key_run_by_csv_cross_section_main41(csv_file, br, seed_array, out_put_path, main131_path):
    print("Running Simulation...")
    LLP_data_path = lp.loop_mass_ctau_br_given_by_csv(csv_file, br, seed_array, out_put_path, main131_path)
    print('The Generation of LLPs is Completed')
    completed_data_dir = detect_folder_files_cross_section(LLP_data_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return LLP_data_path, completed_data_dir, final_files

def calcu_cross_section_and_combine_files(folder_path_date):
    completed_data_dir = detect_folder_files_cross_section(folder_path_date)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return completed_data_dir, final_files

def calcu_cross_section_and_combine_files_CODEX_MATHUSLA(folder_path_date):
    completed_data_dir = detect_folder_files_cross_section_CODEX_MATHUSLA(folder_path_date)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise_CODEX_MATHUSLA(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return completed_data_dir, final_files

def calcu_cross_section_and_combine_files_CODEX_MATHUSLA_SHiP(folder_path_date):
    completed_data_dir = SHiP_CODEX_MATHUSLA(folder_path_date)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise_CODEX_MATHUSLA_SHiP(completed_data_dir)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return completed_data_dir, final_files