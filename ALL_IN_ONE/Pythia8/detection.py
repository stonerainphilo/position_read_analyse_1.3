from one_key_run import detect_folder_files, detect_folder_files_r
import os
import combine as cb
from one_key_run import detect_folder_files_r_no_calcu, detect_folder_files_no_calcu
from one_key_run import detect_folder_files_cross_section
def detect_r_and_combine(LLP_folder_path):
    completed_r_folder_path = detect_folder_files_r(LLP_folder_path)[1]
    print('The LLPs are Judged whether they are Detected or not by r')
    final_files_r = cb.combine_files_precise_r(completed_r_folder_path)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return completed_r_folder_path, final_files_r

def detect_r_and_combine_no_calcu_decay(LLP_folder_path):
    completed_r_folder_path = detect_folder_files_r_no_calcu(LLP_folder_path)[1]
    print('The LLPs are Judged whether they are Detected or not by r')
    final_files_r = cb.combine_files_precise_r(completed_r_folder_path)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return completed_r_folder_path, final_files_r

def detect_percise_and_combine(LLP_folder_path):
    completed_folder_path = detect_folder_files(LLP_folder_path)[1]
    print('The LLPs are Judged whether they are Detected or not')
    final_files = cb.combine_files_precise(completed_folder_path)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return completed_folder_path, final_files

def detect_percise_and_combine_no_calcu_decay(LLP_folder_path):
    completed_folder_path = detect_folder_files_no_calcu(LLP_folder_path)[1]
    print('The LLPs are Judged whether they are Detected or not')
    final_files = cb.combine_files_precise(completed_folder_path)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return completed_folder_path, final_files

def detect_percise_calcu_cross_section_and_combine(LLP_folder_path):
    completed_folder_path = detect_folder_files_cross_section(LLP_folder_path)[1]
    print('The LLPs are Judged whether they are Detected or not, and calculated the cross section')
    final_files = cb.combine_files_precise(completed_folder_path)
    print('The Final Step is Over, See the .csv files for LLPs Completed Data')
    return completed_folder_path, final_files