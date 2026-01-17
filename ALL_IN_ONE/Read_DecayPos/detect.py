import os
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/ALL_IN_ONE")
from Pythia8.functions_for_calculation import whether_in_the_detector_by_position
from Pythia8.cross_section import counting_total_LLP
from Pythia8.functions_for_run import mkdir_1


def detect_folder_files_cross_section_CODEX_MATHUSLA(LLP_data_folder_dir, final_data_folder = None):
    if final_data_folder:
        mkdir_1(final_data_folder)
    else:
        final_data_folder = os.path.dirname(LLP_data_folder_dir) + '/Detect_CODEX-b_Mathusla'

    all_llp = pd.DataFrame()
    all_llp['m'] = []
    all_llp['tau'] = []
    # all_llp['']
    all_llp['CODEX-b_acceptance'] = []
    all_llp['MATHUSLA_acceptance'] = []

    for files in tqdm(os.listdir(LLP_data_folder_dir)):
        filename = os.path.join(LLP_data_folder_dir, files)
        # print(filename)
        if os.path.isfile(filename):
            # print(f"Processing file: {filename}")
            try:
                llp_data = pd.read_csv(filename)
                llp_detect = pd.DataFrame()
                llp_whether_in_detector = whether_in_the_detector_by_position(llp_data['decay_pos_x'], llp_data['decay_pos_y'], llp_data['decay_pos_z'])
                llp_whether_in_detector_MATHUSLA = whether_in_the_detector_by_position(llp_data['decay_pos_x'], llp_data['decay_pos_y'], llp_data['decay_pos_z'], -100000, 100000, 100000, 125000, 100000, 300000)
                # name = f'm{llp_data["m"][0]}_tau{llp_data["tau"][0]}'
                llp_detect['detected'] = llp_whether_in_detector
                llp_detect['detected_MATHUSLA'] = llp_whether_in_detector_MATHUSLA
                llp_detect['detector_acceptance'] = sum(llp_detect['detected']) * llp_data['Cross_section_fb'] / counting_total_LLP(llp_data)
                llp_detect['detector_acceptance_MATHUSLA'] = sum(llp_detect['detected_MATHUSLA']) * llp_data['Cross_section_fb']/ counting_total_LLP(llp_data)

                # final_data_path = os.path.join(final_data_folder + f'/final_data_cross_section_{name}.csv')
                # completed_data_folder = add_whether_in_the_detector_without_Decay_calcu_add_cross_section_CODEX_MATHUSLA(filename, final_data_folder)
            except Exception as e:
                print(f"Error with file: {filename}")
                print(f"Error message: {str(e)}")
                continue

            all_llp = pd.concat([all_llp, pd.DataFrame({
                'm': [llp_data['m'][0]],

                'tau': [llp_data['tau'][0]],
                'CODEX-b_acceptance': [llp_detect['detector_acceptance'].iloc[0]],
                'MATHUSLA_acceptance': [llp_detect['detector_acceptance_MATHUSLA'].iloc[0]]
            })], ignore_index=True)
    all_llp.to_csv(os.path.join(final_data_folder + '/all_llp_detect_CODEX_MATHUSLA_cross_section.csv'), index = False)
    return 'Detection and Calcu Cross-Section Completed'

folder = '/media/ubuntu/SRPPS/CODEX-b/CODEX_B-11/'
out_folder = '/media/ubuntu/SRPPS/CODEX-b/CODEX_b-11/Detect_CODEX-b_Mathusla/'
detect_folder_files_cross_section_CODEX_MATHUSLA(folder, out_folder)