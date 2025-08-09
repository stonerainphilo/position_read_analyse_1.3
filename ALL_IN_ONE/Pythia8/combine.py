import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime

def combine_files_precise_CODEX_MATHUSLA(completed_file_path):
        merged_df = pd.DataFrame()
        df_all = pd.DataFrame()
        date = datetime.now().date()
        completed_file_path = str(completed_file_path)
        out_file_path = os.path.dirname(completed_file_path)
        for file in tqdm(os.listdir(completed_file_path)):
            file_path = os.path.join(completed_file_path, file)
        #     print(file_path)
            if file.endswith('.csv'):

                df = pd.read_csv(file_path)
                
                detected_df = df[(df['detected'] == 1) | (df['detected_MATHUSLA'] == 1)]
                merged_df = pd.concat([merged_df, detected_df], ignore_index=True)
                # df_all = pd.concat(([df_all, df]), ignore_index=True)
                
                # print(file + 'has been combined')
        file_path_combined_detected = out_file_path + '/' + f'{date}' + '_detected_combined_precise_file.csv'
        file_path_combined = out_file_path + '/' + f'{date}' + '_all_combined_precise_file.csv'
        merged_df.to_csv(file_path_combined_detected)
        # df_all.to_csv(file_path_combined)
        return file_path_combined, file_path_combined_detected
    

def combine_files_precise_CODEX_MATHUSLA_SHiP(completed_file_path):
        merged_df = pd.DataFrame()
        df_all = pd.DataFrame()
        date = datetime.now().date()
        completed_file_path = str(completed_file_path)
        out_file_path = os.path.dirname(completed_file_path)
        for file in tqdm(os.listdir(completed_file_path)):
            file_path = os.path.join(completed_file_path, file)
        #     print(file_path)
            if file.endswith('.csv'):

                df = pd.read_csv(file_path)
                
                detected_df = df[(df['detected'] == 1) | (df['detected_MATHUSLA'] == 1) | (df['detected_SHiP'] == 1)]
                merged_df = pd.concat([merged_df, detected_df], ignore_index=True)
                # df_all = pd.concat(([df_all, df]), ignore_index=True)
                
                # print(file + 'has been combined')
        file_path_combined_detected = out_file_path + '/' + f'{date}' + '_detected_combined_precise_file.csv'
        file_path_combined = out_file_path + '/' + f'{date}' + '_all_combined_precise_file.csv'
        merged_df.to_csv(file_path_combined_detected)
        # df_all.to_csv(file_path_combined)
        return file_path_combined, file_path_combined_detected

def combine_files_precise_CODEX_MATHUSLA_SHiP_new_folder(completed_file_path, new_folder_name):
        merged_df = pd.DataFrame()
        df_all = pd.DataFrame()
        date = datetime.now().date()
        completed_file_path = str(completed_file_path)
        out_file_path = new_folder_name
        for file in tqdm(os.listdir(completed_file_path)):
            file_path = os.path.join(completed_file_path, file)
        #     print(file_path)
            if file.endswith('.csv'):

                df = pd.read_csv(file_path)
                
                detected_df = df[(df['detected'] == 1) | (df['detected_MATHUSLA'] == 1) | (df['detected_SHiP'] == 1)]
                merged_df = pd.concat([merged_df, detected_df], ignore_index=True)
                # df_all = pd.concat(([df_all, df]), ignore_index=True)
                
                # print(file + 'has been combined')
        file_path_combined_detected = out_file_path + '/' + f'{date}' + '_detected_combined_precise_file.csv'
        file_path_combined = out_file_path + '/' + f'{date}' + '_all_combined_precise_file.csv'
        merged_df.to_csv(file_path_combined_detected)
        # df_all.to_csv(file_path_combined)
        return file_path_combined, file_path_combined_detected

def combine_files_precise(completed_file_path):
    merged_df = pd.DataFrame()
    df_all = pd.DataFrame()
    date = datetime.now().date()
    completed_file_path = str(completed_file_path)
    out_file_path = os.path.dirname(completed_file_path)
    for file in tqdm(os.listdir(completed_file_path)):
        file_path = os.path.join(completed_file_path, file)
    #     print(file_path)
        if file.endswith('.csv'):

            df = pd.read_csv(file_path)
            
            detected_df = df[df['detected'] == 1]
            merged_df = pd.concat([merged_df, detected_df], ignore_index=True)
            # df_all = pd.concat(([df_all, df]), ignore_index=True)
            
            # print(file + 'has been combined')
    file_path_combined_detected = out_file_path + '/' + f'{date}' + '_detected_combined_precise_file.csv'
    file_path_combined = out_file_path + '/' + f'{date}' + '_all_combined_precise_file.csv'
    merged_df.to_csv(file_path_combined_detected)
    # df_all.to_csv(file_path_combined)
    return file_path_combined, file_path_combined_detected
    
    

def combine_files_precise_r(completed_file_path):
        # print(completed_file_path)
        merged_df = pd.DataFrame()
        df_all = pd.DataFrame()
        completed_file_path = str(completed_file_path)
        # print(completed_file_path)
        date = datetime.now().date()
        out_file_path = os.path.dirname(completed_file_path)
        for file in tqdm(os.listdir(completed_file_path)):
            file_path = os.path.join(completed_file_path, file)
        #     print(file_path)
            if file.endswith('.csv'):
                # with open(file_path, 'rb') as f:
                #     raw_data = f.read()
                #     result = chardet.detect(raw_data)
                #     encoding_code = result['encoding']
                
                df = pd.read_csv(file_path)
                
                detected_df = df[df['detected'] == 1]
                merged_df = pd.concat([merged_df, detected_df], ignore_index=True)
                # df_all = pd.concat(([df_all, df]), ignore_index=True)
                
                # print(file + 'has been combined')
        file_path_combined_detected = out_file_path + '/' + f'{date}' + '_detected_combined_r_file.csv'
        file_path_combined = out_file_path + '/' + f'{date}' + '_all_combined_r_file.csv'
        merged_df.to_csv(file_path_combined_detected)
        # df_all.to_csv(file_path_combined)
        return file_path_combined, file_path_combined_detected


def merge_csv_files(directory, output_file):

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    data_frames = []
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)

    merged_df = pd.concat(data_frames, ignore_index=True)
    
    merged_df.to_csv(output_file, index=False)
    print(f"All CSV files have been merged into {output_file}")