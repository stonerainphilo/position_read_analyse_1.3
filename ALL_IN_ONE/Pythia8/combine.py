import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime
import shutil

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


def combine_files_move_CODEX(completed_file_path):
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
                df_columns = df.filter(regex=r'^detected_CODEXb_size_\d+$').columns
                detected_df = df[(df[df_columns] == 1)]
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

def combine_files_precise_SHiP(completed_file_path):
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
                
                detected_df = df[(df['detected_SHiP'] == 1)]
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




def split(source_folder, files_per_folder=4, prefix='split'):
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    total_files = len(csv_files)
    folder_count = (total_files + files_per_folder - 1) // files_per_folder

    for i in range(folder_count):
        folder_name = os.path.join(source_folder, f"{prefix}_{i+1}")
        os.makedirs(folder_name, exist_ok=True)
        start = i * files_per_folder
        end = start + files_per_folder
        for csv_file in csv_files[start:end]:
            src = os.path.join(source_folder, csv_file)
            dst = os.path.join(folder_name, csv_file)
            shutil.move(src, dst)
    print(f"Split {total_files} files into {folder_count} folders.")

# Example usage:
# split_csv_files_to_folders('/path/to/your/csv_folder', files_per_folder=10)

def undo_split_csv_files(source_folder, prefix='split'):
    # Find all split folders
    split_folders = [os.path.join(source_folder, f) for f in os.listdir(source_folder)
                     if os.path.isdir(os.path.join(source_folder, f)) and f.startswith(prefix)]
    # Move all CSV files back to source_folder
    for folder in split_folders:
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                src = os.path.join(folder, file)
                dst = os.path.join(source_folder, file)
                shutil.move(src, dst)
        # Optionally remove the empty folder
        os.rmdir(folder)
    print(f"All CSV files moved back to {source_folder} and split folders removed.")
