import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import date
# import chardet

def plot_llp_decay_in_the_detector(file_folder_path):
    for file in os.listdir(file_folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(file_folder_path, file)
            
            # with open(file_path, 'rb') as f:
            #     raw_data = f.read()
            #     result = chardet.detect(raw_data)
            #     encoding_code = result['encoding']
            
            df = pd.read_csv(file_path)
            
            detected_df = df[df['detected'] == 1]
            
            plt.scatter(detected_df['br'], detected_df['tau'])
            print(file + 'has been plotted')

    plt.title('Scatter Plot of Detected Data')
    plt.xlabel('br[B->K LLP]')
    plt.ylabel('tau[cm]')
    plt.xscale('log')
    plt.yscale('log')
    # plt.show()
    fig_path = os.path.join(file_folder_path, 'Br_ctau_fig.png')
    plt.savefig(fig_path)
    return fig_path


def find_theta_for_LLP(LLP_file_dir, theta_file):
    all_matched = pd.DataFrame()
    for LLP_file in os.listdir(LLP_file_dir):
        df_LLP = pd.read_csv(LLP_file)
        df_theta = pd.read_csv(theta_file)
        macthed_rows = df_LLP[df_LLP['tau_input'].isin(df_theta['ltime'])]
        all_matched = all_matched.append(macthed_rows)
    macthed_rows.to_csv('test_LLP_theta_mass.csv')