import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import ALL_IN_ONE.Pythia8.functions_for_calculation as fcal
matplotlib.use('Agg')
from scipy.interpolate import interp1d
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

    return 0



def plot_with_envelope_log2(csv_file, interpolation='quadratic', output_file='envelope_plot.png'):

    df = pd.read_csv(csv_file)
    
    df['significance_with_4pi'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], np.sin(df['theta_input'])**2, 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3 * df['visible_br']
    df['significance_without_4pi'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], np.sin(df['theta_input'])**2, 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3 * df['visible_br_without_4pi']
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    df_threshold = df[df['significance_with_4pi'] > 3]
    df_threshold_without_4pi = df[df['significance_without_4pi'] > 3]

    def add_envelope(ax, x, y, label, color, interpolation='cubic'):
        """
        Add an envelope to the given axis and interpolate the upper and lower envelopes.

        Parameters:
        - ax: matplotlib axis object
        - x: x data (m)
        - y: y data (sin^2(theta_input))
        - label: label for the envelope
        - color: color of the envelope
        - interpolation: interpolation method ('cubic' or 'quadratic')
        """
        # Sort data by x
        sorted_indices = np.argsort(x)
        x = np.array(x)[sorted_indices]
        y = np.array(y)[sorted_indices]

        # Sort upper and lower envelope points
        unique_x = np.unique(x)
        upper_envelope_x = []
        upper_envelope_y = []
        lower_envelope_x = []
        lower_envelope_y = []

        for ux in unique_x:
            mask = x == ux
            upper_envelope_x.append(ux)
            upper_envelope_y.append(y[mask].max())  
            lower_envelope_x.append(ux)
            lower_envelope_y.append(y[mask].min())  

        # Interpolate the upper and lower envelope
        x_interp = np.linspace(min(upper_envelope_x), max(upper_envelope_x), 500)
        upper_interp = interp1d(upper_envelope_x, upper_envelope_y, kind=interpolation, fill_value="extrapolate")
        lower_interp = interp1d(lower_envelope_x, lower_envelope_y, kind=interpolation, fill_value="extrapolate")

        # generate interpolated values
        y_upper = upper_interp(x_interp)
        y_lower = lower_interp(x_interp)

        ax.plot(x_interp, y_upper, color=color, linestyle='--', alpha=0.8)
        ax.plot(x_interp, y_lower, color=color, linestyle='--', alpha=0.8)
        ax.fill_between(x_interp, y_lower, y_upper, color=color, alpha=0.1, label=f'Envelope Area ({label})')


    # Subplot 1: No 4Pi
    axs[0, 0].scatter(df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, label='No 4Pi', color='green', s=20, alpha=0.5)
    add_envelope(axs[0, 0], df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, 'No 4Pi', 'green', interpolation)
    axs[0, 0].set_xlim(0.1, 5)
    axs[0, 0].set_ylim(1e-14, 1e-6)
    axs[0, 0].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[0, 0].set_xlabel('mass / GeV', fontsize=15)
    axs[0, 0].set_ylabel('theta^2', fontsize=15)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xscale('log', base=2)
    axs[0, 0].legend()

    # Subplot 2: With 4Pi
    axs[0, 1].scatter(df_threshold['m'], np.sin(df_threshold['theta_input'])**2, label=r'With 4Pi ($3\sigma$)', color='blue', s=20, alpha=0.5)
    add_envelope(axs[0, 1], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'With 4Pi', 'blue', interpolation)
    axs[0, 1].set_xlim(0.1, 5)
    axs[0, 1].set_ylim(1e-14, 1e-6)
    axs[0, 1].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[0, 1].set_xlabel('mass / GeV', fontsize=15)
    axs[0, 1].set_ylabel('theta^2', fontsize=15)
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_xscale('log', base=2)
    axs[0, 1].legend()

    # Subplot 3: Combined
    axs[1, 0].scatter(df_threshold['m'], np.sin(df_threshold['theta_input'])**2, label=r'With 4Pi ($3\sigma$)', color='blue', s=20, alpha=0.5)
    axs[1, 0].scatter(df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, label='No 4Pi with Threshold', color='green', s=20, alpha=0.5)
    add_envelope(axs[1, 0], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'With 4Pi', 'blue', interpolation)
    add_envelope(axs[1, 0], df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, 'No 4Pi', 'green', interpolation)
    axs[1, 0].set_xlim(0.1, 5)
    axs[1, 0].set_ylim(1e-14, 1e-6)
    axs[1, 0].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[1, 0].set_xlabel('mass / GeV', fontsize=15)
    axs[1, 0].set_ylabel('theta^2', fontsize=15)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].legend()

    # Subplot 4: Combined with transparency
    axs[1, 1].scatter(df_threshold['m'], np.sin(df_threshold['theta_input'])**2, label=r'With Threshold ($3\sigma$)', color='blue', s=20, alpha=0.5)
    axs[1, 1].scatter(df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, label='No 4Pi with Threshold', color='green', s=20, alpha=0.2)
    add_envelope(axs[1, 1], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'With 4Pi', 'blue', interpolation)
    add_envelope(axs[1, 1], df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, 'No 4Pi', 'green', interpolation)
    axs[1, 1].set_xlim(0.1, 5)
    axs[1, 1].set_ylim(1e-14, 1e-6)
    axs[1, 1].set_title('Both Data (Low Alpha for with 4Pi)', fontsize=20)
    axs[1, 1].set_xlabel('mass / GeV', fontsize=15)
    axs[1, 1].set_ylabel('theta^2', fontsize=15)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xscale('log', base=2)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()
    return output_file


def plot_with_envelope(csv_file, interpolation='quadratic', output_file='envelope_plot.png'):

    df = pd.read_csv(csv_file)
    
    df['significance_with_4pi'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], np.sin(df['theta_input'])**2, 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3 * df['visible_br']
    df['significance_without_4pi'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], np.sin(df['theta_input'])**2, 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3 * df['visible_br_without_4pi']
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    df_threshold = df[df['significance_with_4pi'] > 3]
    df_threshold_without_4pi = df[df['significance_without_4pi'] > 3]

    def add_envelope(ax, x, y, label, color, interpolation='cubic'):
        """
        Add an envelope to the given axis and interpolate the upper and lower envelopes.

        Parameters:
        - ax: matplotlib axis object
        - x: x data (m)
        - y: y data (sin^2(theta_input))
        - label: label for the envelope
        - color: color of the envelope
        - interpolation: interpolation method ('cubic' or 'quadratic')
        """
        # Sort data by x
        sorted_indices = np.argsort(x)
        x = np.array(x)[sorted_indices]
        y = np.array(y)[sorted_indices]

        # Sort upper and lower envelope points
        unique_x = np.unique(x)
        upper_envelope_x = []
        upper_envelope_y = []
        lower_envelope_x = []
        lower_envelope_y = []

        for ux in unique_x:
            mask = x == ux
            upper_envelope_x.append(ux)
            upper_envelope_y.append(y[mask].max())  
            lower_envelope_x.append(ux)
            lower_envelope_y.append(y[mask].min())  

        # Interpolate the upper and lower envelope
        x_interp = np.linspace(min(upper_envelope_x), max(upper_envelope_x), 500)
        upper_interp = interp1d(upper_envelope_x, upper_envelope_y, kind=interpolation, fill_value="extrapolate")
        lower_interp = interp1d(lower_envelope_x, lower_envelope_y, kind=interpolation, fill_value="extrapolate")

        # generate interpolated values
        y_upper = upper_interp(x_interp)
        y_lower = lower_interp(x_interp)

        ax.plot(x_interp, y_upper, color=color, linestyle='--', alpha=0.8)
        ax.plot(x_interp, y_lower, color=color, linestyle='--', alpha=0.8)
        ax.fill_between(x_interp, y_lower, y_upper, color=color, alpha=0.1, label=f'Envelope Area ({label})')


    # Subplot 1: No 4Pi
    axs[0, 0].scatter(df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, label='No 4Pi', color='green', s=20, alpha=0.5)
    add_envelope(axs[0, 0], df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, 'No 4Pi', 'green', interpolation)
    axs[0, 0].set_xlim(0.1, 5)
    axs[0, 0].set_ylim(1e-14, 1e-6)
    axs[0, 0].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[0, 0].set_xlabel('mass / GeV', fontsize=15)
    axs[0, 0].set_ylabel('theta^2', fontsize=15)
    axs[0, 0].set_yscale('log')
    # axs[0, 0].set_xscale('log', base=2)
    axs[0, 0].legend()

    # Subplot 2: With 4Pi
    axs[0, 1].scatter(df_threshold['m'], np.sin(df_threshold['theta_input'])**2, label=r'With 4Pi ($3\sigma$)', color='blue', s=20, alpha=0.5)
    add_envelope(axs[0, 1], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'With 4Pi', 'blue', interpolation)
    axs[0, 1].set_xlim(0.1, 5)
    axs[0, 1].set_ylim(1e-14, 1e-6)
    axs[0, 1].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[0, 1].set_xlabel('mass / GeV', fontsize=15)
    axs[0, 1].set_ylabel('theta^2', fontsize=15)
    axs[0, 1].set_yscale('log')
    # axs[0, 1].set_xscale('log', base=2)
    axs[0, 1].legend()

    # Subplot 3: Combined
    axs[1, 0].scatter(df_threshold['m'], np.sin(df_threshold['theta_input'])**2, label=r'With 4Pi ($3\sigma$)', color='blue', s=20, alpha=0.5)
    axs[1, 0].scatter(df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, label='No 4Pi with Threshold', color='green', s=20, alpha=0.5)
    add_envelope(axs[1, 0], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'With 4Pi', 'blue', interpolation)
    add_envelope(axs[1, 0], df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, 'No 4Pi', 'green', interpolation)
    axs[1, 0].set_xlim(0.1, 5)
    axs[1, 0].set_ylim(1e-14, 1e-6)
    axs[1, 0].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[1, 0].set_xlabel('mass / GeV', fontsize=15)
    axs[1, 0].set_ylabel('theta^2', fontsize=15)
    axs[1, 0].set_yscale('log')
    # axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].legend()

    # Subplot 4: Combined with transparency
    axs[1, 1].scatter(df_threshold['m'], np.sin(df_threshold['theta_input'])**2, label=r'With Threshold ($3\sigma$)', color='blue', s=20, alpha=0.5)
    axs[1, 1].scatter(df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, label='No 4Pi with Threshold', color='green', s=20, alpha=0.2)
    add_envelope(axs[1, 1], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'With 4Pi', 'blue', interpolation)
    add_envelope(axs[1, 1], df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, 'No 4Pi', 'green', interpolation)
    axs[1, 1].set_xlim(0.1, 5)
    axs[1, 1].set_ylim(1e-14, 1e-6)
    axs[1, 1].set_title('Both Data (Low Alpha for with 4Pi)', fontsize=20)
    axs[1, 1].set_xlabel('mass / GeV', fontsize=15)
    axs[1, 1].set_ylabel('theta^2', fontsize=15)
    axs[1, 1].set_yscale('log')
    # axs[1, 1].set_xscale('log', base=2)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()
