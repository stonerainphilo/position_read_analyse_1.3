import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import ALL_IN_ONE.Pythia8.functions_for_calculation as fcal
matplotlib.use('Agg')
from scipy.interpolate import interp1d
from datetime import date

from scipy.signal import find_peaks
def fine_envelope(x, y):
    x = np.array(x)
    y = np.array(y)


    peaks, _ = find_peaks(y)  # 上包络线
    troughs, _ = find_peaks(-y)  # 下包络线
    return x[peaks], y[peaks], x[troughs], y[troughs]
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


def remove_outliers_per_group(df, group_column, target_column, method='iqr', threshold=1.5):
    """
    对每个 group_column 的唯一值，检测 target_column 的离群点，并删除对应的行。

    参数:
    - df: 输入的 DataFrame
    - group_column: 分组列名（如 'm'）
    - target_column: 检测离群点的目标列名（如 'theta_input'）
    - method: 检测离群点的方法 ('iqr' 或 'zscore')
    - threshold: 离群点的阈值 (对于 IQR 方法，默认为 1.5；对于 Z-score 方法，默认为 3)

    返回:
    - df_clean: 删除离群点后的 DataFrame
    """
    def is_not_outlier(group):
        if method == 'iqr':
            # 基于四分位距 (IQR) 检测离群点
            q1 = group[target_column].quantile(0.25)  # 第 1 四分位数
            q3 = group[target_column].quantile(0.75)  # 第 3 四分位数
            iqr = q3 - q1  # 四分位距
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return (group[target_column] >= lower_bound) & (group[target_column] <= upper_bound)

        elif method == 'zscore':
            # 基于 Z-score 检测离群点
            mean = group[target_column].mean()
            std = group[target_column].std()
            z_scores = (group[target_column] - mean) / std
            return np.abs(z_scores) <= threshold

        else:
            raise ValueError("Invalid method. Use 'iqr' or 'zscore'.")

    # 对每个 group_column 的分组应用离群点检测
    mask = df.groupby(group_column).apply(is_not_outlier).reset_index(drop=True)
    return df[mask]

def remove_outliers_with_gap(df, group_column, target_column, gap_threshold):
    """
    对每个 group_column 的唯一值，检测 target_column 的异常最大值，并删除对应的行。
    进行两次最大值判断。

    参数:
    - df: 输入的 DataFrame
    - group_column: 分组列名（如 'm'）
    - target_column: 检测异常值的目标列名（如 'theta_input'）
    - gap_threshold: 最大值与次大值之间的最小相对差距，超过该差距则认为最大值是异常值

    返回:
    - df_clean: 删除异常值后的 DataFrame
    """
    def is_not_outlier(group):
        # 按 target_column 排序
        sorted_group = group.sort_values(by=target_column, ascending=False)
        valid_indices = sorted_group.index.tolist()

        for _ in range(2):  # 进行两次最大值判断
            if len(valid_indices) > 1:
                max_value = sorted_group.loc[valid_indices[0], target_column]  # 最大值
                
                # 找到第一个与 max_value 不相等的值
                second_max_value = None
                for idx in valid_indices[1:]:
                    if sorted_group.loc[idx, target_column] != max_value:
                        second_max_value = sorted_group.loc[idx, target_column]
                        break

                # 如果找不到不相等的值，则认为没有异常值
                if second_max_value is None:
                    break

                # 检测最大值与次大值之间的相对差距
                if (max_value - second_max_value) / second_max_value > gap_threshold:
                    # 如果相对差距超过阈值，移除所有等于最大值的索引
                    valid_indices = [idx for idx in valid_indices if sorted_group.loc[idx, target_column] != max_value]
                else:
                    break
            else:
                break

        return valid_indices

    # 对每个 group_column 的分组应用检测
    valid_indices = df.groupby(group_column).apply(is_not_outlier).explode().astype(int)
    return df.loc[valid_indices]


def plot_with_envelope_log2_remove(csv_file, interpolation='cubic', output_file='envelope_plot_10_15.png', scatter_size = 20):
    df_ORIGINAL = pd.read_csv(csv_file)
    # 删除离群点
    df = remove_outliers_with_gap(df_ORIGINAL, 'm', 'theta_input', 0.06)


    df['significance_with_4pi'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], np.sin(df['theta_input'])**2, 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3
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
    axs[0, 0].scatter(df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, label='No 4Pi', color='green', s=scatter_size, alpha=0.5)
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
    axs[0, 1].scatter(df_threshold['m'], np.sin(df_threshold['theta_input'])**2, label=r'With 4Pi ($3\sigma$)', color='blue', s=scatter_size, alpha=0.5)
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
    axs[1, 0].scatter(df_threshold['m'], np.sin(df_threshold['theta_input'])**2, label=r'With 4Pi ($3\sigma$)', color='blue', s=scatter_size, alpha=0.5)
    axs[1, 0].scatter(df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, label='No 4Pi with Threshold', color='green', s=scatter_size, alpha=0.5)
    add_envelope(axs[1, 0], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'With 4Pi', 'blue', interpolation)
    add_envelope(axs[1, 0], df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, 'No 4Pi', 'green', interpolation)
    axs[1, 0].set_xlim(0.25, 5)
    axs[1, 0].set_ylim(1e-14, 1e-6)
    axs[1, 0].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[1, 0].set_xlabel('mass / GeV', fontsize=15)
    axs[1, 0].set_ylabel('theta^2', fontsize=15)
    axs[1, 0].set_yscale('log')
    # axs[1, 0].set_xscale('log', base=2)
    axs[1, 0].legend()

    # Subplot 4: Combined with transparency
    add_envelope(axs[1, 1], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'With 4Pi', 'blue', interpolation)
    add_envelope(axs[1, 1], df_threshold_without_4pi['m'], np.sin(df_threshold_without_4pi['theta_input'])**2, 'No 4Pi', 'green', interpolation)
    axs[1, 1].set_xlim(0.25, 5)
    axs[1, 1].set_ylim(1e-14, 1e-6)
    axs[1, 1].set_title('Both Data (Low Alpha for with 4Pi)', fontsize=20)
    axs[1, 1].set_xlabel('mass / GeV', fontsize=15)
    axs[1, 1].set_ylabel('theta^2', fontsize=15)
    axs[1, 1].set_yscale('log')

    # Set x-axis to log scale and display 1, 2, 4 as ticks
    axs[1, 1].set_xscale('log', base=5)
    axs[1, 1].set_xticks([0.5, 1, 5])
    axs[1, 1].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))

    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()



def plot_with_envelope(csv_file, interpolation='cubic', output_file='envelope_plot.png'):
    # 删除离群点
    df = pd.read_csv(csv_file)


    df['significance_with_4pi'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], np.sin(df['theta_input'])**2, 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3
    df['significance_without_4pi'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], np.sin(df['theta_input'])**2, 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3 * df['visible_br_without_4pi']
    df['significance_lowest_br'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], np.sin(df['theta_input'])**2, 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3 * 0.76
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    df_threshold = df[df['significance_with_4pi'] > 3]
    df_threshold_without_4pi = df[df['significance_without_4pi'] > 3]
    df_threshold_lowest_br = df[df['significance_lowest_br'] > 3]

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

    # Remove rows with NaN or infinite values in relevant columns
    df_threshold_lowest_br = df_threshold_lowest_br.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'theta_input', 'detector_acceptance', 'significance_lowest_br'])
    df_threshold = df_threshold.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'theta_input', 'detector_acceptance', 'significance_with_4pi'])
    # df_threshold_without_4pi = df_threshold_without_4pi.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'theta_input', 'detector_acceptance', 'significance_without_4pi'])
    # Subplot 1: No 4Pi
    axs[0, 0].scatter(df_threshold_lowest_br['m'], np.sin(df_threshold_lowest_br['theta_input'])**2, label='Pessimistic', color='green', s=20, alpha=0.5)
    add_envelope(axs[0, 0], df_threshold_lowest_br['m'], np.sin(df_threshold_lowest_br['theta_input'])**2, 'Pessimistic', 'green', interpolation)
    axs[0, 0].set_xlim(0.1, 5)
    axs[0, 0].set_ylim(1e-14, 1e-6)
    axs[0, 0].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[0, 0].set_xlabel('mass / GeV', fontsize=15)
    axs[0, 0].set_ylabel('theta^2', fontsize=15)
    axs[0, 0].set_yscale('log')
    # axs[0, 0].set_xscale('log')
    axs[0, 0].legend()

    # Subplot 2: With 4Pi
    axs[0, 1].scatter(df_threshold['m'], np.sin(df_threshold['theta_input'])**2, label=r'Precise Br ($3\sigma$)', color='blue', s=20, alpha=0.5)
    add_envelope(axs[0, 1], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'Precise Br Envelope', 'blue', interpolation)
    axs[0, 1].set_xlim(0.1, 5)
    axs[0, 1].set_ylim(1e-14, 1e-6)
    axs[0, 1].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[0, 1].set_xlabel('mass / GeV', fontsize=15)
    axs[0, 1].set_ylabel('theta^2', fontsize=15)
    axs[0, 1].set_yscale('log')
    # axs[0, 1].set_xscale('log')
    axs[0, 1].legend()

    # Subplot 3: Combined
    axs[1, 0].scatter(df_threshold['m'], np.sin(df_threshold['theta_input'])**2, label='Precise Br', color='blue', s=20, alpha=0.5)
    axs[1, 0].scatter(df_threshold_lowest_br['m'], np.sin(df_threshold_lowest_br['theta_input'])**2, label='Pessimistic', color='green', s=20, alpha=0.5)
    add_envelope(axs[1, 0], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'Precise Br Envelope', 'blue', interpolation)
    add_envelope(axs[1, 0], df_threshold_lowest_br['m'], np.sin(df_threshold_lowest_br['theta_input'])**2, 'Pessimistic Envelope', 'green', interpolation)
    axs[1, 0].set_xlim(0.25, 5)
    axs[1, 0].set_ylim(1e-14, 1e-6)
    axs[1, 0].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[1, 0].set_xlabel('mass / GeV', fontsize=15)
    axs[1, 0].set_ylabel('theta^2', fontsize=15)
    axs[1, 0].set_yscale('log')
    # axs[1, 0].set_xscale('log')
    axs[1, 0].legend()

    # Subplot 4: Combined with transparency
    add_envelope(axs[1, 1], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'Precise Br', 'blue', interpolation)
    add_envelope(axs[1, 1], df_threshold_lowest_br['m'], np.sin(df_threshold_lowest_br['theta_input'])**2, 'Pessimistic', 'green', interpolation)
    axs[1, 1].set_xlim(0.25, 5)
    axs[1, 1].set_ylim(1e-14, 1e-6)
    axs[1, 1].set_title('Both Data', fontsize=20)
    axs[1, 1].set_xlabel('mass / GeV', fontsize=15)
    axs[1, 1].set_ylabel('theta^2', fontsize=15)
    axs[1, 1].set_yscale('log')
    # axs[1, 1].set_yscale('log')
    # Set x-axis to log scale and display 1, 2, 4 as ticks
    # axs[1, 1].set_xscale('log', base=5)
    # axs[1, 1].set_xticks([0.5, 1, 5])
    # axs[1, 1].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))

    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()

def plot_with_envelope_tanb(csv_file, interpolation='cubic', output_file='envelope_plot.png', visible_br_lowest=0.76):
    # 删除离群点
    df = pd.read_csv(csv_file)


    df['significance_with_4pi'] = df['detector_acceptance'] * fcal.calcu_Br_B_to_H(df['m'], df['tanb'], 1/df['tanb'], 600) * 300 * df['Cross_section_fb'] * 1e3
    df['significance_without_4pi'] = df['detector_acceptance'] * fcal.calcu_Br_B_to_H(df['m'], df['tanb'], 1/df['tanb'], 600) * 300 * df['Cross_section_fb'] * 1e3 * df['visible_br_without_4pi']
    df['significance_lowest_br'] = df['detector_acceptance'] * fcal.calcu_Br_B_to_H(df['m'], df['tanb'], 1/df['tanb'], 600) * 300 * df['Cross_section_fb'] * 1e3 * visible_br_lowest
    df['simple_br'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], df['tanb'], 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3 * df['visible_br_without_4pi']
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    df_threshold = df[df['significance_with_4pi'] > 3]
    df_threshold_without_4pi = df[df['significance_without_4pi'] > 3]
    df_threshold_lowest_br = df[df['significance_lowest_br'] > 3]
    df_threshold_simple_br = df[df['simple_br'] > 3]
    def add_envelope(ax, x, y, label, color, interpolation='cubic'):
        """
        Add an envelope to the given axis and interpolate the upper and lower envelopes.

        Parameters:
        - ax: matplotlib axis object
        - x: x data (m)
        - y: y data (sin^2(tanb))
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

    # Remove rows with NaN or infinite values in relevant columns
    df_threshold_lowest_br = df_threshold_lowest_br.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'tanb', 'detector_acceptance', 'significance_lowest_br'])
    df_threshold = df_threshold.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'tanb', 'detector_acceptance', 'significance_with_4pi'])
    # df_threshold_without_4pi = df_threshold_without_4pi.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'tanb', 'detector_acceptance', 'significance_without_4pi'])
    # Subplot 1: No 4Pi
    axs[0, 0].scatter(df_threshold_lowest_br['m'], df_threshold_lowest_br['tanb'], label='Pessimistic', color='green', s=20, alpha=0.5)
    add_envelope(axs[0, 0], df_threshold_lowest_br['m'], df_threshold_lowest_br['tanb'], 'Pessimistic', 'green', interpolation)
    axs[0, 0].set_xlim(min(df['m']), 5)
    # axs[0, 0].set_ylim(1e-14, 1e-6)
    axs[0, 0].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[0, 0].set_xlabel('mass / GeV', fontsize=15)
    axs[0, 0].set_ylabel(r'tan$\beta$', fontsize=15)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xscale('log')
    axs[0, 0].legend()

    # Subplot 2: With 4Pi
    axs[0, 1].scatter(df_threshold['m'], df_threshold['tanb'], label=r'Precise Br ($3\sigma$)', color='blue', s=20, alpha=0.5)
    add_envelope(axs[0, 1], df_threshold['m'], df_threshold['tanb'], 'Precise Br Envelope', 'blue', interpolation)
    axs[0, 1].set_xlim(min(df['m']), 5)
    # axs[0, 1].set_ylim(1e-14, 1e-6)
    axs[0, 1].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[0, 1].set_xlabel('mass / GeV', fontsize=15)
    axs[0, 1].set_ylabel(r'tan$\beta$', fontsize=15)
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_xscale('log')
    axs[0, 1].legend()

    # Subplot 3: Combined
    axs[1, 0].scatter(df_threshold['m'], df_threshold['tanb'], label='Precise Br', color='blue', s=20, alpha=0.5)
    axs[1, 0].scatter(df_threshold_lowest_br['m'], df_threshold_lowest_br['tanb'], label='Pessimistic', color='green', s=20, alpha=0.5)
    add_envelope(axs[1, 0], df_threshold['m'], df_threshold['tanb'], 'Precise Br Envelope', 'blue', interpolation)
    add_envelope(axs[1, 0], df_threshold_lowest_br['m'], df_threshold_lowest_br['tanb'], 'Pessimistic Envelope', 'green', interpolation)
    axs[1, 0].set_xlim(min(df['m']), 5)
    # axs[1, 0].set_ylim(1e-14, 1e-6)
    axs[1, 0].set_title(r'Threshold $3\sigma$', fontsize=20)
    axs[1, 0].set_xlabel('mass / GeV', fontsize=15)
    axs[1, 0].set_ylabel(r'tan$\beta$', fontsize=15)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xscale('log')
    axs[1, 0].legend()

    # Subplot 4: Combined with transparency
    add_envelope(axs[1, 1], df_threshold['m'], df_threshold['tanb'], 'Precise Br', 'blue', interpolation)
    add_envelope(axs[1, 1], df_threshold_lowest_br['m'], df_threshold_lowest_br['tanb'], 'Pessimistic', 'green', interpolation)
    add_envelope(axs[1, 1], df_threshold_simple_br['m'], df_threshold_simple_br['tanb'], 'Simple Br', 'red', interpolation)
    axs[1, 1].set_xlim(min(df['m']), 5)
    # axs[1, 1].set_ylim(1e-14, 1e-6)
    axs[1, 1].set_title('Both Data', fontsize=20)
    axs[1, 1].set_xlabel('mass / GeV', fontsize=15)
    axs[1, 1].set_ylabel(r'tan$\beta$', fontsize=15)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xscale('log')
    # Set x-axis to log scale and display 1, 2, 4 as ticks
    # axs[1, 1].set_xscale('log', base=5)
    # axs[1, 1].set_xticks([0.5, 1, 5])
    # axs[1, 1].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))

    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()



def plot_with_envelope_tanb_single(csv_file, interpolation='cubic', output_file='envelope_plot.png', visible_br_lowest=0.76):
    # 删除离群点
    df = pd.read_csv(csv_file)


    # df['significance_with_4pi'] = df['detector_acceptance'] * fcal.calcu_Br_B_to_H(df['m'], df['tanb'], 1/df['tanb'], 600) * 300 * df['Cross_section_fb'] * 1e3
    # df['significance_without_4pi'] = df['detector_acceptance'] * fcal.calcu_Br_B_to_H(df['m'], df['tanb'], 1/df['tanb'], 600) * 300 * df['Cross_section_fb'] * 1e3 * df['visible_br_without_4pi']
    df['significance_lowest_br'] = df['detector_acceptance'] * fcal.calcu_Br_B_to_H(df['m'], df['tanb'], 1/df['tanb'], 600) * 300 * df['Cross_section_fb'] * 1e3 * visible_br_lowest
    # df['simple_br'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], df['tanb'], 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3 * df['visible_br_without_4pi']
    # fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    # df_threshold = df[df['significance_with_4pi'] > 3]
    # df_threshold_without_4pi = df[df['significance_without_4pi'] > 3]
    df_threshold_lowest_br = df[df['significance_lowest_br'] > 3]
    # df_threshold_simple_br = df[df['simple_br'] > 3]
    def add_envelope(ax, x, y, label, color, interpolation='cubic'):
        """
        Add an envelope to the given axis and interpolate the upper and lower envelopes.

        Parameters:
        - ax: matplotlib axis object
        - x: x data (m)
        - y: y data (sin^2(tanb))
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

    # Remove rows with NaN or infinite values in relevant columns
    # df_threshold_lowest_br = df_threshold_lowest_br.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'tanb', 'detector_acceptance', 'significance_lowest_br'])
    df_threshold_lowest_br = df_threshold_lowest_br.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'tanb', 'detector_acceptance', 'significance_lowest_br'])
    # df_threshold_without_4pi = df_threshold_without_4pi.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'tanb', 'detector_acceptance', 'significance_without_4pi'])
    # Subplot 1: No 4Pi
    plt.scatter(df_threshold_lowest_br['m'], df_threshold_lowest_br['tanb'], label='Pessimistic', color='green', s=20, alpha=0.5)
    add_envelope(plt, df_threshold_lowest_br['m'], df_threshold_lowest_br['tanb'], 'Pessimistic', 'green', interpolation)
    plt.xlim(min(df['m']), 5)
    # axs[0, 0].set_ylim(1e-14, 1e-6)
    plt.title(r'Threshold $3\sigma$', fontsize=20)
    plt.xlabel('mass / GeV', fontsize=15)
    plt.ylabel(r'tan$\beta$', fontsize=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()




def plot_with_envelope_DDC_PRA(csv_file, DDC_file, interpolation='cubic', output_file='envelope_plot.png'):
    # 删除离群点
    df = pd.read_csv(csv_file)
    df_ddc = pd.read_csv(DDC_file)

    df['significance_with_4pi'] = df['detector_acceptance'] * fcal.calculate_Br(df['m'], np.sin(df['theta_input'])**2, 0.104, 0.653) * 300 * df['Cross_section_fb'] * 1e3 * df['visible_br']
    fig, axs = plt.subplots(1, 2, figsize=(20, 15))
    df_threshold = df[df['significance_with_4pi'] > 3]
    # print(df_threshold[:3])
    print(df_ddc[:3])
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

    # Remove rows with NaN or infinite values in relevant columns
    # df_threshold_lowest_br = df_threshold_lowest_br.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'theta_input', 'detector_acceptance', 'significance_lowest_br'])
    df_threshold = df_threshold.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'theta_input', 'detector_acceptance', 'significance_with_4pi'])
    # df_threshold_without_4pi = df_threshold_without_4pi.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'theta_input', 'detector_acceptance', 'significance_without_4pi'])
    # Subplot 1: No 4Pi
    add_envelope(axs[0], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'PRA Result', 'red', interpolation)
    axs[0].set_xlim(0.1, 5)
    axs[0].set_ylim(1e-12, 1e-6)
    axs[0].set_title(r'PRA Result', fontsize=20)
    axs[0].set_xlabel('mass / GeV', fontsize=15)
    axs[0].set_ylabel(r'sin$\theta^2$', fontsize=15)
    axs[0].set_yscale('log')
    # axs[0, 0].set_xscale('log')
    axs[0].legend()

    # Subplot 2: With 4Pi
    # add_envelope(axs[1], df_ddc['m'], np.sin(df_ddc['theta_input'])**2, 'DDC Result', 'blue', interpolation)
    df_ddc = df_ddc.sort_values(by='m')
    ddc_m_peak, ddc_theta_peak, ddc_m_trough, ddc_theta_trough = fine_envelope(df_ddc['m'], df_ddc['theta_input'])
    upper = np.interp(df_ddc['m'], ddc_m_peak, np.sin(ddc_theta_peak)**2)
    lower = np.interp(df_ddc['m'], ddc_m_trough, np.sin(ddc_theta_trough)**2)
    axs[1].plot(ddc_m_peak, np.sin(ddc_theta_peak)**2, 'o', color='blue', linestyle="--", alpha=0.5)
    axs[1].plot(ddc_m_trough, np.sin(ddc_theta_trough)**2, 'o', color='blue', linestyle="--", alpha=0.5)
    axs[1].fill_between(df_ddc['m'], lower, upper, color="blue", alpha=0.2, label="DDC Result")
    # axs[1].scatter(df_ddc['m'], np.sin(df_ddc['theta_input'])**2, color='blue', label='DDC Result', alpha=0.5)
    print('fine')
    axs[1].plot()
    
    add_envelope(axs[1], df_threshold['m'], np.sin(df_threshold['theta_input'])**2, 'PRA Result', 'red', interpolation)
    axs[1].set_xlim(0.1, 5)
    axs[1].set_ylim(1e-12, 1e-6)
    axs[1].set_title(r'Compare of PRA and DDC', fontsize=20)
    axs[1].set_xlabel('mass / GeV', fontsize=15)
    axs[1].set_ylabel(r'sin$\theta^2$', fontsize=15)
    axs[1].set_yscale('log')
    # axs[0, 1].set_xscale('log')
    axs[1].legend()


    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()



def plot_with_envelope_tanb_good(csv_file, interpolation='cubic', output_file='envelope_plot.png', visible_br_lowest=0.76):
    # 删除离群点
    df = pd.read_csv(csv_file)


    df['significance_lowest_br'] = df['detector_acceptance'] * fcal.calcu_Br_B_to_H(df['m'], df['tanb'], 1/df['tanb'], 600) * 300 * df['Cross_section_fb'] * 1e3 * visible_br_lowest
    plt.figure(figsize=(20, 15))
    df_threshold = df[df['significance_lowest_br'] > 3]
    # print(df_threshold[:3])
    
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

    # Remove rows with NaN or infinite values in relevant columns
    # df_threshold_lowest_br = df_threshold_lowest_br.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'theta_input', 'detector_acceptance', 'significance_lowest_br'])
    # df_threshold = df_threshold.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'tanb', 'detector_acceptance', 'significance_with_4pi'])
    # df_threshold_without_4pi = df_threshold_without_4pi.replace([np.inf, -np.inf], np.nan).dropna(subset=['m', 'theta_input', 'detector_acceptance', 'significance_without_4pi'])
    # Subplot 1: No 4Pi

    # Subplot 2: With 4Pi
    # add_envelope(axs[1], df_ddc['m'], np.sin(df_ddc['theta_input'])**2, 'DDC Result', 'blue', interpolation)
    df_threshold = df_threshold.sort_values(by='m')
    ddc_m_peak, ddc_theta_peak, ddc_m_trough, ddc_theta_trough = fine_envelope(df_threshold['m'], df_threshold['tanb'])
    upper = np.interp(df_threshold['m'], ddc_m_peak, np.sin(ddc_theta_peak)**2)
    lower = np.interp(df_threshold['m'], ddc_m_trough, np.sin(ddc_theta_trough)**2)
    plt.plot(ddc_m_peak, ddc_theta_peak, 'o', color='blue', linestyle="--", alpha=0.5)
    plt.plot(ddc_m_trough, ddc_theta_trough, 'o', color='blue', linestyle="--", alpha=0.5)
    plt.fill_between(df_threshold['m'], lower, upper, color="blue", alpha=0.2, label="2HDM-H Result")
    # axs[1].scatter(df_ddc['m'], np.sin(df_ddc['theta_input'])**2, color='blue', label='DDC Result', alpha=0.5)
    print('fine')
    
    # plt.xlim(0.1, 5)
    # plt.ylim(1e-12, 1e-6)
    plt.title(r'2HDM', fontsize=20)
    plt.xlabel('mass / GeV', fontsize=15)
    plt.ylabel(r'tanb', fontsize=15)
    plt.yscale('log')
    # axs[0, 1].set_xscale('log')
    plt.legend()

    plt.savefig(output_file)
    plt.show()
    plt.close()