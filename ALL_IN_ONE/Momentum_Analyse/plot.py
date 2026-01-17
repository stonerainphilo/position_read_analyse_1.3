import matplotlib.pyplot as plt
import sys
sys.path.append("/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/ALL_IN_ONE")
# sys.path.append("ALL_IN_ONE/LSD")
# sys.path.append("ALL_IN_ONE/FIT_FUNC")
import pandas as pd
import Pythia8.functions_for_calculation as fcal
import numpy as np
import mplhep as hep
plt.style.use(hep.style.LHCb)
def plot_Detector(detector: str, data_path: str, output_path: str, luminosity: float, confidence: float, 
                  other_lables: str = None, llp_coupling_para: str = 'tanb', llp_coupling_name: str = 'mH'):
    df = pd.read_csv(data_path)
    df['significance_lowest'] = df['fraction_in_region'] * fcal.calculate_Br(df[llp_coupling_name], df[llp_coupling_para], 0.104) * df['Cross_section_fb'] * df['cross_section']  * df['vis_br'] * luminosity

    # size_scatter = 30
    fig, axs = plt.subplots(1, 2, figsize=(20, 15))
    # df_threshold = df[df['significance_with_4pi'] > 3]
    df_lowest = df[df['significance_lowest'] > confidence]
    plt.figure(figsize=(15, 15))
    plt.scatter(df_lowest['m'], np.sin(df_lowest['theta'])**2, label='2HDM-S SHiP', color='green', s=20, alpha=0.3)
    plt.xlim(0.1, 5)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('mass / GeV', fontsize=15)
    plt.ylabel(r'sin\theta ^2$', fontsize=15)
    plt.title(r'Higgs Scalar', fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('SHiP_2HDM-S_test.png')
    plt.show()
    plt.close()
    
    
    return 0

def plot_for_test(file_path, outdir):
    df_original = pd.read_csv(file_path)
    df = df_original[df_original['decays_in_region'] > 0]
    plt.figure(figsize=(15, 15))
    plt.scatter(df['mass'], df['tanb'], label='2HDM-H CODEX-b', color='green', s=20, alpha=0.3)
    plt.xlim(0.1, 5)
    plt.ylim(1,1e6)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('mass / GeV', fontsize=15)
    plt.ylabel(r'tanb', fontsize=15)
    plt.title(r'Higgs Doublet CP-Even', fontsize=20)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'{outdir}/Higgs_CPEven_CODEX-b_test.png')
    # plt.show()
    plt.close()
    
    return 0

def plot_for_Res(file_path, outdir, Detector = 'CODEX-b'):
    df_original = pd.read_csv(file_path)
    df = df_original[df_original['probability'] > 0]
    plt.figure(figsize=(15, 15))
    plt.scatter(df['mass'], df['tanb'], label=f'2HDM-H {Detector}', color='green', s=20, alpha=0.3)
    plt.xlim(0.1, 5)
    plt.ylim(1,1e6)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('mass / GeV', fontsize=15)
    plt.ylabel(r'tanb', fontsize=15)
    plt.title(r'Higgs Doublet CP-Even', fontsize=20)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'{outdir}/Higgs_CPEven_{Detector}_test.png')
    # plt.show()
    plt.close()
    
    return 0

# plot_for_test('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_178/llp_simulation_results/incremental_summary_all.csv', '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_178')
# plot_for_test('/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/Summing_up/all.csv', '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/Summing_up/')
def plot(file, out):
    df = pd.read_csv(file)
    plt.figure(figsize=(15, 15))
    plt.plot(df['m'], df['tanb'], color='green', markersize=5, linestyle='-', alpha=0.3)
    plt.xlim(0.1, 5)
    plt.ylim(1,1e6)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('mass / GeV', fontsize=15)
    plt.ylabel(r'tanb', fontsize=15)
    plt.title(r'Higgs Doublet CP-Eve New Detector', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{out}/Higgs_CPEven_New_Example.png')
    plt.close()
    return 0

plot('/media/ubuntu/SRPPS/Results/C.csv', '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/Summing_up/')
# csv_file = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/14TeV_LLP_Distribution/New Detector Example_detailed.csv'
# out_path = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/14TeV_LLP_Distribution/'
# plot_for_Res(csv_file, out_path, Detector="New_Detector_Sample")