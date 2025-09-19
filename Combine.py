import sys
sys.path.append("ALL_IN_ONE/Pythia8")
import ALL_IN_ONE.Pythia8.SHiP as ship
import ALL_IN_ONE.Pythia8.combine as combine

folder = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Original_DATA/SHiP-A/2025-09-15_Scalar_SHiP/LLP_data'
folder2 = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Original_DATA/SHiP-A/2025-09-15_Scalar_SHiP/Completed_llp_data_precise_cross_section'
# ship.SHiP(folder)
combine.combine_files_precise_SHiP(folder2)