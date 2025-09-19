import sys
import os
import product_test as pt
from numba import jit
import cross_section as cs
from tqdm import tqdm
import pandas as pd
import numpy as np


def mkdir_1(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path

# Convert planes data to Numba-compatible format
def prepare_planes_data():
    """准备Numba兼容的平面数据"""
    planes_normals = []
    planes_D = []
    
    # make sure planes are calculated
    if not hasattr(pt, 'planes'):
        # if not, calculate them
        SHiPvertices = np.array([
            [750, 2150, 45000],  # Bottom Vertices
            [750, -2150, 45000],
            [-750, -2150, 45000],
            [-750, 2150, 45000],
            [2500, 5000, 95000],   # Top Vertices
            [2500, -5000, 95000],
            [-2500, -5000, 95000],
            [-2500, 5000, 95000]])
        
        SHiPfaces = [
            [0, 1, 2, 3],  # Bottom face
            [4, 5, 6, 7],  # Top face
            [0, 1, 5, 4],  # Side faces
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7]]
        
        # Calculate frustum center
        center = np.mean(SHiPvertices, axis=0)
        
        # Calculate planes
        pt.planes = []
        for face in SHiPfaces:
            p0 = SHiPvertices[face[0]]
            p1 = SHiPvertices[face[1]]
            p2 = SHiPvertices[face[2]]
            
            v1 = p1 - p0
            v2 = p2 - p0
            
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            
            D = -np.dot(normal, p0)
            
            # make sure normal points outward
            distance_to_center = np.dot(normal, center) + D
            if distance_to_center > 0:
                normal = -normal
                D = -np.dot(normal, p0)
            
            pt.planes.append((normal, D))
    
    # transfer to Numba-compatible format
    for normal, D in pt.planes:
        planes_normals.append(np.array(normal, dtype=np.float64))
        planes_D.append(np.float64(D))
    
    return planes_normals, planes_D

# Prepare planes data once
planes_normals, planes_D = prepare_planes_data()

@jit(nopython=True)
def check_points_inside_frustum(points_array, planes_normals, planes_D):
    """Chack multiple points inside frustum using Numba"""
    results = np.empty(points_array.shape[0], dtype=np.bool_)
    
    for i in range(points_array.shape[0]):
        point = points_array[i]
        inside = True
        
        for j in range(len(planes_normals)):
            normal = planes_normals[j]
            D_val = planes_D[j]
            distance = normal[0]*point[0] + normal[1]*point[1] + normal[2]*point[2] + D_val
            
            if distance > 1e-6:
                inside = False
                break
        
        results[i] = inside
    
    return results

def add_whether_in_the_detector_without_Decay_calcu_add_cross_section_SHiP(filename, out_folder_path):
    mkdir_1(out_folder_path)
    
    file_path_only, file_name_only = os.path.split(filename)
    file_parent_path_only = os.path.dirname(file_path_only)
    
    llp_data = pd.read_csv(filename)
    
    points = llp_data[['decay_pos_x', 'decay_pos_y', 'decay_pos_z']].values
    
    llp_whether_in_detector_SHiP = check_points_inside_frustum(points, planes_normals, planes_D)
    cross_section = cs.calculate_cross_section(llp_data)
    
    llp_data['detected_SHiP'] = llp_whether_in_detector_SHiP
    llp_data['cross_section'] = cross_section
    
    total_llp = cs.counting_total_LLP(llp_data)
    if total_llp > 0:
        llp_data['detector_acceptance_SHiP'] = sum(llp_data['detected_SHiP']) / total_llp
    else:
        llp_data['detector_acceptance_SHiP'] = 0.0
    
    final_data_folder = os.path.join(file_parent_path_only, 'Completed_llp_data_precise_cross_section')
    mkdir_1(final_data_folder)
    
    final_data_path = os.path.join(final_data_folder, f'final_data_cross_section_{file_name_only}')
    llp_data.to_csv(final_data_path, index=False)
    
    return final_data_folder

def SHiP(LLP_data_folder_dir):
    completed_data_folder = None
    csv_files = [f for f in os.listdir(LLP_data_folder_dir) if f.endswith('.csv')]
    
    for files in tqdm(csv_files, desc="Processing files"):
        file_path_all = os.path.join(LLP_data_folder_dir, files)
        try:
            completed_data_folder = add_whether_in_the_detector_without_Decay_calcu_add_cross_section_SHiP(
                file_path_all, LLP_data_folder_dir
            )
        except Exception as e:
            print(f"Error with file: {file_path_all}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return 'Detection and Calcu Cross-Section Completed', completed_data_folder

