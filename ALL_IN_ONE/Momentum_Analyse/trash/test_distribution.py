import h5py
import json
import numpy as np
from pathlib import Path

def debug_data_structure(data_dir: str):
    """调试数据结构"""
    data_path = Path(data_dir)
    
    print("=" * 70)
    print("DEBUGGING DATA STRUCTURE")
    print("=" * 70)
    
    # 查找第一个HDF5文件
    h5_files = list(data_path.rglob("*.h5"))
    print(f"Found {len(h5_files)} HDF5 files")
    
    if not h5_files:
        print("No HDF5 files found!")
        return
    
    # 检查前3个文件
    for i, h5_file in enumerate(h5_files[:3]):
        print(f"\n{'='*40}")
        print(f"File {i+1}: {h5_file.relative_to(data_path)}")
        print('='*40)
        
        try:
            with h5py.File(h5_file, 'r') as f:
                print(f"File size: {h5_file.stat().st_size / 1024**2:.2f} MB")
                print(f"Datasets: {list(f.keys())}")
                
                # 检查每个数据集
                for dataset_name in f.keys():
                    dataset = f[dataset_name]
                    print(f"\n  Dataset: {dataset_name}")
                    
                    if isinstance(dataset, h5py.Dataset):
                        print(f"    Shape: {dataset.shape}")
                        print(f"    Dtype: {dataset.dtype}")
                        
                        # 如果是positions数据集，查看一些样本
                        if dataset_name == 'positions':
                            if len(dataset) > 0:
                                print(f"    First 3 positions:")
                                for j in range(min(3, len(dataset))):
                                    print(f"      [{dataset[j][0]:.1f}, {dataset[j][1]:.1f}, {dataset[j][2]:.1f}]")
                    
                    elif isinstance(dataset, h5py.Group):
                        print(f"    Group with items: {list(dataset.keys())}")
                        
                        # 如果是parameters组，查看属性
                        if dataset_name == 'parameters':
                            print(f"    Attributes: {dict(dataset.attrs)}")
                
                # 检查根属性
                if f.attrs:
                    print(f"\n  Root attributes: {dict(f.attrs)}")
                    
        except Exception as e:
            print(f"Error reading {h5_file}: {e}")
    
    # 检查JSON文件
    json_files = list(data_path.rglob("*result.json"))
    print(f"\n{'='*40}")
    print(f"Found {len(json_files)} result JSON files")
    print('='*40)
    
    if json_files:
        json_file = json_files[0]
        print(f"\nFirst result file: {json_file.relative_to(data_path)}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                print(f"Keys: {list(data.keys())}")
                print(f"Values:")
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {type(value).__name__}")
        except Exception as e:
            print(f"Error reading JSON: {e}")

# 运行调试
data_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_40/llp_simulation_results/incremental_results"
debug_data_structure(data_dir)