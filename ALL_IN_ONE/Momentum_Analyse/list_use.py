# usage_example.py
"""
完整的使用示例 - 支持批量处理多个LLP参数文件
"""
import numpy as np
import pandas as pd
import os
import glob
import gc
import psutil
import humanize
from ALL_IN_ONE.Momentum_Analyse.Block_decay_new import LLPDecaySimulationPipeline, LLPBlockConfig
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def print_memory_usage(label=""):
    """打印当前内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{label} Memory usage: {humanize.naturalsize(mem_info.rss)}")

def clear_memory():
    """清理Python内存"""
    gc.collect()
    print("Memory cleared via garbage collection")

def process_llp_param_file(llp_params_file, pipeline_config):
    """处理单个LLP参数文件"""
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(llp_params_file)}")
    print(f"{'='*80}")
    
    # 创建输出子目录，以参数文件名命名
    base_name = os.path.splitext(os.path.basename(llp_params_file))[0]
    output_dir = os.path.join(
        pipeline_config['output_base_dir'],
        base_name
    )
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建配置对象
    decay_config = LLPBlockConfig(
        x_range=pipeline_config['x_range'],
        y_range=pipeline_config['y_range'],
        z_range=pipeline_config['z_range'],
        nx=pipeline_config['nx'],
        ny=pipeline_config['ny'],
        nz=pipeline_config['nz'],
        target_region=pipeline_config['target_region'],
        min_decays_in_region=pipeline_config['min_decays_in_region'],
        store_full_positions=True
    )
    
    # 创建并运行管道
    pipeline = LLPDecaySimulationPipeline(
        particle_blocks_dir=pipeline_config['particle_blocks_dir'],
        llp_params_file=llp_params_file,
        output_dir=output_dir,
        decay_config=decay_config
    )
    
    # 运行模拟
    result = pipeline.simulate_llp_decays_incremental(
        samples_per_block=pipeline_config['samples_per_block'],
        max_llp_params=pipeline_config['max_llp_params'],
        target_region=pipeline_config['target_region']
    )
    
    # 清理管道对象
    del pipeline
    return result

def batch_process_llp_files(params_folder, pipeline_config):
    """批量处理LLP参数文件夹中的所有文件"""
    
    print("🚀 Batch LLP Decay Analysis Workflow")
    print("="*80)
    
    # 查找所有CSV文件（或根据需要修改扩展名）
    pattern = os.path.join(params_folder, "*.csv")
    llp_files = sorted(glob.glob(pattern))
    
    if not llp_files:
        print(f"❌ No CSV files found in: {params_folder}")
        return []
    
    print(f"Found {len(llp_files)} LLP parameter files:")
    for i, file_path in enumerate(llp_files, 1):
        print(f"  {i:3d}. {os.path.basename(file_path)}")
    
    results = []
    processed_files = []
    
    try:
        for i, llp_file in enumerate(llp_files, 1):
            print(f"\n{'='*80}")
            print(f"File {i}/{len(llp_files)}")
            print(llp_file)
            print_memory_usage("Before processing")
            
            try:
                # 处理当前文件
                result = process_llp_param_file(llp_file, pipeline_config)
                results.append((llp_file, result))
                processed_files.append(llp_file)
                
                print(f"✓ Successfully processed: {os.path.basename(llp_file)}")
                
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(llp_file)}: {str(e)}")
                continue
            
            finally:
                # 每次处理后都清理内存
                print_memory_usage("After processing")
                clear_memory()
                print_memory_usage("After cleanup")
    
    except KeyboardInterrupt:
        print("\n⚠️ Batch processing interrupted by user")
    
    print(f"\n{'='*80}")
    print("Batch Processing Summary:")
    print(f"  Total files: {len(llp_files)}")
    print(f"  Successfully processed: {len(processed_files)}")
    print(f"  Failed: {len(llp_files) - len(processed_files)}")
    
    return results

def example_complete_workflow():
    """完整的LLP衰变分析工作流示例"""
    
    print("🚀 Complete LLP Decay Analysis Workflow")
    print("="*80)
    
    # ============================================
    # 步骤1: 配置目标区域
    # ============================================
    # target_region = {
    #     'x_min': 26000,   # mm
    #     'x_max': 36000,   # mm
    #     'y_min': -7000,   # mm
    #     'y_max': 3000,    # mm
    #     'z_min': 5000,    # mm
    #     'z_max': 15000    # mm
    # }
    
    target_region = {
        'x_min': 100000,   # mm
        'x_max': 120000,   # mm
        'y_min': -100000,   # mm
        'y_max': 100000,    # mm
        'z_min': 100000,    # mm
        'z_max': 300000    # mm
    }

    min_decays_threshold = 0  # 目标区域最小衰变数
    
    # ============================================
    # 步骤2: 管道配置（所有文件共享的配置）
    # ============================================
    pipeline_config = {
        'particle_blocks_dir': '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/14TeV/14TeV_Block',
        'output_base_dir': '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/14TeV',
        # 'x_range': (80000, 200000),
        # 'y_range': (-200000, 200000),
        # 'z_range': (100000, 300000),
        # 'nx': 300,
        # 'ny': 160,
        # 'nz': 200,
        'x_range': (16000, 46000),
        'y_range': (-10000, 10000),
        'z_range': (3000, 23000),
        'nx': 300,
        'ny': 200,
        'nz': 200,
        'target_region': target_region,
        'min_decays_in_region': min_decays_threshold,
        'samples_per_block': 50,
        'max_llp_params': None,  # 设为None处理所有参数
    }
    
    # ============================================
    # 步骤3: 批量处理文件夹中的所有LLP参数文件
    # ============================================
    llp_params_folder = '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/LLP_paras'
    
    print(f"\nStep 1: Batch processing LLP parameter files from: {llp_params_folder}")
    print_memory_usage("Initial")
    
    results = batch_process_llp_files(llp_params_folder, pipeline_config)
    
    print("\n" + "="*80)
    print("Workflow Complete!")
    print(f"Results saved in: {pipeline_config['output_base_dir']}/")
    print("="*80)
    
    return results

def process_specific_files(file_list, pipeline_config):
    """处理指定的文件列表"""
    
    print("🚀 Processing Specific LLP Files")
    print("="*80)
    
    results = []
    
    for i, llp_file in enumerate(file_list, 1):
        print(f"\n{'='*80}")
        print(f"File {i}/{len(file_list)}: {os.path.basename(llp_file)}")
        
        # 检查文件是否存在
        if not os.path.exists(llp_file):
            print(f"❌ File not found: {llp_file}")
            continue
        
        print_memory_usage("Before processing")
        
        try:
            # 处理当前文件
            result = process_llp_param_file(llp_file, pipeline_config)
            results.append((llp_file, result))
            
            print(f"✓ Successfully processed: {os.path.basename(llp_file)}")
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(llp_file)}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理内存
            print_memory_usage("After processing")
            clear_memory()
    
    return results

if __name__ == "__main__":
    # 运行完整示例
    print("Starting batch LLP decay analysis workflow...")
    
    # 方法1: 处理文件夹中的所有文件
    results = example_complete_workflow()
    
    # 方法2: 处理指定的文件列表（如果需要）
    """
    pipeline_config = {
        'particle_blocks_dir': '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Decay_B/13TeV/13TeV_Block',
        'output_base_dir': '/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/13TeV_LLP/test_scan_B/llp_simulation_results',
        'x_range': (16000, 46000),
        'y_range': (-10000, 6000),
        'z_range': (3000, 23000),
        'nx': 300,
        'ny': 160,
        'nz': 200,
        'target_region': {
            'x_min': 26000,
            'x_max': 36000,
            'y_min': -7000,
            'y_max': 3000,
            'z_min': 5000,
            'z_max': 15000
        },
        'min_decays_in_region': 0,
        'samples_per_block': 50,
        'max_llp_params': 2,
    }
    
    specific_files = [
        '/path/to/file1.csv',
        '/path/to/file2.csv',
        # ... 添加更多文件
    ]
    
    results = process_specific_files(specific_files, pipeline_config)
    """