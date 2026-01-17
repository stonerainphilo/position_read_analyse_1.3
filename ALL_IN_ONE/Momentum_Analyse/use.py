# usage_example.py
"""
完整的使用示例
"""
import numpy as np
import pandas as pd
from ALL_IN_ONE.Momentum_Analyse.Block_decay_new import LLPDecaySimulationPipeline, LLPBlockConfig
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
def example_complete_workflow():
    """完整的LLP衰变分析工作流示例"""
    
    print("🚀 Complete LLP Decay Analysis Workflow")
    print("="*70)
    
    # ============================================
    # 步骤1: 配置目标区域
    # ============================================
    target_region = {
        'x_min': 26000,   # mm
        'x_max': 36000,   # mm
        'y_min': -7000,   # mm
        'y_max': 3000,    # mm
        'z_min': 5000,    # mm
        'z_max': 15000    # mm
    }
    
    min_decays_threshold = 0  # 目标区域最小衰变数
    
    # ============================================
    # 步骤2: 配置LLP衰变分块
    # ============================================
    decay_config = LLPBlockConfig(
        x_range=(16000, 46000),
        y_range=(-10000, 6000),
        z_range=(3000, 23000),
        nx=300,
        ny=160,
        nz=200,
        target_region=target_region,
        min_decays_in_region=min_decays_threshold,
        store_full_positions=True
    )
    
    # ============================================
    # 步骤3: 创建模拟管道
    # ============================================
    pipeline = LLPDecaySimulationPipeline(
        particle_blocks_dir='/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Decay_B/13TeV/13TeV_Block',  # 母粒子分块目录
        llp_params_file='/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/LLP_paras/2HDM_H_B_decay_A1.csv',     # LLP参数文件
        output_dir='/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/13TeV_LLP/test_scan_B/llp_simulation_results',
        decay_config=decay_config
    )
    # ============================================
    # 步骤4: 运行模拟
    # ============================================
    print("\nStep 1: Running LLP decay simulation...")
    pipeline.simulate_llp_decays_incremental(
        samples_per_block=50,        # 每个母粒子块抽样50个
        max_llp_params=2,         # 处理所有参数（设为10用于快速测试）
        target_region=target_region
    )
    pipeline.__init__()

    print("\n" + "="*70)
    print("Workflow Complete!")
    print("Results saved in: ./llp_simulation_results/")
    print("="*70)
    
    return pipeline


if __name__ == "__main__":
    # 运行完整示例
    print("Starting complete LLP decay analysis workflow...")
    pipeline = example_complete_workflow()
