# usage_example.py
"""
完整的使用示例
"""
import numpy as np
import pandas as pd
from Block_decay_new import LLPDecaySimulationPipeline, LLPBlockConfig
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
        particle_blocks_dir='/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18',  # 母粒子分块目录
        llp_params_file='/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/test/2HDM_H_B_decay_1.csv',     # LLP参数文件
        output_dir='/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_18/llp_simulation_results',
        decay_config=decay_config
    )
    
    # ============================================
    # 步骤4: 运行模拟
    # ============================================
    print("\nStep 1: Running LLP decay simulation...")
    pipeline.simulate_llp_decays(
        samples_per_block=20,        # 每个母粒子块抽样50个
        max_llp_params=2,         # 处理所有参数（设为10用于快速测试）
        target_region=target_region
    )
    
    # ============================================
    # 步骤5: 分析结果
    # ============================================
    print("\nStep 2: Analyzing results...")
    pipeline.analyze_and_visualize(
        region=target_region,
        min_decays=min_decays_threshold
    )
    
    # ============================================
    # 步骤6: 详细分析特定LLP参数
    # ============================================
    print("\nStep 3: Detailed analysis of specific LLP parameters...")
    
    # 加载分析器
    analyzer = pipeline.analyzer
    
    # 获取衰变数最多的参数
    filtered_df = analyzer.get_blocks_in_region(target_region, min_decays_threshold)
    if len(filtered_df) > 0:
        top_param = filtered_df.nlargest(1, 'decays_in_region').iloc[0]
        
        print(f"\nTop performing LLP parameter:")
        print(f"  Mass: {top_param['mass']:.3f} GeV")
        print(f"  tanβ: {top_param['tanb']:.2f}")
        print(f"  Decays in region: {top_param['decays_in_region']:.0f}")
        print(f"  Block ID: {top_param['block_id']}")
        
        # 加载该块进行详细分析
        try:
            llp_block = analyzer.load_block(top_param['block_id'])
            
            print(f"\nDetailed block analysis:")
            print(f"  Total weighted events: {llp_block.total_weighted_events:.0f}")
            print(f"  Unique positions: {len(llp_block.positions)}")
            
            if llp_block.density_map:
                print(f"  Mean density: {llp_block.density_map['mean_density']:.2e} decays/mm³")
                print(f"  Max density: {llp_block.density_map['max_density']:.2e} decays/mm³")
            
            # 绘制该LLP的空间分布
            fig = plt.figure(figsize=(15, 5))
            
            # X-Z投影
            ax1 = fig.add_subplot(131)
            h_xz, x_edges, z_edges = np.histogram2d(
                llp_block.positions[:, 0], llp_block.positions[:, 2],
                bins=[decay_config.nx, decay_config.nz],
                weights=llp_block.weights,
                range=[[decay_config.x_range[0], decay_config.x_range[1]], 
                      [decay_config.z_range[0], decay_config.z_range[1]]]
            )
            im1 = ax1.pcolormesh(x_edges, z_edges, h_xz.T, cmap='hot', norm=LogNorm())
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Z (mm)')
            ax1.set_title(f'X-Z Projection\nm={top_param["mass"]:.2f}GeV, tanβ={top_param["tanb"]:.2f}')
            plt.colorbar(im1, ax=ax1, label='Decays')
            
            # Y-Z投影
            ax2 = fig.add_subplot(132)
            h_yz, y_edges, z_edges = np.histogram2d(
                llp_block.positions[:, 1], llp_block.positions[:, 2],
                bins=[decay_config.ny, decay_config.nz],
                weights=llp_block.weights,
                range=[[decay_config.y_range[0], decay_config.y_range[1]], 
                      [decay_config.z_range[0], decay_config.z_range[1]]]
            )
            im2 = ax2.pcolormesh(y_edges, z_edges, h_yz.T, cmap='hot', norm=LogNorm())
            ax2.set_xlabel('Y (mm)')
            ax2.set_ylabel('Z (mm)')
            ax2.set_title('Y-Z Projection')
            plt.colorbar(im2, ax=ax2, label='Decays')
            
            # X-Y投影（在目标Z范围内）
            ax3 = fig.add_subplot(133)
            z_mask = (llp_block.positions[:, 2] >= target_region['z_min']) & \
                    (llp_block.positions[:, 2] <= target_region['z_max'])
            positions_in_region = llp_block.positions[z_mask]
            weights_in_region = llp_block.weights[z_mask]
            
            if len(positions_in_region) > 0:
                h_xy, x_edges, y_edges = np.histogram2d(
                    positions_in_region[:, 0], positions_in_region[:, 1],
                    bins=[50, 50],
                    weights=weights_in_region,
                    range=[[target_region['x_min'], target_region['x_max']], 
                          [target_region['y_min'], target_region['y_max']]]
                )
                im3 = ax3.pcolormesh(x_edges, y_edges, h_xy.T, cmap='viridis')
                ax3.set_xlabel('X (mm)')
                ax3.set_ylabel('Y (mm)')
                ax3.set_title(f'X-Y Projection in Target Z\n({target_region["z_min"]}-{target_region["z_max"]} mm)')
                plt.colorbar(im3, ax=ax3, label='Decays')
            
            plt.tight_layout()
            plt.savefig('./llp_simulation_results/visualization/top_llp_spatial_dist.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error loading block: {e}")
    
    print("\n" + "="*70)
    print("Workflow Complete!")
    print("Results saved in: ./llp_simulation_results/")
    print("="*70)
    
    return pipeline


if __name__ == "__main__":
    # 运行完整示例
    print("Starting complete LLP decay analysis workflow...")
    pipeline = example_complete_workflow()