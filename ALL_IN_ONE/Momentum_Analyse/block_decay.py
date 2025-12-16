import numpy as np
import pandas as pd
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import h5py
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from decay_sim import two_body_decay_lab, generate_decay_position
from blockify_sample import BlockConfig, ParticleBlock, HierarchicalParticleBlocking
from dataloader import BlockDataLoader
from decay_weight import compute_weighted_statistics
class LLPAnalysisPipeline:
    """
    LLP衰变分析完整流程
    """
    
    def __init__(self, 
                 particle_data_file: str,
                 llp_params_file: str,
                 blocks_output_dir: str = './blocks_output',
                 llp_output_dir: str = './llp_results'):
        self.particle_data_file = particle_data_file
        self.llp_params_file = llp_params_file
        self.blocks_output_dir = blocks_output_dir
        self.llp_output_dir = llp_output_dir
        
        # 创建输出目录
        Path(llp_output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_full_analysis(self, 
                         blocking_config: BlockConfig = None,
                         samples_per_block: int = 100,
                         llp_mass_range: Tuple[float, float] = None):
        """
        运行完整分析流程
        
        Args:
            blocking_config: 分块配置
            samples_per_block: 每块抽样数
            llp_mass_range: LLP质量范围筛选
        """
        print("=" * 60)
        print("Starting LLP Analysis Pipeline")
        print("=" * 60)
        
        # # 步骤1: 数据分块
        # print("\n[Step 1] Creating hierarchical particle blocks...")
        # blocker = HierarchicalParticleBlocking(
        #     data_path=self.particle_data_file,
        #     output_dir=self.blocks_output_dir,
        #     config=blocking_config or BlockConfig()
        # )
        
        # blocks = blocker.create_blocks()
        # blocker.save_blocks(format='parquet')
        
        # 步骤2: 加载块数据
        print("\n[Step 2] Loading block data for LLP analysis...")
        loader = BlockDataLoader(self.blocks_output_dir)
        
        # 步骤3: 加载LLP参数
        llp_params = pd.read_csv(self.llp_params_file)
        
        if llp_mass_range:
            llp_params = llp_params[
                (llp_params['mH'] >= llp_mass_range[0]) & 
                (llp_params['mH'] <= llp_mass_range[1])
            ]
        
        print(f"Loaded {len(llp_params)} LLP parameter sets")
        
        # 步骤4: 对每个LLP参数进行分析
        results = []
        
        for _, llp_row in tqdm(llp_params.iterrows(), total=len(llp_params), desc="Processing LLP parameters"):
            llp_mass = llp_row['mH']
            llp_lifetime = llp_row['ltime']
            
            # 为当前LLP参数筛选合适的块
            # 例如，根据能量阈值筛选
            suitable_blocks = loader.get_blocks_by_momentum(
                min_energy=llp_mass + 0.1  # 能量必须大于LLP质量
            )
            
            if not suitable_blocks:
                continue
            
            # 从每个块抽样并计算LLP衰变
            all_decay_positions = []
            all_weights = []
            
            for block_id in suitable_blocks:
                # 抽样粒子
                sampled_particles = loader.sample_from_block(
                    block_id=block_id,
                    n_samples=samples_per_block,
                    strategy='importance'  # 重要性抽样
                )
                
                # 对每个抽样粒子计算LLP衰变
                for _, particle in sampled_particles.iterrows():
                    birth_position = particle[['decay_x', 'decay_y', 'decay_z']].values
                    momentum = particle[['e', 'px', 'py', 'pz']].values
                    
                    try:
                        llp_momentum, _ = two_body_decay_lab(momentum, 5.279, llp_mass, 0.494)
                        decay_pos, _ = generate_decay_position(llp_lifetime, llp_momentum, birth_position)
                        
                        if not np.any(np.isnan(decay_pos)):
                            all_decay_positions.append(decay_pos)
                            # 权重 = 1 / 抽样率
                            all_weights.append(len(sampled_particles) / samples_per_block)
                            
                    except (ValueError, ZeroDivisionError):
                        continue
            
            if all_decay_positions:
                # 计算统计
                stats = compute_weighted_statistics(
                    np.array(all_decay_positions),
                    np.array(all_weights)
                )
                
                results.append({
                    'mass': llp_mass,
                    'lifetime': llp_lifetime,
                    'tanb': llp_row.get('tanb', 0.0),
                    'vis_br': llp_row.get('Br_visible', 0.0),
                    **stats,
                    'blocks_used': len(suitable_blocks),
                    'total_samples': len(all_decay_positions)
                })
        
        # 保存结果
        results_df = pd.DataFrame(results)
        results_file = Path(self.llp_output_dir) / 'llp_analysis_results.csv'
        results_df.to_csv(results_file, index=False)
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {results_file}")
        print(f"Total LLP parameter sets processed: {len(results)}")
        
        return results_df
    












#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Process particle data for LLP analysis')
    parser.add_argument('--input', type=str, required=True, help='Input particle data file')
    parser.add_argument('--llp-params', type=str, required=True, help='LLP parameters file')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--grid-size', type=int, default=10, help='Position grid size')
    parser.add_argument('--momentum-clusters', type=int, default=8, help='Number of momentum clusters')
    parser.add_argument('--samples-per-block', type=int, default=100, help='Samples per block for LLP analysis')
    parser.add_argument('--min-mass', type=float, default=0.1, help='Minimum LLP mass to analyze')
    parser.add_argument('--max-mass', type=float, default=5.0, help='Maximum LLP mass to analyze')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input file: {args.input}")
    print(f"LLP params: {args.llp_params}")
    print(f"Output directory: {args.output}")
    
    # 配置
    config = BlockConfig(
        momentum_n_clusters=args.momentum_clusters,
        position_bin_size=args.grid_size,
        use_position_clustering=False,
        compression='gzip'
    )
    
    try:
        # 步骤1: 数据分块
        print("\nStep 1: Creating particle blocks...")
        blocker = HierarchicalParticleBlocking(
            data_path=args.input,
            output_dir=str(output_dir / 'particle_blocks'),
            config=config
        )
        blocks = blocker.create_blocks()
        blocker.save_blocks(format='parquet')
        
        # 步骤2: LLP分析
        print("\nStep 2: Running LLP analysis...")
        pipeline = LLPAnalysisPipeline(
            particle_data_file=args.input,
            llp_params_file=args.llp_params,
            blocks_output_dir=str(output_dir / 'particle_blocks'),
            llp_output_dir=str(output_dir / 'llp_results')
        )
        
        results = pipeline.run_full_analysis(
            blocking_config=config,
            samples_per_block=args.samples_per_block,
            llp_mass_range=(args.min_mass, args.max_mass)
        )
        
        print(f"\nProcessing complete!")
        print(f"Results saved in: {output_dir}")
        
        # 生成报告
        report = f"""
        Processing Report
        =================
        Input file: {args.input}
        LLP parameters: {args.llp_params}
        Total blocks created: {len(blocks)}
        Total LLP parameters analyzed: {len(results) if results is not None else 0}
        Output directory: {output_dir}
        
        Generated files:
        - {output_dir}/particle_blocks/global_index.parquet
        - {output_dir}/particle_blocks/blocks/*/particles.parquet
        - {output_dir}/llp_results/llp_analysis_results.csv
        """
        
        with open(output_dir / 'processing_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()