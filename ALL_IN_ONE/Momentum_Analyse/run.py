#!/usr/bin/env python3
"""
批量处理脚本：run.py
用法：run.py --input /path/to/data --output ./results
"""

import argparse
import sys
from pathlib import Path
from blockify_sample import HierarchicalParticleBlocking, BlockConfig
from block_decay import LLPAnalysisPipeline
def main():
    parser = argparse.ArgumentParser(description='Process particle data for LLP analysis')
    parser.add_argument('--input', type=str, required=True, help='Input particle data file')
    parser.add_argument('--llp-params', type=str, required=True, help='LLP parameters file')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--grid-size', type=int, default=10, help='Position grid size')
    parser.add_argument('--momentum-clusters', type=int, default=3000, help='Number of momentum clusters')
    parser.add_argument('--samples-per-block', type=int, default=10000, help='Samples per block for LLP analysis')
    parser.add_argument('--min-mass', type=float, default=0.01, help='Minimum LLP mass to analyze')
    parser.add_argument('--max-mass', type=float, default=5.01, help='Maximum LLP mass to analyze')
    
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