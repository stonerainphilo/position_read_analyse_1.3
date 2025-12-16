from decay_sim import two_body_decay_lab, generate_decay_position

import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, List, Optional
import json
from tqdm import tqdm

def calculate_llp_decay_from_block(
    block_row: pd.Series,
    llp_mass: float,
    llp_lifetime: float,
    num_samples_per_block: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate LLP decay positions for a given block
    
    Parameters:
        block_row: Row containing block data
        llp_mass: LLP mass in GeV
        llp_lifetime: LLP lifetime in mm/c
        num_samples_per_block: Number of Monte Carlo samples per block
    
    Returns:
        decay_positions: Array of decay positions (n, 3)
        weights: Array of weights for each position (n,)
    """
    # Get block parameters
    birth_position = block_row[['decay_x_center', 'decay_y_center', 'decay_z_center']].values.astype(np.float64)
    momentum = block_row[['e_511_center', 'px_511_center', 'py_511_center', 'pz_511_center']].values.astype(np.float64)
    data_count = int(block_row['data_count'])
    
    # Calculate weight for each sample
    weight = data_count / num_samples_per_block
    
    # Generate multiple samples for this block to account for stochastic variations
    decay_positions = []
    weights = []
    
    for _ in range(num_samples_per_block):
        try:
            # Generate LLP from B meson decay
            llp_momentum, _ = two_body_decay_lab(momentum, 5.279, llp_mass, 0.494)
            
            # Generate decay position
            decay_pos, _ = generate_decay_position(llp_lifetime, llp_momentum, birth_position)
            
            if not np.any(np.isnan(decay_pos)):
                decay_positions.append(decay_pos)
                weights.append(weight)
                
        except (ValueError, ZeroDivisionError):
            continue
    
    if not decay_positions:
        return np.empty((0, 3)), np.empty(0)
    
    return np.array(decay_positions), np.array(weights)

def compute_weighted_statistics(
    positions: np.ndarray,
    weights: np.ndarray,
    reference_point: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute weighted statistics for decay positions
    
    Parameters:
        positions: Array of positions (n, 3)
        weights: Array of weights (n,)
        reference_point: Reference point for relative positions
    
    Returns:
        Dictionary of statistics
    """
    if reference_point is None:
        reference_point = np.zeros(3)
    
    # Convert to relative positions
    rel_positions = positions - reference_point
    
    # Calculate weighted statistics
    weighted_sum = np.sum(weights)
    
    if weighted_sum == 0:
        return {
            'total_weighted_events': 0,
            'mean_position': [0, 0, 0],
            'std_position': [0, 0, 0],
            'min_position': [0, 0, 0],
            'max_position': [0, 0, 0]
        }
    
    # Weighted mean
    weighted_mean = np.average(rel_positions, axis=0, weights=weights)
    
    # Weighted variance (using unbiased estimator)
    weighted_var = np.average((rel_positions - weighted_mean) ** 2, 
                             axis=0, weights=weights)
    weighted_var *= weighted_sum / (weighted_sum - 1)  # Bessel's correction
    
    # Weighted percentiles (approximate using sorted positions)
    sorted_indices = np.argsort(rel_positions[:, 0])
    median_idx = np.searchsorted(np.cumsum(weights[sorted_indices]), 
                                weighted_sum / 2)
    median_x = rel_positions[sorted_indices[median_idx], 0] if median_idx < len(positions) else 0
    
    return {
        'total_weighted_events': float(weighted_sum),
        'mean_position': weighted_mean.tolist(),
        'std_position': np.sqrt(weighted_var).tolist(),
        'median_x': float(median_x),
        'min_position': np.min(rel_positions, axis=0).tolist(),
        'max_position': np.max(rel_positions, axis=0).tolist(),
        'total_blocks': len(np.unique(weights))  # Approximate number of contributing blocks
    }

def compute_weighted_histograms(
    positions: np.ndarray,
    weights: np.ndarray,
    bins_config: Optional[Dict] = None
) -> Dict:
    """
    Compute weighted histograms for decay positions
    
    Parameters:
        positions: Array of positions (n, 3)
        weights: Array of weights (n,)
        bins_config: Configuration for histogram bins
    
    Returns:
        Dictionary with histogram data
    """
    if bins_config is None:
        # Auto-determine bin ranges based on weighted percentiles
        weighted_positions = positions * weights[:, np.newaxis]
        pos_ranges = []
        
        for i in range(3):
            # Use weighted 1st and 99th percentiles to determine range
            sorted_idx = np.argsort(positions[:, i])
            cum_weights = np.cumsum(weights[sorted_idx])
            total_weight = cum_weights[-1]
            
            if total_weight > 0:
                # Find indices for percentiles
                p1_idx = np.searchsorted(cum_weights, total_weight * 0.01)
                p99_idx = np.searchsorted(cum_weights, total_weight * 0.99)
                
                min_val = positions[sorted_idx[max(0, p1_idx-1)], i]
                max_val = positions[sorted_idx[min(len(positions)-1, p99_idx)], i]
                
                # Add 10% margin
                margin = (max_val - min_val) * 0.1
                pos_ranges.append((min_val - margin, max_val + margin))
            else:
                pos_ranges.append((-1000, 1000))
    
    histograms = {}
    axis_names = ['x', 'y', 'z']
    
    for i, axis in enumerate(axis_names):
        if bins_config and axis in bins_config:
            bin_edges = np.linspace(*bins_config[axis])
        else:
            bin_edges = np.linspace(pos_ranges[i][0], pos_ranges[i][1], 101)
        
        # Compute weighted histogram
        hist, edges = np.histogram(positions[:, i], bins=bin_edges, weights=weights)
        
        # Normalize to probability density
        bin_widths = np.diff(edges)
        density = hist / (bin_widths * np.sum(weights)) if np.sum(weights) > 0 else hist
        
        histograms[axis] = {
            'bin_edges': edges.astype(np.float32).tolist(),
            'counts': hist.astype(np.uint32).tolist(),
            'density': density.astype(np.float32).tolist()
        }
    
    # Compute radial distribution
    r = np.sqrt(np.sum(positions**2, axis=1))
    r_max = np.max(r) * 1.1 if len(r) > 0 else 1000
    r_edges = np.linspace(0, r_max, 101)
    
    hist_r, edges_r = np.histogram(r, bins=r_edges, weights=weights)
    bin_widths_r = np.diff(edges_r)
    density_r = hist_r / (bin_widths_r * np.sum(weights)) if np.sum(weights) > 0 else hist_r
    
    histograms['r'] = {
        'bin_edges': edges_r.astype(np.float32).tolist(),
        'counts': hist_r.astype(np.uint32).tolist(),
        'density': density_r.astype(np.float32).tolist()
    }
    
    return histograms

def process_blocks_llp_decay(
    blocks_file: str,
    llp_params_file: str,
    output_dir: str,
    samples_per_block: int = 50,
    store_histograms: bool = True,
    store_summary: bool = True,
    reference_point: Optional[Tuple[float, float, float]] = None
):
    """
    Process block data to compute LLP decay position distributions
    
    Parameters:
        blocks_file: Path to CSV file with block data
        llp_params_file: Path to CSV file with LLP parameters
        output_dir: Output directory for results
        samples_per_block: Monte Carlo samples per block
        store_histograms: Whether to store histogram data
        store_summary: Whether to store summary statistics
        reference_point: Reference point for relative coordinates
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    print(f"Reading block data from {blocks_file}")
    blocks_df = pd.read_csv(blocks_file)
    print(f"Reading LLP parameters from {llp_params_file}")
    llp_params_df = pd.read_csv(llp_params_file)
    
    if reference_point is None:
        # Use average block position as reference
        reference_point = [0, 0, 0]
    
    all_results = []
    
    # Get counts for progress bar
    total_llp_sets = len(llp_params_df)
    total_blocks = len(blocks_df)
    
    print(f"\nProcessing {total_llp_sets} LLP parameter sets with {total_blocks} blocks each")
    print(f"Samples per block: {samples_per_block}")
    print(f"Total Monte Carlo samples: {total_llp_sets * total_blocks * samples_per_block:,}")
    print("-" * 50)
    
    # Process each LLP parameter set with progress bar
    pbar_llp = tqdm(llp_params_df.iterrows(), total=total_llp_sets, 
                   desc="Processing LLP parameters", unit="LLP")
    
    for _, llp_row in pbar_llp:
        llp_mass = llp_row['mH']
        llp_lifetime = llp_row['ltime']
        # tanb = llp_row.get('tanb', 0.0)
        tanb = llp_row['tanb']
        vis_br = llp_row['Br_visible']
        
        # Update progress bar description with current parameters
        pbar_llp.set_description(f"LLP: m={llp_mass:.3f}, τ={llp_lifetime:.2e}")
        
        # Initialize accumulators
        all_positions = []
        all_weights = []
        
        # Process each block with a nested progress bar (only if not too many blocks)
        if total_blocks <= 100:  # Only show for manageable number of blocks
            block_pbar = tqdm(blocks_df.iterrows(), total=total_blocks, 
                            desc="Processing blocks", unit="block", leave=False)
            block_iter = block_pbar
        else:
            block_iter = blocks_df.iterrows()
        
        for _, block_row in block_iter:
            positions, weights = calculate_llp_decay_from_block(
                block_row, llp_mass, llp_lifetime, samples_per_block
            )
            
            if len(positions) > 0:
                all_positions.append(positions)
                all_weights.append(weights)
        
        if hasattr(block_iter, 'close'):  # Close nested progress bar if it exists
            block_iter.close()
        
        if not all_positions:
            tqdm.write(f"Warning: No valid decay positions for mass={llp_mass}, lifetime={llp_lifetime}")
            continue
        
        # Combine all samples
        all_positions = np.vstack(all_positions)
        all_weights = np.concatenate(all_weights)
        
        # Compute statistics
        stats = compute_weighted_statistics(all_positions, all_weights, reference_point)
        
        # Compute histograms if requested
        histograms = {}
        if store_histograms:
            histograms = compute_weighted_histograms(all_positions, all_weights)
        
        # Prepare result entry
        result = {
            'mass': float(llp_mass),
            'lifetime': float(llp_lifetime),
            'tanb': float(tanb),
            'vis_br': float(vis_br),
            **stats
        }
        
        all_results.append(result)
        
        # Save individual LLP results if histograms are stored
        if store_histograms:
            # Create filename
            mass_str = f"{llp_mass:.6e}".replace('.', 'p').replace('+', '').replace('-', 'm')
            tau_str = f"{llp_lifetime:.6e}".replace('.', 'p').replace('+', '').replace('-', 'm')
            filename = f"llp_m{mass_str}_t{tau_str}_tb{tanb:.2f}"
            
            # Save histogram data as compressed JSON
            hist_data = {
                'parameters': {
                    'mass': llp_mass,
                    'lifetime': llp_lifetime,
                    'tanb': tanb,
                    'vis_br': vis_br
                },
                'statistics': stats,
                'histograms': histograms
            }
            
            with open(os.path.join(output_dir, f"{filename}_hist.json"), 'w') as f:
                json.dump(hist_data, f, separators=(',', ':'))  # Compact JSON
            
            # Also save as compressed CSV for histogram data
            hist_csv_data = []
            for axis, hist in histograms.items():
                edges = hist['bin_edges']
                counts = hist['counts']
                density = hist['density']
                
                for i in range(len(counts)):
                    hist_csv_data.append({
                        'axis': axis,
                        'bin_low': edges[i],
                        'bin_high': edges[i+1],
                        'count': counts[i],
                        'density': density[i]
                    })
            
            hist_csv_df = pd.DataFrame(hist_csv_data)
            hist_csv_df.to_csv(
                os.path.join(output_dir, f"{filename}_hist.csv.gz"),
                index=False,
                compression='gzip'
            )
    
    pbar_llp.close()
    
    # Save summary statistics for all LLP parameters
    if store_summary and all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Optimize column types for storage
        for col in ['mean_position_x', 'mean_position_y', 'mean_position_z',
                   'std_position_x', 'std_position_y', 'std_position_z',
                   'min_position_x', 'min_position_y', 'min_position_z',
                   'max_position_x', 'max_position_y', 'max_position_z']:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].astype(np.float32)
        
        # Save as compressed CSV
        summary_path = os.path.join(output_dir, "llp_decay_summary.csv.gz")
        summary_df.to_csv(summary_path, index=False, compression='gzip')
        
        # Also save as Parquet for better compression
        try:
            summary_df.to_parquet(
                os.path.join(output_dir, "llp_decay_summary.parquet"),
                compression='gzip',
                index=False
            )
        except ImportError:
            pass  # Parquet not available
        
        print(f"\n{'='*50}")
        print(f"Analysis Complete!")
        print(f"Summary saved with {len(all_results)} LLP parameter sets")
        print(f"Total weighted events: {summary_df['total_weighted_events'].sum():.0f}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*50}")
    
    return all_results

def quick_llp_distribution(
    blocks_file: str,
    llp_params_file: str,
    output_file: str,
    samples_per_block: int = 20
):
    """
    Quick analysis with minimal storage
    
    Parameters:
        blocks_file: Path to block data CSV
        llp_params_file: Path to LLP parameters CSV
        output_file: Output file path
        samples_per_block: Samples per block for Monte Carlo
    """
    
    # Read data
    print(f"Reading block data from {blocks_file}")
    blocks_df = pd.read_csv(blocks_file)
    print(f"Reading LLP parameters from {llp_params_file}")
    llp_params_df = pd.read_csv(llp_params_file)
    
    total_llp_sets = len(llp_params_df)
    total_blocks = len(blocks_df)
    
    print(f"\nQuick Analysis: {total_llp_sets} LLP sets, {total_blocks} blocks")
    print(f"Samples per block: {samples_per_block}")
    print("-" * 50)
    
    results = []
    
    # Use tqdm for progress bar
    pbar = tqdm(llp_params_df.iterrows(), total=total_llp_sets, 
                desc="Processing LLP parameters", unit="LLP")
    
    for _, llp_row in pbar:
        llp_mass = llp_row['mH']
        llp_lifetime = llp_row['ltime']
        tanb = llp_row.get('tanb', 0.0)
        
        # Update progress bar
        pbar.set_description(f"LLP: m={llp_mass:.3f}, τ={llp_lifetime:.2e}")
        
        # Use block centers directly (no Monte Carlo for quick analysis)
        decay_positions = []
        weights = []
        
        for _, block_row in blocks_df.iterrows():
            birth_position = block_row[['decay_x_center', 'decay_y_center', 'decay_z_center']].values
            momentum = block_row[['e_511_center', 'px_511_center', 'py_511_center', 'pz_511_center']].values
            data_count = int(block_row['data_count'])
            
            try:
                llp_momentum, _ = two_body_decay_lab(momentum, 5.279, llp_mass, 0.494)
                decay_pos, _ = generate_decay_position(llp_lifetime, llp_momentum, birth_position)
                
                if not np.any(np.isnan(decay_pos)):
                    decay_positions.append(decay_pos)
                    weights.append(data_count)
                    
            except (ValueError, ZeroDivisionError):
                continue
        
        if not decay_positions:
            continue
        
        decay_positions = np.array(decay_positions)
        weights = np.array(weights)
        
        # Simple statistics
        weighted_mean = np.average(decay_positions, axis=0, weights=weights)
        weighted_std = np.sqrt(
            np.average((decay_positions - weighted_mean)**2, axis=0, weights=weights)
        )
        
        # Radial distribution
        r = np.sqrt(np.sum(decay_positions**2, axis=1))
        r_mean = np.average(r, weights=weights)
        r_std = np.sqrt(np.average((r - r_mean)**2, weights=weights))
        
        results.append({
            'mass': float(llp_mass),
            'lifetime': float(llp_lifetime),
            'tanb': float(tanb),
            'mean_x': float(weighted_mean[0]),
            'mean_y': float(weighted_mean[1]),
            'mean_z': float(weighted_mean[2]),
            'std_x': float(weighted_std[0]),
            'std_y': float(weighted_std[1]),
            'std_z': float(weighted_std[2]),
            'mean_r': float(r_mean),
            'std_r': float(r_std),
            'total_weight': float(np.sum(weights)),
            'n_blocks': len(decay_positions),
            'vis_br': float(llp_row['Br_visible'])
        })
    
    pbar.close()
    
    # Save minimal results
    results_df = pd.DataFrame(results)
    
    # Use efficient numeric types
    float_cols = results_df.select_dtypes(include=[np.float64]).columns
    results_df[float_cols] = results_df[float_cols].astype(np.float32)
    
    # Save as compressed CSV
    results_df.to_csv(output_file, index=False, compression='gzip')
    
    print(f"\n{'='*50}")
    print(f"Quick Analysis Complete!")
    print(f"Processed {len(results)} LLP parameters")
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    print(f"{'='*50}")
    
    return results_df

# Example usage
if __name__ == "__main__":
    # Paths to your data files
    BLOCKS_FILE = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Decay_B/B_511/B_511_block_stats_grid.csv"  # Contains: block_id,decay_x_center,decay_y_center,decay_z_center,data_count,density,px_511_center,py_511_center,pz_511_center,e_511_center
    # BLOCKS_FILE = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/PRA/Github/position_read_analyse_1.3/ALL_IN_ONE/Momentum_Analyse/block_stats_grid.csv"
    LLP_PARAMS_FILE = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Program/Light_scalar_decay/Combined_Code/2HDM_H_B_decay_A.csv"  # Contains: mH, ltime, tanb (optional)
    OUTPUT_DIR = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Block/Test/LLP_DECAY/B_511_2HDMH_A/"
    
    print("LLP Decay Position Analysis")
    print("=" * 50)
    
    # 首先检查tqdm是否可用
    try:
        from tqdm import tqdm
        print("✓ tqdm progress bar available")
    except ImportError:
        print("⚠ tqdm not available. Installing with: pip install tqdm")
        import subprocess
        subprocess.check_call(["pip", "install", "tqdm"])
        from tqdm import tqdm
        print("✓ tqdm installed successfully")
    
    print(f"Blocks file: {BLOCKS_FILE}")
    print(f"LLP params file: {LLP_PARAMS_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    # Option 1: Full analysis with histograms
    print("\n[Option 1] Full Analysis with Histograms")
    print("-" * 30)
    results = process_blocks_llp_decay(
        blocks_file=BLOCKS_FILE,
        llp_params_file=LLP_PARAMS_FILE,
        output_dir=OUTPUT_DIR,
        samples_per_block=100000,  # Adjust based on desired accuracy
        store_histograms=True,
        store_summary=True
    )
    
    # Option 2: Quick minimal analysis
    # print("\n[Option 2] Quick Minimal Analysis")
    # print("-" * 30)
    # quick_output = os.path.join(OUTPUT_DIR, "llp_quick_results.csv.gz")
    # quick_llp_distribution(
    #     blocks_file=BLOCKS_FILE,
    #     llp_params_file=LLP_PARAMS_FILE,
    #     output_file=quick_output,
    #     samples_per_block=2000
    # )
    
    print("\n" + "=" * 50)
    print("All analyses completed successfully!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 50)
