def quick_diagnostic(data_dir, llp_ids=None):
    """
    快速诊断几个LLP的分布问题
    """
    # 加载分析器
    analyzer = LLPDistributionAnalyzer(data_dir)
    analyzer.load_all_data()
    
    # 如果没有指定LLP，使用前几个
    if llp_ids is None:
        llp_ids = list(analyzer.llp_data.keys())[:5]
    
    print(f"\nRunning quick diagnostic for {len(llp_ids)} LLPs...")
    
    for llp_id in llp_ids:
        if llp_id not in analyzer.llp_data:
            print(f"Warning: {llp_id} not found in data")
            continue
        
        data = analyzer.llp_data[llp_id]
        positions = data['positions']
        weights = data['weights']
        params = data['params']
        
        print(f"\n{'='*60}")
        print(f"LLP: {llp_id}")
        print(f"Mass: {params.get('mass', 'N/A'):.3f} GeV")
        print(f"Lifetime: {params.get('lifetime', 'N/A'):.2e} mm")
        print('='*60)
        
        for coord_idx, coord in enumerate(['x', 'y', 'z']):
            coord_data = positions[:, coord_idx]
            weights_norm = weights / np.sum(weights)
            
            # 基本统计
            mean = np.average(coord_data, weights=weights)
            std = np.sqrt(np.average((coord_data - mean)**2, weights=weights))
            skew = stats.skew(coord_data)
            kurtosis = stats.kurtosis(coord_data)
            
            print(f"\n{coord.upper()}:")
            print(f"  Samples: {len(coord_data):,}")
            print(f"  Mean ± Std: {mean:.3f} ± {std:.3f} mm")
            print(f"  Range: [{np.min(coord_data):.3f}, {np.max(coord_data):.3f}] mm")
            print(f"  Skewness: {skew:.3f} (0 for symmetric)")
            print(f"  Kurtosis: {kurtosis:.3f} (0 for normal)")
            
            # 检查是否常数
            if std < 1e-6:
                print(f"  ⚠ WARNING: Essentially constant!")
                continue
            
            # 正态性测试
            if len(coord_data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(coord_data)
                print(f"  Shapiro-Wilk test: p={shapiro_p:.3e} ({'NORMAL' if shapiro_p > 0.05 else 'NOT NORMAL'})")
            
            # 创建快速可视化
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # 直方图
            axes[0].hist(coord_data, bins=30, weights=weights_norm, 
                        density=True, alpha=0.7)
            axes[0].set_xlabel(f'{coord.upper()} (mm)')
            axes[0].set_ylabel('Density')
            axes[0].set_title(f'{coord.upper()} Distribution')
            axes[0].grid(True, alpha=0.3)
            
            # Q-Q图
            stats.probplot(coord_data, dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q Plot vs Normal')
            axes[1].grid(True, alpha=0.3)
            
            # 对数尺度
            hist, bins = np.histogram(coord_data, bins=30, weights=weights_norm, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            axes[2].semilogy(bin_centers, hist, 'o-')
            axes[2].set_xlabel(f'{coord.upper()} (mm)')
            axes[2].set_ylabel('Log Density')
            axes[2].set_title('Log-scale')
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'{llp_id} - {coord.upper()} Coordinate', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'quick_diagnostic_{llp_id}_{coord}.png', 
                       dpi=120, bbox_inches='tight')
            plt.close()
            
            print(f"  Plot saved: quick_diagnostic_{llp_id}_{coord}.png")
    
    print(f"\n✓ Quick diagnostic completed!")


# 使用示例
if __name__ == "__main__":
    data_dir = "/media/ubuntu/6156e08b-fdb1-4cde-964e-431f74a6078e/Files/LLP_DATA/Test/B_blocks/test_scan_40/llp_simulation_results/incremental_results"
    
    # 诊断特定LLP
    quick_diagnostic(data_dir, llp_ids=['llp_0117', 'llp_0123', 'llp_0135'])