import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path

# Import module by file path to be independent of package layout
spec = importlib.util.spec_from_file_location(
    "blockify_sample",
    str(Path(__file__).resolve().parents[1] / 'ALL_IN_ONE' / 'Momentum_Analyse' / 'blockify_sample.py')
)
blockify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blockify)

BlockConfig = blockify.BlockConfig
HierarchicalParticleBlocking = blockify.HierarchicalParticleBlocking


def make_small_df(n=100):
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'decay_x': rng.uniform(-1000, 1000, size=n),
        'decay_y': rng.uniform(-1000, 1000, size=n),
        'decay_z': rng.uniform(-1000, 1000, size=n),
        'px': rng.normal(0, 1, size=n),
        'py': rng.normal(0, 1, size=n),
        'pz': rng.normal(0, 1, size=n),
        'e': rng.uniform(0.1, 10.0, size=n)
    })
    return df


def test_momentum_log_clustering_runs():
    df = make_small_df(200)
    config = BlockConfig(
        momentum_n_clusters=10,
        momentum_scale='log',
        momentum_log_eps=1e-12,
        position_bin_size=500.0,
        use_position_clustering=False
    )
    hb = HierarchicalParticleBlocking(data_path=df, config=config, output_dir='./tmp_blocks_test')
    labels = hb.create_momentum_clusters()
    assert len(labels) == len(df)
    # should give cluster centers
    assert hasattr(hb, 'momentum_cluster_centers')


def test_momentum_log_magnitude_mode_runs():
    df = make_small_df(300)
    config = BlockConfig(
        momentum_n_clusters=8,
        momentum_scale='log',
        momentum_log_mode='magnitude',
        momentum_log_eps=1e-12,
        position_bin_size=500.0,
        use_position_clustering=False
    )
    hb = HierarchicalParticleBlocking(data_path=df, config=config, output_dir='./tmp_blocks_test')
    labels = hb.create_momentum_clusters()
    assert len(labels) == len(df)
    assert hasattr(hb, 'momentum_cluster_centers')
    # In magnitude mode, centers are stored as a dict with 'p_mag_log'
    assert isinstance(hb.momentum_cluster_centers, dict)
    assert 'p_mag_log' in hb.momentum_cluster_centers


def test_position_log_magnitude_binning_runs():
    df = make_small_df(500)
    config = BlockConfig(
        momentum_n_clusters=8,
        momentum_scale='linear',
        position_bin_size=500.0,
        position_scale='log',
        position_log_mode='magnitude',
        position_log_n_bins=5,
        use_position_clustering=False
    )
    hb = HierarchicalParticleBlocking(data_path=df, config=config, output_dir='./tmp_blocks_test_pos')
    blocks = hb.create_blocks()
    assert len(blocks) > 0
    # check that at least one block id uses radial bin naming
    assert any('pos_r_' in bid for bid in blocks.keys())


if __name__ == '__main__':
    test_momentum_log_clustering_runs()
    test_momentum_log_magnitude_mode_runs()
    test_position_log_magnitude_binning_runs()
    print('all tests passed')
