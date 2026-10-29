import numpy as np
import pandas as pd
import pytest
from occ.features.extractor import FEATURE_COLUMNS, compute_single_link_stats, compute_window_features

def test_single_link_feature_math():
    t = np.arange(0.0, 5.0, 0.1)  # 50 points
    # Simple linear ramp
    vals = 2.0 * t - 50.0
    stats = compute_single_link_stats(vals, t)
    
    assert abs(stats["slope"] - 2.0) < 1e-3
    assert abs(stats["min"] - (-50.0)) < 1e-3
    assert abs(stats["max"] - (-40.2)) < 1e-3
    assert stats["std"] > 0.0

def test_window_features_exact_columns_and_shape():
    t = np.arange(0.0, 5.0, 0.1)
    df_window = pd.concat([
        pd.DataFrame({"t_host": t, "receiver_id": "R1", "rssi_dbm": -45.0 + np.random.normal(0, 1, len(t))}),
        pd.DataFrame({"t_host": t, "receiver_id": "R2", "rssi_dbm": -52.0 + np.random.normal(0, 1, len(t))})
    ], ignore_index=True)

    feats = compute_window_features(df_window, baseline={"R1": -45.0, "R2": -52.0})
    assert feats is not None
    assert len(feats) == len(FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        assert col in feats, f"Missing feature column: {col}"
        assert not np.isnan(feats[col]), f"NaN value found in feature: {col}"

def test_relative_baseline_features():
    t = np.arange(0.0, 5.0, 0.1)
    # R1 at -55 dBm with baseline at -45 dBm => delta should be ~ -10 dBm
    df_window = pd.concat([
        pd.DataFrame({"t_host": t, "receiver_id": "R1", "rssi_dbm": -55.0}),
        pd.DataFrame({"t_host": t, "receiver_id": "R2", "rssi_dbm": -60.0})
    ], ignore_index=True)

    feats = compute_window_features(df_window, baseline={"R1": -45.0, "R2": -50.0})
    assert abs(feats["R1_delta_mean"] - (-10.0)) < 1e-3
    assert abs(feats["R2_delta_mean"] - (-10.0)) < 1e-3
