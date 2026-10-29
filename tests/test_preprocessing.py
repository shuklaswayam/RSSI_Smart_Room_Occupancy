import numpy as np
import pandas as pd
import pytest
from occ.preprocess.cleaner import clean_rssi_stream, ema_filter, median_filter, resample_grid

def test_median_filter_spike_rejection():
    # Sequence with clean baseline and a 15 dB impulsive spike
    raw = pd.Series([-50.0, -50.2, -35.0, -50.1, -49.8])
    filtered = median_filter(raw, k=3)
    # The spike at index 2 should be completely suppressed
    assert abs(filtered.iloc[2] - (-50.1)) < 0.5
    assert filtered.iloc[2] < -45.0

def test_ema_smoothing():
    raw = pd.Series([-50.0, -60.0, -50.0, -60.0, -50.0])
    ema = ema_filter(raw, alpha=0.25)
    # Variance of EMA should be substantially lower than raw
    assert np.var(ema) < np.var(raw) * 0.5

def test_resample_grid_and_gap_interpolation():
    # Asynchronous timestamps around 10 Hz
    t = [0.00, 0.11, 0.22, 0.31, 0.62]  # Gap of ~0.3s (3 missing slots) between 0.31 and 0.62
    rssi = [-45.0, -46.0, -45.0, -47.0, -48.0]
    df = pd.DataFrame({"t_host": t, "receiver_id": "R1", "rssi_dbm": rssi})
    
    grid_df = resample_grid(df, target_freq=10.0, max_gap_slots=5)
    assert not grid_df.empty
    # Timestamps should be rounded to 0.1s increments
    diffs = np.diff(grid_df["t_host"].to_numpy())
    assert np.allclose(diffs, 0.1, atol=1e-3)
    # Interpolated slots should not be NaN
    assert grid_df["rssi_dbm"].isna().sum() == 0

def test_clean_rssi_stream_pipeline():
    df_raw = pd.DataFrame({
        "t_host": [0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3, 0.4],
        "receiver_id": ["R1", "R1", "R1", "R1", "R1", "R2", "R2", "R2", "R2", "R2"],
        "rssi_dbm": [-45.0, -45.0, -30.0, -45.0, -45.0, -52.0, -52.0, -51.0, -53.0, -52.0]
    })
    df_clean = clean_rssi_stream(df_raw)
    assert len(df_clean) > 0
    assert set(df_clean["receiver_id"].unique()) == {"R1", "R2"}
    assert df_clean["rssi_dbm"].isna().sum() == 0
