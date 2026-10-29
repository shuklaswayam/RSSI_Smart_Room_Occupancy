import numpy as np
import pandas as pd
import pytest
from occ.config import load_config
from occ.features.extractor import FEATURE_COLUMNS, extract_dataset_features

def test_pipeline_shape_equality_real_and_sim():
    cfg = load_config()
    t = np.arange(0.0, 30.0, 0.1)
    
    clean_df = pd.concat([
        pd.DataFrame({"t_host": t, "receiver_id": "R1", "rssi_dbm": -45.0 + np.random.normal(0, 1, len(t))}),
        pd.DataFrame({"t_host": t, "receiver_id": "R2", "rssi_dbm": -52.0 + np.random.normal(0, 1, len(t))})
    ], ignore_index=True)

    events_df = pd.DataFrame([
        {"t_host": 0.0, "event_type": "state_start", "value": "Empty"}
    ])

    meta_real = {"run_id": "REAL_01", "origin": "real"}
    meta_sim = {"run_id": "SIM_01", "origin": "sim", "seed": 1}

    df_feats_real = extract_dataset_features(clean_df, events_df, meta_real, cfg)
    df_feats_sim = extract_dataset_features(clean_df, events_df, meta_sim, cfg)

    # Schema and column shapes must be identical
    assert list(df_feats_real.columns) == list(df_feats_sim.columns)
    assert len(df_feats_real) == len(df_feats_sim)
    assert set(FEATURE_COLUMNS).issubset(set(df_feats_real.columns))
