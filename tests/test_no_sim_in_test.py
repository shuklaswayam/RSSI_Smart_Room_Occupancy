import pandas as pd
import pytest
from occ.features.extractor import FEATURE_COLUMNS
from scripts.validate import load_real_dataset

def test_leakage_prevention_no_sim_in_test(tmp_path):
    # Create mixed dataframe with both real and sim rows
    df_mixed = pd.DataFrame({
        "window_id": ["w1", "w2", "w3", "w4"],
        "run_id": ["run_real_01", "run_real_02", "SIM_S1_seed01", "SIM_S2_seed01"],
        "origin": ["real", "real", "sim", "sim"],
        "label": ["Empty", "Occupied", "Empty", "Occupied"],
        **{c: [1.0, 2.0, 1.0, 2.0] for c in FEATURE_COLUMNS}
    })
    
    test_csv = tmp_path / "mixed_features.csv"
    df_mixed.to_csv(test_csv, index=False)

    # load_real_dataset must strictly filter out all origin == 'sim' records
    df_clean = load_real_dataset(test_csv)
    assert len(df_clean) == 2
    assert (df_clean["origin"] == "real").all()
    assert "SIM" not in "".join(df_clean["run_id"])
