import pytest
from occ.calibration.engine import AdaptiveCalibrationEngine

def test_adaptive_calibration_drift_recovery():
    # Initial baseline R1=-45, R2=-52
    engine = AdaptiveCalibrationEngine(
        initial_baseline={"R1": -45.0, "R2": -52.0},
        gamma=0.1,
        min_empty_windows=3,
        confidence_thresh=0.85
    )

    # Injected environmental drift: room is empty, but level shifted by +3 dB to R1=-42, R2=-49
    drifted_empty_feats = {"R1_mean": -42.0, "R2_mean": -49.0, "R1_std": 1.2, "R2_std": 1.3}

    # First 2 windows: under min_empty_windows threshold (no update yet)
    for _ in range(2):
        status = engine.update("Empty", 0.95, drifted_empty_feats)
        assert not status["updated"]

    # Subsequent windows: should adaptively update towards drifted level
    for _ in range(10):
        status = engine.update("Empty", 0.95, drifted_empty_feats)

    curr_b = engine.get_baseline()
    # Baseline should have adapted significantly towards -42 / -49
    assert curr_b["R1"] > -44.0
    assert curr_b["R2"] > -51.0

def test_adaptive_calibration_anti_poisoning_lock():
    # Verify that during Occupied or high variance windows, the baseline is strictly LOCKED
    engine = AdaptiveCalibrationEngine(
        initial_baseline={"R1": -45.0, "R2": -52.0},
        gamma=0.1,
        min_empty_windows=3
    )

    # Occupied window (stationary person absorbing signal)
    occupied_feats = {"R1_mean": -52.0, "R2_mean": -58.0, "R1_std": 1.4, "R2_std": 1.5}
    for _ in range(10):
        status = engine.update("Occupied", 0.95, occupied_feats)
        assert not status["updated"]

    curr_b = engine.get_baseline()
    # Baseline must remain strictly unchanged
    assert curr_b["R1"] == -45.0
    assert curr_b["R2"] == -52.0
