import time
import numpy as np
import pytest
from occ.config import load_config
from occ.db import fetch_predictions, init_db
from occ.inference.engine import RealtimeInferenceEngine
from occ.models.baselines import ThresholdClassifier
from occ.models.classifier import get_random_forest, load_model, save_model, validate_feature_vector

def test_end_to_end_streaming_pipeline_threshold():
    """Verify live sliding window streaming with heuristic baseline."""
    cfg = load_config()
    init_db(cfg)
    
    # Initialize engine with baseline model
    model = ThresholdClassifier()
    engine = RealtimeInferenceEngine(cfg=cfg, model=model)

    # Ingest 60 samples at 10 Hz (6.0 seconds of data)
    t_base = time.time()
    for i in range(60):
        t_pkt = t_base + (i * 0.1)
        # Empty room signal levels
        engine.push_sample(t_pkt, "R1", -45.0 + np.random.normal(0, 1.0))
        engine.push_sample(t_pkt + 0.005, "R2", -52.0 + np.random.normal(0, 1.0))

    # Trigger window evaluation
    res = engine.process_window(run_id="e2e_test_thresh", origin="real")

    assert res is not None
    assert res["voted_label"] in ["Empty", "Occupied", "Transition"]
    assert res["latency_ms"] < 2000.0  # Must satisfy O5 latency budget!
    assert res["meets_budget"] is True

    # Verify prediction persisted in SQLite
    preds_df = fetch_predictions(run_id="e2e_test_thresh", limit=5)
    assert not preds_df.empty
    assert preds_df.iloc[0]["run_id"] == "e2e_test_thresh"

def test_end_to_end_streaming_pipeline_random_forest():
    """Verify live sliding window streaming with primary trained Random Forest."""
    cfg = load_config()
    init_db(cfg)

    # Train a minimal RF or load serialized
    rf = get_random_forest(n_estimators=10)
    X_synth = np.random.randn(30, 21)
    y_synth = np.array(["Empty"] * 10 + ["Occupied"] * 10 + ["Transition"] * 10)
    rf.fit(X_synth, y_synth)

    engine = RealtimeInferenceEngine(cfg=cfg, model=rf)

    # Ingest 60 samples representing occupied state
    t_base = time.time()
    for i in range(60):
        t_pkt = t_base + (i * 0.1)
        engine.push_sample(t_pkt, "R1", -58.0 + np.random.normal(0, 2.5))
        engine.push_sample(t_pkt + 0.005, "R2", -64.0 + np.random.normal(0, 2.5))

    res = engine.process_window(run_id="e2e_test_rf", origin="real")
    assert res is not None
    assert res["voted_label"] in ["Empty", "Occupied", "Transition"]
    assert res["confidence"] >= 0.0
    assert res["latency_ms"] < 2000.0

def test_mle_model_bundle_lifecycle(tmp_path):
    """Test model serialization, bundle metadata packaging, and loading parity."""
    rf = get_random_forest(n_estimators=5)
    X = np.random.randn(20, 21)
    y = np.array(["Empty"] * 10 + ["Occupied"] * 10)
    rf.fit(X, y)

    artifact_path = tmp_path / "test_rf.joblib"
    meta = {"version": "2.0.0", "author": "MLE Reviewer", "origin": "hybrid"}
    saved = save_model(rf, out_path=str(artifact_path), metadata=meta)
    assert saved.is_file()
    assert (tmp_path / "test_rf.bundle.joblib").is_file()

    # Load back and verify behavior
    loaded_model = load_model(str(artifact_path))
    preds = loaded_model.predict(X[:2])
    assert len(preds) == 2

def test_feature_vector_validation_and_sanitization():
    """Test defensive schema checks and NaN/Inf cleaning."""
    # Valid 21-dim vector
    v_clean = np.ones((1, 21))
    out = validate_feature_vector(v_clean)
    assert out.shape == (1, 21)

    # 1D vector auto-reshaped
    v_1d = np.ones(21)
    out_1d = validate_feature_vector(v_1d)
    assert out_1d.shape == (1, 21)

    # NaN / Inf sanitization
    v_dirty = np.array([np.nan, np.inf, -np.inf] + [1.0] * 18)
    out_sanitized = validate_feature_vector(v_dirty)
    assert not np.isnan(out_sanitized).any()
    assert not np.isinf(out_sanitized).any()
    assert out_sanitized[0, 0] == 0.0

    # Wrong dimension must fail closed
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        validate_feature_vector(np.ones((1, 10)))
