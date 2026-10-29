from collections import deque
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from occ.calibration.engine import AdaptiveCalibrationEngine
from occ.config import load_config
from occ.db import log_prediction
from occ.features.extractor import FEATURE_COLUMNS, compute_window_features
from occ.inference.vote import MajorityVoter
from occ.models.classifier import load_model
from occ.preprocess.cleaner import clean_rssi_stream

class RealtimeInferenceEngine:
    """
    Near-Real-Time Inference Service (Hero 1).
    Ingests live or replayed RSSI packets, buffers them in sliding windows,
    extracts features, executes classification, applies majority voting,
    and updates the Hero 3 adaptive calibration engine.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, model: Optional[Any] = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        
        # Load classifier
        if model is not None:
            self.model = model
        else:
            try:
                self.model = load_model()
            except Exception:
                from occ.models.baselines import ThresholdClassifier
                self.model = ThresholdClassifier()

        self.window_s = float(cfg.get("windowing", {}).get("window_s", 5.0))
        self.step_s = float(cfg.get("windowing", {}).get("step_s", 2.5))
        self.min_samples = int(cfg.get("windowing", {}).get("min_samples", 30))
        
        # Buffer of raw incoming packets: dicts of {'t_host', 'receiver_id', 'rssi_dbm'}
        self.packet_buffer = deque(maxlen=500)
        self.last_inference_time = 0.0
        
        # Helpers
        vote_k = int(cfg.get("inference", {}).get("vote_window_size", 3))
        self.voter = MajorityVoter(window_size=vote_k)
        self.calibrator = AdaptiveCalibrationEngine()
        
    def push_sample(self, t_host: float, receiver_id: str, rssi_dbm: float):
        """Ingest a single RSSI packet from hardware serial or mock stream."""
        self.packet_buffer.append({
            "t_host": float(t_host),
            "receiver_id": str(receiver_id),
            "rssi_dbm": float(rssi_dbm)
        })

    def process_window(self, run_id: str = "live", origin: str = "real") -> Optional[Dict[str, Any]]:
        """
        Evaluate currently buffered window if sufficient time and samples exist.
        Measures total compute latency from window close to prediction return.
        """
        if len(self.packet_buffer) < self.min_samples:
            return None

        t_now = self.packet_buffer[-1]["t_host"]
        if t_now - self.last_inference_time < self.step_s:
            return None

        t_start_compute = time.time()
        t_window_start = t_now - self.window_s

        # Slice window samples
        window_samples = [p for p in self.packet_buffer if p["t_host"] >= t_window_start]
        if len(window_samples) < self.min_samples:
            return None

        df_raw = pd.DataFrame(window_samples)
        
        # 1. Condition signal
        df_clean = clean_rssi_stream(df_raw, self.cfg)
        
        # 2. Extract features against current adaptive baseline
        curr_baseline = self.calibrator.get_baseline()
        feats = compute_window_features(df_clean, baseline=curr_baseline)
        if feats is None:
            return None

        # 3. Model inference with schema validation & NaN protection
        feat_vector = np.array([[feats.get(c, 0.0) for c in FEATURE_COLUMNS]], dtype=float)
        feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=0.0, neginf=0.0)
        pred_label = str(self.model.predict(feat_vector)[0])
        
        # Determine confidence
        confidence = 0.90
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feat_vector)[0]
            confidence = float(np.max(probs))
            
        # 4. Temporal majority voting filter
        voted_label = self.voter.add(pred_label)
        
        # 5. Adaptive calibration step
        calib_status = self.calibrator.update(voted_label, confidence, feats)
        
        # 6. Measure latency in milliseconds
        latency_ms = round((time.time() - t_start_compute) * 1000.0, 2)
        self.last_inference_time = t_now
        
        # 7. Log to SQLite
        try:
            log_prediction(
                run_id=run_id,
                timestamp=t_now,
                predicted_label=pred_label,
                confidence=round(confidence, 3),
                voted_label=voted_label,
                latency_ms=latency_ms,
                origin=origin,
                cfg=self.cfg
            )
        except Exception:
            pass

        return {
            "timestamp": t_now,
            "predicted_label": pred_label,
            "voted_label": voted_label,
            "confidence": round(confidence, 3),
            "latency_ms": latency_ms,
            "meets_budget": bool(latency_ms < 2000.0),
            "baseline": curr_baseline,
            "calib_status": calib_status,
            "features": feats
        }
