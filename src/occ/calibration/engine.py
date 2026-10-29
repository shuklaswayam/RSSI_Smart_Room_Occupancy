from typing import Any, Dict, Optional
import numpy as np

class AdaptiveCalibrationEngine:
    """
    Hero 3: Adaptive Calibration Engine.
    Dynamically counters environmental temperature, multipath, and hardware drift
    by updating empty-room baselines during confirmed-empty periods.
    """

    def __init__(
        self,
        initial_baseline: Optional[Dict[str, float]] = None,
        gamma: float = 0.05,
        confidence_thresh: float = 0.90,
        min_empty_windows: int = 10,
        max_var_ratio: float = 1.5
    ):
        self.gamma = gamma
        self.confidence_thresh = confidence_thresh
        self.min_empty_windows = min_empty_windows
        self.max_var_ratio = max_var_ratio
        
        # Current adaptive baseline
        self.baseline = initial_baseline.copy() if initial_baseline else {"R1": -45.0, "R2": -52.0}
        self.static_baseline = self.baseline.copy()
        
        # State tracking
        self.consecutive_empty_count = 0
        self.base_variance = {"R1": 1.5, "R2": 1.8}
        self.update_history = []

    def get_baseline(self) -> Dict[str, float]:
        """Return current estimated empty baseline."""
        return self.baseline.copy()

    def update(
        self,
        predicted_label: str,
        confidence: float,
        window_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate single window prediction and update baseline if conditions are met.
        """
        is_empty = (predicted_label == "Empty") and (confidence >= self.confidence_thresh)
        r1_std = window_features.get("R1_std", 1.0)
        r2_std = window_features.get("R2_std", 1.0)
        
        # Check variance guard (prevent absorbing still human occupants)
        var_safe = (r1_std <= self.base_variance["R1"] * self.max_var_ratio) and \
                   (r2_std <= self.base_variance["R2"] * self.max_var_ratio)

        updated = False
        if is_empty and var_safe:
            self.consecutive_empty_count += 1
            if self.consecutive_empty_count >= self.min_empty_windows:
                # Update baseline via Exponential Moving Average
                r1_curr_mean = window_features.get("R1_mean", self.baseline["R1"])
                r2_curr_mean = window_features.get("R2_mean", self.baseline["R2"])
                
                old_r1 = self.baseline["R1"]
                old_r2 = self.baseline["R2"]
                
                self.baseline["R1"] = (1.0 - self.gamma) * old_r1 + self.gamma * r1_curr_mean
                self.baseline["R2"] = (1.0 - self.gamma) * old_r2 + self.gamma * r2_curr_mean
                updated = True
        else:
            self.consecutive_empty_count = 0

        status = {
            "updated": updated,
            "baseline_r1": round(self.baseline["R1"], 2),
            "baseline_r2": round(self.baseline["R2"], 2),
            "drift_from_static_r1": round(self.baseline["R1"] - self.static_baseline["R1"], 2),
            "drift_from_static_r2": round(self.baseline["R2"] - self.static_baseline["R2"], 2),
            "consecutive_empty_count": self.consecutive_empty_count
        }
        self.update_history.append(status)
        return status
