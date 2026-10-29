from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from occ.config import load_config

FEATURE_COLUMNS: List[str] = [
    "R1_mean", "R1_std", "R1_min", "R1_max", "R1_iqr", "R1_slope", "R1_zcr",
    "R2_mean", "R2_std", "R2_min", "R2_max", "R2_iqr", "R2_slope", "R2_zcr",
    "R1R2_mean_diff", "R1R2_std_diff",
    "R1_delta_mean", "R2_delta_mean", "R1_delta_min", "R2_delta_min", "R1_delta_max"
]

def compute_single_link_stats(vals: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """Compute 7 statistical features for a single receiver link."""
    n = len(vals)
    if n < 2:
        return {
            "mean": float(np.mean(vals)) if n == 1 else 0.0,
            "std": 0.0, "min": float(vals[0]) if n == 1 else 0.0,
            "max": float(vals[0]) if n == 1 else 0.0,
            "iqr": 0.0, "slope": 0.0, "zcr": 0.0
        }
    
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1))
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    q75, q25 = np.percentile(vals, [75, 25])
    iqr_val = float(q75 - q25)
    
    # Linear slope via OLS
    t_norm = t - t[0]
    denom = np.sum((t_norm - np.mean(t_norm)) ** 2)
    if denom > 1e-9:
        slope_val = float(np.sum((t_norm - np.mean(t_norm)) * (vals - mean_val)) / denom)
    else:
        slope_val = 0.0
        
    # Zero-crossing rate of mean-subtracted signal
    centered = vals - mean_val
    signs = np.sign(centered)
    signs[signs == 0] = 1.0
    crossings = np.sum(signs[:-1] * signs[1:] < 0)
    zcr_val = float(crossings / (n - 1))
    
    return {
        "mean": mean_val, "std": std_val, "min": min_val,
        "max": max_val, "iqr": iqr_val, "slope": slope_val,
        "zcr": zcr_val
    }

def compute_window_features(
    w_df: pd.DataFrame,
    baseline: Optional[Dict[str, float]] = None
) -> Optional[Dict[str, float]]:
    """
    Extract 21-dimensional feature dict from a window DataFrame
    containing columns ['t_host', 'receiver_id', 'rssi_dbm'].
    """
    r1_data = w_df[w_df["receiver_id"] == "R1"].sort_values("t_host")
    r2_data = w_df[w_df["receiver_id"] == "R2"].sort_values("t_host")
    
    if len(r1_data) < 10 or len(r2_data) < 10:
        return None
    
    r1_vals = r1_data["rssi_dbm"].to_numpy(dtype=float)
    r1_t = r1_data["t_host"].to_numpy(dtype=float)
    r2_vals = r2_data["rssi_dbm"].to_numpy(dtype=float)
    r2_t = r2_data["t_host"].to_numpy(dtype=float)
    
    f1 = compute_single_link_stats(r1_vals, r1_t)
    f2 = compute_single_link_stats(r2_vals, r2_t)
    
    feat: Dict[str, float] = {}
    for k, v in f1.items():
        feat[f"R1_{k}"] = v
    for k, v in f2.items():
        feat[f"R2_{k}"] = v
        
    # Cross-link differential features
    feat["R1R2_mean_diff"] = feat["R1_mean"] - feat["R2_mean"]
    feat["R1R2_std_diff"] = feat["R1_std"] - feat["R2_std"]
    
    # Baseline-relative features
    b_r1 = baseline.get("R1", feat["R1_mean"]) if baseline else feat["R1_mean"]
    b_r2 = baseline.get("R2", feat["R2_mean"]) if baseline else feat["R2_mean"]
    
    feat["R1_delta_mean"] = feat["R1_mean"] - b_r1
    feat["R2_delta_mean"] = feat["R2_mean"] - b_r2
    feat["R1_delta_min"] = feat["R1_min"] - b_r1
    feat["R2_delta_min"] = feat["R2_min"] - b_r2
    feat["R1_delta_max"] = feat["R1_max"] - b_r1
    
    return feat

def extract_windows(
    clean_df: pd.DataFrame,
    window_s: float = 5.0,
    step_s: float = 2.5,
    min_samples: int = 40
) -> List[Tuple[float, float, pd.DataFrame]]:
    """Slice time-series into sliding overlapping windows."""
    if clean_df.empty:
        return []
    
    t_start_global = clean_df["t_host"].min()
    t_end_global = clean_df["t_host"].max()
    
    windows = []
    t_curr = t_start_global
    while t_curr + window_s <= t_end_global + 1e-4:
        t_w_end = t_curr + window_s
        sub = clean_df[(clean_df["t_host"] >= t_curr) & (clean_df["t_host"] < t_w_end)]
        if len(sub) >= min_samples:
            windows.append((round(t_curr, 3), round(t_w_end, 3), sub))
        t_curr += step_s
        
    return windows

def assign_window_label(t_start: float, t_end: float, events_df: pd.DataFrame) -> str:
    """
    Assign ground-truth label to window from events.csv.
    Priority: Transition if any door_cross occurs in window; else dominant state.
    """
    if events_df.empty:
        return "Empty"
    
    # Check for door crossing event
    crosses = events_df[
        (events_df["t_host"] >= t_start) &
        (events_df["t_host"] <= t_end) &
        (events_df["event_type"].str.startswith("door_cross"))
    ]
    if not crosses.empty:
        return "Transition"
    
    # Check state at midpoint
    t_mid = (t_start + t_end) / 2.0
    states = events_df[
        (events_df["event_type"] == "state_start") &
        (events_df["t_host"] <= t_mid)
    ].sort_values("t_host")
    
    if not states.empty:
        lbl = str(states.iloc[-1].get("value", "Empty")).strip()
        if lbl in ["Empty", "Occupied", "Transition", "Near", "Far"]:
            return lbl
        if "Occupied" in lbl:
            return "Occupied"
    
    return "Empty"

def extract_dataset_features(
    clean_df: pd.DataFrame,
    events_df: pd.DataFrame,
    meta: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Process full clean run into window features DataFrame.
    Calculates initial empty baseline from the first 20 Empty windows if available.
    """
    if cfg is None:
        cfg = load_config()
    
    wc = cfg.get("windowing", {})
    win_s = wc.get("window_s", 5.0)
    step_s = wc.get("step_s", 2.5)
    min_samp = wc.get("min_samples", 40)
    
    windows = extract_windows(clean_df, window_s=win_s, step_s=step_s, min_samples=min_samp)
    if not windows:
        return pd.DataFrame()
    
    # 1. Label all windows first to establish baseline from initial Empty blocks
    raw_windows_info = []
    for t_s, t_e, sub in windows:
        lbl = assign_window_label(t_s, t_e, events_df)
        raw_windows_info.append((t_s, t_e, sub, lbl))
        
    # Baseline estimation (median of first up to 20 Empty windows)
    empty_subs = [sub for _, _, sub, lbl in raw_windows_info if lbl == "Empty"][:20]
    if empty_subs:
        all_empty = pd.concat(empty_subs, ignore_index=True)
        baseline = {
            "R1": float(all_empty[all_empty["receiver_id"] == "R1"]["rssi_dbm"].median()),
            "R2": float(all_empty[all_empty["receiver_id"] == "R2"]["rssi_dbm"].median())
        }
    else:
        baseline = {
            "R1": float(clean_df[clean_df["receiver_id"] == "R1"]["rssi_dbm"].median()),
            "R2": float(clean_df[clean_df["receiver_id"] == "R2"]["rssi_dbm"].median())
        }
        
    rows = []
    run_id = meta.get("run_id", "UNKNOWN")
    origin = meta.get("origin", "real")
    seed = meta.get("seed", np.nan)
    
    for idx, (t_s, t_e, sub, lbl) in enumerate(raw_windows_info):
        feat = compute_window_features(sub, baseline=baseline)
        if feat is None:
            continue
            
        row = {
            "window_id": f"{run_id}_w{idx:04d}",
            "run_id": run_id,
            "t_start": t_s,
            "t_end": t_e,
            "origin": origin,
            "seed": seed,
            "label": lbl,
            **feat
        }
        rows.append(row)
        
    return pd.DataFrame(rows)
