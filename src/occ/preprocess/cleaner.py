from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from occ.config import load_config

def median_filter(series: pd.Series, k: int = 3) -> pd.Series:
    """Apply 1D median filter of kernel size k (must be odd) to reject spikes."""
    if len(series) < k:
        return series.copy()
    valid_mask = series.notna()
    if not valid_mask.any():
        return series.copy()
    
    # Fill internal NaNs for filtering, then mask back
    filled = series.interpolate(method="linear", limit_direction="both").bfill().ffill()
    filtered = medfilt(filled.to_numpy(), kernel_size=k)
    res = pd.Series(filtered, index=series.index, name=series.name)
    res[~valid_mask] = np.nan
    return res

def ema_filter(series: pd.Series, alpha: float = 0.25) -> pd.Series:
    """Apply Exponential Moving Average smoothing."""
    return series.ewm(alpha=alpha, adjust=False).mean()

def resample_grid(
    df: pd.DataFrame,
    target_freq: float = 10.0,
    max_gap_slots: int = 5
) -> pd.DataFrame:
    """
    Snap asynchronous sniffer packets to a uniform target frequency grid.
    Uses exact integer slot indices to prevent floating-point alignment mismatches.
    """
    if df.empty:
        return pd.DataFrame(columns=["t_host", "receiver_id", "rssi_dbm"])
    
    step = 1.0 / target_freq
    t_min = float(df["t_host"].min())
    t_max = float(df["t_host"].max())
    num_ticks = int(np.round((t_max - t_min) / step)) + 1
    
    full_grid = pd.DataFrame({
        "tick_idx": np.arange(num_ticks),
        "t_host": (t_min + np.arange(num_ticks) * step).round(3)
    })
    
    dfs_out = []
    for rx, grp in df.groupby("receiver_id"):
        grp = grp.sort_values("t_host").drop_duplicates(subset=["t_host"])
        ticks = np.round((grp["t_host"].to_numpy() - t_min) / step).astype(int)
        ticks = np.clip(ticks, 0, num_ticks - 1)
        
        grp_snapped = pd.DataFrame({
            "tick_idx": ticks,
            "rssi_dbm": grp["rssi_dbm"].to_numpy()
        }).groupby("tick_idx").mean().reset_index()
        
        merged = pd.merge(full_grid, grp_snapped, on="tick_idx", how="left")
        merged["rssi_dbm"] = merged["rssi_dbm"].interpolate(method="linear", limit=max_gap_slots)
        merged["receiver_id"] = rx
        dfs_out.append(merged[["t_host", "receiver_id", "rssi_dbm"]])
    
    if not dfs_out:
        return pd.DataFrame(columns=["t_host", "receiver_id", "rssi_dbm"])
    
    res = pd.concat(dfs_out, ignore_index=True).sort_values(["t_host", "receiver_id"]).reset_index(drop=True)
    return res

def clean_rssi_stream(df_raw: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Complete signal conditioning pipeline:
    1. Uniform grid resampling
    2. Outlier rejection (Median filter)
    3. EMA noise smoothing
    """
    if cfg is None:
        cfg = load_config()
    sc = cfg.get("sampling", {})
    freq = sc.get("target_freq_hz", 10.0)
    k = sc.get("median_k", 3)
    alpha = sc.get("ema_alpha", 0.25)
    max_gap = sc.get("max_gap_slots", 5)
    
    # 1. Resample to 10 Hz uniform grid
    grid_df = resample_grid(df_raw, target_freq=freq, max_gap_slots=max_gap)
    if grid_df.empty:
        return grid_df
    
    # 2. Filter per receiver
    cleaned_parts = []
    for rx, grp in grid_df.groupby("receiver_id"):
        grp = grp.sort_values("t_host").copy()
        s_med = median_filter(grp["rssi_dbm"], k=k)
        s_ema = ema_filter(s_med, alpha=alpha)
        grp["rssi_dbm"] = s_ema
        cleaned_parts.append(grp)
    
    return pd.concat(cleaned_parts, ignore_index=True).sort_values(["t_host", "receiver_id"]).reset_index(drop=True)
