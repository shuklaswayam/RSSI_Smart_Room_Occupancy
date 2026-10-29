import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from occ.config import get_root_dir

def parse_network_trace(trace_path: str, gateway_id: int = 0) -> pd.DataFrame:
    """Parse standard old-format NS2 wireless trace for application layer packets."""
    rows = []
    with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            # Format check: s/r/d/f, time, node, layer, uid, packet_type, size
            if len(p) >= 8 and p[3] == "AGT" and p[6] == "cbr":
                try:
                    event = p[0]
                    t = float(p[1])
                    node = int(p[2].strip("_"))
                    uid = int(p[5])
                    size = int(p[7])
                    rows.append((event, t, node, uid, size))
                except (ValueError, IndexError):
                    continue
    return pd.DataFrame(rows, columns=["event", "t", "node", "uid", "size"])

def compute_network_metrics(df_tr: pd.DataFrame, gateway_id: int = 0) -> Dict[str, float]:
    """Compute PDR, delay mean/p95, and throughput from trace DataFrame."""
    if df_tr.empty:
        return {"pdr": 0.0, "delay_mean_ms": 0.0, "delay_p95_ms": 0.0, "throughput_Bps": 0.0, "max_gap": 0}
        
    sends = df_tr[df_tr["event"] == "s"].set_index("uid")
    sends = sends[sends["node"] != gateway_id]
    
    recvs = df_tr[(df_tr["event"] == "r") & (df_tr["node"] == gateway_id)].set_index("uid")
    
    if sends.empty:
        return {"pdr": 0.0, "delay_mean_ms": 0.0, "delay_p95_ms": 0.0, "throughput_Bps": 0.0, "max_gap": 0}
        
    joined = sends.join(recvs, lsuffix="_s", rsuffix="_r", how="left")
    delays = (joined["t_r"] - joined["t_s"]).dropna().to_numpy() * 1000.0  # ms
    
    pdr = float(joined["t_r"].notna().mean())
    delay_mean = float(np.mean(delays)) if len(delays) > 0 else 0.0
    delay_p95 = float(np.percentile(delays, 95)) if len(delays) > 0 else 0.0
    
    dur = float(df_tr["t"].max() - df_tr["t"].min())
    total_bytes = float(recvs["size"].sum()) if not recvs.empty else 0.0
    throughput = (total_bytes / dur) if dur > 0 else 0.0
    
    # Calculate maximum consecutive drop gap
    missing_mask = joined["t_r"].isna().to_numpy(dtype=int)
    max_gap = 0
    curr_gap = 0
    for m in missing_mask:
        if m:
            curr_gap += 1
            if curr_gap > max_gap:
                max_gap = curr_gap
        else:
            curr_gap = 0
            
    return {
        "pdr": round(pdr, 4),
        "delay_mean_ms": round(delay_mean, 2),
        "delay_p95_ms": round(delay_p95, 2),
        "throughput_Bps": round(throughput, 1),
        "max_gap": max_gap
    }

def analytical_network_model(num_rx: int, rate_hz: float, extra_devices: int) -> Dict[str, float]:
    """
    Theoretical 802.11 CSMA/CA capacity model:
    Each 100-byte frame consumes ~2.0 ms of channel air time (preamble + MAC + ACK + backoff).
    """
    total_pps = (num_rx * rate_hz) + (extra_devices * 50.0)
    channel_capacity_pps = 480.0  # 1 Mb/s basic rate ceiling
    
    load = total_pps / channel_capacity_pps
    if load < 0.6:
        pdr = 1.0 - (0.005 * load)
        delay_mean = 2.2 + 1.5 * load
        delay_p95 = 4.0 + 3.0 * load
    elif load < 1.0:
        pdr = 1.0 - 0.15 * ((load - 0.6) / 0.4) ** 2
        delay_mean = 5.0 + 25.0 * (load - 0.6)
        delay_p95 = 12.0 + 60.0 * (load - 0.6)
    else:
        pdr = max(0.15, 0.85 / load)
        delay_mean = 35.0 + 50.0 * min(load, 2.0)
        delay_p95 = 90.0 + 150.0 * min(load, 2.0)
        
    throughput = total_pps * pdr * 40.0
    max_gap = int(np.ceil((1.0 - pdr) * 10))
    
    return {
        "pdr": round(pdr, 4),
        "delay_mean_ms": round(delay_mean, 2),
        "delay_p95_ms": round(delay_p95, 2),
        "throughput_Bps": round(throughput, 1),
        "max_gap": max_gap
    }
