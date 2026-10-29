import argparse
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from occ.config import get_root_dir, load_config
from occ.sim.netstudy import analytical_network_model, compute_network_metrics, parse_network_trace

def run_network_sweeps(cfg=None, out_csv="results/tables/e15_network_study.csv") -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    root = get_root_dir()
    out_path = root / out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    rates = cfg.get("netstudy", {}).get("rates_hz", [5, 10, 20, 50, 100])
    receivers = cfg.get("netstudy", {}).get("receivers", [2, 3, 5, 8])
    extra_devs = cfg.get("netstudy", {}).get("extra_devices", [0, 2, 4, 8])
    
    print("[*] Running 802.11 RSSI Collection Network Study (O10, E15)...")
    results = []
    
    for rx_count in receivers:
        for rate in rates:
            for devs in extra_devs:
                run_id = f"net_r{rx_count}_f{rate}_d{devs}"
                trace_file = root / "ns2" / "out" / f"{run_id}.tr"
                
                # Check if trace exists or if ns is available
                if trace_file.is_file():
                    df_tr = parse_network_trace(str(trace_file))
                    m = compute_network_metrics(df_tr)
                else:
                    # Use calibrated 802.11 analytical model
                    m = analytical_network_model(rx_count, rate, devs)
                    
                row = {
                    "receiver_count": rx_count,
                    "sampling_rate_hz": rate,
                    "background_devices": devs,
                    "pdr": m["pdr"],
                    "delay_mean_ms": m["delay_mean_ms"],
                    "delay_p95_ms": m["delay_p95_ms"],
                    "throughput_Bps": m["throughput_Bps"],
                    "max_gap": m["max_gap"],
                    "meets_latency_budget": bool(m["delay_p95_ms"] < 2000.0 and m["pdr"] >= 0.95)
                }
                results.append(row)
                
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_path, index=False)
    print(f"[SUCCESS] Network study completed! Saved {len(df_res)} configurations to {out_path}")
    
    # Highlight prototype configuration (2 receivers, 10 Hz, 0 devices)
    proto_row = df_res[(df_res["receiver_count"] == 2) & (df_res["sampling_rate_hz"] == 10) & (df_res["background_devices"] == 0)]
    if not proto_row.empty:
        r = proto_row.iloc[0]
        print(f"  [+] Prototype Config (2 RX @ 10 Hz): PDR={r['pdr']*100:.1f}% | Delay p95={r['delay_p95_ms']:.1f} ms | Max Gap={r['max_gap']} slots")
        
    return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 802.11 network scaling sweeps (O10, E15)")
    parser.add_argument("--out", type=str, default="results/tables/e15_network_study.csv", help="Output CSV path")
    args = parser.parse_args()
    
    run_network_sweeps(out_csv=args.out)
