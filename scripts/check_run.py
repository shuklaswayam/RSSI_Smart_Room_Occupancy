import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

def check_run(run_dir_path: str) -> bool:
    run_dir = Path(run_dir_path)
    print(f"[*] Checking run directory: {run_dir.resolve()}")
    
    if not run_dir.is_dir():
        print(f"[ERROR] Directory does not exist: {run_dir}")
        return False
        
    req_files = ["rssi.csv", "events.csv", "meta.json"]
    for f in req_files:
        if not (run_dir / f).is_file():
            print(f"[ERROR] Missing required file: {f}")
            return False
            
    # Check meta.json
    try:
        with open(run_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[+] Loaded meta.json | Run ID: {meta.get('run_id')} | Origin: {meta.get('origin')} | Day: {meta.get('day_id')}")
    except Exception as e:
        print(f"[ERROR] Failed to parse meta.json: {e}")
        return False
        
    # Check events.csv
    try:
        events_df = pd.read_csv(run_dir / "events.csv")
        req_event_cols = {"t_host", "event_type"}
        if not req_event_cols.issubset(set(events_df.columns)):
            print(f"[ERROR] events.csv missing required columns: {req_event_cols - set(events_df.columns)}")
            return False
        print(f"[+] events.csv valid with {len(events_df)} logged events.")
    except Exception as e:
        print(f"[ERROR] Failed to parse events.csv: {e}")
        return False
        
    # Check rssi.csv
    try:
        rssi_df = pd.read_csv(run_dir / "rssi.csv")
        req_rssi_cols = {"t_host", "receiver_id", "rssi_dbm"}
        if not req_rssi_cols.issubset(set(rssi_df.columns)):
            print(f"[ERROR] rssi.csv missing required columns: {req_rssi_cols - set(rssi_df.columns)}")
            return False
            
        print(f"[+] rssi.csv contains {len(rssi_df)} samples across receivers: {rssi_df['receiver_id'].unique().tolist()}")
        
        # Check rates per receiver
        for rx, grp in rssi_df.groupby("receiver_id"):
            t = grp["t_host"].to_numpy()
            if not np.all(np.diff(t) >= 0):
                print(f"[WARNING] Timestamps for {rx} are not strictly non-decreasing!")
                
            dur = t[-1] - t[0] if len(t) > 1 else 0
            if dur > 0:
                rate = len(t) / dur
                print(f"    - Receiver {rx}: Duration = {dur:.1f} s | Rate = {rate:.2f} Hz | RSSI range = [{grp['rssi_dbm'].min()}, {grp['rssi_dbm'].max()}] dBm")
                if rate < 7.0 or rate > 20.0:
                    print(f"    [WARNING] Rate {rate:.2f} Hz deviates from expected ~10 Hz")
                    
        print(f"[SUCCESS] Run verification passed for {run_dir.name}!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to parse rssi.csv: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check run data folder")
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    args = parser.parse_args()
    
    success = check_run(args.run_dir)
    sys.exit(0 if success else 1)
