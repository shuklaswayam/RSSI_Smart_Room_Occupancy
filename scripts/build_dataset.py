import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from occ.config import get_root_dir, load_config
from occ.db import init_db, insert_features_df, insert_raw_rssi_df, insert_run
from occ.features.extractor import extract_dataset_features
from occ.preprocess.cleaner import clean_rssi_stream

def process_single_run(run_dir: Path, cfg: Dict) -> Tuple[pd.DataFrame, Dict]:
    meta_file = run_dir / "meta.json"
    rssi_file = run_dir / "rssi.csv"
    events_file = run_dir / "events.csv"
    
    if not (meta_file.is_file() and rssi_file.is_file() and events_file.is_file()):
        return pd.DataFrame(), {}
        
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    df_raw = pd.read_csv(rssi_file)
    df_events = pd.read_csv(events_file)
    
    # Ingest into SQLite database
    try:
        insert_run(meta, cfg)
        df_raw_db = df_raw.copy()
        df_raw_db["run_id"] = meta.get("run_id")
        df_raw_db["origin"] = meta.get("origin", "real")
        insert_raw_rssi_df(df_raw_db, cfg)
    except Exception as e:
        print(f"  [!] DB ingestion note for {run_dir.name}: {e}")
        
    # Preprocess & Extract Features
    df_clean = clean_rssi_stream(df_raw, cfg)
    df_feats = extract_dataset_features(df_clean, df_events, meta, cfg)
    
    if not df_feats.empty:
        try:
            insert_features_df(df_feats, cfg)
        except Exception as e:
            pass
            
    return df_feats, meta

def build_all_datasets(cfg=None):
    if cfg is None:
        cfg = load_config()
    root = get_root_dir()
    
    init_db(cfg)
    
    raw_dir = root / cfg["paths"]["raw_dir"]
    sim_dir = root / cfg["paths"]["sim_dir"]
    proc_dir = root / cfg["paths"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    print("[*] Compiling feature datasets from raw and sim repositories...")
    
    real_dfs = []
    sim_dfs = []
    
    # 1. Process real runs
    if raw_dir.is_dir():
        for r_dir in sorted(raw_dir.iterdir()):
            if r_dir.is_dir() and (r_dir / "rssi.csv").is_file():
                df_f, meta = process_single_run(r_dir, cfg)
                if not df_f.empty:
                    real_dfs.append(df_f)
                    print(f"  [+] Processed Real Run: {r_dir.name} ({len(df_f)} windows)")
                    
    # 2. Process simulated runs
    if sim_dir.is_dir():
        for s_dir in sorted(sim_dir.iterdir()):
            if s_dir.is_dir() and (s_dir / "rssi.csv").is_file():
                df_f, meta = process_single_run(s_dir, cfg)
                if not df_f.empty:
                    sim_dfs.append(df_f)
                    print(f"  [+] Processed Sim Run: {s_dir.name} ({len(df_f)} windows)")
                    
    # Save real features dataset
    if real_dfs:
        df_real_all = pd.concat(real_dfs, ignore_index=True)
        real_csv = proc_dir / "windows_features_real.csv"
        df_real_all.to_csv(real_csv, index=False)
        print(f"[SUCCESS] Real dataset compiled: {real_csv} ({len(df_real_all)} windows)")
        print(f"  Class breakdown:\n{df_real_all['label'].value_counts().to_string()}")
    else:
        print("[!] No real runs found yet in data/raw/")
        
    # Save sim features dataset
    if sim_dfs:
        df_sim_all = pd.concat(sim_dfs, ignore_index=True)
        sim_csv = proc_dir / "windows_features_sim.csv"
        df_sim_all.to_csv(sim_csv, index=False)
        print(f"[SUCCESS] Sim dataset compiled: {sim_csv} ({len(df_sim_all)} windows)")
        print(f"  Class breakdown:\n{df_sim_all['label'].value_counts().to_string()}")
    else:
        print("[!] No sim runs found yet in data/sim/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile feature datasets from data/raw and data/sim")
    args = parser.parse_args()
    build_all_datasets()
