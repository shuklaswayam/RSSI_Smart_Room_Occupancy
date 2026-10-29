from pathlib import Path
import json
import pandas as pd
import pytest
from occ.config import get_root_dir, load_config
from occ.sim.nsparse import parse_ns2_run

def test_nsparse_roundtrip(tmp_path):
    root = get_root_dir()
    ns2_out = root / "ns2" / "out"
    ns2_out.mkdir(parents=True, exist_ok=True)
    
    test_run_id = "SIM_S2_seed99"
    rssi_file = ns2_out / f"{test_run_id}.rssi"
    events_file = ns2_out / f"{test_run_id}.events.csv"

    # Write dummy .rssi trace
    with open(rssi_file, "w") as f:
        f.write("0.100000 1 0 -45.234\n")
        f.write("0.105000 2 0 -52.789\n")
        f.write("0.200000 1 0 -46.100\n")

    # Write dummy .events.csv
    with open(events_file, "w") as f:
        f.write("t_host,event_type,value,person_count,activity,position\n")
        f.write("0.0,run_start,,,,\n")
        f.write("0.0,state_start,Empty,0,none,none\n")
        f.write("60.0,run_end,,,,\n")

    cfg = load_config()
    out_dir = parse_ns2_run(test_run_id, cfg)

    assert out_dir.is_dir()
    assert (out_dir / "rssi.csv").is_file()
    assert (out_dir / "events.csv").is_file()
    assert (out_dir / "meta.json").is_file()

    df_rssi = pd.read_csv(out_dir / "rssi.csv")
    assert len(df_rssi) == 3
    assert set(df_rssi["receiver_id"].unique()) == {"R1", "R2"}
    # Quantized dBm check
    assert df_rssi["rssi_dbm"].dtype in [int, "int64"]

    with open(out_dir / "meta.json") as f:
        meta = json.load(f)
    assert meta["origin"] == "sim"
    assert meta["seed"] == 99
