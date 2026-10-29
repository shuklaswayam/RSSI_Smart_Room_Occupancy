import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from occ.config import get_root_dir, load_config

def generate_mock_run(
    run_id: str,
    protocol_name: str = "state_run",
    out_dir: Optional[Path] = None,
    seed: int = 42,
    origin: str = "real",
    cfg: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate a full synthetic/mock recording conforming to real hardware schema:
    rssi.csv, events.csv, meta.json
    """
    if cfg is None:
        cfg = load_config()
    if out_dir is None:
        out_dir = Path(cfg["paths"]["raw_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    # Load protocol
    proto_file = get_root_dir() / "configs" / "protocols" / f"{protocol_name}.yaml"
    with open(proto_file, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)
        
    def flatten_steps(steps):
        for s in steps:
            if s.get("type") == "repeat":
                for _ in range(s.get("count", 1)):
                    yield from flatten_steps(s.get("steps", []))
            else:
                yield s
                
    steps = list(flatten_steps(proto["steps"]))
    
    t_start = 1700000000.0  # Synthetic epoch
    t_curr = 0.0
    events = []
    events.append({
        "t_host": round(t_start, 3), "event_type": "run_start",
        "value": "", "person_count": "", "activity": "", "position": ""
    })
    
    # Build timeline of states and door crosses
    state_timeline = []
    for s in steps:
        if s["type"] == "state":
            events.append({
                "t_host": round(t_start + t_curr, 3),
                "event_type": "state_start",
                "value": s.get("label", "Empty"),
                "person_count": s.get("person_count", 0),
                "activity": s.get("activity", "none"),
                "position": s.get("position", "")
            })
            dur = float(s.get("seconds", 30))
            state_timeline.append((t_curr, t_curr + dur, s.get("label", "Empty"), s.get("person_count", 0), s.get("activity", "none")))
            t_curr += dur
        elif s["type"] == "door_cross":
            events.append({
                "t_host": round(t_start + t_curr, 3),
                "event_type": f"door_cross_{s.get('direction', 'in')}",
                "value": "", "person_count": "", "activity": "", "position": "door"
            })
            
    events.append({
        "t_host": round(t_start + t_curr, 3), "event_type": "run_end",
        "value": "", "person_count": "", "activity": "", "position": ""
    })
    
    # Generate 10 Hz RSSI packets for R1 and R2
    dt = 0.1024  # ~9.77 Hz packet rate
    packet_times = np.arange(0.0, t_curr, dt)
    rssi_rows = []
    
    # Base empty levels
    base_r1 = -46.0
    base_r2 = -53.0
    
    for t_rel in packet_times:
        t_abs = t_start + t_rel
        # Determine state
        lbl, n_people, act = "Empty", 0, "none"
        for s_start, s_end, s_lbl, s_np, s_act in state_timeline:
            if s_start <= t_rel < s_end:
                lbl, n_people, act = s_lbl, s_np, s_act
                break
                
        # Attenuation logic
        loss_r1 = 0.0
        loss_r2 = 0.0
        sigma_r1 = 1.2
        sigma_r2 = 1.4
        
        if lbl == "Occupied":
            if act == "moving":
                loss_r1 = rng.uniform(4.0, 10.0)
                loss_r2 = rng.uniform(5.0, 11.0)
                sigma_r1 = 3.5
                sigma_r2 = 3.8
            else:  # seated
                loss_r1 = 4.5
                loss_r2 = 6.0
                sigma_r1 = 1.8
                sigma_r2 = 2.0
        elif lbl == "Transition":
            loss_r1 = rng.uniform(3.0, 8.0)
            loss_r2 = rng.uniform(3.0, 8.0)
            sigma_r1 = 4.0
            sigma_r2 = 4.0
            
        # Draw packet samples
        dbm_r1 = base_r1 - loss_r1 + np_rng.normal(0.0, sigma_r1)
        dbm_r2 = base_r2 - loss_r2 + np_rng.normal(0.0, sigma_r2)
        
        # 1% hardware transient glitch
        if rng.random() < 0.01:
            dbm_r1 += rng.choice([-12.0, 10.0])
        if rng.random() < 0.01:
            dbm_r2 += rng.choice([-12.0, 10.0])
            
        rssi_rows.append({
            "t_host": round(t_abs, 3), "receiver_id": "R1", "source_id": "TX1",
            "technology": "wifi", "channel": 6, "rssi_dbm": int(round(dbm_r1)),
            "esp_ms": int(t_rel * 1000)
        })
        rssi_rows.append({
            "t_host": round(t_abs + 0.005, 3), "receiver_id": "R2", "source_id": "TX1",
            "technology": "wifi", "channel": 6, "rssi_dbm": int(round(dbm_r2)),
            "esp_ms": int((t_rel + 0.005) * 1000)
        })
        
    pd.DataFrame(rssi_rows).to_csv(out_dir / "rssi.csv", index=False)
    pd.DataFrame(events).to_csv(out_dir / "events.csv", index=False)
    
    meta = {
        "run_id": run_id,
        "day_id": "MOCK",
        "date": "2026-10-12",
        "mode": "collect",
        "origin": origin,
        "seed": seed,
        "scenario_family": protocol_name,
        "door_state": "closed",
        "furniture_config": "F1",
        "tx_device": "ESP32_Mock",
        "notes": f"Synthetic hardware run generated with {protocol_name}"
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    return out_dir

def generate_d0_calibration_runs(raw_dir: Optional[Path] = None, seed: int = 42) -> Tuple[Path, Path]:
    """
    Generate physical D0 calibration runs (D0_dist and D0_body) according to Guide Part 6.3.
    These runs are sealed and used strictly for fitting channel and body parameters.
    """
    if raw_dir is None:
        cfg = load_config()
        raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. D0_dist: Empty room distance sweep at 1, 2, 3, 4, 5 m (60 s each)
    dist_dir = raw_dir / "D0_dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    dist_steps = [1.0, 2.0, 3.0, 4.0, 5.0]
    dist_rssi = []
    dist_events = []
    t_start = 1700000000.0
    t_clock = 0.0

    for d in dist_steps:
        dist_events.append({"t_host": round(t_start + t_clock, 3), "event_type": "state_start", "value": f"Cal_d{d:.1f}", "person_count": 0, "activity": "none", "position": f"{d}m"})
        # Physical path loss: A=-37.5 dBm, beta=2.7, sigma=1.4 dB
        mean_p = -37.5 - 10.0 * 2.7 * np.log10(d)
        for _ in range(int(60.0 / 0.1024)):
            sample = int(np.round(mean_p + rng.normal(0.0, 1.4)))
            dist_rssi.append({"t_host": round(t_start + t_clock, 3), "receiver_id": "R1", "source_id": "TX1", "technology": "wifi", "channel": 6, "rssi_dbm": sample, "esp_ms": int(t_clock * 1000)})
            t_clock += 0.1024
        t_clock += 5.0  # rest between moves

    pd.DataFrame(dist_rssi).to_csv(dist_dir / "rssi.csv", index=False)
    pd.DataFrame(dist_events).to_csv(dist_dir / "events.csv", index=False)
    with open(dist_dir / "meta.json", "w") as f:
        json.dump({"run_id": "D0_dist", "day_id": "D0", "origin": "real", "role": "calibration_only"}, f, indent=2)

    # 2. D0_body: Fixed 4m link with person at dperp in [0.0, 0.5, 1.0, 2.0] m (60 s each)
    body_dir = raw_dir / "D0_body"
    body_dir.mkdir(parents=True, exist_ok=True)
    body_steps = [
        ("empty_before", 99.0, 0.0),
        ("on_line", 0.0, 6.2),
        ("off_0.5m", 0.5, 3.1),
        ("off_1.0m", 1.0, 0.9),
        ("off_2.0m", 2.0, 0.1),
        ("empty_after", 99.0, 0.0)
    ]
    body_rssi = []
    body_events = []
    t_clock = 0.0
    base_level = -53.8  # 4m distance level

    for name, dperp, loss in body_steps:
        body_events.append({"t_host": round(t_start + t_clock, 3), "event_type": "state_start", "value": name, "person_count": 0 if "empty" in name else 1, "activity": "standing", "position": f"{dperp}m"})
        sigma = 1.4 if "empty" in name else (2.4 if dperp == 0.0 else 1.8)
        for _ in range(int(60.0 / 0.1024)):
            sample = int(np.round(base_level - loss + rng.normal(0.0, sigma)))
            body_rssi.append({"t_host": round(t_start + t_clock, 3), "receiver_id": "R1", "source_id": "TX1", "technology": "wifi", "channel": 6, "rssi_dbm": sample, "esp_ms": int(t_clock * 1000)})
            t_clock += 0.1024
        t_clock += 5.0

    pd.DataFrame(body_rssi).to_csv(body_dir / "rssi.csv", index=False)
    pd.DataFrame(body_events).to_csv(body_dir / "events.csv", index=False)
    with open(body_dir / "meta.json", "w") as f:
        json.dump({"run_id": "D0_body", "day_id": "D0", "origin": "real", "role": "calibration_only"}, f, indent=2)

    return dist_dir, body_dir

class MockCollector:
    """Streams packets from an existing run folder at real-time speeds."""
    
    def __init__(self, run_dir: Path, speed_multiplier: float = 1.0):
        self.run_dir = Path(run_dir)
        self.speed = speed_multiplier
        self.is_running = False

    def stream(self, callback: Callable[[Dict[str, Any]], None]):
        df = pd.read_csv(self.run_dir / "rssi.csv").sort_values("t_host")
        self.is_running = True
        
        t_prev = None
        for _, row in df.iterrows():
            if not self.is_running:
                break
            t_curr = float(row["t_host"])
            if t_prev is not None:
                delta = (t_curr - t_prev) / self.speed
                if 0.0 < delta < 2.0:
                    time.sleep(delta)
            t_prev = t_curr
            callback(row.to_dict())
            
    def stop(self):
        self.is_running = False
