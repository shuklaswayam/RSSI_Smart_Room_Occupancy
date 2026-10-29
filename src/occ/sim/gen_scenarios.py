import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml
from occ.config import get_root_dir, load_config
from occ.sim.nsparse import parse_ns2_run

SPEED = 1.0  # m/s walking speed
DOOR = (0.2, 2.25)
OUTSIDE = (-2.0, 2.25)

SCN_MAP = {
    "S1": {"id": "S1", "protocol": "state_run", "fixed_people": 0, "family": "empty"},
    "S2": {"id": "S2", "protocol": "state_run", "fixed_people": 1, "family": "1p_seated"},
    "S3": {"id": "S3", "protocol": "state_run", "fixed_people": 3, "family": "3p_moving"},
    "S4": {"id": "S4", "protocol": "drill",     "fixed_people": 1, "family": "drill"},
    "S5": {"id": "S5", "protocol": "nearfar",   "fixed_people": 1, "family": "nearfar"}
}

def flatten_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for s in steps:
        if s.get("type") == "repeat":
            for _ in range(s.get("count", 1)):
                out.extend(flatten_steps(s.get("steps", [])))
        else:
            out.append(s)
    return out

def compile_run_trajectory(
    steps: List[Dict[str, Any]],
    room_dims: Tuple[float, float],
    seats: List[List[float]],
    rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, int, float, float, float]], float, int]:
    """Compile high-level protocol steps into events and person waypoint moves."""
    seq = flatten_steps(steps)
    t = 0.0
    events = []
    moves = []  # (t, person_idx, target_x, target_y, speed)
    pos: Dict[int, Tuple[float, float]] = {}
    inside: List[int] = []

    def schedule_move(t0: float, p: int, dest: Tuple[float, float]) -> float:
        curr = pos.get(p, OUTSIDE)
        d = math.dist(curr, dest)
        pos[p] = dest
        moves.append((round(t0, 3), p, round(dest[0], 2), round(dest[1], 2), SPEED))
        return d / SPEED

    for k, s in enumerate(seq):
        if s["type"] == "state":
            events.append({
                "t_host": round(t, 3), "event_type": "state_start",
                "value": s.get("label", ""), "person_count": s.get("person_count", 0),
                "activity": s.get("activity", ""), "position": s.get("position", "")
            })
            dur = float(s.get("seconds", 30))
            if s.get("activity") == "moving" and inside:
                for p in inside:
                    tt = t + 5.0
                    while tt < t + dur - 5.0:
                        dest = (rng.uniform(0.6, room_dims[0] - 0.6), rng.uniform(0.6, room_dims[1] - 0.6))
                        tt += schedule_move(tt, p, dest) + rng.uniform(1.0, 3.5)
            t += dur
        elif s["type"] == "door_cross":
            direction = s.get("direction", "in")
            events.append({
                "t_host": round(t, 3), "event_type": f"door_cross_{direction}",
                "value": "", "person_count": "", "activity": "", "position": "door"
            })
            if direction == "in":
                # Lookahead for target person count
                nxt_state = next((x for x in seq[k + 1:] if x["type"] == "state"), {})
                n_target = int(nxt_state.get("person_count", 1))
                inside = list(range(n_target))
                for i, p in enumerate(inside):
                    pos[p] = OUTSIDE
                    t_door = t - (math.dist(OUTSIDE, DOOR) / SPEED) - 0.4 * i
                    schedule_move(t_door, p, DOOR)
                    seat = tuple(seats[i % len(seats)])
                    schedule_move(t + 0.4 * i, p, seat)
            else:
                for i, p in enumerate(inside):
                    t_door = t - (math.dist(pos.get(p, DOOR), DOOR) / SPEED) - 0.4 * i
                    schedule_move(t_door, p, DOOR)
                    schedule_move(t + 0.4 * i, p, OUTSIDE)
                inside = []

    num_people = len({m[1] for m in moves})
    return events, sorted(moves), t, num_people

def generate_scenario_tcl(
    scn_key: str,
    seed: int,
    cfg: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Path] = None
) -> str:
    """Generate .tcl scenario script and matching .events.csv for NS2."""
    if cfg is None:
        cfg = load_config()
    scn_info = SCN_MAP[scn_key]
    run_id = f"SIM_{scn_info['id']}_seed{seed:02d}"

    root = get_root_dir()
    if out_dir is None:
        out_dir = root / "ns2" / "tcl"
    out_dir.mkdir(parents=True, exist_ok=True)
    ns2_out = root / "ns2" / "out"
    ns2_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    proto_path = root / "configs" / "protocols" / f"{scn_info['protocol']}.yaml"
    with open(proto_path, "r", encoding="utf-8") as f:
        proto_data = yaml.safe_load(f)

    room_dims = tuple(cfg["room"]["dimensions_m"])
    seats = cfg["room"]["seats"]
    events, moves, duration, n_people = compile_run_trajectory(
        proto_data["steps"], room_dims, seats, rng
    )

    c = cfg["sim"]["channel"]
    b = cfg["sim"]["body"]

    base_tcl_path = root / "ns2" / "tcl" / "room_base.tcl"
    with open(base_tcl_path, "r", encoding="utf-8") as f:
        tcl_content = f.read()

    substitutions = {
        "@RUN@": run_id,
        "@SEED@": str(seed),
        "@NN@": str(3 + n_people),
        "@STOP@": f"{duration:.2f}",
        "@X@": str(room_dims[0]),
        "@Y@": str(room_dims[1]),
        "@BETA@": str(c["pathloss_exp"]),
        "@STD@": str(c["shadow_std_db"]),
        "@PT@": str(c["tx_power_w"]),
        "@LOSS@": str(b["loss_db"]),
        "@XSTD@": str(b["extra_std_db"]),
        "@FRES@": str(b["fresnel_scale"]),
        "@INT@": str(cfg["sim"]["tx_interval_s"])
    }
    for k, v in substitutions.items():
        tcl_content = tcl_content.replace(k, v)

    # Initialize person nodes outside room
    init_lines = []
    for p in range(n_people):
        init_lines.append(f"$n({3+p}) set X_ {OUTSIDE[0]}; $n({3+p}) set Y_ {OUTSIDE[1]}; $n({3+p}) set Z_ 0.0")
    tcl_content = tcl_content.replace("# PERSON_INIT", "\n".join(init_lines))

    # Append movement commands and run
    move_lines = [f'$ns at {t:.3f} "$n({3+p}) setdest {x:.2f} {y:.2f} {sp:.2f}"' for t, p, x, y, sp in moves]
    tcl_content += "\n" + "\n".join(move_lines)
    tcl_content += "\n$ns run\n"

    target_tcl = out_dir / f"{run_id}.tcl"
    with open(target_tcl, "w", encoding="utf-8") as f:
        f.write(tcl_content)

    # Write events.csv
    target_events = ns2_out / f"{run_id}.events.csv"
    with open(target_events, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_host", "event_type", "value", "person_count", "activity", "position"])
        writer.writerow([0.0, "run_start", "", "", "", ""])
        for ev in events:
            writer.writerow([ev["t_host"], ev["event_type"], ev["value"], ev["person_count"], ev["activity"], ev["position"]])
        writer.writerow([round(duration, 3), "run_end", "", "", "", ""])

    return run_id

def simulate_scenario_python(
    scn_key: str,
    seed: int,
    cfg: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Pure Python simulation engine executing the exact BodyShadow Fresnel-zone
    attenuation and log-distance propagation model.
    Used for instant local execution without Docker container dependencies.
    """
    if cfg is None:
        cfg = load_config()
    scn_info = SCN_MAP[scn_key]
    run_id = f"SIM_{scn_info['id']}_seed{seed:02d}"

    root = get_root_dir()
    ns2_out = root / "ns2" / "out"
    ns2_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    proto_path = root / "configs" / "protocols" / f"{scn_info['protocol']}.yaml"
    with open(proto_path, "r", encoding="utf-8") as f:
        proto_data = yaml.safe_load(f)

    room_dims = tuple(cfg["room"]["dimensions_m"])
    seats = cfg["room"]["seats"]
    events, moves, duration, n_people = compile_run_trajectory(
        proto_data["steps"], room_dims, seats, rng
    )

    # Save events.csv
    events_path = ns2_out / f"{run_id}.events.csv"
    with open(events_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_host", "event_type", "value", "person_count", "activity", "position"])
        writer.writerow([0.0, "run_start", "", "", "", ""])
        for ev in events:
            writer.writerow([ev["t_host"], ev["event_type"], ev["value"], ev["person_count"], ev["activity"], ev["position"]])
        writer.writerow([round(duration, 3), "run_end", "", "", "", ""])

    # Node positions
    tx = np.array(cfg["room"]["tx_pos"], dtype=float)
    receivers = {
        1: np.array(cfg["room"]["r1_pos"], dtype=float),
        2: np.array(cfg["room"]["r2_pos"], dtype=float)
    }

    c = cfg["sim"]["channel"]
    b = cfg["sim"]["body"]
    beta = float(c["pathloss_exp"])
    sigma_shadow = float(c["shadow_std_db"])
    loss_db = float(b["loss_db"])
    extra_std = float(b["extra_std_db"])
    fresnel_scale = float(b["fresnel_scale"])
    body_radius = float(b.get("body_radius", 0.2))
    lambda_rf = 3e8 / float(c.get("freq_hz", 2.437e9))

    # Base reference RSSI at 1.0 m from Tx power
    # Friis at 1 m
    pt_w = float(c["tx_power_w"])
    pr_1m_w = pt_w * ((lambda_rf / (4 * math.pi * 1.0)) ** 2)
    p_r0_dbm = 10.0 * math.log10(pr_1m_w * 1000.0)

    # Track person positions over time
    person_pos = {p: np.array(OUTSIDE, dtype=float) for p in range(n_people)}
    # Convert moves into time segments
    # moves: (t, p, x, y, speed)
    move_queue = list(moves)

    dt = float(cfg["sim"]["tx_interval_s"])
    times = np.arange(0.1, duration, dt)
    rssi_lines = []

    curr_move_idx = 0
    for t in times:
        # Update person locations
        while curr_move_idx < len(move_queue) and move_queue[curr_move_idx][0] <= t:
            _, p_idx, dest_x, dest_y, _ = move_queue[curr_move_idx]
            person_pos[p_idx] = np.array([dest_x, dest_y], dtype=float)
            curr_move_idx += 1

        for rx_id, rx_pos in receivers.items():
            link_vec = rx_pos - tx
            L = float(np.linalg.norm(link_vec))
            if L < 1e-4:
                continue

            # Base shadowing path loss
            mean_loss = -10.0 * beta * math.log10(max(L, 0.1))
            base_dbm = p_r0_dbm + mean_loss + float(np_rng.normal(0.0, sigma_shadow))

            # Fresnel zone excess body attenuation
            extra_db = 0.0
            var_db2 = 0.0

            for p_idx, pos_p in person_pos.items():
                p_vec = pos_p - tx
                s = float(np.dot(p_vec, link_vec) / (L * L))
                if s <= 0.0 or s >= 1.0:
                    continue  # Not between transmitter and receiver

                dperp = float(abs(p_vec[0] * link_vec[1] - p_vec[1] * link_vec[0]) / L)
                d1 = s * L
                d2 = (1.0 - s) * L
                rF = math.sqrt(lambda_rf * d1 * d2 / (d1 + d2))
                w = fresnel_scale * rF + body_radius
                g = math.exp(-((dperp / w) ** 2))

                extra_db += loss_db * g
                var_db2 += (extra_std * extra_std) * (g * g)

            body_noise = float(np_rng.normal(0.0, math.sqrt(var_db2))) if var_db2 > 1e-6 else 0.0
            total_rssi = base_dbm - extra_db + body_noise

            if cfg["sim"].get("quantize_dbm", True):
                total_rssi = float(round(total_rssi))

            # format: time receiver_node sender_node rssi_dbm
            rssi_lines.append(f"{t:.6f} {rx_id} 0 {total_rssi:.3f}\n")

    rssi_file = ns2_out / f"{run_id}.rssi"
    with open(rssi_file, "w", encoding="utf-8") as f:
        f.writelines(rssi_lines)

    # Immediately parse into standard data/sim/ folder
    parse_ns2_run(run_id, cfg)
    return root / "data" / "sim" / run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NS2 Tcl scenarios or execute Python simulation")
    parser.add_argument("--scenarios", type=str, default="S1,S2,S3,S4,S5", help="Comma-separated scenario IDs")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to generate per scenario")
    parser.add_argument("--engine", type=str, choices=["tcl", "python"], default="python", help="Generation target")
    args = parser.parse_args()

    cfg = load_config()
    scns = [s.strip().upper() for s in args.scenarios.split(",")]

    print(f"[*] Generating scenarios: {scns} | Seeds: 1..{args.seeds} | Engine: {args.engine}")
    for scn in scns:
        for seed in range(1, args.seeds + 1):
            if args.engine == "tcl":
                run_id = generate_scenario_tcl(scn, seed, cfg)
                print(f"  [+] Created Tcl scenario: {run_id}.tcl")
            else:
                out = simulate_scenario_python(scn, seed, cfg)
                print(f"  [+] Simulated {scn} (seed {seed}) -> {out}")
    print("[SUCCESS] Scenario generation complete!")
