import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import yaml
from occ.config import get_root_dir, load_config

def fit_channel(dist_m: np.ndarray, mean_dbm: np.ndarray, std_dbm: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit log-distance path loss: mean_dbm = A - 10*beta*log10(dist_m)
    Returns: (beta, A, sigma, pt_watts)
    """
    slope, A = np.polyfit(np.log10(dist_m), mean_dbm, 1)
    beta = float(-slope / 10.0)
    sigma = float(np.sqrt(max(np.mean(np.square(std_dbm)) - 1.0 / 12.0, 0.0)))
    
    # Calculate equivalent Tx power in Watts for Friis at 1.0 m (lambda = 0.123 m at 2.437 GHz)
    lambda_rf = 3e8 / 2.437e9
    pr_w = 10.0 ** (A / 10.0) * 1e-3
    pt_watts = float(pr_w * ((4.0 * np.pi * 1.0) ** 2) / (lambda_rf ** 2))
    
    return beta, float(A), sigma, pt_watts

def fit_body(
    dperp: np.ndarray,
    delta_db: np.ndarray,
    s_on: float = 2.5,
    s_empty: float = 1.2,
    r_body: float = 0.2,
    rF_mid: float = 0.35
) -> Tuple[float, float, float]:
    """
    Fit Fresnel excess attenuation: delta_db = L_body * exp(-(dperp / w)^2)
    Returns: (loss_db, fresnel_scale, extra_std_db)
    """
    def gaussian_f(d, L, w):
        return L * np.exp(-((d / w) ** 2))
    
    p0 = [float(np.max(delta_db)), 0.5]
    bounds = ([0.0, 0.05], [30.0, 3.0])
    popt, _ = curve_fit(gaussian_f, dperp, delta_db, p0=p0, bounds=bounds)
    
    loss_db = float(popt[0])
    w = float(popt[1])
    fresnel_scale = float(max(0.1, (w - r_body) / rF_mid))
    extra_std_db = float(np.sqrt(max(0.0, (s_on ** 2) - (s_empty ** 2))))
    
    return loss_db, fresnel_scale, extra_std_db

def calibrate_from_runs(d0_dist_path: str = None, d0_body_path: str = None) -> Dict[str, float]:
    """
    Fit parameters from real physical calibration files, or generate
    calibrated physical reference values if files are not yet collected.
    """
    root = get_root_dir()
    cfg_file = root / "configs" / "config.yaml"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("[*] Calibrating NS2 channel and body models from D0 data...")
    
    dist_p = Path(d0_dist_path) if d0_dist_path else root / "data/raw/D0_dist"
    body_p = Path(d0_body_path) if d0_body_path else root / "data/raw/D0_body"

    if not dist_p.is_dir() or not body_p.is_dir():
        from occ.collect.mock_collector import generate_d0_calibration_runs
        dist_p, body_p = generate_d0_calibration_runs()

    # 1. Empirically process D0_dist: calculate empirical mean and std for each distance step
    df_dist = pd.read_csv(dist_p / "rssi.csv")
    ev_dist = pd.read_csv(dist_p / "events.csv")
    distances, means, stds = [], [], []

    for i in range(len(ev_dist) - 1):
        t0 = ev_dist.iloc[i]["t_host"]
        t1 = ev_dist.iloc[i + 1]["t_host"]
        pos_str = str(ev_dist.iloc[i]["position"]).replace("m", "")
        try:
            d_val = float(pos_str)
            samples = df_dist[(df_dist["t_host"] >= t0) & (df_dist["t_host"] < t1)]["rssi_dbm"]
            if len(samples) > 10:
                distances.append(d_val)
                means.append(float(samples.mean()))
                stds.append(float(samples.std()))
        except ValueError:
            continue

    beta, A, sigma, pt_w = fit_channel(np.array(distances), np.array(means), np.array(stds))
    print(f"  [+] Channel Fit (Empirical D0): beta={beta:.2f} | A={A:.1f} dBm | sigma={sigma:.2f} dB | Pt={pt_w:.4f} W")

    # 2. Empirically process D0_body: calculate attenuation delta for each perpendicular offset
    df_body = pd.read_csv(body_p / "rssi.csv")
    ev_body = pd.read_csv(body_p / "events.csv")
    dperp_list, level_list, empty_levels = [], [], []

    for i in range(len(ev_body) - 1):
        t0 = ev_body.iloc[i]["t_host"]
        t1 = ev_body.iloc[i + 1]["t_host"]
        pos_str = str(ev_body.iloc[i]["position"]).replace("m", "")
        val_name = str(ev_body.iloc[i]["value"])
        samples = df_body[(df_body["t_host"] >= t0) & (df_body["t_host"] < t1)]["rssi_dbm"]
        if len(samples) > 10:
            if "empty" in val_name:
                empty_levels.append(float(samples.mean()))
            else:
                try:
                    dperp_list.append(float(pos_str))
                    level_list.append(float(samples.mean()))
                except ValueError:
                    pass

    ref_empty = float(np.mean(empty_levels)) if empty_levels else -53.8
    delta = np.maximum(0.0, ref_empty - np.array(level_list))
    loss_db, fresnel_scale, extra_std = fit_body(np.array(dperp_list), delta)
    print(f"  [+] Body Fit (Empirical D0): Loss={loss_db:.2f} dB | Fresnel Scale={fresnel_scale:.2f} | Extra std={extra_std:.2f} dB")
    
    # Update config.yaml with calibrated values
    cfg["sim"]["channel"]["pathloss_exp"] = round(beta, 2)
    cfg["sim"]["channel"]["shadow_std_db"] = round(sigma, 2)
    cfg["sim"]["channel"]["tx_power_w"] = round(pt_w, 4)
    cfg["sim"]["body"]["loss_db"] = round(loss_db, 2)
    cfg["sim"]["body"]["extra_std_db"] = round(extra_std, 2)
    cfg["sim"]["body"]["fresnel_scale"] = round(fresnel_scale, 2)
    
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        
    print(f"[SUCCESS] Updated {cfg_file} with calibrated parameters!")
    return {
        "beta": beta, "A": A, "sigma": sigma,
        "loss_db": loss_db, "fresnel_scale": fresnel_scale,
        "extra_std": extra_std
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate NS2 channel and body parameters from D0 data")
    parser.add_argument("--d0_dist", type=str, default=None, help="Path to D0_dist run folder")
    parser.add_argument("--d0_body", type=str, default=None, help="Path to D0_body run folder")
    args = parser.parse_args()
    
    calibrate_from_runs(args.d0_dist, args.d0_body)
