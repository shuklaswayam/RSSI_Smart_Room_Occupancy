import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from occ.config import get_root_dir

DEFAULT_FEATS = ("R1_mean", "R2_mean", "R1_std", "R2_std", "R1R2_mean_diff")

def compute_fidelity_metrics(
    real_vals: pd.Series,
    sim_vals: pd.Series
) -> Tuple[float, float]:
    """Compute Kolmogorov-Smirnov statistic D and 1st Wasserstein distance in dB."""
    r = real_vals.dropna().to_numpy()
    s = sim_vals.dropna().to_numpy()
    if len(r) == 0 or len(s) == 0:
        return 1.0, 99.0
    ks_stat = float(ks_2samp(r, s).statistic)
    w1_dist = float(wasserstein_distance(r, s))
    return ks_stat, w1_dist

def fidelity_table(
    df_real: pd.DataFrame,
    df_sim: pd.DataFrame,
    feats: Tuple[str, ...] = DEFAULT_FEATS,
    by: Tuple[str, ...] = ("label",)
) -> pd.DataFrame:
    """
    Generate comprehensive fidelity evaluation table comparing
    real-world vs simulated feature distributions across classes.
    """
    rows = []
    by_list = list(by)
    
    for key, g_real in df_real.groupby(by_list):
        if not isinstance(key, tuple):
            key = (key,)
        # Filter matching group in simulation
        cond = pd.Series(True, index=df_sim.index)
        for col, val in zip(by_list, key):
            if col in df_sim.columns:
                cond = cond & (df_sim[col] == val)
        g_sim = df_sim[cond]
        
        if len(g_real) < 5 or len(g_sim) < 5:
            continue
            
        group_meta = dict(zip(by_list, key))
        for f in feats:
            if f in g_real.columns and f in g_sim.columns:
                ks, w1 = compute_fidelity_metrics(g_real[f], g_sim[f])
                rows.append({
                    **group_meta,
                    "feature": f,
                    "n_real": len(g_real),
                    "n_sim": len(g_sim),
                    "real_mean": round(float(g_real[f].mean()), 2),
                    "sim_mean": round(float(g_sim[f].mean()), 2),
                    "KS_statistic": round(ks, 4),
                    "Wasserstein_dB": round(w1, 3)
                })
                
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate simulation fidelity against real measurements")
    parser.add_argument("--real", type=str, default="data/processed/windows_features_real.csv", help="Path to real features CSV")
    parser.add_argument("--sim", type=str, default="data/processed/windows_features_sim.csv", help="Path to sim features CSV")
    parser.add_argument("--out", type=str, default="results/tables/e14_fidelity.csv", help="Output CSV path")
    args = parser.parse_args()

    root = get_root_dir()
    real_path = root / args.real if not Path(args.real).is_absolute() else Path(args.real)
    sim_path = root / args.sim if not Path(args.sim).is_absolute() else Path(args.sim)
    out_path = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if not real_path.is_file() or not sim_path.is_file():
        raise FileNotFoundError(
            f"Features file missing (real={real_path.exists()}, sim={sim_path.exists()}). "
            "Please run scripts/build_dataset.py first."
        )

    df_real = pd.read_csv(real_path)
    df_sim = pd.read_csv(sim_path)

    tbl = fidelity_table(df_real, df_sim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_path, index=False)
    print(f"[SUCCESS] Fidelity evaluation saved to {out_path}:\n{tbl.to_string(index=False)}")
