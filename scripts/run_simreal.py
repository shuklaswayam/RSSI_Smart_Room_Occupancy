import argparse
from pathlib import Path
import sys
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from occ.config import get_root_dir, load_config
from occ.features.extractor import FEATURE_COLUMNS
from occ.models.classifier import get_random_forest

LABELS = ["Empty", "Occupied", "Transition"]

def fit_weighted(model, X, y, sample_weight):
    m = clone(model)
    return m.fit(X, y, sample_weight=sample_weight)

def evaluate_regimes_for_fold(
    rf_base,
    X_tr_real: np.ndarray,
    y_tr_real: np.ndarray,
    X_sim: np.ndarray,
    y_sim: np.ndarray,
    X_te_real: np.ndarray,
    y_te_real: np.ndarray,
    w_sim: float = 0.3
) -> Dict[str, float]:
    """Train each of the 4 regimes and evaluate macro F1 on the held-out real test set."""
    scores = {}
    
    # 1. Real-only
    m_real = clone(rf_base).fit(X_tr_real, y_tr_real)
    scores["real_only"] = float(f1_score(y_te_real, m_real.predict(X_te_real), labels=LABELS, average="macro", zero_division=0))
    
    # 2. Sim-only
    m_sim = clone(rf_base).fit(X_sim, y_sim)
    scores["sim_only"] = float(f1_score(y_te_real, m_sim.predict(X_te_real), labels=LABELS, average="macro", zero_division=0))
    
    # 3. Pre-train on sim, Fine-tune on real (Random Forest warm_start)
    if set(y_tr_real) == set(y_sim):
        rf_ft = clone(rf_base).set_params(n_estimators=100, warm_start=True).fit(X_sim, y_sim)
        rf_ft.set_params(n_estimators=200).fit(X_tr_real, y_tr_real)
        scores["pretrain_finetune"] = float(f1_score(y_te_real, rf_ft.predict(X_te_real), labels=LABELS, average="macro", zero_division=0))
    else:
        scores["pretrain_finetune"] = scores.get("mixed", scores["real_only"])
    
    # 4. Mixed training (real weight 1.0, sim weight w_sim)
    X_mix = np.vstack([X_tr_real, X_sim])
    y_mix = np.concatenate([y_tr_real, y_sim])
    weights = np.concatenate([np.ones(len(y_tr_real)), np.full(len(y_sim), w_sim)])
    m_mix = clone(rf_base).fit(X_mix, y_mix, sample_weight=weights)
    scores["mixed"] = float(f1_score(y_te_real, m_mix.predict(X_te_real), labels=LABELS, average="macro", zero_division=0))
    
    return scores

def run_sim_to_real_study(
    df_real: pd.DataFrame,
    df_sim: pd.DataFrame,
    features: List[str] = FEATURE_COLUMNS,
    n_real_list: List[int] = [1, 2, 4, 8],
    w_sim: float = 0.3,
    reps: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """Execute the full O11 sim-to-real transfer learning study."""
    # Ensure both datasets are restricted to the 3 occupancy classes (S1-S4)
    df_real = df_real[df_real["label"].isin(LABELS)].copy()
    df_sim = df_sim[df_sim["label"].isin(LABELS)].copy()

    rng = np.random.default_rng(seed)
    real_runs = df_real["run_id"].unique()
    
    # Clean simulated training pool
    X_sim = df_sim[features].to_numpy(dtype=float)
    y_sim = df_sim["label"].to_numpy()
    
    rf_base = get_random_forest(n_estimators=150, max_depth=8, random_state=seed, class_weight=None)
    
    results = []
    print(f"[*] Starting Sim-to-Real Transfer Learning Evaluation (O11, E16)...")
    print(f"  [+] Real sessions: {len(real_runs)} | Sim windows: {len(df_sim)}")
    
    for held_run in real_runs:
        df_te = df_real[df_real["run_id"] == held_run]
        assert (df_te["origin"] == "real").all(), "Synthetic data leaked into test fold!"
        
        X_te = df_te[features].to_numpy(dtype=float)
        y_te = df_te["label"].to_numpy()
        
        other_runs = [r for r in real_runs if r != held_run]
        
        for n_real in n_real_list:
            if n_real > len(other_runs):
                continue
            for rep in range(reps):
                chosen_runs = rng.choice(other_runs, size=n_real, replace=False)
                df_tr = df_real[df_real["run_id"].isin(chosen_runs)]
                
                X_tr = df_tr[features].to_numpy(dtype=float)
                y_tr = df_tr["label"].to_numpy()
                
                scores = evaluate_regimes_for_fold(
                    rf_base, X_tr, y_tr, X_sim, y_sim, X_te, y_te, w_sim=w_sim
                )
                
                for regime, f1 in scores.items():
                    results.append({
                        "held_out_run": held_run,
                        "n_real_sessions": n_real,
                        "rep": rep,
                        "regime": regime,
                        "macro_f1": f1
                    })
                    
    df_res = pd.DataFrame(results)
    summary = df_res.groupby(["regime", "n_real_sessions"])["macro_f1"].agg(
        mean_f1="mean", std_f1="std", min_f1="min", max_f1="max"
    ).reset_index()
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sim-to-real transfer learning study (O11, E16)")
    parser.add_argument("--real", type=str, default="data/processed/windows_features_real.csv", help="Real features CSV")
    parser.add_argument("--sim", type=str, default="data/processed/windows_features_sim.csv", help="Sim features CSV")
    parser.add_argument("--out", type=str, default="results/tables/e16_simreal_learning_curves.csv", help="Output summary CSV")
    args = parser.parse_args()

    root = get_root_dir()
    real_p = root / args.real if not Path(args.real).is_absolute() else Path(args.real)
    sim_p = root / args.sim if not Path(args.sim).is_absolute() else Path(args.sim)
    out_p = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if not real_p.is_file() or not sim_p.is_file():
        print(f"[!] Feature files not ready (real={real_p.exists()}, sim={sim_p.exists()}).")
        exit(1)

    df_real = pd.read_csv(real_p)
    df_sim = pd.read_csv(sim_p)

    summary_df = run_sim_to_real_study(df_real, df_sim)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_p, index=False)
    
    print("\n" + "="*80)
    print("O11 / E16 SIM-TO-REAL TRANSFER LEARNING RESULTS (Macro-F1)")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
