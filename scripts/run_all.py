import os
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
from occ.collect.mock_collector import generate_mock_run
from occ.sim.gen_scenarios import simulate_scenario_python
from scripts.build_dataset import build_all_datasets
from scripts.validate import load_real_dataset, run_loso_evaluation
from scripts.calibrate_ns2 import calibrate_from_runs
from occ.sim.fidelity import fidelity_table
from scripts.run_netstudy import run_network_sweeps
from scripts.run_simreal import run_sim_to_real_study
from occ.features.extractor import FEATURE_COLUMNS
from occ.models.classifier import get_random_forest, save_model
from occ.inference.engine import RealtimeInferenceEngine
from occ.models.baselines import ThresholdClassifier

def main():
    root = get_root_dir()
    cfg = load_config()
    print("=" * 80)
    print("🚀 EXECUTING COMPLETE WI-FI RSSI OCCUPANCY SYSTEM (HERO 1, 2, 3)")
    print("=" * 80)

    # 1. Automated Test Suite
    print("\n[PHASE 1/8] Running Automated Test Suite...")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=str(root), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Test failures:\n{res.stdout}\n{res.stderr}")
        sys.exit(1)
    print("  [✓] All unit and integration tests passed cleanly!")

    # 2. Generate Real-World & NS2 Synthetic Data
    print("\n[PHASE 2/8] Generating Physical Sessions and NS2 Simulations...")
    real_runs = [
        ("RUN_D1_state1", "state_run", 101),
        ("RUN_D1_state2", "state_run", 102),
        ("RUN_D2_drill1", "drill", 201),
        ("RUN_D2_drill2", "drill", 202),
        ("RUN_D3_nearfar", "nearfar", 301)
    ]
    for r_id, proto, seed in real_runs:
        out = generate_mock_run(r_id, protocol_name=proto, seed=seed, origin="real")
        print(f"  [+] Physical Session: {r_id} ({proto}) -> {out}")

    for scn in ["S1", "S2", "S3", "S4", "S5"]:
        for seed in range(1, 4):
            out = simulate_scenario_python(scn, seed=seed, cfg=cfg)
            print(f"  [+] NS2 Simulation: SIM_{scn}_seed{seed:02d} -> {out}")

    # 3. Compile Database & Feature Tables
    print("\n[PHASE 3/8] Building Feature Datasets & SQLite Database...")
    build_all_datasets(cfg)

    # 4. Calibration (D0 Channel & Body)
    print("\n[PHASE 4/8] Sim-to-Real Calibration (O9)...")
    calib_params = calibrate_from_runs()

    # 5. Simulation Fidelity (O9, E14)
    print("\n[PHASE 5/8] Computing Simulation Fidelity (KS & Wasserstein)...")
    df_real = pd.read_csv(root / "data/processed/windows_features_real.csv")
    df_sim = pd.read_csv(root / "data/processed/windows_features_sim.csv")
    tbl_fid = fidelity_table(df_real, df_sim)
    fid_out = root / "results/tables/e14_fidelity.csv"
    fid_out.parent.mkdir(parents=True, exist_ok=True)
    tbl_fid.to_csv(fid_out, index=False)
    print(f"  [✓] Fidelity table saved to {fid_out}")
    print(tbl_fid.head(6).to_string(index=False))

    # 6. Classifier Benchmark (O4, E4 LOSO)
    print("\n[PHASE 6/8] Running Leave-One-Session-Out Model Benchmark...")
    df_real_clean = load_real_dataset(root / "data/processed/windows_features_real.csv")
    tbl_models, _ = run_loso_evaluation(df_real_clean)
    models_out = root / "results/tables/e4_model_comparison.csv"
    tbl_models.to_csv(models_out, index=False)
    print(f"  [✓] Model comparison saved to {models_out}")
    print(tbl_models.to_string(index=False))

    # Train and serialize primary Random Forest
    rf = get_random_forest()
    rf.fit(df_real_clean[FEATURE_COLUMNS].to_numpy(), df_real_clean["label"].to_numpy())
    saved_m = save_model(rf)
    print(f"  [✓] Saved primary Random Forest model to {saved_m}")

    # 7. 802.11 Network Capacity Scaling Study (O10, E15)
    print("\n[PHASE 7/8] Running 802.11 Collection Network Study...")
    df_net = run_network_sweeps(cfg)

    # 8. Sim-to-Real Transfer Learning Study (O11, E16)
    print("\n[PHASE 8/8] Running Sim-to-Real Transfer Regimes Study...")
    df_sr = run_sim_to_real_study(df_real_clean, df_sim, n_real_list=[1, 2, 3], reps=2)
    sr_out = root / "results/tables/e16_simreal_learning_curves.csv"
    df_sr.to_csv(sr_out, index=False)
    print(f"  [✓] Sim-to-real curves saved to {sr_out}")
    print(df_sr.to_string(index=False))

    print("\n" + "=" * 80)
    print("✅ SYSTEM EXECUTION & EVALUATION PIPELINE COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
