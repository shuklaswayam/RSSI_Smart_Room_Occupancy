import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from occ.config import get_root_dir, load_config
from occ.features.extractor import FEATURE_COLUMNS
from occ.models.baselines import (
    ThresholdClassifier,
    get_knn,
    get_logistic_regression,
    get_svm
)
from occ.models.classifier import get_random_forest, save_model

LABELS = ["Empty", "Occupied", "Transition"]

def load_real_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    df = pd.read_csv(path)
    # Strictly enforce that test/validation data contains only real origins
    if "origin" in df.columns:
        df = df[df["origin"] == "real"]
    # Filter to primary 3-class occupancy states (S5 near/far is evaluated separately)
    if "label" in df.columns:
        df = df[df["label"].isin(LABELS)]
    return df

def run_loso_evaluation(
    df: pd.DataFrame,
    features: List[str] = FEATURE_COLUMNS,
    label_col: str = "label",
    group_col: str = "run_id"
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Perform Leave-One-Session-Out (LOSO) cross-validation across the 5 models.
    Guarantees no window leakage across training and test folds.
    """
    unique_runs = df[group_col].unique()
    if len(unique_runs) < 2:
        raise ValueError(f"Need at least 2 distinct runs for LOSO validation; found {len(unique_runs)}")

    models = {
        "Threshold": ThresholdClassifier(),
        "LogisticRegression": get_logistic_regression(),
        "KNN": get_knn(),
        "SVM_RBF": get_svm(),
        "RandomForest": get_random_forest()
    }

    results = []
    confusion_matrices = {}

    for name, model_tmpl in models.items():
        y_true_all = []
        y_pred_all = []

        for held_out_run in unique_runs:
            train_mask = df[group_col] != held_out_run
            test_mask = df[group_col] == held_out_run

            df_train = df[train_mask]
            df_test = df[test_mask]

            # Security assertion: ensure test fold has zero synthetic windows
            if "origin" in df_test.columns:
                assert (df_test["origin"] == "real").all(), "Simulated windows leaked into test fold!"

            X_tr = df_train[features].to_numpy(dtype=float)
            y_tr = df_train[label_col].to_numpy()

            X_te = df_test[features].to_numpy(dtype=float)
            y_te = df_test[label_col].to_numpy()

            from sklearn.base import clone
            clf = clone(model_tmpl)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)

            y_true_all.extend(y_te)
            y_pred_all.extend(preds)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        acc = float(accuracy_score(y_true_all, y_pred_all))
        prec = float(precision_score(y_true_all, y_pred_all, labels=LABELS, average="macro", zero_division=0))
        rec = float(recall_score(y_true_all, y_pred_all, labels=LABELS, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true_all, y_pred_all, labels=LABELS, average="macro", zero_division=0))

        # Per-class F1
        f1_per_class = f1_score(y_true_all, y_pred_all, labels=LABELS, average=None, zero_division=0)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=LABELS)
        confusion_matrices[name] = cm

        results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "macro_precision": round(prec, 4),
            "macro_recall": round(rec, 4),
            "macro_f1": round(f1_macro, 4),
            "f1_empty": round(float(f1_per_class[0]), 4),
            "f1_occupied": round(float(f1_per_class[1]), 4),
            "f1_transition": round(float(f1_per_class[2]), 4) if len(f1_per_class) > 2 else 0.0
        })

    df_results = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return df_results, confusion_matrices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-model LOSO validation on real physical dataset")
    parser.add_argument("--input", type=str, default="data/processed/windows_features_real.csv", help="Input feature CSV")
    parser.add_argument("--out", type=str, default="results/tables/e4_model_comparison.csv", help="Output metrics CSV")
    args = parser.parse_args()

    root = get_root_dir()
    csv_path = root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out_path = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    print(f"[*] Loading real features from {csv_path}...")
    if not csv_path.is_file():
        print(f"[!] File not found: {csv_path}. Please run scripts/build_dataset.py first.")
        exit(1)

    df_data = load_real_dataset(csv_path)
    print(f"  [+] Loaded {len(df_data)} windows from {df_data['run_id'].nunique()} real sessions.")

    tbl, cms = run_loso_evaluation(df_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_path, index=False)
    
    print("\n" + "="*80)
    print("O4 / E4 FIVE-MODEL LEAVE-ONE-SESSION-OUT BENCHMARK RESULTS")
    print("="*80)
    print(tbl.to_string(index=False))
    print("="*80)
    
    # Train full Random Forest model and serialize
    print("[*] Training and serializing full Random Forest model to models/rf_primary.joblib...")
    rf = get_random_forest()
    rf.fit(df_data[FEATURE_COLUMNS].to_numpy(), df_data["label"].to_numpy())
    saved_path = save_model(rf)
    print(f"[SUCCESS] Serialized primary model to {saved_path}")
