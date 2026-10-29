from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from occ.config import get_root_dir, load_config
from occ.features.extractor import FEATURE_COLUMNS

def get_random_forest(
    n_estimators: int = 200,
    max_depth: Optional[int] = 8,
    min_samples_split: int = 5,
    random_state: int = 42,
    warm_start: bool = False,
    class_weight: Optional[Any] = "balanced"
) -> RandomForestClassifier:
    """Instantiate primary Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state,
        warm_start=warm_start,
        n_jobs=-1
    )

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rf",
    sample_weight: Optional[np.ndarray] = None,
    cfg: Optional[Dict[str, Any]] = None
) -> Any:
    """Train classifier on feature matrix."""
    if model_type == "rf":
        clf = get_random_forest()
    else:
        from occ.models.baselines import get_logistic_regression
        clf = get_logistic_regression()
        
    if sample_weight is not None:
        clf.fit(X, y, sample_weight=sample_weight)
    else:
        clf.fit(X, y)
    return clf

def validate_feature_vector(X: np.ndarray, expected_dim: int = 21) -> np.ndarray:
    """Validate and sanitize feature vector for safe production inference."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != expected_dim:
        raise ValueError(f"Feature dimension mismatch: expected {expected_dim} features, received {arr.shape[1]}")
    # Sanitize NaNs and Infs to prevent inference crashes
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def save_model(
    model: Any,
    out_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """Serialize model artifact and MLE production bundle using joblib."""
    if out_path is None:
        cfg = load_config()
        out_path = cfg["paths"]["models_dir"] + "/rf_primary.joblib"
    path = Path(out_path)
    if not path.is_absolute():
        path = get_root_dir() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Direct model dump for backwards compatibility
    joblib.dump(model, str(path))
    
    # Metadata bundle for MLE production governance
    bundle = {
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "classes": getattr(model, "classes_", np.array(["Empty", "Occupied", "Transition"])).tolist()
                   if hasattr(getattr(model, "classes_", None), "tolist")
                   else getattr(model, "classes_", ["Empty", "Occupied", "Transition"]),
        "metadata": metadata or {"version": "1.0.0", "origin": "real", "framework": "scikit-learn"},
        "expected_dim": len(FEATURE_COLUMNS)
    }
    bundle_path = path.with_suffix(".bundle.joblib")
    joblib.dump(bundle, str(bundle_path))
    return path

def load_model(in_path: Optional[str] = None) -> Any:
    """Load serialized model artifact, transparently handling both raw estimators and MLE bundles."""
    if in_path is None:
        cfg = load_config()
        in_path = cfg["paths"]["models_dir"] + "/rf_primary.joblib"
    path = Path(in_path)
    if not path.is_absolute():
        path = get_root_dir() / path
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    loaded = joblib.load(str(path))
    if isinstance(loaded, dict) and "model" in loaded:
        m = loaded["model"]
        m._bundle_metadata = loaded
        return m
    return loaded

def get_feature_importances(model: Any) -> pd.DataFrame:
    """Extract feature importance ranking from Random Forest model."""
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        df_fi = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": fi
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_fi
    return pd.DataFrame(columns=["feature", "importance"])
