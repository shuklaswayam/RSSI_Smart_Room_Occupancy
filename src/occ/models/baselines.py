from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    Heuristic rule-based baseline:
    Uses variance and baseline level shift to distinguish Empty vs Occupied/Transition.
    """
    def __init__(self, var_thresh: float = 2.0, delta_thresh: float = 3.0):
        self.var_thresh = var_thresh
        self.delta_thresh = delta_thresh
        self.classes_ = np.array(["Empty", "Occupied", "Transition"])

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Unsupervised rule heuristic, no training required
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Expects X feature columns: col 1 is R1_std, col 8 is R2_std, col 16 is R1_delta_mean
        preds = []
        for row in X:
            # Check R1_std (idx 1), R2_std (idx 8) and R1_delta_mean (idx 16)
            std_avg = (row[1] + row[8]) / 2.0 if len(row) > 8 else row[1]
            delta = abs(row[16]) if len(row) > 16 else 0.0
            
            if std_avg > 3.0 and delta > 4.0:
                preds.append("Transition")
            elif std_avg > self.var_thresh or delta > self.delta_thresh:
                preds.append("Occupied")
            else:
                preds.append("Empty")
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        probs = []
        for p in preds:
            if p == "Empty":
                probs.append([0.85, 0.10, 0.05])
            elif p == "Occupied":
                probs.append([0.10, 0.80, 0.10])
            else:
                probs.append([0.05, 0.20, 0.75])
        return np.array(probs)

def get_logistic_regression(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state
        ))
    ])

def get_knn(n_neighbors: int = 5) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance"
        ))
    ])

def get_svm(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ))
    ])
