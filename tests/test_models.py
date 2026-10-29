import numpy as np
import pytest
from occ.models.baselines import (
    ThresholdClassifier,
    get_knn,
    get_logistic_regression,
    get_svm
)
from occ.models.classifier import get_random_forest

def test_threshold_classifier():
    clf = ThresholdClassifier(var_thresh=2.0, delta_thresh=3.0)
    # Synthetic feature row: idx 1=R1_std, idx 8=R2_std, idx 16=R1_delta_mean
    empty_row = np.zeros(21)
    empty_row[1] = 1.0  # low std
    empty_row[8] = 1.2
    empty_row[16] = 0.5  # low delta

    occupied_row = np.zeros(21)
    occupied_row[1] = 3.5  # high std
    occupied_row[8] = 4.0
    occupied_row[16] = -7.0  # high delta

    preds = clf.predict(np.array([empty_row, occupied_row]))
    assert preds[0] == "Empty"
    assert preds[1] in ["Occupied", "Transition"]

def test_scikit_learn_classifiers():
    # Synthetic small dataset
    np.random.seed(42)
    X = np.random.randn(30, 21)
    y = np.array(["Empty"] * 10 + ["Occupied"] * 10 + ["Transition"] * 10)

    models = [
        ("LogisticRegression", get_logistic_regression()),
        ("KNN", get_knn(n_neighbors=3)),
        ("SVM", get_svm()),
        ("RandomForest", get_random_forest(n_estimators=10))
    ]

    for name, m in models:
        m.fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({"Empty", "Occupied", "Transition"})
