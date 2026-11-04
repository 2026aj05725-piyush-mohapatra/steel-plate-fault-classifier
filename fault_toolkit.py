"""
fault_toolkit.py
----------------
Shared definitions for the Steel Plate Fault Classifier.

Both the training pipeline (train_models.py / model_training.ipynb) and the
Streamlit app (app.py) import from here, so the feature order, class labels,
model file names and -- most importantly -- the metric calculations are
identical in both places. That is what guarantees the numbers shown in the
app for test_data.csv match the numbers reported in the README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------
# Dataset schema (UCI Steel Plates Faults, DOI 10.24432/C5J88N)
# --------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum",
    "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
    "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
    "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400",
    "Steel_Plate_Thickness", "Edges_Index", "Empty_Index", "Square_Index",
    "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]

# The 7 one-hot target columns, in the order they appear in Faults.NNA.
# ("K_Scatch" is spelled this way in the original UCI file.)
FAULT_CLASSES: list[str] = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults",
]

TARGET_COLUMN = "Fault_Type"

FAULT_NOTES: dict[str, str] = {
    "Pastry": "Small flaky patches on the plate surface",
    "Z_Scratch": "Z-type scratch defect",
    "K_Scatch": "K-type scratch defect",
    "Stains": "Discoloured / stained regions",
    "Dirtiness": "Dirt or foreign residue on the surface",
    "Bumps": "Raised bumps on the surface",
    "Other_Faults": "Every other surface fault not in the six named types",
}

# --------------------------------------------------------------------------
# Model registry: display name -> file name inside model/
# --------------------------------------------------------------------------
MODEL_DIR_NAME = "model"
MODEL_REGISTRY: dict[str, str] = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes_gaussian.joblib",
    "Support Vector Machine": "svm_rbf.joblib",
}

METRIC_ORDER: list[str] = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_uci_faults(nna_path: str | Path) -> pd.DataFrame:
    """Read the raw UCI ``Faults.NNA`` file (whitespace separated, no header,
    27 feature columns + 7 one-hot fault columns) and return a tidy frame with
    the 27 features plus a single categorical ``Fault_Type`` column."""
    raw = pd.read_csv(
        nna_path, sep=r"\s+", header=None,
        names=FEATURE_COLUMNS + FAULT_CLASSES,
    )
    one_hot = raw[FAULT_CLASSES]
    if not (one_hot.sum(axis=1) == 1).all():
        raise ValueError("Each row must belong to exactly one fault class.")
    tidy = raw[FEATURE_COLUMNS].copy()
    tidy[TARGET_COLUMN] = one_hot.idxmax(axis=1)
    return tidy


def one_hot_to_label(frame: pd.DataFrame) -> pd.DataFrame:
    """If a frame carries the 7 original one-hot fault columns instead of a
    ``Fault_Type`` column, collapse them into ``Fault_Type``."""
    if TARGET_COLUMN in frame.columns:
        return frame
    if all(c in frame.columns for c in FAULT_CLASSES):
        out = frame.drop(columns=FAULT_CLASSES).copy()
        out[TARGET_COLUMN] = frame[FAULT_CLASSES].idxmax(axis=1)
        return out
    return frame


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def one_vs_rest_auc(y_true, proba: np.ndarray, class_order, average: str = "macro") -> float:
    """One-vs-rest ROC-AUC for a multi-class problem.

    Equivalent to ``roc_auc_score(..., multi_class="ovr")`` when every class is
    present, but it also works on an uploaded subset in which some classes are
    missing: classes without both positive and negative rows are skipped
    instead of raising an error. ``average`` is "macro" (plain mean) or
    "weighted" (weighted by class support)."""
    y_true = np.asarray(y_true)
    per_class, support = [], []
    for col, label in enumerate(class_order):
        is_pos = y_true == label
        if is_pos.any() and (~is_pos).any():
            per_class.append(roc_auc_score(is_pos.astype(int), proba[:, col]))
            support.append(int(is_pos.sum()))
    if not per_class:
        return float("nan")
    if average == "weighted":
        return float(np.average(per_class, weights=support))
    return float(np.mean(per_class))


def score_predictions(y_true, y_pred, proba: np.ndarray, class_order,
                      average: str = "macro") -> dict[str, float]:
    """The six metrics required by the assignment, computed one way only.

    Precision / recall / F1 are averaged over the fault types that actually occur
    in ``y_true``. On the full test split (all 7 types present) this is exactly
    scikit-learn's default; on an uploaded subset it stops classes that are absent
    from the file from dragging the average down with undefined zeros."""
    present = sorted(pd.unique(np.asarray(y_true)))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": one_vs_rest_auc(y_true, proba, class_order, average),
        "Precision": precision_score(y_true, y_pred, labels=present, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, labels=present, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, labels=present, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
