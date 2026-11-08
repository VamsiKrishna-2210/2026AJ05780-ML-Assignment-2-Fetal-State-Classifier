"""Shared pieces for training and the Streamlit app.

The five classifiers are fit once by train_models.py and saved under model/.
The app only loads those files. It does not fit them again.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ctg.csv"
TEST_PATH = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
METRICS_PATH = ROOT / "results" / "metrics.csv"

TARGET = "FetalState"
CLASS_ORDER = ["Normal", "Suspect", "Pathologic"]
FEATURES = [
    "LB",
    "AC",
    "FM",
    "UC",
    "DL",
    "DS",
    "DP",
    "ASTV",
    "MSTV",
    "ALTV",
    "MLTV",
    "Width",
    "Min",
    "Max",
    "Nmax",
    "Nzeros",
    "Mode",
    "Mean",
    "Median",
    "Variance",
    "Tendency",
]

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def make_models():
    """Five classifiers on the same feature table.

    Logistic regression, KNN and the SVM sit behind a scaler because their
    distances or coefficients depend on units. The tree and Gaussian Naive
    Bayes use the raw measurements.
    """
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=11)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def aligned_probabilities(estimator, frame, labels):
    """Return predict_proba columns in `labels` order, not sklearn's class order."""
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = estimator.predict_proba(frame)
    class_list = [str(name) for name in estimator.classes_]
    order = [class_list.index(label) for label in labels]
    return proba[:, order]


def compute_metrics(y_true, y_pred, y_proba, labels):
    """Accuracy, AUC, and macro precision / recall / F1, plus MCC.

    Macro averages give Suspect and Pathologic the same weight as Normal.
    AUC is one-vs-rest macro. It is left blank when the file has one class.
    """
    y_true = pd.Series(y_true).astype(str).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(str).reset_index(drop=True)
    present = [label for label in labels if label in set(y_true)]
    score_labels = present or None

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(
                y_true, y_pred, average="macro", zero_division=0, labels=score_labels
            )
        ),
        "Recall": float(
            recall_score(
                y_true, y_pred, average="macro", zero_division=0, labels=score_labels
            )
        ),
        "F1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, labels=score_labels)
        ),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "AUC": float("nan"),
    }

    if y_proba is None or len(present) < 2:
        return metrics

    # roc_auc_score requires `labels` in sorted order, and the probability
    # columns must follow that same order.
    auc_labels = sorted(present)
    column_index = [list(labels).index(label) for label in auc_labels]
    metrics["AUC"] = float(
        roc_auc_score(
            y_true,
            y_proba[:, column_index],
            multi_class="ovr",
            average="macro",
            labels=auc_labels,
        )
    )
    return metrics


def prepare_table(frame):
    """Keep the 21 features and FetalState. Reject anything else the app cannot score."""
    required = FEATURES + [TARGET]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        return None, "Missing columns: " + ", ".join(missing)

    table = frame[required].copy()
    for name in FEATURES:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    if table[FEATURES].isna().any().any():
        return None, (
            "Every feature must be numeric. Blank cells or text in a feature column "
            "cannot be scored."
        )

    table[TARGET] = table[TARGET].astype(str).str.strip()
    unknown = sorted(set(table[TARGET]) - set(CLASS_ORDER))
    if unknown:
        return None, (
            "Unknown FetalState values: "
            + ", ".join(unknown)
            + ". Use Normal, Suspect or Pathologic."
        )
    if table.empty:
        return None, "The file has no rows."
    return table, None
