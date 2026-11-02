"""
Model evaluation: metrics, chart data and choosing the best model.

Every metric is computed on the held-out TEST set, which the model never saw
during training. A metric that cannot be computed is stored as None together
with a plain-English note explaining why (shown as "N/A" in the UI) — we never
invent a number.

Averaging choice for precision / recall / F1:
    * Binary problems: average="binary" — the score for the "positive" class.
      The positive class is the label that sorts LAST (1 over 0, "yes" over
      "no", True over False), which matches the usual convention.
    * Multiclass problems: average="weighted" — the per-class scores averaged,
      weighted by how many test examples each class has. This reflects overall
      performance without letting a tiny class dominate the number.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

MAX_SCATTER_POINTS = 500

# Primary metric used to rank models, and whether bigger numbers are better.
PRIMARY_METRICS = {
    "classification": ("f1", True),
    "regression": ("rmse", False),
}


def _clean(value: Any) -> float | None:
    """Convert to a plain float, mapping NaN/inf to None so it can be stored as JSON."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def compute_roc_auc(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, positive_class: Any = None) -> tuple[float | None, str | None]:
    """
    ROC-AUC measures how well the model *ranks* positives above negatives using
    its predicted probabilities (not the hard 0/1 predictions).

    Returns (value, None) on success or (None, reason) when it is not valid.
    """
    classes = list(pipeline.classes_)
    present = set(y_test.unique())
    if len(present) < 2:
        return None, "ROC-AUC needs at least two classes in the test set, but only one is present."
    if not present.issubset(classes):
        return None, "The test set contains a class that never appeared in training."

    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X_test)
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_test)
    else:
        return None, "This model does not provide probabilities or decision scores."

    try:
        if len(classes) == 2:
            positive_label = positive_class if positive_class is not None else classes[-1]
            positive_index = classes.index(positive_label)
            positive_scores = scores[:, positive_index] if scores.ndim == 2 else scores
            return float(roc_auc_score(y_test == positive_label, positive_scores)), None
        if present != set(classes):
            missing = ", ".join(str(c) for c in classes if c not in present)
            return None, f"Multiclass ROC-AUC needs every class in the test set; missing: {missing}."
        # One-vs-rest: each class against all others, then weighted by class size.
        value = roc_auc_score(y_test, scores, multi_class="ovr", average="weighted", labels=classes)
        return float(value), None
    except ValueError as error:
        return None, f"ROC-AUC could not be calculated: {error}"


def evaluate_classification(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, positive_class: Any = None
) -> tuple[dict[str, float | None], dict[str, str], dict[str, Any]]:
    """Return (metrics, metric_notes, chart_data) for a fitted classifier."""
    classes = list(pipeline.classes_)
    is_binary = len(classes) == 2
    selected_positive = positive_class if positive_class is not None else (classes[-1] if is_binary else None)
    average_kwargs: dict[str, Any] = (
        {"average": "binary", "pos_label": selected_positive} if is_binary else {"average": "weighted"}
    )
    # zero_division=0: if a class is never predicted, its precision is 0 instead of a warning + NaN.
    metrics = {
        "accuracy": _clean(accuracy_score(y_test, y_pred)),
        "precision": _clean(precision_score(y_test, y_pred, zero_division=0, **average_kwargs)),
        "recall": _clean(recall_score(y_test, y_pred, zero_division=0, **average_kwargs)),
        "f1": _clean(f1_score(y_test, y_pred, zero_division=0, **average_kwargs)),
    }
    notes: dict[str, str] = {
        "averaging": (
            f"Binary averaging with positive class '{selected_positive}'."
            if is_binary
            else "Weighted averaging across classes (weighted by test-set support)."
        )
    }
    roc_auc, roc_note = compute_roc_auc(pipeline, X_test, y_test, selected_positive)
    metrics["roc_auc"] = _clean(roc_auc)
    if roc_note:
        notes["roc_auc"] = roc_note

    # Include every label seen in training, the test set or the predictions so the matrix is square.
    labels = sorted(set(classes) | set(y_test.unique()) | set(np.unique(y_pred)))
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = [
        {
            "label": str(label),
            "precision": _clean(report[str(label)]["precision"]),
            "recall": _clean(report[str(label)]["recall"]),
            "f1": _clean(report[str(label)]["f1-score"]),
            "support": int(report[str(label)]["support"]),
        }
        for label in labels
        if str(label) in report
    ]
    chart_data = {
        "confusion_matrix": {"labels": [str(label) for label in labels], "matrix": matrix.tolist()},
        "per_class": per_class,
    }
    return metrics, notes, chart_data


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def evaluate_regression(
    y_test: pd.Series, y_pred: np.ndarray, random_seed: int
) -> tuple[dict[str, float | None], dict[str, str], dict[str, Any]]:
    """
    MAE  - average absolute error, in the target's own units. Easy to explain.
    MSE  - average squared error; punishes large mistakes heavily.
    RMSE - square root of MSE, back in the target's units. Our primary metric.
    R²   - share of the target's variance explained (1 = perfect, 0 = no better
           than always predicting the mean, negative = worse than that).
    """
    notes: dict[str, str] = {}
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "mae": _clean(mean_absolute_error(y_test, y_pred)),
        "mse": _clean(mse),
        "rmse": _clean(math.sqrt(mse)),
        "r2": None,
    }
    if len(y_test) < 2:
        notes["r2"] = "R² needs at least two test rows."
    elif y_test.nunique() < 2:
        notes["r2"] = "R² is undefined because every test target has the same value."
    else:
        metrics["r2"] = _clean(r2_score(y_test, y_pred))
        if metrics["r2"] is not None and metrics["r2"] < 0:
            notes["r2"] = "Negative R² means this model does worse than always predicting the average."

    # Down-sample the scatter plot for large test sets so the response stays small.
    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if len(actual) > MAX_SCATTER_POINTS:
        rng = np.random.default_rng(random_seed)
        idx = np.sort(rng.choice(len(actual), MAX_SCATTER_POINTS, replace=False))
        actual, predicted = actual[idx], predicted[idx]
        notes["chart"] = f"Scatter plot shows a random sample of {MAX_SCATTER_POINTS} test rows."
    chart_data = {
        "actual_vs_predicted": [
            {"actual": _clean(a), "predicted": _clean(p)} for a, p in zip(actual, predicted)
        ]
    }
    return metrics, notes, chart_data


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def pick_best_model(results: list[dict[str, Any]], task_type: str) -> str | None:
    """
    Return the name of the best successful model by the primary metric, or None.

    Models whose primary metric is missing are ignored rather than ranked —
    we never compare against an invalid number.
    """
    metric, higher_is_better = PRIMARY_METRICS[task_type]
    candidates = []
    for result in results:
        if result.get("status") != "success" or not result.get("metrics"):
            continue
        value = result.get("selection_metric_value")
        if value is None:  # Backward compatibility for older saved result shapes/tests.
            value = result["metrics"].get(metric)
        if value is not None and math.isfinite(float(value)):
            candidates.append((result, float(value)))
    if not candidates:
        return None
    chooser = max if higher_is_better else min
    winner, _ = chooser(candidates, key=lambda item: item[1])
    return winner["model_name"]
