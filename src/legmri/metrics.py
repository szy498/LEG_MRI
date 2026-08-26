"""Participant-level resampling and paired predictive comparisons."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve


def binary_metrics(y, probability, threshold=0.5):
    y, probability = np.asarray(y), np.asarray(probability)
    tn, fp, fn, tp = confusion_matrix(y, probability >= threshold, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y, probability),
        "Accuracy": (tp + tn) / len(y),
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
        "Brier": brier_score_loss(y, probability),
    }


def participant_predictions(predictions):
    """Average repeated held-out predictions; never count one subject twice for CIs."""
    keys = ["strategy", "model", "Patient_ID"]
    if predictions.groupby(keys)["y"].nunique().max() != 1:
        raise ValueError("Inconsistent participant labels across repetitions")
    return (
        predictions.groupby(keys, sort=False)
        .agg(
            y=("y", "first"),
            probability=("probability", "mean"),
            n_predictions=("probability", "size"),
        )
        .reset_index()
    )


def bootstrap_metrics(y, probability, n_bootstrap=2000, seed=42, threshold=0.5):
    y, probability = np.asarray(y), np.asarray(probability)
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == value) for value in (0, 1)]
    if any(len(c) < 2 for c in classes):
        raise ValueError("At least two participants per class are needed")
    draws = []
    for _ in range(n_bootstrap):
        ix = np.concatenate([rng.choice(c, len(c), replace=True) for c in classes])
        draws.append(binary_metrics(y[ix], probability[ix], threshold))
    draws = pd.DataFrame(draws)
    return {
        k: {"estimate": v, "low": draws[k].quantile(0.025), "high": draws[k].quantile(0.975)}
        for k, v in binary_metrics(y, probability, threshold).items()
    }


def decision_curve(y, probability, thresholds=None):
    thresholds = np.arange(0.01, 0.96, 0.01) if thresholds is None else np.asarray(thresholds)
    if np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("DCA thresholds must lie strictly between 0 and 1")
    y, probability = np.asarray(y), np.asarray(probability)
    rows = []
    for t in thresholds:
        positive = probability >= t
        tp, fp = np.sum(positive & (y == 1)), np.sum(positive & (y == 0))
        rows.append(
            {
                "threshold": t,
                "model": tp / len(y) - fp / len(y) * t / (1 - t),
                "all": y.mean() - (1 - y.mean()) * t / (1 - t),
                "none": 0,
            }
        )
    return pd.DataFrame(rows)


def roc_band(y, probability, n_bootstrap=1000, seed=42):
    y, probability = np.asarray(y), np.asarray(probability)
    grid, rng, curves = np.linspace(0, 1, 101), np.random.default_rng(seed), []
    classes = [np.flatnonzero(y == c) for c in (0, 1)]
    for _ in range(n_bootstrap):
        ix = np.concatenate([rng.choice(c, len(c)) for c in classes])
        fpr, tpr, _ = roc_curve(y[ix], probability[ix])
        curves.append(np.interp(grid, fpr, tpr))
    return grid, np.percentile(curves, [2.5, 97.5], axis=0)
