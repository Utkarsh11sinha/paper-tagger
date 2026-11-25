"""Multi-label evaluation helpers shared by both models."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support, precision_score, recall_score

from shared.config import tag_name


def predict_from_probs(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold the probabilities, but always give every paper at least one tag."""
    y_pred = (y_prob >= threshold).astype(int)
    empty = y_pred.sum(axis=1) == 0
    y_pred[empty, y_prob[empty].argmax(axis=1)] = 1
    return y_pred


def precision_at_k(y_true, y_prob, k):
    top = np.argsort(-y_prob, axis=1)[:, :k]
    return float(np.take_along_axis(y_true, top, axis=1).mean())


def recall_at_k(y_true, y_prob, k):
    top = np.argsort(-y_prob, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top, axis=1).sum(axis=1)
    return float(np.mean(hits / np.maximum(y_true.sum(axis=1), 1)))


def tune_threshold(y_true, y_prob) -> float:
    """Pick the single global threshold that maximises micro-F1 on validation data."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.025):
        f1 = f1_score(y_true, predict_from_probs(y_prob, t), average="micro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return round(best_t, 3)


def evaluate(y_true, y_prob, threshold) -> dict:
    y_pred = predict_from_probs(y_prob, threshold)
    return {
        "threshold": threshold,
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "precision@1": precision_at_k(y_true, y_prob, 1),
        "precision@3": precision_at_k(y_true, y_prob, 3),
        "recall@3": recall_at_k(y_true, y_prob, 3),
        "exact_match": float((y_pred == y_true).all(axis=1).mean()),
    }


def per_tag_table(y_true, y_prob, threshold, labels) -> pd.DataFrame:
    y_pred = predict_from_probs(y_prob, threshold)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    return pd.DataFrame({
        "tag": labels, "name": [tag_name(t) for t in labels],
        "precision": p, "recall": r, "f1": f, "support": s,
    }).sort_values("f1", ascending=False)


def save_report(report_dir: Path, metrics: dict, table: pd.DataFrame, extra: dict | None = None):
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": {k: float(v) for k, v in metrics.items()}, **(extra or {})}
    (report_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table.to_csv(report_dir / "per_tag.csv", index=False, float_format="%.4f")


def print_metrics(title: str, metrics: dict):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"  {k:<16} {v:.4f}")
