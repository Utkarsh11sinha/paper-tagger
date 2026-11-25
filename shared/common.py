"""Helpers shared by the training scripts, the tagger and the app."""
import json
import re

import numpy as np
import pandas as pd

from shared.config import LABELS_PATH, PROCESSED_DIR

_MATH_RE = re.compile(r"\$[^$]*\$")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light cleaning: drop inline LaTeX math, collapse whitespace."""
    text = _MATH_RE.sub(" ", str(text or ""))
    return _WS_RE.sub(" ", text).strip()


def combine(title: str, abstract: str) -> str:
    """Single text field used by the TF-IDF model."""
    return f"{clean_text(title)}. {clean_text(abstract)}"


def get_device():
    """NVIDIA GPU -> Apple Silicon GPU -> CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device, enabled: bool):
    """fp16 autocast on CUDA only; a no-op everywhere else."""
    import contextlib
    import torch
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run data_prep/prepare_data.py first.")
    return pd.read_json(path, lines=True)


def load_labels() -> list:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def binarize(tag_lists, labels) -> np.ndarray:
    index = {t: i for i, t in enumerate(labels)}
    y = np.zeros((len(tag_lists), len(labels)), dtype=np.float32)
    for row, tags in enumerate(tag_lists):
        for t in tags:
            if t in index:
                y[row, index[t]] = 1.0
    return y


def select_tags(probs, labels, threshold, top_k):
    """Tags above the threshold, best first, capped at top_k; never empty."""
    from shared.config import tag_name
    order = np.argsort(-probs)
    chosen = [i for i in order[:top_k] if probs[i] >= threshold] or [order[0]]
    return [{"tag": labels[i], "name": tag_name(labels[i]), "score": float(probs[i])} for i in chosen]
