"""Load the trained models and suggest tags for a user's title + abstract."""
import json
from pathlib import Path

import numpy as np

from common import clean_text, combine, get_device
from config import BERT_DIR, HF_REPO_ID, ML_MODEL_PATH, tag_name


def select_tags(probs, labels, threshold, top_k):
    """Tags above the threshold, best first, capped at top_k; never empty."""
    order = np.argsort(-probs)
    chosen = [i for i in order[:top_k] if probs[i] >= threshold] or [order[0]]
    return [{"tag": labels[i], "name": tag_name(labels[i]), "score": float(probs[i])} for i in chosen]


class MLTagger:
    name = "TF-IDF + Logistic Regression"

    def __init__(self, path: Path = ML_MODEL_PATH):
        import joblib
        bundle = joblib.load(path)
        self.vectorizer, self.clf = bundle["vectorizer"], bundle["clf"]
        self.labels, self.threshold = bundle["labels"], bundle["threshold"]

    def predict_proba(self, title: str, abstract: str) -> np.ndarray:
        return self.clf.predict_proba(self.vectorizer.transform([combine(title, abstract)]))[0]

    def suggest(self, title, abstract, top_k=5, threshold=None):
        thr = self.threshold if threshold is None else threshold
        return select_tags(self.predict_proba(title, abstract), self.labels, thr, top_k)


class BertTagger:
    name = "Fine-tuned SciBERT"

    def __init__(self, path: Path = BERT_DIR):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        cfg = json.loads((Path(path) / "tagger_config.json").read_text(encoding="utf-8"))
        self.labels, self.threshold, self.max_len = cfg["labels"], cfg["threshold"], cfg["max_len"]
        self.name = f"Fine-tuned {cfg['base_model'].split('/')[-1]}"
        self.torch = torch
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # Shared checkpoints are stored in fp16 to halve the download; run in fp32 for portability.
        self.model = AutoModelForSequenceClassification.from_pretrained(path).float().to(self.device).eval()

    def predict_proba(self, title: str, abstract: str) -> np.ndarray:
        enc = self.tokenizer(clean_text(title), clean_text(abstract), truncation="longest_first",
                             max_length=self.max_len, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits
        return self.torch.sigmoid(logits.float())[0].cpu().numpy()

    def suggest(self, title, abstract, top_k=5, threshold=None):
        thr = self.threshold if threshold is None else threshold
        return select_tags(self.predict_proba(title, abstract), self.labels, thr, top_k)


def models_present() -> bool:
    return ML_MODEL_PATH.exists() or (BERT_DIR / "tagger_config.json").exists()


def load_available(auto_download: bool = True):
    """Return {"ml": tagger, "bert": tagger} for whichever models exist locally.

    If no models exist and HF_REPO_ID is configured, download them first.
    """
    if auto_download and not models_present() and HF_REPO_ID:
        from hub import download
        download(HF_REPO_ID)
    taggers = {}
    if ML_MODEL_PATH.exists():
        taggers["ml"] = MLTagger()
    if (BERT_DIR / "tagger_config.json").exists():
        taggers["bert"] = BertTagger()
    return taggers
