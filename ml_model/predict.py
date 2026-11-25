"""Model 1 inference: load the TF-IDF + Logistic Regression model and suggest tags."""
from pathlib import Path

import numpy as np

from shared.common import combine, select_tags
from shared.config import ML_MODEL_PATH


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


def is_trained() -> bool:
    return ML_MODEL_PATH.exists()
