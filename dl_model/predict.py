"""Model 2 inference: load the fine-tuned BERT (MiniLM) model and suggest tags."""
import json
from pathlib import Path

import numpy as np

from shared.common import clean_text, get_device, select_tags
from shared.config import BERT_DIR


class BertTagger:
    name = "Fine-tuned BERT"

    def __init__(self, path: Path = BERT_DIR):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        cfg = json.loads((Path(path) / "tagger_config.json").read_text(encoding="utf-8"))
        self.labels, self.threshold, self.max_len = cfg["labels"], cfg["threshold"], cfg["max_len"]
        self.name = f"Fine-tuned {cfg['base_model'].split('/')[-1]}"
        self.torch = torch
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # Weights are stored in fp16 to halve the file size; run in fp32 for portability.
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


def is_trained() -> bool:
    return (BERT_DIR / "tagger_config.json").exists()
