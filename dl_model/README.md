# Model 2: Deep Learning (fine-tuned BERT / MiniLM)

A transformer model, fine-tuned end to end, that suggests arXiv tags from a paper's
title and abstract.

## How it works

1. **Base model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
   a distilled BERT with 6 layers, 384 hidden size and 22M parameters.
2. **Input:** `[CLS] title [SEP] abstract [SEP]`, up to 256 tokens.
3. **Head:** a linear layer on the `[CLS]` representation with one sigmoid output per tag (40 tags).
4. **Training:** all weights fine-tuned with binary cross-entropy, AdamW (lr 5e-5, weight decay 0.01),
   linear warm-up/decay, batch 32, fp16 mixed precision, 4 epochs on an RTX 3050 Laptop GPU (~5 min/epoch).
   The epoch with the best validation micro-F1 is kept.
5. **Threshold:** tags with probability ≥ 0.30 (tuned on the validation set) are suggested;
   the top tag is always returned.

Weights are saved in fp16 (46 MB) and loaded back as fp32, so it runs on any CPU or GPU.

## Files

| File | What it is |
|---|---|
| `train.py` | Fine-tunes the model, picks the best epoch and threshold, evaluates on the test set |
| `predict.py` | `BertTagger`: loads the saved model and suggests tags |
| `model/model.safetensors` | Fine-tuned weights (fp16), 45 MB |
| `model/config.json` | Model architecture and tag names |
| `model/tokenizer.json`, `model/tokenizer_config.json` | Tokenizer |
| `model/tagger_config.json` | Tag list, tuned threshold, max length, base model name |
| `reports/metrics.json` | Test-set metrics |
| `reports/per_tag.csv` | Precision / recall / F1 for each tag |

## Test results (5,231 papers)

| Metric | Score |
|---|---:|
| Micro F1 | 0.627 |
| Macro F1 | 0.528 |
| Precision@1 (top tag correct) | 0.753 |
| Recall@3 (true tags in top 3) | 0.790 |
| Hamming loss | 0.033 |

## Usage

From the project root:

```bash
python dl_model/train.py          # retrain (GPU recommended, ~20 min on an RTX 3050)
python dl_model/train.py --limit 500 --epochs 1   # quick check
```

```python
from dl_model.predict import BertTagger
tagger = BertTagger()
tagger.suggest("Deep reinforcement learning for robot grasping",
               "We train a robotic arm to grasp unseen objects ...")
# [{'tag': 'cs.RO', 'name': 'Robotics', 'score': 0.66}]
```
