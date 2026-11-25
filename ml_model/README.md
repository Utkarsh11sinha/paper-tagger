# Model 1: Machine Learning (TF-IDF + Logistic Regression)

A classical machine-learning model that suggests arXiv tags from a paper's title and abstract.

## How it works

1. **TF-IDF features:** the title and abstract become a vector of 50,000 weighted
   words and word pairs (uni- and bi-grams, English stop words removed, sublinear TF).
2. **One-vs-rest Logistic Regression:** one binary classifier per tag (40 tags), each
   giving the probability that the tag applies.
3. **Threshold:** tags with probability ≥ 0.25 (tuned on the validation set) are
   suggested; the top tag is always returned.

Regularisation strength `C` was chosen from {1, 4, 10} on the validation set (best: 4).

## Files

| File | What it is |
|---|---|
| `train.py` | Trains the model, tunes `C` and the threshold, evaluates on the test set |
| `predict.py` | `MLTagger`: loads the saved model and suggests tags |
| `model/tfidf_logreg.joblib` | Trained model (vectorizer + 40 classifiers + threshold), 8 MB |
| `reports/metrics.json` | Test-set metrics |
| `reports/per_tag.csv` | Precision / recall / F1 for each tag |
| `reports/top_terms.txt` | The most indicative words for each tag (explains the model) |

## Test results (5,231 papers)

| Metric | Score |
|---|---:|
| Micro F1 | 0.666 |
| Macro F1 | 0.650 |
| Precision@1 (top tag correct) | 0.820 |
| Recall@3 (true tags in top 3) | 0.849 |
| Hamming loss | 0.031 |

## Usage

From the project root:

```bash
python ml_model/train.py          # retrain (a few minutes on CPU)
```

```python
from ml_model.predict import MLTagger
tagger = MLTagger()
tagger.suggest("Deep reinforcement learning for robot grasping",
               "We train a robotic arm to grasp unseen objects ...")
# [{'tag': 'cs.RO', 'name': 'Robotics', 'score': 1.0}, {'tag': 'cs.LG', ...}]
```
