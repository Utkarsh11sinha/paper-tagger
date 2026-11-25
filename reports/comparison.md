# Test-set comparison

| Metric | TF-IDF + OneVsRest LogisticRegression | sentence-transformers/all-MiniLM-L6-v2 (fine-tuned) |
|---|---:|---:|
| Micro F1 | 0.6658 | 0.6274 |
| Macro F1 | 0.6505 | 0.5277 |
| Samples F1 | 0.6794 | 0.6489 |
| Micro precision | 0.6418 | 0.6408 |
| Micro recall | 0.6916 | 0.6146 |
| Precision@1 | 0.8203 | 0.7530 |
| Precision@3 | 0.4774 | 0.4379 |
| Recall@3 | 0.8487 | 0.7898 |
| Exact match | 0.3041 | 0.3026 |
| Hamming loss (lower is better) | 0.0312 | 0.0328 |
| Tuned threshold | 0.2500 | 0.3000 |
