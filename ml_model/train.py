"""Model 1 (classical ML): TF-IDF + one-vs-rest Logistic Regression.

Each tag gets its own binary logistic regression on TF-IDF word uni/bi-grams.
The regularisation strength C and the decision threshold are tuned on the
validation split; the final scores are reported on the test split.

Usage:
    python ml_model/train.py
    python ml_model/train.py --c-grid 2 5 10 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python <folder>/<script>.py`

import argparse
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

from shared.common import binarize, combine, load_labels, load_split
from shared.config import ML_MODEL_PATH, ML_REPORTS_DIR, tag_name
from shared.metrics import evaluate, per_tag_table, predict_from_probs, print_metrics, save_report, tune_threshold


def texts(df):
    return [combine(t, a) for t, a in zip(df["title"], df["abstract"])]


def top_terms(vectorizer, clf, labels, n=12):
    vocab = np.array(vectorizer.get_feature_names_out())
    lines = []
    for tag, est in zip(labels, clf.estimators_):
        if not hasattr(est, "coef_"):
            continue
        best = np.argsort(est.coef_[0])[::-1][:n]
        lines.append(f"{tag} ({tag_name(tag)}): " + ", ".join(vocab[best]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c-grid", type=float, nargs="+", default=[1.0, 4.0, 10.0])
    ap.add_argument("--max-features", type=int, default=50_000)
    args = ap.parse_args()

    labels = load_labels()
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    y_train, y_val, y_test = (binarize(d["tags"], labels) for d in (train, val, test))
    print(f"{len(train)} train / {len(val)} val / {len(test)} test papers, {len(labels)} tags")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=args.max_features,
        sublinear_tf=True, stop_words="english", strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform(texts(train))
    x_val, x_test = vectorizer.transform(texts(val)), vectorizer.transform(texts(test))
    # Parallel workers get read-only copies of large matrices; sort indices up front so
    # scikit-learn never needs to modify them in place.
    for x in (x_train, x_val, x_test):
        x.sort_indices()
    print(f"TF-IDF vocabulary: {len(vectorizer.vocabulary_)} features")

    best = None
    for c in args.c_grid:
        t0 = time.time()
        clf = OneVsRestClassifier(LogisticRegression(C=c, solver="liblinear", max_iter=1000), n_jobs=-1)
        clf.fit(x_train, y_train)
        p_val = clf.predict_proba(x_val)
        thr = tune_threshold(y_val, p_val)
        f1 = f1_score(y_val, predict_from_probs(p_val, thr), average="micro", zero_division=0)
        print(f"  C={c:<6} val micro-F1={f1:.4f}  threshold={thr}  ({time.time() - t0:.1f}s)")
        if best is None or f1 > best[0]:
            best = (f1, c, thr, clf)

    _, best_c, threshold, clf = best
    print(f"Best C={best_c}, threshold={threshold}")

    p_test = clf.predict_proba(x_test)
    metrics = evaluate(y_test, p_test, threshold)
    print_metrics("TF-IDF + Logistic Regression (test)", metrics)
    save_report(ML_REPORTS_DIR, metrics, per_tag_table(y_test, p_test, threshold, labels),
                {"model": "TF-IDF + OneVsRest LogisticRegression", "C": best_c})

    (ML_REPORTS_DIR / "top_terms.txt").write_text(top_terms(vectorizer, clf, labels), encoding="utf-8")

    # Keep the saved file small: stop_words_ lists every pruned n-gram and is not
    # needed for prediction; float32 weights are plenty.
    vectorizer.stop_words_ = None
    for est in clf.estimators_:
        if hasattr(est, "coef_"):
            est.coef_ = est.coef_.astype(np.float32)
            est.intercept_ = est.intercept_.astype(np.float32)
    ML_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "clf": clf, "labels": labels, "threshold": threshold},
                ML_MODEL_PATH, compress=9)
    print(f"\nSaved model to {ML_MODEL_PATH} ({ML_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
