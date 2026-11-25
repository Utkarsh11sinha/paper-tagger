"""Clean the raw papers, choose the tag set and make train/val/test splits.

Usage:
    python data_prep/prepare_data.py                         # from fetch_arxiv.py output
    python data_prep/prepare_data.py --kaggle-csv arxiv_data.csv
        # alternative: Kaggle "arXiv Paper Abstracts" (columns: titles, summaries, terms)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python <folder>/<script>.py`

import argparse
import ast
import json
from collections import Counter

import numpy as np
import pandas as pd

from shared.common import clean_text
from shared.config import LABELS_PATH, PROCESSED_DIR, RAW_PATH, SEED, tag_name


def load_raw(args) -> pd.DataFrame:
    if args.kaggle_csv:
        df = pd.read_csv(args.kaggle_csv)
        df = pd.DataFrame({
            "title": df["titles"],
            "abstract": df["summaries"],
            "tags": df["terms"].apply(ast.literal_eval),
        })
    else:
        if not RAW_PATH.exists():
            raise SystemExit(f"{RAW_PATH} not found. Run data_prep/fetch_arxiv.py first.")
        df = pd.read_json(RAW_PATH, lines=True)
        df = df.rename(columns={"categories": "tags"})[["title", "abstract", "tags"]]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaggle-csv", help="use the Kaggle arXiv CSV instead of fetched data")
    ap.add_argument("--max-tags", type=int, default=40, help="keep at most this many tags")
    ap.add_argument("--min-tag-count", type=int, default=150, help="drop tags rarer than this")
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    args = ap.parse_args()

    df = load_raw(args)
    n_raw = len(df)
    df["title"] = df["title"].map(clean_text)
    df["abstract"] = df["abstract"].map(clean_text)
    df = df[(df["abstract"].str.len() > 50) & (df["title"].str.len() > 0)]
    df = df.drop_duplicates(subset="title")

    counts = Counter(t for tags in df["tags"] for t in tags)
    labels = [t for t, c in counts.most_common(args.max_tags) if c >= args.min_tag_count]
    if len(labels) < 2:
        raise SystemExit("Fewer than 2 tags left; lower --min-tag-count or fetch more data.")
    keep = set(labels)
    df["tags"] = df["tags"].apply(lambda ts: [t for t in ts if t in keep])
    df = df[df["tags"].str.len() > 0].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * args.test_size)
    n_val = int(len(df) * args.val_size)
    splits = {
        "test": df.iloc[idx[:n_test]],
        "val": df.iloc[idx[n_test:n_test + n_val]],
        "train": df.iloc[idx[n_test + n_val:]],
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_json(PROCESSED_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    # ---- quick EDA ----
    print(f"Raw papers: {n_raw}  ->  kept: {len(df)}")
    print("Split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"Tags kept: {len(labels)}   avg tags/paper: {df['tags'].str.len().mean():.2f}")
    print(f"Abstract length (words): median {int(df['abstract'].str.split().str.len().median())}")
    print("\nTag frequencies:")
    kept_counts = Counter(t for tags in df["tags"] for t in tags)
    for t in labels:
        print(f"  {t:<16} {kept_counts[t]:>6}  {tag_name(t)}")


if __name__ == "__main__":
    main()
