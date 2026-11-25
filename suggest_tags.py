"""Command-line tag suggestions.

Usage:
    python suggest_tags.py --title "Attention is all you need" --abstract "The dominant sequence ..."
    python suggest_tags.py --abstract-file abstract.txt --model bert --top-k 3
"""
import argparse
from pathlib import Path

from tagger import load_available


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title", default="")
    ap.add_argument("--abstract", default="")
    ap.add_argument("--abstract-file")
    ap.add_argument("--model", choices=["ml", "bert", "both"], default="both")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, help="override the tuned threshold")
    args = ap.parse_args()

    abstract = Path(args.abstract_file).read_text(encoding="utf-8") if args.abstract_file else args.abstract
    if not (args.title or abstract).strip():
        ap.error("give at least --title or --abstract")

    taggers = load_available()
    wanted = ["ml", "bert"] if args.model == "both" else [args.model]
    missing = [m for m in wanted if m not in taggers]
    if missing:
        print(f"Not trained yet: {', '.join(missing)} (run train_ml.py / train_bert.py)")
    for key in wanted:
        if key not in taggers:
            continue
        tagger = taggers[key]
        print(f"\n{tagger.name}:")
        for t in tagger.suggest(args.title, abstract, args.top_k, args.threshold):
            print(f"  {t['score']:.2f}  {t['tag']:<16} {t['name']}")


if __name__ == "__main__":
    main()
