"""Print and save a side-by-side comparison of the two models' test metrics."""
import json

from config import REPORTS_DIR

ROWS = [
    ("micro_f1", "Micro F1"), ("macro_f1", "Macro F1"), ("samples_f1", "Samples F1"),
    ("micro_precision", "Micro precision"), ("micro_recall", "Micro recall"),
    ("precision@1", "Precision@1"), ("precision@3", "Precision@3"), ("recall@3", "Recall@3"),
    ("exact_match", "Exact match"), ("hamming_loss", "Hamming loss (lower is better)"),
    ("threshold", "Tuned threshold"),
]


def load(prefix):
    path = REPORTS_DIR / f"{prefix}_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def comparison_markdown():
    ml, bert = load("ml"), load("bert")
    if not (ml or bert):
        return None
    cols = [r for r in (ml, bert) if r]
    lines = ["| Metric | " + " | ".join(r["model"] for r in cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for key, label in ROWS:
        lines.append(f"| {label} | " + " | ".join(f"{r['metrics'][key]:.4f}" for r in cols) + " |")
    return "\n".join(lines)


def main():
    md = comparison_markdown()
    if md is None:
        raise SystemExit("No reports found. Train the models first.")
    print(md)
    (REPORTS_DIR / "comparison.md").write_text("# Test-set comparison\n\n" + md + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORTS_DIR / 'comparison.md'}")


if __name__ == "__main__":
    main()
