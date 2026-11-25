"""Share trained models through the Hugging Face Hub (GitHub can't hold files > 100 MB).

Upload (once, after training; needs a free huggingface.co account):
    hf auth login                      # paste a token with "write" access
    python hub.py upload --repo your-username/paper-tagger

Download (anyone, no account needed for a public repo):
    python hub.py download --repo your-username/paper-tagger

The BERT weights are uploaded in fp16, which halves the download (~220 MB for
SciBERT) with no meaningful change in predictions.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from config import BERT_DIR, ML_MODEL_PATH, MODELS_DIR, REPORTS_DIR

MODEL_CARD = """---
library_name: transformers
pipeline_tag: text-classification
tags: [multi-label-classification, arxiv, scibert, tf-idf]
---
# Research Paper Tagger

Suggests arXiv subject tags from a paper's title and abstract.

* `bert/` - fine-tuned {base_model} (multi-label, sigmoid per tag), fp16 weights
* `ml/tfidf_logreg.joblib` - TF-IDF + one-vs-rest Logistic Regression (scikit-learn)

Tags ({n_tags}): {tags}

## Test-set results
{results}

Code and usage: see the project's GitHub repository.
"""


def export_bundle(dst: Path) -> Path:
    """Write a shareable copy of the trained models into dst (ml/, bert/, README.md)."""
    dst.mkdir(parents=True, exist_ok=True)
    have_ml, have_bert = ML_MODEL_PATH.exists(), (BERT_DIR / "tagger_config.json").exists()
    if not (have_ml or have_bert):
        raise SystemExit("No trained models found. Run train_ml.py and/or train_bert.py first.")

    cfg = {"base_model": "n/a", "labels": []}
    if have_ml:
        (dst / "ml").mkdir(exist_ok=True)
        shutil.copy2(ML_MODEL_PATH, dst / "ml" / ML_MODEL_PATH.name)
    if have_bert:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).half()
        model.save_pretrained(dst / "bert")
        AutoTokenizer.from_pretrained(BERT_DIR).save_pretrained(dst / "bert")
        shutil.copy2(BERT_DIR / "tagger_config.json", dst / "bert" / "tagger_config.json")
        cfg = json.loads((BERT_DIR / "tagger_config.json").read_text(encoding="utf-8"))

    comparison = REPORTS_DIR / "comparison.md"
    results = comparison.read_text(encoding="utf-8").split("\n", 2)[-1] if comparison.exists() else "_not available_"
    labels = cfg["labels"] or __import__("joblib").load(ML_MODEL_PATH)["labels"]
    (dst / "README.md").write_text(MODEL_CARD.format(
        base_model=cfg["base_model"], n_tags=len(labels), tags=", ".join(labels), results=results,
    ), encoding="utf-8")
    return dst


def upload(repo: str, private: bool = False):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        export_bundle(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=repo, repo_type="model",
                          commit_message="Upload paper tagger models")
    print(f"Uploaded to https://huggingface.co/{repo}")
    print(f'Set HF_REPO_ID = "{repo}" in config.py so the app downloads it automatically.')


def download(repo: str):
    from huggingface_hub import snapshot_download
    print(f"Downloading models from huggingface.co/{repo} ...")
    snapshot_download(repo_id=repo, repo_type="model", local_dir=MODELS_DIR,
                      allow_patterns=["ml/*", "bert/*"])
    print(f"Models saved to {MODELS_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upload")
    up.add_argument("--repo", required=True, help="your-username/paper-tagger")
    up.add_argument("--private", action="store_true")
    down = sub.add_parser("download")
    down.add_argument("--repo", required=True)
    ex = sub.add_parser("export", help="write the shareable bundle to a local folder")
    ex.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "upload":
        upload(args.repo, args.private)
    elif args.cmd == "download":
        download(args.repo)
    else:
        print(f"Exported to {export_bundle(Path(args.out))}")


if __name__ == "__main__":
    main()
