# Research Paper Tagger

Suggests arXiv subject tags (e.g. `cs.CV` Computer Vision, `stat.ML`, `quant-ph`)
for a paper from its **title + abstract**. It is a multi-label NLP problem: one paper
can get several tags, each with a confidence score.

Two models are trained on the same data and compared:

| | Model 1: Machine Learning | Model 2: Deep Learning |
|---|---|---|
| Text representation | TF-IDF word uni+bi-grams | SciBERT contextual embeddings (`[CLS] title [SEP] abstract [SEP]`) |
| Classifier | One-vs-rest Logistic Regression (one per tag) | Linear layer + sigmoid per tag, BCE loss |
| Tuning | `C` grid search + decision threshold on validation | Best epoch + decision threshold on validation |
| Size | ~50 MB | ~220 MB (fp16 download) |
| Script | `train_ml.py` | `train_bert.py` |

Runs on **Windows, macOS (Intel and Apple Silicon) and Linux**, with or without a GPU.
Training BERT needs a GPU to be practical; *using* the trained models does not.

---

## Quick start: use the trained models (any computer, no GPU)

Requires Python 3.10 or newer.

**Windows (PowerShell)**
```powershell
git clone https://github.com/<your-username>/paper-tagger.git
cd paper-tagger
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PAPER_TAGGER_HF_REPO = "<hf-username>/paper-tagger"
streamlit run app.py
```

**macOS / Linux**
```bash
git clone https://github.com/<your-username>/paper-tagger.git
cd paper-tagger
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PAPER_TAGGER_HF_REPO="<hf-username>/paper-tagger"
streamlit run app.py
```

The first run downloads the models (~270 MB) from the Hugging Face Hub, then opens the
app at http://localhost:8501. (If `HF_REPO_ID` is already set in `config.py`, you can
skip the environment-variable line.)

**Docker (any OS)**
```bash
docker build -t paper-tagger .
docker run -p 8501:8501 -e PAPER_TAGGER_HF_REPO=<hf-username>/paper-tagger paper-tagger
```

**Command line instead of the app**
```bash
python hub.py download --repo <hf-username>/paper-tagger
python suggest_tags.py --title "Vision transformers for medical image segmentation" --abstract "We replace the CNN encoder of a U-Net with a ViT ..."
```

---

## Train the models yourself

### 1. GPU setup (optional, for BERT training)

| Your machine | What to do |
|---|---|
| NVIDIA GPU, Windows | `pip install torch --index-url https://download.pytorch.org/whl/cu128` **before** `pip install -r requirements.txt` |
| NVIDIA GPU, Linux | nothing extra: the default `torch` wheel includes CUDA |
| Apple Silicon Mac | nothing extra: the Apple GPU (MPS) is used automatically |
| No GPU | use Google Colab (free T4 GPU), or train the lighter model: `--model distilbert-base-uncased --limit 5000` |

Check: `python -c "import torch; print(torch.cuda.is_available())"`.
If `cu128` doesn't match your setup, pick the CUDA build at pytorch.org → Get Started.

### 2. Get the data

```bash
python fetch_arxiv.py
python prepare_data.py
```

`fetch_arxiv.py` downloads ~2000 recent papers for each of 30 categories (~45–55k papers
after de-duplication) from the free arXiv API. It takes about 5–10 minutes because of
arXiv's rate limit and can be stopped and resumed. Use `--per-category 500` for a
quicker run.

`prepare_data.py` keeps the 40 most frequent tags (each with ≥150 papers), splits the
data 80/10/10 and prints dataset statistics.

*Alternative data:* download the Kaggle dataset "arXiv Paper Abstracts" (`arxiv_data.csv`)
and run `python prepare_data.py --kaggle-csv arxiv_data.csv`.

### 3. Train

```bash
python train_ml.py      # a few minutes on any CPU
python train_bert.py    # GPU recommended
python compare.py       # side-by-side test metrics -> reports/comparison.md
```

`train_bert.py` defaults fit a 4 GB GPU (e.g. RTX 3050 Laptop): batch 8 × 4 gradient
accumulation, 256 tokens, fp16, 3 epochs. Useful options:

- CUDA out of memory: `--grad-checkpointing`, `--max-len 192`, or `--batch-size 4 --grad-accum 8`
- Faster and lighter: `--model distilbert-base-uncased --batch-size 16 --grad-accum 2`
- Quick check that everything works: `--limit 500 --epochs 1`

### 4. Share your trained models

GitHub rejects files over 100 MB, so models go to the Hugging Face Hub (free) and the
code goes to GitHub. `data/` and `models/` are already excluded by `.gitignore`.

```bash
hf auth login                                     # token with "write" access from huggingface.co/settings/tokens
python hub.py upload --repo <hf-username>/paper-tagger
```

Then set `HF_REPO_ID = "<hf-username>/paper-tagger"` in `config.py` and commit it.
Anyone who clones the repo can run `streamlit run app.py` and the models download
automatically. `reports/` (metrics, per-tag scores) is small and is meant to be committed.

> Note: `ml/tfidf_logreg.joblib` is a pickle file. Only load model files from repos you trust.

---

## Project layout

```
config.py         paths, Hugging Face repo id, default categories, tag names
fetch_arxiv.py    download papers from the public arXiv API
prepare_data.py   clean text, choose tag set, train/val/test split, EDA printout
common.py         text cleaning, data loading, device selection (CUDA / MPS / CPU)
metrics.py        multi-label metrics and threshold tuning
train_ml.py       Model 1: TF-IDF + Logistic Regression
train_bert.py     Model 2: SciBERT fine-tuning (PyTorch)
compare.py        side-by-side test metrics
tagger.py         loads models (auto-downloads if needed), suggest(title, abstract)
hub.py            upload / download models via the Hugging Face Hub
suggest_tags.py   command-line tag suggestions
app.py            Streamlit demo
Dockerfile        CPU image for the demo
```

## Metrics reported

- **Micro / macro / samples F1**: micro is dominated by frequent tags; macro weights all tags equally (shows rare-tag performance)
- **Precision@1, Precision@3, Recall@3**: quality of the ranked tag suggestions
- **Hamming loss**: fraction of wrong tag decisions (lower is better)
- **Exact match**: the predicted tag set is exactly right
- Per-tag precision / recall / F1 in `reports/*_per_tag.csv`

Every paper always receives at least one tag (the highest-scoring one), even if no
score passes the threshold.

## Data

Paper metadata comes from the [arXiv API](https://info.arxiv.org/help/api/index.html).
Thank you to arXiv for use of its open access interoperability.
