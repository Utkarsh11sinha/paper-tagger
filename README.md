# Research Paper Tagger

Suggests arXiv subject tags (e.g. `cs.CV` Computer Vision, `stat.ML`, `quant-ph`)
for a paper from its **title + abstract**. It is a multi-label NLP problem: one paper
can get several tags, each with a confidence score.

Two models are trained on the same data and compared:

| | Model 1: Machine Learning | Model 2: Deep Learning |
|---|---|---|
| Text representation | TF-IDF word uni+bi-grams (50k features) | Contextual embeddings from MiniLM, a distilled BERT (`[CLS] title [SEP] abstract [SEP]`) |
| Classifier | One-vs-rest Logistic Regression (one per tag) | Fine-tuned transformer + linear layer, sigmoid per tag, BCE loss |
| Tuning | `C` grid search + decision threshold on validation | Best epoch + decision threshold on validation |
| Parameters | ~2M weights | 22M |
| Size on disk | a few MB | ~46 MB (fp16) |
| Script | `train_ml.py` | `train_bert.py` |

**The trained models are included in this repo** (`models/`), so you can use them
right after cloning. Both are small enough for GitHub and run on any computer
(Windows, macOS, Linux) without a GPU.

---

## Quick start (any computer, no GPU needed)

Requires Python 3.10 or newer.

**Windows (PowerShell)**
```powershell
git clone https://github.com/Utkarsh11sinha/paper-tagger.git
cd paper-tagger
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**
```bash
git clone https://github.com/Utkarsh11sinha/paper-tagger.git
cd paper-tagger
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Paste a title and abstract and both models
suggest tags side by side.

**Command line**
```bash
python suggest_tags.py --title "Vision transformers for medical image segmentation" --abstract "We replace the CNN encoder of a U-Net with a ViT ..."
```

**Docker**
```bash
docker build -t paper-tagger .
docker run -p 8501:8501 paper-tagger
```

---

## Retrain the models yourself

### 1. GPU setup (optional, makes BERT training much faster)

| Your machine | What to do |
|---|---|
| NVIDIA GPU, Windows | `pip install torch --index-url https://download.pytorch.org/whl/cu128` **before** `pip install -r requirements.txt` |
| NVIDIA GPU, Linux | nothing extra: the default `torch` wheel includes CUDA |
| Apple Silicon Mac | nothing extra: the Apple GPU (MPS) is used automatically |
| No GPU | works, but BERT training takes hours on CPU; use `--limit 5000` for a smaller run |

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

### 3. Train and compare

```bash
python train_ml.py      # a few minutes on any CPU
python train_bert.py    # GPU recommended
python compare.py       # side-by-side test metrics -> reports/comparison.md
```

`train_bert.py` options:

- CUDA out of memory: `--batch-size 16`, `--max-len 192`, or `--grad-checkpointing`
- Quick check that everything works: `--limit 500 --epochs 1`
- Bigger model: `--model allenai/scibert_scivocab_uncased --batch-size 8 --grad-accum 4 --lr 3e-5`
  (more accurate, but ~220 MB, which is over GitHub's 100 MB file limit)

### 4. Publish the new models

```bash
git add models reports
git commit -m "Retrain models"
git push
```

`data/` is ignored by git; `models/` and `reports/` are committed.

> Note: `models/ml/tfidf_logreg.joblib` is a pickle file. Only load model files from sources you trust.

---

## Project layout

```
config.py         paths, default categories, tag names
fetch_arxiv.py    download papers from the public arXiv API
prepare_data.py   clean text, choose tag set, train/val/test split, EDA printout
common.py         text cleaning, data loading, device selection (CUDA / MPS / CPU)
metrics.py        multi-label metrics and threshold tuning
train_ml.py       Model 1: TF-IDF + Logistic Regression
train_bert.py     Model 2: BERT-family fine-tuning (PyTorch)
compare.py        side-by-side test metrics
tagger.py         loads the models, suggest(title, abstract)
suggest_tags.py   command-line tag suggestions
app.py            Streamlit demo
Dockerfile        CPU image for the demo
models/           trained models (committed)
reports/          test metrics, per-tag scores, top TF-IDF terms per tag
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
