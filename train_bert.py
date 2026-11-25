"""Model 2 (deep learning): fine-tune SciBERT for multi-label tag prediction.

SciBERT is BERT pretrained on scientific papers, so it suits arXiv abstracts.
The [CLS] representation feeds a linear layer with one sigmoid output per tag,
trained with binary cross-entropy. Title and abstract are encoded as a
sentence pair: [CLS] title [SEP] abstract [SEP].

Defaults are sized for a 4 GB GPU (RTX 3050 Laptop): batch 8 x 4 accumulation
steps (effective batch 32), max 256 tokens, fp16 mixed precision.

Usage:
    python train_bert.py
    python train_bert.py --model distilbert-base-uncased --batch-size 16 --grad-accum 2
    python train_bert.py --grad-checkpointing      # if you hit CUDA out-of-memory
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import amp_context, binarize, get_device, load_labels, load_split
from config import BERT_DIR, SEED
from metrics import evaluate, per_tag_table, print_metrics, save_report, tune_threshold


class PaperDataset(Dataset):
    def __init__(self, df, tokenizer, labels, max_len):
        self.enc = tokenizer(list(df["title"]), list(df["abstract"]),
                             truncation="longest_first", max_length=max_len)
        self.y = binarize(df["tags"], labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


def make_collate(tokenizer):
    def collate(batch):
        labels = torch.tensor(np.stack([b.pop("labels") for b in batch]))
        out = tokenizer.pad(batch, return_tensors="pt")
        out["labels"] = labels
        return out
    return collate


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    probs, golds = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp_context(device, use_amp):
            logits = model(**batch).logits
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        golds.append(labels.numpy())
    return np.concatenate(golds), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="use only N train papers (quick tests)")
    ap.add_argument("--out", default=str(BERT_DIR))
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_fp16
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)}), fp16={use_amp}")
    elif device.type == "mps":
        print("Device: Apple Silicon GPU (mps)")
    else:
        print("Device: cpu  -- WARNING: no GPU found, training will be very slow. "
              "Consider Google Colab, or --model distilbert-base-uncased --limit 5000")

    labels = load_labels()
    train_df, val_df, test_df = load_split("train"), load_split("val"), load_split("test")
    if args.limit:
        train_df = train_df.head(args.limit)
        val_df = val_df.head(max(args.limit // 4, 16))
        test_df = test_df.head(max(args.limit // 4, 16))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(labels)),
        label2id={t: i for i, t in enumerate(labels)},
    ).to(device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    print("Tokenizing...")
    collate = make_collate(tokenizer)
    train_loader = DataLoader(PaperDataset(train_df, tokenizer, labels, args.max_len),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PaperDataset(val_df, tokenizer, labels, args.max_len),
                            batch_size=args.eval_batch_size, collate_fn=collate)
    test_loader = DataLoader(PaperDataset(test_df, tokenizer, labels, args.max_len),
                             batch_size=args.eval_batch_size, collate_fn=collate)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    out_dir = Path(args.out)
    best_f1, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(bar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            with amp_context(device, use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), y) / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            bar.set_postfix(loss=f"{running / step:.4f}")

        y_val, p_val = predict(model, val_loader, device, use_amp)
        thr = tune_threshold(y_val, p_val)
        val_metrics = evaluate(y_val, p_val, thr)
        print(f"epoch {epoch}: train loss {running / len(train_loader):.4f} | "
              f"val micro-F1 {val_metrics['micro_f1']:.4f} (thr {thr}) | {time.time() - t0:.0f}s")

        if val_metrics["micro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["micro_f1"], epoch
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            (out_dir / "tagger_config.json").write_text(json.dumps({
                "labels": labels, "threshold": thr, "max_len": args.max_len, "base_model": args.model,
            }, indent=2), encoding="utf-8")
            print(f"  saved best model to {out_dir}")

    # Final evaluation on the test split with the best checkpoint.
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).float().to(device)
    threshold = json.loads((out_dir / "tagger_config.json").read_text(encoding="utf-8"))["threshold"]
    y_test, p_test = predict(model, test_loader, device, use_amp)
    metrics = evaluate(y_test, p_test, threshold)
    print_metrics(f"{args.model} fine-tuned (test, best epoch {best_epoch})", metrics)
    save_report("bert", metrics, per_tag_table(y_test, p_test, threshold, labels),
                {"model": f"{args.model} (fine-tuned)", "best_epoch": best_epoch, "epochs": args.epochs})


if __name__ == "__main__":
    main()
