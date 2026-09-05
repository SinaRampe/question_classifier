#!/usr/bin/env python3
"""Reproduce the notebook's UMAP visualization of the hidden states.

Projects the 768-dim [CLS] hidden states of the training split down to 2D
(UMAP, cosine metric, after MinMaxScaling) and plots the per-class point
density as hexbin panels — the notebook's chapter 2 figure.

Writes docs/umap_hidden_states.png. Contains no question texts.

Run:  python visualize.py    # needs data/*.csv (run build_data.py first)
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

MODEL_CKPT = "distilbert-base-uncased"


def extract(texts, tokenizer, model, batch_size=32, max_length=64):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            hs = model(**enc).last_hidden_state[:, 0]
        out.append(hs.numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="docs/umap_hidden_states.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(f"{args.data}/classifier_train.csv")))
    texts = [r["prompt"] for r in rows]
    labels = np.array([int(r["completion"]) for r in rows])
    print(f"train split: {len(texts)} questions")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    model = AutoModel.from_pretrained(MODEL_CKPT)
    model.eval()
    print("encoding ...")
    X = extract(texts, tokenizer, model)

    print("scaling + UMAP (cosine) ...")
    X_scaled = MinMaxScaler().fit_transform(X)
    mapper = UMAP(n_components=2, metric="cosine", random_state=42).fit(X_scaled)
    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = labels

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    cmaps = ["Blues", "Reds"]
    titles = ["not health-related questions", "health-related questions"]
    for i, (title, cmap) in enumerate(zip(titles, cmaps)):
        sub = df_emb[df_emb["label"] == i]
        axes[i].hexbin(sub["X"], sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
        axes[i].set_title(title)
        axes[i].set_xticks([]), axes[i].set_yticks([])
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
