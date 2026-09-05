#!/usr/bin/env python3
"""Evaluate the classifier: frozen DistilBERT [CLS] states + logistic regression.

Reproduces the notebook's method as a plain script so the result can be
verified without Jupyter:

  1. tokenize with distilbert-base-uncased
  2. extract frozen [CLS] hidden states (768-dim)
  3. fit LogisticRegression on the train split, score the test split

Also reports the notebook-style variant (fit on train+test, score test)
for comparison with the original 2022 numbers.

Run:  python evaluate.py            # needs data/*.csv (run build_data.py first)
      python evaluate.py --skip-full
"""

import argparse
import csv
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModel, AutoTokenizer

MODEL_CKPT = "distilbert-base-uncased"


def extract_hidden_states(texts, tokenizer, model, device, batch_size=32, max_length=64):
    """Return np.ndarray (N, 768) of frozen [CLS] hidden states."""
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            hs = model(**enc).last_hidden_state[:, 0]  # [CLS] token
        out.append(hs.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            done = i + len(batch)
            print(f"  encoded {done}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    return np.vstack(out)


def load_split(path):
    rows = list(csv.DictReader(open(path)))
    return [r["prompt"] for r in rows], np.array([int(r["completion"]) for r in rows])


def report(name, y_true, y_pred):
    print(f"\n== {name} ==")
    print("accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("confusion matrix (rows: true other/health):")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["other", "health"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--skip-full", action="store_true",
                    help="skip the notebook-style fit on train+test")
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"Loading {MODEL_CKPT} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    model = AutoModel.from_pretrained(MODEL_CKPT).to(device)
    model.eval()

    X_tr_text, y_tr = load_split(f"{args.data}/classifier_train.csv")
    X_te_text, y_te = load_split(f"{args.data}/classifier_test.csv")
    print(f"train: {len(X_tr_text)}  test: {len(X_te_text)}")

    print("\nEncoding train ...")
    X_tr = extract_hidden_states(X_tr_text, tokenizer, model, device)
    print("Encoding test ...")
    X_te = extract_hidden_states(X_te_text, tokenizer, model, device)

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_tr, y_tr)
    report("Fit on train, score on test (held-out)", y_te, clf.predict(X_te))

    if not args.skip_full:
        X_full = np.vstack([X_tr, X_te])
        y_full = np.concatenate([y_tr, y_te])
        clf_full = LogisticRegression(max_iter=5000)
        clf_full.fit(X_full, y_full)
        acc = accuracy_score(y_te, clf_full.predict(X_te))
        print(f"\nNotebook-style fit on train+test, scored on test: {round(acc, 4)} "
              "(optimistic — test was part of the fit)")


if __name__ == "__main__":
    main()
