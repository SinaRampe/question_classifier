#!/usr/bin/env python3
"""Build the train/test CSVs from upstream sources + labels.csv.

The question texts are NOT stored in this repository (licensing, see README).
This script downloads the upstream sources, resolves every label in
labels.csv to its question text, and writes:

    data/classifier_train.csv   (prompt, completion)
    data/classifier_test.csv    (prompt, completion)

Sources:
  - SQuAD v1.1 (train + dev), CC BY-SA 4.0
  - AskDocs (train_en + external_en), GPL-3.0

Usage:  python build_data.py [--out data]
"""

import argparse
import csv
import json
import os
import sys
import urllib.request

SQUAD_BASE = "https://rajpurkar.github.io/SQuAD-explorer/dataset"
ASKD_BASE = "https://github.com/ju-resplande/askD/releases/download/v0.0.0"

SQUAD_FILES = ["train-v1.1.json", "dev-v1.1.json"]
ASKD_FILES = {
    "askd:train_en": "train_en.json",
    "askd:external_en": "external_en.json",
}


def fetch(url, dest):
    if os.path.exists(dest):
        print(f"  cached: {dest}")
        return
    print(f"  downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "question-classifier-build"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def load_questions(cache_dir):
    """Return {source_key: {source_id: question_text}}."""
    questions = {}

    # SQuAD: stable per-question ids (e.g. 5733be284776f41900661182),
    # globally unique across train + dev -> single "squad" key
    squad_qs = {}
    for fname in SQUAD_FILES:
        path = os.path.join(cache_dir, fname)
        fetch(f"{SQUAD_BASE}/{fname}", path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for article in data["data"]:
            for para in article["paragraphs"]:
                for qa in para["qas"]:
                    squad_qs[qa["id"]] = qa["question"]
    questions["squad"] = squad_qs
    print(f"  squad: {len(squad_qs)} questions")

    # AskDocs: train_en carries stable reddit q_ids (e.g. 3yzn3p);
    # external_en only has positional ids (0..n) -> id = position in the file
    for key, fname in ASKD_FILES.items():
        path = os.path.join(cache_dir, fname)
        fetch(f"{ASKD_BASE}/{fname}", path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        qs = {}
        for i, row in enumerate(data):
            qid = row.get("q_id") or str(i)
            if row.get("title"):
                qs[str(qid)] = row["title"]
        questions[key] = qs
        print(f"  {key}: {len(qs)} questions")

    return questions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--cache", default=".source_cache")
    ap.add_argument("--out", default="data")
    ap.add_argument("--include-unused", action="store_true",
                    help="also write rows labeled split=unused (default: skip; "
                         "'full' is always train+test only)")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    print("Loading upstream sources:")
    questions = load_questions(args.cache)

    with open(args.labels, newline="", encoding="utf-8") as f:
        label_rows = list(csv.DictReader(f))
    print(f"\nlabels.csv: {len(label_rows)} rows")

    splits = {"train": [], "test": []}
    missing = 0
    skipped = 0
    for row in label_rows:
        src, sid, label, split = row["source"], row["source_id"], int(row["label"]), row["split"]
        if split == "unused" and not args.include_unused:
            skipped += 1
            continue
        text = questions.get(src, {}).get(sid)
        if text is None:
            missing += 1
            print(f"  WARN unresolved: {src}/{sid}", file=sys.stderr)
            continue
        if split in splits:
            splits[split].append((text, label))

    if missing:
        sys.exit(f"ERROR: {missing} labels could not be resolved — aborting")

    counts = {}
    splits["full"] = splits["train"] + splits["test"]
    for split, rows_ in splits.items():
        fname = "df_full.csv" if split == "full" else f"classifier_{split}.csv"
        out = os.path.join(args.out, fname)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["prompt", "completion"])
            w.writerows(rows_)
        pos = sum(l for _, l in rows_)
        counts[split] = (len(rows_), len(rows_) - pos, pos)
        print(f"  wrote {out}: {len(rows_)} rows (health={pos}, other={len(rows_)-pos})")

    print(f"\nSkipped split=unused: {skipped}")
    print("Done.")
    print("NOTE: the downloaded source files remain in "
          f"{args.cache}/ — they are NOT licensed under this repo's license. "
          "See README for attribution and license terms.")


if __name__ == "__main__":
    main()
