# Question Classifier

A lightweight text classifier that decides whether a question is about a
health(care) topic. Built on a method from
[*Natural Language Processing with Transformers*](https://github.com/nlp-with-transformers/notebooks)
(O'Reilly): DistilBERT's frozen hidden states + a logistic-regression head —
no fine-tuning, small and fast, ~98% accuracy on the held-out split.

This project was my portfolio piece in a 2022 data-science bootcamp; I built
and labeled it on my own. In 2026 the repository was reworked for licensing
compliance: the raw question texts were removed and replaced by a
reproducible build pipeline (see below for what changed and why).

## How it works

1. `build_data.py` downloads the upstream datasets, resolves `labels.csv`
   and writes `data/classifier_train.csv`, `data/classifier_test.csv` and
   `data/df_full.csv` (train+test combined; the classifier is fitted on `df_full`).
2. `train_classifier.ipynb` encodes the questions with DistilBERT
   (`distilbert-base-uncased`), extracts the `[CLS]` hidden states,
   projects them with UMAP for visualization, and fits a logistic
   regression on top.

```bash
python build_data.py        # downloads sources, writes data/*.csv
python train_classifier.ipynb  # or open in Colab/Jupyter
```

## Data sources & licenses

The question texts are **not** stored in this repository. Instead,
`labels.csv` stores one row per labeled question — `source`, `source_id`,
`label` (1 = health, 0 = other), `split` — and `build_data.py` pulls the
texts from the upstream datasets, which retain their own licenses:

| Source | What was used | License |
|---|---|---|
| [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar et al.) | 1,355 questions (1,292 non-health, 63 health) | CC BY-SA 4.0 |
| [AskDocs](https://github.com/ju-resplande/askD) (Gomes) | 871 questions (health, from r/AskDocs titles) | GPL-3.0 |

`labels.csv` itself (the labeling decisions) is my own work and is part of
this repository under the Apache-2.0 license. If you redistribute the
*reconstructed datasets* (the CSVs produced by `build_data.py`), the
upstream licenses above apply to the question texts.

### Why the texts are not in the repo

An earlier version of this repository shipped the combined CSVs directly.
A provenance review found that ~41% of those rows could not be traced to
a licensable source with certainty, so they were dropped; the remaining
2,226 rows are reproduced deterministically from the two sources above.
This keeps the repo clean for portfolio use without silently redistributing
third-party forum content.

## Repository layout

```
train_classifier.ipynb   # end-to-end: encode -> features -> logistic regression
build_data.py            # dataset build from upstream sources
labels.csv               # labeling decisions (source, source_id, label, split)
data/                    # generated output (git-ignored)
```

## Credits & license

- The training-notebook method is adapted from the book
  [*Natural Language Processing with Transformers*](https://github.com/nlp-with-transformers/notebooks)
  (O'Reilly), Apache-2.0 — see NOTICE.
- Dataset licenses: SQuAD v1.1 CC BY-SA 4.0; AskDocs GPL-3.0.
- Everything else in this repository (code, labels, documentation):
  Apache-2.0 (see LICENSE), © 2022–2026 Sina Rampe.
