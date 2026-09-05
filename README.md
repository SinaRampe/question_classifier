# Question Classifier

A lightweight text classifier that decides whether a question is about a
health(care) topic. Built on a method from
[*Natural Language Processing with Transformers*](https://github.com/nlp-with-transformers/notebooks)
(O'Reilly): DistilBERT's frozen hidden states + a logistic-regression head —
a full training run that trains only 769 parameters (the encoder's 66M stay
frozen), runs on CPU, ~98% accuracy on the held-out split.

This classifier was part of a team portfolio project in my 2022 data-science
bootcamp; the classifier itself — concept, labeling, implementation — I built
on my own. In 2026 the repository was reworked for licensing
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

The 768-dimensional hidden states, projected to 2D with UMAP (cosine metric)
— the two classes fall apart cleanly, which is why 769 trained parameters
are enough:

![UMAP projection of the frozen DistilBERT [CLS] hidden states on the
training split. Left: non-health questions (blue density), right: health
questions (red density). The two densities occupy distinct regions of the
2D embedding.](docs/umap_hidden_states.png)

```bash
python build_data.py        # downloads sources, writes data/*.csv
python evaluate.py           # trains + scores the classifier (CPU, ~5 min)
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

The cleanup cost nothing in accuracy: the reduced dataset scores **97.8%**
on the held-out test split with the notebook's method (frozen DistilBERT
+ logistic regression), vs. 98.0% for the original 2022 dataset with ~70%
more rows — the classifier was already in its saturation region.
`evaluate.py` reproduces this number without Jupyter.

## Repository layout

```
train_classifier.ipynb   # end-to-end: encode -> features -> logistic regression
build_data.py            # dataset build from upstream sources
evaluate.py              # same method as a script: train + score, no Jupyter
visualize.py             # UMAP 2D projection of the hidden states -> docs/*.png
labels.csv               # labeling decisions (source, source_id, label, split)
data/                    # generated output (git-ignored)
docs/                    # generated figures (committed)
```

## Credits & license

- The training-notebook method is adapted from the book
  [*Natural Language Processing with Transformers*](https://github.com/nlp-with-transformers/notebooks)
  (O'Reilly), Apache-2.0 — see NOTICE.
- Dataset licenses: SQuAD v1.1 CC BY-SA 4.0; AskDocs GPL-3.0.
- Everything else in this repository (code, labels, documentation):
  Apache-2.0 (see LICENSE), © 2022–2026 Sina Rampe.
