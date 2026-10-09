# Business Entity Resolution

For every **Source 1** business, find the **Source 2 / Source 3** records that describe the same
real-world business (US and India in training, plus unseen countries such as France in test).
Built for the Amazon ML Challenge entity-resolution task, scored on **macro F0.5**.

The pipeline uses only the provided data: no external databases, APIs or geocoding. The one
pretrained component is an open multilingual sentence-embedding model used for candidate
generation. It is optional and can be switched off in `src/config.py`.

## The problem

The same business shows up differently in different databases:

| Source | Name | Address |
|---|---|---|
| Source 1 | Sharma Traders Pvt. Ltd. | Shop 12, MG Road, Near City Mall, Pune 411001 |
| Source 2 | SHARMA TRADERS PRIVATE LIMITED | 12 M.G. Rd, Pune - 411001 |
| Source 3 | Sharma Trdrs | MG Road, Pune |

A person can tell these are one business. A computer comparing strings can't, and checking
every record against every other record is impossible at this scale (2.2M × 10.3M is about
22 trillion pairs). This project matches such records automatically. It handles abbreviations,
typos, legal suffixes, transliterated Indian names, reordered addresses and landmarks. It is
deliberately conservative: when it isn't sure, it doesn't merge, because a wrong merge is worse
than a missed one.

## Use cases

Entity resolution (also called record linkage or deduplication) is a core data problem
wherever records about the same thing come from more than one place:

- **E-commerce and marketplaces:** merge seller, supplier and store listings that come from
  different onboarding systems, so each business has one profile, one rating and one payout
  account.
- **Maps and local search:** remove duplicate places so a business doesn't appear three times
  with different hours or phone numbers.
- **Banking and fintech (KYC / AML):** link a merchant or company across applications and
  partner feeds to spot duplicate accounts, shell companies and fraud rings.
- **CRM and sales:** clean customer and lead databases after a merger, an acquisition or a
  data import, so sales teams don't contact the same company twice.
- **Supply chain and procurement:** build one vendor master list from many ERP systems, to
  see total spend per supplier and catch duplicate invoices.
- **Government and public data:** join business registries, tax records and licence databases
  that were never designed to share a common ID.
- **Data integration in general:** any pipeline that combines datasets with no shared key.

The methods here (normalisation, blocking, learned similarity, and a decision rule tuned to the
business cost of errors) carry over to all of these settings, including countries the model
was never trained on.

## Results (cross-validated, 2% sample of training regions)

| Metric | Score |
|---|---|
| **Macro F0.5 (out-of-fold, 5 folds)** | **0.9847** |
| Pair precision | 0.9915 |
| Pair recall | 0.9796 |
| Blocking recall (true pairs kept as candidates) | 0.9978 |
| US entities (n = 43,776) | F0.5 0.9871 |
| India entities (n = 4,353) | F0.5 0.9609 |
| Singletons (n = 12,922) | F0.5 0.9847 |

Tuned decision rule: match if `p ≥ 0.475` **and** `p ≥ 0.7 × best p for that Source 1 record`,
with one-to-one assignment. On the run above, about 2.88M candidate pairs were scored (about 60
per Source 1 record, 2.29% positives).

## How it works

```
3 sources ─► normalise ─► blocking (7 blockers, union, cap 60) ─► candidate_pairs.tsv
                                                                         │
            decision rule (tuned for F0.5) ◄── LightGBM probability ◄── ~55 pair features
                   │
                   ▼
         matching_results.tsv
```

| Stage | File | What it does |
|---|---|---|
| Normalise | `src/normalize.py` | lower-case, accent strip, transliteration, abbreviation canonicalisation, legal-suffix split, DBA split, landmark / postal / house-number extraction |
| Block | `src/blocking.py` | union of 7 blockers, capped at 60 candidates per Source 1 record (see below) |
| Embed | `src/embed.py`, `src/knn.py` | multilingual MiniLM name embeddings, FAISS nearest-neighbour search per country, cached to disk |
| Features | `src/features.py` | ~55 country-agnostic pair features: string similarities, TF-IDF cosines, postal / number agreement, legal-form conflict, acronym match, rank-in-group context |
| Model | `src/model.py` | LightGBM binary classifier |
| Decide | `src/decide.py` | threshold + relative-to-best + optional one-to-one rule, tuned on out-of-fold macro F0.5 |
| Validate | `src/validate_local.py` | checks every submission rule |

**Blockers** (the bit numbers appear in the run log):

| Bit | Signal |
|---|---|
| 0 | char n-gram TF-IDF on the cleaned core name (within region) |
| 1 | word TF-IDF on core name + address (within region) |
| 2 | char n-gram TF-IDF on the address (within region) |
| 3 | same postal code + shared distinctive name token |
| 4 | exact compact core name (`abctraders` = `abc traders`) |
| 5 | same house number + street token + country |
| 6 | multilingual name embedding, FAISS search over the whole country |

## Setup

Python 3.10 to 3.12.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Apple Silicon Macs, run `brew install libomp` first (LightGBM needs it).

## Data (not included in this repo)

The competition dataset is **not** committed. Download it separately and point `--data-dir` at
the `dataset/` folder:

```
dataset/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

## Run

**Quick experiment** (2% of regions, about 15 to 20 minutes on a laptop, no test prediction):

```bash
OMP_NUM_THREADS=1 python src/run_pipeline.py --data-dir /path/to/dataset --sample 0.02
```

**Full run** (train → tune → predict → validate, writes the submission):

```bash
OMP_NUM_THREADS=1 python src/run_pipeline.py --data-dir /path/to/dataset --out-dir output
```

The full dataset is large (about 2.2M Source 1 and 10.3M pool records). Run it on a machine
with plenty of RAM, or use `train_on_colab.ipynb`.

> **macOS note:** keep `OMP_NUM_THREADS=1`. PyTorch (via sentence-transformers) and `faiss-cpu`
> each ship their own OpenMP library. Without this setting, the FAISS search step crashes with
> a `segmentation fault`.

## Outputs

| File | Contents |
|---|---|
| `output/matching_results.tsv` | final matches (the submission) |
| `output/candidate_pairs.tsv` | exactly the pairs the model scored |
| `artifacts/decision.json` | tuned rule and cross-validated macro F0.5 |
| `artifacts/feature_importance.tsv` | LightGBM feature importances |
| `artifacts/oof_errors.tsv` | missed matches and false merges from CV, for error analysis (local only, not committed) |
| `artifacts/model.joblib` | trained model (local only, not committed) |
| `artifacts/emb_cache/` | cached embeddings, about 200 MB (local only, not committed) |

## Smoke test without the real data

```bash
python tools/make_synthetic.py --out /tmp/synth
python src/run_pipeline.py --data-dir /tmp/synth --out-dir /tmp/synth/output
python tools/score.py /tmp/synth/output/matching_results.tsv /tmp/synth/test_hidden_ground_truth.tsv /tmp/synth/test/test_source1.tsv
```

## Next steps

- **India** is the weakest segment (F0.5 0.961, blocking recall 0.967). Error analysis on
  `artifacts/oof_errors.tsv` (1,204 missed, 558 false merges, 144 never reached candidates)
  should guide better transliteration and address handling.
- Make the full-data run memory-safe (chunked features and embedding).

## Project layout

```
src/         pipeline code (run_pipeline.py is the entry point)
tools/       EDA, synthetic data generator, local scorer
artifacts/   model outputs (mostly git-ignored)
WORKFLOW.md  step-by-step working guide
METHODOLOGY_DRAFT.md  write-up draft for the submission documentation
```
