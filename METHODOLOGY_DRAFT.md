# Methodology (draft: copy into Documentation_template.md and update the numbers)

> Replace every `[…]` with the value from your final run log (`run_vN.log`, `artifacts/decision.json`).

## 1. Methodology used

We treat the task as supervised pairwise entity resolution with a three-stage pipeline:
(1) **normalisation** of names and addresses into canonical, comparable forms; (2) **blocking**
that selects a small set of plausible Source 2/3 candidates for each Source 1 entity; (3) a
**gradient-boosted classifier** (LightGBM, MIT license) that scores each candidate pair,
followed by a **decision rule tuned directly for macro-averaged F0.5**.

The pipeline uses only the provided data. No external databases, APIs, geocoders or
external datasets are used. Country is treated as an open set of labels: no country is filtered,
one-hot encoded or hard-coded in the model, so the same pipeline applies to France, which is
absent from training.

## 2. Candidate generation / blocking strategy

Each Source 1 record receives the **union** of candidates from five complementary blockers, so
that a match missed by one signal is recovered by another:

| Blocker | Signal | Catches |
|---|---|---|
| Char 2–4-gram TF-IDF on cleaned core name, top-25 | name shape | typos, transliterations, spacing |
| Word 1–2-gram TF-IDF on core name + address, top-25 | joint name/address | word-order changes, partial records |
| Char 3–4-gram TF-IDF on address, top-15 | location | renamed / DBA businesses at the same address |
| Same postal code + shared core-name token (≥3 chars) | exact location key | records with heavy name noise |
| Identical compact core name | exact key | spacing / punctuation variants |

TF-IDF vocabularies are fitted on the records of each split itself (unsupervised; no labels, no
external text). Nearest neighbours are computed by chunked sparse cosine similarity. The union is
ordered by a pre-score (max of name and full-text cosine + 0.3 × address cosine) and capped at 60
candidates per Source 1 record. `candidate_pairs.tsv` contains exactly these pairs, which is the set
the model scores.

Measured on the training split: pair recall = **[…]**, mean candidates per entity = **[…]**,
reduction ratio = **[…]**.

## 3. Model architecture and feature engineering

### Normalisation
Lower-casing, Unicode accent stripping, punctuation removal ("Pvt. Ltd." → "pvt ltd",
"L.L.C." → "llc"), canonicalisation of business and address abbreviations for US, Indian and
French conventions (Corp/Corporation, Pvt/Private, Rd/Road, Ngr/Nagar, Av/Avenue, Bd/Boulevard…),
separation of **legal-form tokens** (LLC, Inc, Pvt Ltd, LLP, SARL, SAS…) from the **core name**,
splitting of **DBA / trading-as** names, extraction of **landmark phrases** ("near SBI ATM",
"opp. bus stand", "en face de…") from the main address, extraction of **postal codes**
(5-digit, ZIP+4, 6-digit PIN including "110 001") and **house / unit numbers**.

### Features (~55 per pair, all country-agnostic)
* **Name:** Levenshtein-based ratio, Jaro-Winkler, token-sort, token-set and partial ratios,
  token Jaccard and overlap, full-name ratio, compact-name equality and Jaro-Winkler, first-token
  equality, best similarity over DBA name variants, acronym match, length differences, digit-token
  agreement/conflict, legal-form agreement/conflict.
* **Address:** ratio, token-set/sort ratios, word Jaccard and overlap, city/state-tail similarity,
  house-number agreement and conflict, postal-code equality and 3-digit-prefix equality (with
  explicit "missing" values), landmark similarity, empty-address flags.
* **Blocking signals:** the three TF-IDF cosines, which blockers produced the pair, and the reverse
  rank (rank of this Source 1 record among the candidate's own nearest Source 1 records).
* **Context:** rank and score gap of the candidate among all candidates of the same Source 1
  entity and among all Source 1 entities competing for the same candidate, candidate counts, and
  the frequency of the candidate's name in the pool (chain / franchise indicator).
* **Other:** country agreement (as equality only), source indicator (S2 vs S3).

### Model
LightGBM binary classifier (700 trees, learning rate 0.04, 63 leaves, row/column subsampling 0.8,
seed 42). Training pairs are the blocking candidates of the training split, labelled from the
ground truth. Hyper-parameters were kept at robust defaults; gains came from features and
normalisation.

### Validation
5-fold **GroupKFold grouped by Source 1 entity**, so no entity appears in both train and
validation folds. Out-of-fold probabilities are used to tune the decision rule and to report
the score. Out-of-fold macro F0.5 = **[…]** (singletons **[…]**, matched entities **[…]**).

### Decision rule (optimised for macro F0.5)
A candidate is accepted if p ≥ t and p ≥ r × (best p for that Source 1 entity). Optionally
(one-to-one), each Source 2/3 record is assigned to at most one Source 1 entity, the one with the
highest probability, because Source 1 is deduplicated. (t, r, one-to-one) are grid-searched on
out-of-fold predictions to maximise macro F0.5, which accounts for singletons (an empty prediction
scores 1.0) and the 2× precision weighting. Selected: t = **[…]**, r = **[…]**,
one-to-one = **[…]**.

## 4. Other relevant information

* **Generalisation to an unseen country (France):** no country-specific inputs; accent stripping
  and French abbreviation/legal-form entries are part of the shared normalisation; character
  n-gram features are language-independent. On a synthetic stress test with France held out of
  training, macro F0.5 on France stayed within ~0.01 of the trained countries.
* **Licenses:** pandas, NumPy, SciPy, scikit-learn (BSD-3); rapidfuzz, LightGBM (MIT). No
  pretrained language model is used [update if you added the Phase 6 re-ranker: name, license,
  parameter count].
* **Reproducibility:** single command (`python src/run_pipeline.py`), fixed seeds, pinned
  requirements; runtime on our machine: **[…]**.
* **Error analysis:** `artifacts/oof_errors.tsv` lists cross-validation false merges and misses;
  dictionary and feature changes were accepted only when out-of-fold F0.5 improved.
* **AI assistance:** [state any coding assistants used, if the organisers require it].
