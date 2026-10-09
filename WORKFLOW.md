# Business Entity Resolution: the complete workflow

Follow the phases in order. Each step has an exact command, what you should see, and what to
do if you don't. Do not skip the checkpoints: they catch the mistakes that cost places on the
leaderboard.

---

## The strategy in one paragraph

Entity resolution is a funnel. **Blocking** cuts the millions of possible record pairs down to
~50 plausible candidates per Source 1 business without losing true matches. A **matching
model** (LightGBM on ~55 hand-built similarity features) scores each candidate. A **decision
rule**, tuned directly on the competition metric, picks the final matches. The metric is
**macro F0.5**: precision counts twice as much as recall, and a correctly predicted *empty* list
for a business with no matches (a singleton) is worth a full 1.0. So the whole pipeline is built
to be conservative: when in doubt, do not merge.

```
3 sources ─► normalise ─► blocking (5 blockers, union) ─► candidate_pairs.tsv
                                                                │
              decision rule (tuned for F0.5) ◄── LightGBM probability ◄── ~55 pair features
                     │
                     ▼
           matching_results.tsv
```

---

## Phase 0: Setup (30 minutes)

**0.1** Put the official files in this layout (create folders if needed):

```
student_resource/
├── dataset/train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
├── dataset/test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
├── utils/validate_submission.py
├── Documentation_template.md
└── code/business_entity_resolution/        ← unzip this package here
```

**0.2** Create a clean environment (Python 3.10–3.12):

```bash
cd student_resource/code/business_entity_resolution
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import rapidfuzz, lightgbm; print('fast path OK')"
```

✅ **Checkpoint:** it prints `fast path OK`. If `rapidfuzz` or `lightgbm` fails to install, the
pipeline still runs, but slower and slightly weaker. Fix the install before the real runs
(on Windows, `pip install lightgbm` needs the Microsoft C++ redistributable; on a Mac with
Apple Silicon run `brew install libomp` first).

**0.3** Smoke-test the pipeline on synthetic data (proves your environment works before touching
the real data):

```bash
python tools/make_synthetic.py --out /tmp/synth
python src/run_pipeline.py --data-dir /tmp/synth --out-dir /tmp/synth/output
python tools/score.py /tmp/synth/output/matching_results.tsv /tmp/synth/test_hidden_ground_truth.tsv /tmp/synth/test/test_source1.tsv
```

✅ **Checkpoint:** ends with `[validate] PASS` and a macro F0.5 around 0.99. (Synthetic data
is easy; real data will score lower. This only proves the plumbing works.)

**0.4** Start a Git repo now (`git init`, commit after every phase). The final zip must
reproduce your results, so you need to know exactly which code produced which submission.

---

## Phase 1: Understand the real data (1–2 hours, do not skip)

**1.1** Run the exploration script:

```bash
python tools/eda.py --data-dir ../../dataset > ../../eda_report.txt
```

**1.2** Read `eda_report.txt` and write down these numbers in a notes file. Later decisions
depend on them:

| Look for | Why it matters |
|---|---|
| Sizes of S1, S2, S3 (train and test) | Runtime; if the pool is over ~200k rows, lower `CHUNK` in `src/config.py` |
| Singleton share | Typically 20–50%. Every singleton is a free 1.0 if you predict nothing, so a high share means be stricter |
| Matches-per-entity distribution | If most entities have 1–2 matches, lists of 5+ are almost always wrong |
| "pool records matched to >1 Source 1 entity" | If **0**, the one-to-one rule is safe (the tuner will decide; this just tells you why) |
| Country labels in test | Confirms France appears; check its exact spelling (e.g. `France` vs `FR`) |
| Empty names or addresses per source | Sources with many empty addresses need name-only matching to carry them |

**1.3** Read the 15 printed match examples slowly. For each one, ask: *what would a human use to
decide this is the same business?* Note every abbreviation, local word or format you see that is
**not** in the dictionaries at the top of `src/normalize.py` (`NAME_ABBR`, `LEGAL`, `ADDR_ABBR`,
`LANDMARK_RE`). You'll add them in Phase 4.

**1.4** Open the three test files and look at ~30 **France** rows by eye. Note French legal
forms (SARL, SAS, SASU, EURL, SCI, SA…), street words (rue, avenue, boulevard, chemin, allée,
impasse, quai…) and postcode format (5 digits). Most are already handled; add anything missing
to the dictionaries. Adding generic language knowledge is fine. **Never look anything up
externally** (see the rules box at the end).

---

## Phase 2: Baseline run and first submission (half a day)

**2.1** Run everything:

```bash
python src/run_pipeline.py --data-dir ../../dataset --out-dir ../../output 2>&1 | tee ../../run_v1.log
```

**2.2** In the log, read and record:

* `[blocking] pair recall = …`: share of true matches that survived blocking. **Target ≥ 0.98.**
* `[decide] best rule …  macro F0.5 = …`: your cross-validated score. **This is your real
  score estimate.** Trust it more than the public leaderboard.
* The per-country and singleton/matched breakdown under `[cv] out-of-fold result`.

**2.3** Run the official validator:

```bash
cd ../..
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

✅ **Checkpoint:** prints `PASS`. Then upload `output/matching_results.tsv` to the portal and
record the public score next to your CV score in a table:

| Version | Change | Blocking recall | CV F0.5 | Public LB |
|---|---|---|---|---|
| v1 | baseline | | | |

**2.4** `git commit -am "v1 baseline"` and copy `output/` to `submissions/v1/`.

---

## Phase 3: Fix blocking first (it caps everything)

If blocking misses a true match, nothing later can recover it.

**3.1** Look at the blocking recall from 2.2.

* **≥ 0.99:** fine, go to Phase 4.
* **0.95–0.99:** open `artifacts/oof_errors.tsv`, filter `type = NOT_IN_CANDIDATES`, and read
  30 rows. Classify why each was missed:
  * Names very different but address the same (renamed / DBA) → raise `K_ADDR_CHAR` to 30.
  * Abbreviations or transliterations → add them to the dictionaries in `normalize.py`.
  * Heavy typos in short names → raise `K_NAME_CHAR` to 40.
  * Just ranked too low → raise `MAX_CANDIDATES` to 80–100.
* **< 0.95:** something is structurally wrong (wrong column read, a source with mostly empty
  addresses, a different ID format). Re-check Phase 1 before tuning.

**3.2** After each change, re-run `python src/train.py --data-dir ../../dataset` (training only
is enough, and faster) and compare blocking recall. Keep the candidate count per Source 1 record
under ~100. More candidates means more chances for false merges and slower runs.

---

## Phase 4: The error-analysis loop (where most of the score comes from)

Repeat this loop 4–8 times. Each round takes 30–60 minutes.

**4.1** Open `artifacts/oof_errors.tsv` in a spreadsheet. It lists cross-validation mistakes:

* `FALSE_MERGE`: the model matched two *different* businesses. These hurt most under F0.5.
* `MISSED`: a true match the model rejected.
* `NOT_IN_CANDIDATES`: lost in blocking (Phase 3).

**4.2** Sort `FALSE_MERGE` by `p` descending and read the top 50. Group them into patterns:

| Pattern you see | Fix |
|---|---|
| Same chain/brand, different branch (different house number, street or postal code) | Features already exist (`a_num_conflict`, `postal_eq`, `pool_name_freq`). If still failing, add a feature such as "same name AND different postal code" |
| Generic names ("Sai Traders", "City Pharmacy") matched on name alone | Add a name-rarity feature: average IDF of the core-name tokens |
| Different legal entity types ("X Pvt Ltd" vs "X LLP") | Check `LEGAL` covers both forms |
| Label noise (looks like a true match to you too) | Ignore. Don't chase noise |

**4.3** Sort `MISSED` by `p` descending and read the top 50:

| Pattern | Fix |
|---|---|
| Abbreviation not normalised ("Mfg" / "Manufacturing", "Ngr" / "Nagar") | Add to `NAME_ABBR` / `ADDR_ABBR` |
| Landmark-only address ("Near SBI ATM, Sadar Bazar") vs full address | Check `LANDMARK_RE` catches the phrase; add new trigger words |
| DBA / trade name | Add the trigger word to `DBA_RE` |
| Transliteration ("Lakshmi" / "Laxmi") | Char n-grams usually handle this; for frequent ones add a canonical mapping |
| Word order, punctuation | Already handled by token-sort features; check normalisation didn't drop the key word |

**4.4** Make **one category of change at a time**, re-run `src/train.py`, and keep the change
only if CV F0.5 goes up. Log it in your version table. Commit.

**4.5** Every 2–3 rounds, run the full pipeline and submit. Use the public leaderboard only as a
sanity check: if CV goes up and LB goes clearly down, something leaked or broke.

---

## Phase 5: Squeeze the decision rule

**5.1** `artifacts/decision.json` holds the tuned rule (`t` threshold, `r` relative-to-best,
`one_to_one`). It's re-tuned automatically on every training run. Don't hand-edit it.

**5.2** If CV shows singletons scoring much lower than matched entities, the model is too eager.
Add stricter thresholds to the search grid in `src/config.py` (extend `THRESHOLDS` up to 0.97).

**5.3** **Seed ensembling** (a reliable +0.002–0.01): in `src/model.py`, train 3–5 LightGBM
models with different `random_state` values and average their probabilities, in both the CV
loop and the final model. Re-tune the rule on the averaged OOF predictions.

---

## Phase 6: Optional upgrade, a small transformer re-ranker (only if you have a GPU and time)

Do this only after Phases 3–5 have stopped giving gains. It helps most with transliteration
and the unseen French data.

**6.1** Model choice must satisfy the rules: **MIT or Apache-2.0 license, ≤ 8B parameters.**
Good fits are small Apache-2.0 sentence-embedding or cross-encoder checkpoints (tens to hundreds
of millions of parameters), ideally a multilingual one for French. **Confirm the license on the
model card before using it and note it in your documentation.**

**6.2** How to use it: turn each record into one string (`name | address | country`), fine-tune
a cross-encoder on your training candidate pairs (labels = ground truth, grouped folds by Source 1
id exactly like `train.py`), and add its out-of-fold score as **one extra feature** column in
LightGBM. Don't replace LightGBM. Stacking keeps all the hand-built signals.

**6.3** Keep the folds identical to `train.py`, or the extra feature will leak labels and your
CV will lie.

---

## Phase 7: Choosing the final submission

**7.1** The final ranking uses the **private** leaderboard (the part of the test set you never see
scored). Pick your final submission by **best CV F0.5**, not best public score. Public-LB
chasing overfits the visible slice.

**7.2** Re-run the chosen version from a clean checkout to prove it reproduces:

```bash
git checkout <chosen commit>
rm -rf artifacts ../../output
python src/run_pipeline.py --data-dir ../../dataset --out-dir ../../output
```

Confirm the new `matching_results.tsv` is identical to the one you submitted (`diff` or compare
file hashes).

---

## Phase 8: Final submission package

**8.1** Freeze your environment: `pip freeze > requirements.txt`.

**8.2** Fill in `Documentation_template.md`. `METHODOLOGY_DRAFT.md` in this package already
contains the text for every required section (methodology, blocking strategy, model and features,
other details). Copy it across, then update the numbers (blocking recall, reduction ratio, CV
F0.5, final features) from your own logs.

**8.3** Build the zip with exactly this structure:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   ├── tools/
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md
```

```bash
cd student_resource
mkdir -p pkg/code && cp -r output pkg/ && cp Documentation_template.md pkg/
rsync -a --exclude .venv --exclude artifacts --exclude __pycache__ code/business_entity_resolution pkg/code/
cd pkg && zip -r ../<team_name>_submission.zip . && cd ..
```

**8.4** Final checks before uploading:

- [ ] Official validator prints `PASS` on the two files inside the zip
- [ ] README commands regenerate both files from scratch
- [ ] No API keys, no internet calls, no external data files anywhere in the code
- [ ] Every model and library used is MIT / Apache-2.0 / BSD; any pretrained model is ≤ 8B parameters and its license is stated in the documentation
- [ ] Documentation numbers match your final run

---

## ⚠️ Rules that get teams disqualified

1. **No external data lookup of any kind**: no business registries, no geocoding or address APIs, no web
   scraping, no commercial entity-resolution services, no external datasets. Dictionaries of
   generic abbreviations (Rd = Road, SARL = legal form) written by you are fine.
2. **Final model: MIT/Apache-2.0, ≤ 8B parameters.**
3. **Every test Source 1 id exactly once**; only Source 2/3 ids that exist in the test set; no
   duplicates in any list.
4. Final matches must be a **subset of candidate_pairs**. The pipeline guarantees this; don't
   edit one file without the other.
5. Don't hard-code countries. The code treats country as an open set and uses no country-specific
   model inputs.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| A single column after reading a file | You read it without `sep="\t"`. Always use `io_utils.read_tsv` |
| `MemoryError` in blocking | Lower `CHUNK` in `src/config.py` (e.g. 128) |
| Very slow features | `rapidfuzz` is not installed; install it |
| Validator: "ids not in test set" | You ran predict on the train folder or mixed versions. Re-run `run_pipeline.py` |
| CV high, public LB much lower | Check France: compare `[predict] share with a match, by country`. If France is far off, revisit Phase 1.4 dictionaries |
| CV drops after a dictionary change | A mapping was too aggressive (e.g. mapping a short token that is also a common word). Revert it |

---

## Suggested schedule (adjust to your deadline)

| Day | Work |
|---|---|
| 1 | Phases 0–2: setup, data exploration, baseline submission |
| 2 | Phase 3 (blocking) + first two error-analysis rounds |
| 3 | Error-analysis rounds 3–6, seed ensembling |
| 4 | Optional Phase 6, or more error analysis; pick the final version by CV |
| Last day | Phase 7–8: reproduce from clean, documentation, zip, validator, submit. Don't change the model on the last day |
