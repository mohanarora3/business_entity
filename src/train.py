"""Step 1: train the matcher on the training split and tune the decision rule.

    python src/train.py --data-dir ../../dataset

Outputs (in artifacts/):
  model.joblib        model trained on ALL training pairs
  decision.json       tuned threshold rule + out-of-fold macro F0.5
  oof_errors.tsv      false merges / missed matches from cross-validation, for error analysis
  feature_importance.tsv
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # faiss + lightgbm on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config as C
import decide
import features
import model as M
from blocking import blocking_report, generate_candidates
from io_utils import load_ground_truth, load_split
from normalize import normalize_frame


def prepare(data_dir, split, sample=None):
    t0 = time.time()
    s1, pool = load_split(data_dir, split, sample)
    print(f"[{split}] Source 1: {len(s1):,} | Source 2+3 pool: {len(pool):,} "
          f"(S2 {int((pool.source == 'S2').sum()):,}, S3 {int((pool.source == 'S3').sum()):,})")
    print(f"[{split}] countries in Source 1: {s1['country'].value_counts().to_dict()}")
    s1n, pooln = normalize_frame(s1), normalize_frame(pool)
    pairs, _ = generate_candidates(s1n, pooln)
    X = features.build_features(pairs, s1n, pooln)
    pairs["s1_id"] = s1n.entity_id.values[pairs.i.values]
    pairs["pool_id"] = pooln.entity_id.values[pairs.j.values]
    print(f"[{split}] prepared in {time.time() - t0:.0f}s")
    return s1n, pooln, pairs, X


def main(data_dir, sample=None):
    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    s1n, pooln, pairs, X = prepare(data_dir, "train", sample)
    truth = load_ground_truth(f"{data_dir}/train/train_ground_truth.tsv")
    s1_ids = s1n.entity_id.tolist()
    if sample:
        # only judge matches whose partner record is inside the sampled regions
        kept = set(pooln.entity_id)
        truth = {s: truth.get(s, set()) & kept for s in s1_ids}

    # ---- sanity facts about the labels
    n_single = sum(1 for s in s1_ids if not truth.get(s))
    multi = pd.Series([p for s in s1_ids for p in truth.get(s, ())]).value_counts()
    print(f"[labels] singletons: {n_single:,}/{len(s1_ids):,} ({n_single / len(s1_ids):.1%}) | "
          f"pool records matched to >1 Source 1 entity: {int((multi > 1).sum())}")
    blocking_report(pairs, s1n, pooln, truth)

    y = np.array([p in truth.get(s, ()) for s, p in zip(pairs.s1_id, pairs.pool_id)], dtype=np.int8)
    print(f"[train] {len(y):,} pairs, {int(y.sum()):,} positives ({y.mean():.2%}) | model: {M.name()}")

    # ---- out-of-fold predictions, grouped by Source 1 entity (no leakage between folds)
    oof = np.zeros(len(y), dtype=np.float32)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=C.N_FOLDS).split(X, y, groups=pairs.i.values)):
        m = M.make_model(int(y[tr].sum()), int(len(tr) - y[tr].sum()))
        m.fit(X.iloc[tr], y[tr])
        oof[va] = M.predict_proba(m, X.iloc[va])
        print(f"[cv] fold {k + 1}/{C.N_FOLDS} done")

    scored = pd.DataFrame({"s1_id": pairs.s1_id, "pool_id": pairs.pool_id, "p": oof})
    rule = decide.tune(scored, truth, s1_ids)
    country_of = dict(zip(s1n.entity_id, s1n.country_norm))
    pred = decide.apply_rule(scored, rule["t"], rule["r"], rule["one_to_one"])
    print("[cv] out-of-fold result with the tuned rule:")
    decide.detail_report(pred, truth, s1_ids, country_of)

    # ---- error analysis file
    name_a = dict(zip(s1n.entity_id, s1n.business_name))
    addr_a = dict(zip(s1n.entity_id, s1n.business_address))
    name_b = dict(zip(pooln.entity_id, pooln.business_name))
    addr_b = dict(zip(pooln.entity_id, pooln.business_address))
    pred_set = {(s, p) for s, ps in pred.items() for p in ps}
    rows = []
    for s, p, prob, lab in zip(scored.s1_id, scored.pool_id, scored.p, y):
        kind = ("FALSE_MERGE" if (s, p) in pred_set and not lab else
                "MISSED" if lab and (s, p) not in pred_set else None)
        if kind:
            rows.append((kind, round(float(prob), 3), s, name_a[s], addr_a[s], p, name_b[p], addr_b[p]))
    cand_set = set(zip(scored.s1_id, scored.pool_id))
    for s in s1_ids:
        for p in truth.get(s, ()):
            if (s, p) not in cand_set and p in name_b:
                rows.append(("NOT_IN_CANDIDATES", -1, s, name_a[s], addr_a[s], p, name_b[p], addr_b[p]))
    err = pd.DataFrame(rows, columns=["type", "p", "s1_id", "s1_name", "s1_address",
                                      "other_id", "other_name", "other_address"])
    err.sort_values(["type", "p"], ascending=[True, False]).to_csv(
        C.ARTIFACT_DIR / "oof_errors.tsv", sep="\t", index=False)
    print(f"[cv] wrote {len(err):,} error rows to artifacts/oof_errors.tsv "
          f"({err.type.value_counts().to_dict()})")

    # ---- final model on all training pairs
    final = M.make_model(int(y.sum()), int(len(y) - y.sum()))
    final.fit(X, y)
    joblib.dump({"model": final, "features": list(X.columns)}, C.ARTIFACT_DIR / "model.joblib")
    with open(C.ARTIFACT_DIR / "decision.json", "w") as f:
        json.dump({**rule, "model": M.name()}, f, indent=2)
    if hasattr(final, "feature_importances_"):
        pd.Series(final.feature_importances_, index=X.columns).sort_values(ascending=False) \
          .to_csv(C.ARTIFACT_DIR / "feature_importance.tsv", sep="\t", header=["importance"])
    print(f"[train] saved model + decision rule to {C.ARTIFACT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--sample", type=float, default=None,
                    help="fraction of geographic regions to train on, e.g. 0.02 (default: all data)")
    a = ap.parse_args()
    main(a.data_dir, a.sample)
