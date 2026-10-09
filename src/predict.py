"""Step 2: predict matches for the test split and write both submission files.

    python src/predict.py --data-dir ../../dataset --out-dir ../../output

Writes:
  <out-dir>/candidate_pairs.tsv   exactly the pairs the model scored (the final candidate set)
  <out-dir>/matching_results.tsv  the final matches (a subset of the candidates)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # faiss + lightgbm on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import json

import joblib
import pandas as pd

import config as C
import decide
import model as M
from io_utils import write_id_lists
from train import prepare
from validate_local import validate


def main(data_dir, out_dir):
    art = joblib.load(C.ARTIFACT_DIR / "model.joblib")
    rule = json.load(open(C.ARTIFACT_DIR / "decision.json"))
    s1n, pooln, pairs, X = prepare(data_dir, "test")
    X = X[art["features"]]
    pairs["p"] = M.predict_proba(art["model"], X)

    s1_ids = s1n.entity_id.tolist()
    candidates = pairs.groupby("s1_id").pool_id.apply(list).to_dict()
    scored = pairs[["s1_id", "pool_id", "p"]]
    matches = decide.apply_rule(scored, rule["t"], rule["r"], rule["one_to_one"])

    write_id_lists(f"{out_dir}/candidate_pairs.tsv", s1_ids, candidates, "candidate_entity_ids")
    write_id_lists(f"{out_dir}/matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    pairs[["s1_id", "pool_id", "p"]].to_csv(C.ARTIFACT_DIR / "test_scores.tsv", sep="\t", index=False)

    n_match = sum(1 for s in s1_ids if matches.get(s))
    print(f"[predict] {len(s1_ids):,} Source 1 entities | {n_match:,} with >=1 match | "
          f"{len(s1_ids) - n_match:,} predicted singletons")
    by_c = pd.Series({s: bool(matches.get(s)) for s in s1_ids}).groupby(
        s1n.set_index("entity_id").country_norm).mean()
    print(f"[predict] share with a match, by country: {by_c.round(3).to_dict()}")
    validate(f"{out_dir}/matching_results.tsv", f"{out_dir}/candidate_pairs.tsv", f"{data_dir}/test")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--out-dir", default=str(C.OUTPUT_DIR))
    a = ap.parse_args()
    main(a.data_dir, a.out_dir)
