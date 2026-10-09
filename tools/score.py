"""Score a matching_results.tsv against a ground-truth file with the official macro F0.5.

    python tools/score.py <matching_results.tsv> <ground_truth.tsv> [source1.tsv for per-country]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from decide import f05  # noqa: E402
from io_utils import load_ground_truth, read_tsv  # noqa: E402

p = read_tsv(sys.argv[1])
pred = {s: {x for x in ids.split(",") if x} for s, ids in zip(p.iloc[:, 0], p.iloc[:, 1])}
truth = load_ground_truth(sys.argv[2])
rows = [(s, f05(pred.get(s, set()), t), bool(t)) for s, t in truth.items()]
df = pd.DataFrame(rows, columns=["s1", "f", "has_match"])
print(f"macro F0.5 = {df.f.mean():.4f}  (n={len(df):,})")
print(df.groupby("has_match").f.agg(["mean", "size"]).rename(index={True: "matched", False: "singleton"}))
if len(sys.argv) > 3:
    c = read_tsv(sys.argv[3]).set_index("entity_id")["country"]
    df["country"] = df.s1.map(c)
    print(df.groupby("country").f.agg(["mean", "size"]))
