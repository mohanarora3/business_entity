"""Quick look at the real data before modelling.

    python tools/eda.py --data-dir ../../dataset
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from io_utils import load_ground_truth, load_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
d = ap.parse_args().data_dir
pd.set_option("display.width", 220, "display.max_colwidth", 70)

for split in ("train", "test"):
    s1, pool = load_split(d, split)
    print(f"\n===== {split.upper()} =====")
    print(f"Source 1: {len(s1):,} | Source 2: {(pool.source == 'S2').sum():,} | Source 3: {(pool.source == 'S3').sum():,}")
    print("Country labels (Source 1):", s1.country.value_counts().to_dict())
    print("Country labels (Source 2+3):", pool.country.value_counts().to_dict())
    for name, df in (("S1", s1), ("S2", pool[pool.source == "S2"]), ("S3", pool[pool.source == "S3"])):
        print(f"{name}: empty name {(df.business_name == '').mean():.1%} | empty address "
              f"{(df.business_address == '').mean():.1%} | median name len "
              f"{df.business_name.str.len().median():.0f} | median address len {df.business_address.str.len().median():.0f}")

truth = load_ground_truth(f"{d}/train/train_ground_truth.tsv")
s1, pool = load_split(d, "train")
n = pd.Series({k: len(v) for k, v in truth.items()})
print("\n===== LABELS =====")
print(f"singletons: {(n == 0).mean():.1%} | matches per entity distribution: {n.value_counts().sort_index().to_dict()}")
src = pd.Series([p[:2] for v in truth.values() for p in v]).value_counts().to_dict()
print("matched ids by source:", src)
allm = pd.Series([p for v in truth.values() for p in v])
print(f"pool records matched to >1 Source 1 entity: {(allm.value_counts() > 1).sum()}")
print(f"pool records never matched: {len(set(pool.entity_id) - set(allm)):,} / {len(pool):,}")

print("\n===== 15 RANDOM MATCHED EXAMPLES (read these carefully) =====")
a = s1.set_index("entity_id")
b = pool.set_index("entity_id")
ex = [(k, p) for k, v in truth.items() for p in v]
for k, p in pd.Series(ex).sample(min(15, len(ex)), random_state=1):
    print(f"\n{k} | {a.at[k, 'business_name']} | {a.at[k, 'business_address']} | {a.at[k, 'country']}")
    print(f"{p} | {b.at[p, 'business_name']} | {b.at[p, 'business_address']} | {b.at[p, 'country']}")
