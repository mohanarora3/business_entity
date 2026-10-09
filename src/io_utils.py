"""Reading and writing the competition's TSV files safely."""

import csv
import zlib
from pathlib import Path

import pandas as pd

from config import ADDR, COUNTRY, ID, NAME


def read_tsv(path) -> pd.DataFrame:
    """Read a TSV exactly as given: tab separator, everything as string, no NaN magic,
    no quote handling (business names contain quotes and commas)."""
    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        quoting=csv.QUOTE_NONE, encoding="utf-8",
    )
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df


_ZIP_RE = r"(?<!\d)(\d{5,6})(?!\d)"


def region_key(df: pd.DataFrame) -> pd.Series:
    """Coarse region of each record: country + first 3 digits of its last 5/6-digit number
    (US ZIP3 / India PIN3). Records without one fall back to country + last address word."""
    zips = df[ADDR].str.findall(_ZIP_RE).str[-1].fillna("")
    last = df[ADDR].str.lower().str.replace(r"[^a-z ]", " ", regex=True).str.split().str[-1].fillna("")
    key = zips.str[:3].where(zips != "", "w:" + last)
    return df[COUNTRY].str.lower().str.strip() + "|" + key


def sample_regions(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Keep every record whose region hashes into the sampled fraction. The same regions are
    kept in every source, so true matches stay together and candidate density stays realistic."""
    keys = region_key(df)
    cut = int(frac * 10_000)
    keep = keys.map(lambda k: zlib.crc32(k.encode()) % 10_000 < cut)
    return df[keep.values].reset_index(drop=True)


def load_split(data_dir, split: str, sample: float = None):
    """Return (source1_df, pool_df). pool = Source 2 + Source 3 stacked, with a 'source' column.
    sample: optional fraction (e.g. 0.02) of geographic regions to keep, for fast experiments."""
    d = Path(data_dir) / split
    s1 = read_tsv(d / f"{split}_source1.tsv")
    s2 = read_tsv(d / f"{split}_source2.tsv")
    s3 = read_tsv(d / f"{split}_source3.tsv")
    s2["source"] = "S2"
    s3["source"] = "S3"
    pool = pd.concat([s2, s3], ignore_index=True)
    if sample:
        n1, n2 = len(s1), len(pool)
        s1, pool = sample_regions(s1, sample), sample_regions(pool, sample)
        print(f"[{split}] regional sample {sample:.1%}: Source 1 {n1:,} -> {len(s1):,} | "
              f"pool {n2:,} -> {len(pool):,}")
    for df in (s1, pool):
        for col in (ID, NAME, ADDR, COUNTRY):
            if col not in df.columns:
                raise ValueError(f"Missing column {col!r}; found {list(df.columns)}")
    if s1[ID].duplicated().any():
        raise ValueError("Duplicate Source 1 ids")
    if pool[ID].duplicated().any():
        raise ValueError("Duplicate Source 2/3 ids")
    return s1.reset_index(drop=True), pool.reset_index(drop=True)


def load_ground_truth(path) -> dict:
    """{source1_entity_id: set(matched ids)} — empty set for singletons."""
    gt = read_tsv(path)
    out = {}
    for s1_id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1_id] = {x.strip() for x in ids.split(",") if x.strip()}
    return out


def write_id_lists(path, s1_ids, mapping: dict, col: str):
    """Write one row per Source 1 id, ids comma-joined, no quoting, no duplicates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1_id in s1_ids:
            ids = sorted(set(mapping.get(s1_id, ())))
            f.write(f"{s1_id}\t{','.join(ids)}\n")
