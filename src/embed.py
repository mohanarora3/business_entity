"""Multilingual name embeddings.

A small sentence-embedding model (Apache-2.0, ~118M parameters) maps names written in
different scripts to nearby vectors: "अरिहंत इन्फ्रा प्राइवेट लिमिटेड" lands near
"Arihant Infra Private Limited". String similarity cannot do that.

Only unique strings are encoded, results are cached in artifacts/emb_cache/, and the
model runs on a GPU automatically when one is available (much faster than CPU).
If sentence-transformers is not installed the pipeline runs without embeddings.
"""

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

# only check that the package exists: torch is loaded in a separate process (embed_worker.py)
HAVE_ST = importlib.util.find_spec("sentence_transformers") is not None


def available() -> bool:
    return C.USE_EMBEDDINGS and HAVE_ST


def _encode_subprocess(texts) -> np.ndarray:
    worker = Path(__file__).with_name("embed_worker.py")
    with tempfile.TemporaryDirectory() as d:
        inp, out = Path(d) / "texts.txt", Path(d) / "emb.npy"
        inp.write_text("\n".join(t.replace("\n", " ").replace("\r", " ") for t in texts), encoding="utf-8")
        subprocess.run([sys.executable, str(worker), str(inp), str(out), C.EMB_MODEL, str(C.EMB_BATCH)],
                       check=True)
        return np.load(out)


class Embeddings:
    """Unique-string embedding table + one code per record (saves memory for repeated names)."""

    def __init__(self, table: np.ndarray, codes: np.ndarray):
        self.table = table            # (n_unique, dim) float16, L2-normalised
        self.codes = codes            # (n_records,) int64 index into table

    def rows(self, idx) -> np.ndarray:
        return self.table[self.codes[idx]].astype(np.float32)

    def pair_cos(self, other: "Embeddings", i: np.ndarray, j: np.ndarray, step=1_000_000) -> np.ndarray:
        out = np.empty(len(i), dtype=np.float32)
        for s in range(0, len(i), step):
            a = self.table[self.codes[i[s:s + step]]].astype(np.float32)
            b = other.table[other.codes[j[s:s + step]]].astype(np.float32)
            out[s:s + step] = (a * b).sum(axis=1)
        return out


def encode(texts, tag: str) -> Embeddings:
    """Embed a list of strings (one per record)."""
    codes, uniques = pd.factorize(pd.Series(list(texts), dtype=object).fillna(""), sort=False)
    uniques = [u if u else "_" for u in uniques]
    h = hashlib.md5((C.EMB_MODEL + "\n" + "\n".join(uniques)).encode("utf-8")).hexdigest()[:16]
    cache = C.ARTIFACT_DIR / "emb_cache" / f"{tag}_{h}.npy"
    if cache.exists():
        table = np.load(cache)
        print(f"[embed] {tag}: loaded {len(uniques):,} cached embeddings")
    else:
        t0 = time.time()
        table = _encode_subprocess(uniques).astype(np.float16)
        assert len(table) == len(uniques), "embedding count mismatch"
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, table)
        print(f"[embed] {tag}: encoded {len(uniques):,} unique names in {time.time() - t0:.0f}s")
    return Embeddings(table, codes.astype(np.int64))
