"""Candidate generation (blocking), scalable version.

For every Source 1 record, collect a short list of plausible Source 2/3 records from the
UNION of several independent blockers, so that a record missed by one blocker is caught
by another:

  bit 0  char n-gram TF-IDF on the cleaned core name    } searched only inside the same
  bit 1  word TF-IDF on core name + address             } region (country + first 3 postal
  bit 2  char n-gram TF-IDF on the address              } digits), in parallel on all cores
  bit 3  same postal code AND a shared distinctive name token
  bit 4  exact compact core name ("abctraders" == "abc traders")
  bit 5  same house number + same street token + same country
  bit 6  multilingual name embedding, FAISS search over the whole country
         (catches other scripts, typos in the postal code, missing postal codes)

The union is cut to MAX_CANDIDATES per Source 1 record by a cheap pre-score.
"""

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer

import config as C
import embed
import knn

N_BITS = 7


def region_of(df: pd.DataFrame) -> np.ndarray:
    """country + first 3 postal digits (US ZIP3 / India PIN3 / French departement);
    without a postal code: country + last address word (usually city or state)."""
    pc = df["addr_postal"].str[:3].values.astype(str)
    tail = df["addr_words"].str.split().str[-1].fillna("").values.astype(str)
    key = np.where(pc != "", pc, np.char.add("w:", tail))
    return np.char.add(np.char.add(df["country_norm"].values.astype(str), "|"), key.astype(str))


def _groups(keys: np.ndarray) -> dict:
    """key -> array of row positions."""
    s = pd.Series(np.arange(len(keys))).groupby(keys, sort=False)
    return {k: v.values for k, v in s}


def _region_task(A, B, k, ia, jb, min_score):
    idx, sc = knn.sparse_topk(A, B, k)
    ok = sc > min_score
    rows = np.repeat(ia, idx.shape[1])[ok.ravel()]
    cols = jb[idx[ok]]
    return rows, cols


def _region_rev_task(A, B, k, ia, jb):
    """reverse direction: for pool rows (A) the k nearest Source 1 rows (B)."""
    idx, _ = knn.sparse_topk(A, B, k)
    out = np.full((len(ia), k), -1, np.int64)
    out[:, :idx.shape[1]] = jb[idx]
    return ia, out


class Blocker:
    """Fits everything that depends on the full record sets once (TF-IDF, embeddings,
    regions); candidates can then be generated for all or part of Source 1."""

    def __init__(self, s1n: pd.DataFrame, pooln: pd.DataFrame):
        t0 = time.time()
        self.s1n, self.pooln = s1n, pooln
        specs = {
            "name_char": ("block_name", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)),
            "full_word": ("block_full", dict(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                             token_pattern=r"(?u)\b\w+\b")),
            "addr_char": ("addr_norm", dict(analyzer="char_wb", ngram_range=(3, 4), min_df=1, sublinear_tf=True)),
        }
        self.s1, self.pool = {}, {}
        for key, (col, kw) in specs.items():
            vec = TfidfVectorizer(dtype=np.float32, **kw)
            corpus = pd.concat([s1n[col], pooln[col]]).fillna("").replace("", "_empty_")
            vec.fit(corpus)
            self.s1[key] = vec.transform(s1n[col].replace("", "_empty_")).tocsr()
            self.pool[key] = vec.transform(pooln[col].replace("", "_empty_")).tocsr()
        self.reg_s1, self.reg_pool = region_of(s1n), region_of(pooln)
        self.pool_regions = _groups(self.reg_pool)
        self.s1_regions = _groups(self.reg_s1)
        # secondary regions (country + a city/state word) for records WITHOUT a postal code,
        # so "..., Mumbai, Maharashtra" still meets "..., Mumbai, MH 400024"
        self.s1_has_postal = s1n["addr_postal"].values != ""
        pr, pk = _sec_keys(pooln)
        nop = pooln["addr_postal"].values[pr] == ""
        self.sec_pool_all = {k: v for k, v in _groups_of(pr, pk).items() if len(v) <= C.SEC_REGION_MAX}
        self.sec_pool_nop = {k: v for k, v in _groups_of(pr[nop], pk[nop]).items() if len(v) <= C.SEC_REGION_MAX}
        self.s1_sec_rows, self.s1_sec_keys = _sec_keys(s1n)
        print(f"[blocking] TF-IDF fitted, {len(self.pool_regions):,} pool regions ({time.time() - t0:.0f}s)")

        self.emb_s1 = self.emb_pool = None
        if embed.available():
            self.emb_s1 = embed.encode(s1n["business_name"].values, "s1")
            self.emb_pool = embed.encode(pooln["business_name"].values, "pool")
        else:
            print("[blocking] embeddings OFF (install sentence-transformers to enable)")

        # dictionary blockers
        self.by_pt = {}
        for j, (p, name) in enumerate(zip(pooln["addr_postal"].values, pooln["name_core"].values)):
            if p:
                for t in set(name.split()):
                    if len(t) >= 3:
                        self.by_pt.setdefault((p, t), []).append(j)
        self.by_compact = {}
        for j, c in enumerate(pooln["name_compact"].values):
            if c:
                self.by_compact.setdefault(c, []).append(j)
        self.by_spot = {}
        for j, k in enumerate(_spot(pooln)):
            if k:
                self.by_spot.setdefault(k, []).append(j)
        print(f"[blocking] indexes ready in {time.time() - t0:.0f}s | similarity search: {knn.backend()}")

    # ------------------------------------------------------------------ candidates
    def candidates(self, rows=None, verbose=True) -> pd.DataFrame:
        """Candidate pairs for Source 1 rows `rows` (default: all). Columns: i, j, blockers,
        cos_name, cos_full, cos_addr, emb_name, rev_rank."""
        t0 = time.time()
        s1n, pooln = self.s1n, self.pooln
        n1 = len(s1n)
        rows = np.arange(n1) if rows is None else np.asarray(rows)
        nB = len(pooln)
        I, J, BITS = [], [], []

        def push(i, j, bit):
            if len(i):
                I.append(np.asarray(i, np.int64))
                J.append(np.asarray(j, np.int64))
                BITS.append(np.full(len(i), bit, np.int8))

        # bits 0-2: TF-IDF inside each region, all cores
        reg_rows = _groups(self.reg_s1[rows])
        in_rows = np.zeros(n1, bool)
        in_rows[rows] = True
        m = in_rows[self.s1_sec_rows]
        sec_r, sec_k = self.s1_sec_rows[m], self.s1_sec_keys[m]
        has_p = self.s1_has_postal[sec_r]
        sec_tasks = [(self.sec_pool_all, _groups_of(sec_r[~has_p], sec_k[~has_p])),   # S1 has no postal
                     (self.sec_pool_nop, _groups_of(sec_r[has_p], sec_k[has_p]))]      # pool has no postal
        for bit, (key, k) in enumerate([("name_char", C.K_NAME_CHAR), ("full_word", C.K_FULL_WORD),
                                        ("addr_char", C.K_ADDR_CHAR)]):
            A, B = self.s1[key], self.pool[key]
            tasks = []
            for reg, local in reg_rows.items():
                jb = self.pool_regions.get(reg)
                if jb is None:
                    continue
                ia = rows[local]
                tasks.append(delayed(_region_task)(A[ia], B[jb], k, ia, jb, 0.05))
            for pool_groups, s1_groups in sec_tasks:
                for reg, ia in s1_groups.items():
                    jb = pool_groups.get(reg)
                    if jb is not None:
                        tasks.append(delayed(_region_task)(A[ia], B[jb], k, ia, jb, 0.05))
            for i, j in Parallel(n_jobs=C.N_JOBS, batch_size=8)(tasks):
                push(i, j, bit)
            if verbose:
                print(f"[blocking]   {key}: done ({time.time() - t0:.0f}s)")

        # bit 3: postal + shared name token
        postal, core = s1n["addr_postal"].values, s1n["name_core"].values
        ii, jj = [], []
        for i in rows:
            p = postal[i]
            if not p:
                continue
            seen = set()
            for t in set(core[i].split()):
                if len(t) >= 3:
                    seen.update(self.by_pt.get((p, t), ()))
            for j in sorted(seen)[: C.POSTAL_BLOCK_CAP]:
                ii.append(i)
                jj.append(j)
        push(ii, jj, 3)
        # bit 4: exact compact name
        ii, jj = [], []
        comp = s1n["name_compact"].values
        for i in rows:
            for j in self.by_compact.get(comp[i], ())[:50]:
                ii.append(i)
                jj.append(j)
        push(ii, jj, 4)
        # bit 5: same house number + street
        ii, jj = [], []
        spots = _spot(s1n)
        for i in rows:
            if spots[i]:
                for j in self.by_spot.get(spots[i], ())[: C.SPOT_BLOCK_CAP]:
                    ii.append(i)
                    jj.append(j)
        push(ii, jj, 5)

        # bit 6: FAISS on name embeddings, per country
        if self.emb_s1 is not None:
            c1 = s1n["country_norm"].values[rows]
            cp = pooln["country_norm"].values
            for c in pd.unique(c1):
                q = rows[c1 == c]
                x = np.flatnonzero(cp == c)
                if len(x) == 0:
                    continue
                idx, sc = knn.dense_topk(self.emb_s1.rows(q), self.emb_pool.rows(x), C.K_EMB_NAME)
                ok = (idx >= 0) & (sc > C.EMB_MIN_SCORE)
                push(np.repeat(q, idx.shape[1])[ok.ravel()], x[idx[ok]], 6)
            if verbose:
                print(f"[blocking]   embeddings (FAISS): done ({time.time() - t0:.0f}s)")

        # union with a bitmask of which blockers found each pair
        I, J, BITS = np.concatenate(I), np.concatenate(J), np.concatenate(BITS)
        key = I * nB + J
        uniq = np.unique(key)
        bits = np.zeros(len(uniq), np.int16)
        for b in range(N_BITS):
            kb = np.unique(key[BITS == b])
            bits[np.searchsorted(uniq, kb)] |= np.int16(1 << b)
        del I, J, BITS, key
        pairs = pd.DataFrame({"i": uniq // nB, "j": uniq % nB, "blockers": bits})
        pi, pj = pairs.i.values, pairs.j.values
        pairs["cos_name"] = rowdot(self.s1["name_char"], self.pool["name_char"], pi, pj)
        pairs["cos_full"] = rowdot(self.s1["full_word"], self.pool["full_word"], pi, pj)
        pairs["cos_addr"] = rowdot(self.s1["addr_char"], self.pool["addr_char"], pi, pj)
        pairs["emb_name"] = (self.emb_s1.pair_cos(self.emb_pool, pi, pj) if self.emb_s1 is not None
                             else np.full(len(pairs), -1.0, np.float32))

        # cap per Source 1 record by a cheap pre-score
        emb_scaled = np.clip((pairs.emb_name.values - 0.5) / 0.5, 0, 1)
        pre = (np.maximum.reduce([pairs.cos_name.values, pairs.cos_full.values, emb_scaled])
               + 0.3 * pairs.cos_addr.values + 0.3 * ((bits >> 5) & 1))
        order = np.lexsort((-pre, pi))
        pi_sorted = pi[order]
        starts = np.r_[0, np.flatnonzero(np.diff(pi_sorted)) + 1]
        rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
        pairs = pairs.iloc[np.sort(order[rank < C.MAX_CANDIDATES])].reset_index(drop=True)

        # reverse direction: rank of this S1 record among the pool record's nearest S1 records
        pairs["rev_rank"] = self._rev_rank(pairs)
        if verbose:
            per = pairs.groupby("i").size()
            print(f"[blocking] {len(pairs):,} candidate pairs | {per.mean():.1f} per Source 1 record "
                  f"(max {per.max()}) | {len(rows) - per.shape[0]} records with no candidates "
                  f"| {time.time() - t0:.0f}s")
        return pairs

    def _rev_rank(self, pairs):
        K = C.K_REVERSE
        js = np.unique(pairs.j.values)
        A, B = self.pool["name_char"], self.s1["name_char"]
        tasks = []
        for reg, local in _groups(self.reg_pool[js]).items():
            ib = self.s1_regions.get(reg)
            if ib is None:
                continue
            ja = js[local]
            tasks.append(delayed(_region_rev_task)(A[ja], B[ib], K, ja, ib))
        table = np.full((len(js), K), -1, np.int64)
        for ja, out in Parallel(n_jobs=C.N_JOBS, batch_size=8)(tasks):
            table[np.searchsorted(js, ja)] = out
        r = table[np.searchsorted(js, pairs.j.values)]
        hit = r == pairs.i.values[:, None]
        return np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, K + 1).astype(np.float32)


def _groups_of(rows: np.ndarray, keys: np.ndarray) -> dict:
    """key -> array of the given row numbers."""
    if len(rows) == 0:
        return {}
    s = pd.Series(rows).groupby(keys, sort=False)
    return {k: v.values for k, v in s}


def _sec_keys(df):
    """(row, key) for every distinctive word of the address end (city / district / state)."""
    rr, kk = [], []
    for r, (c, tail) in enumerate(zip(df["country_norm"].values, df["addr_tail"].values)):
        for t in set(tail.split()):
            if len(t) >= 3:
                rr.append(r)
                kk.append(f"{c}|{t}")
    return np.array(rr, np.int64), np.array(kk, dtype=object)


def _spot(df):
    return [f"{c}|{h}|{st}" if h and st else "" for c, h, st in
            zip(df["country_norm"].values, df["addr_house"].values, df["addr_street"].values)]


def rowdot(A: sp.csr_matrix, B: sp.csr_matrix, i: np.ndarray, j: np.ndarray, step=200_000):
    """Cosine of row A[i[n]] with row B[j[n]] for every pair n (matrices are L2-normalised)."""
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), step):
        a = A[i[s:s + step]]
        b = B[j[s:s + step]]
        out[s:s + step] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def generate_candidates(s1n: pd.DataFrame, pooln: pd.DataFrame, verbose=True):
    """One-shot call: (pairs, blocker)."""
    b = Blocker(s1n, pooln)
    return b.candidates(verbose=verbose), b


def blocking_report(pairs, s1n, pooln, truth: dict):
    """Recall ceiling of blocking: share of true pairs that survived, overall and per country."""
    s1_ids = s1n["entity_id"].values
    pool_ids = pooln["entity_id"].values
    cand = {}
    for i, j in zip(pairs.i.values, pairs.j.values):
        cand.setdefault(s1_ids[i], set()).add(pool_ids[j])
    pool_set = set(pool_ids)
    rows = []
    for i, s1_id in enumerate(s1_ids):
        t = truth.get(s1_id, set()) & pool_set
        c = cand.get(s1_id, set())
        rows.append((s1n["country_norm"].iat[i], len(t), len(t & c)))
    df = pd.DataFrame(rows, columns=["country", "true", "found"])
    tot = df[["true", "found"]].sum()
    print(f"[blocking] pair recall = {tot.found / max(tot.true, 1):.4f} "
          f"({tot.found:,}/{tot.true:,} true pairs kept)")
    for c, g in df.groupby("country"):
        s = g[["true", "found"]].sum()
        print(f"           {c:>10}: recall {s.found / max(s.true, 1):.4f}  ({s.true:,} true pairs)")
    found_by = {b: int(((pairs.blockers.values >> b) & 1).sum()) for b in range(N_BITS)}
    print(f"[blocking] pairs found per blocker (bit: count): {found_by}")
    return tot.found / max(tot.true, 1)
