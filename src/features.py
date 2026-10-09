"""Pair features for the matching model.

All features are country-agnostic (no country one-hot), so the model transfers to
countries never seen in training (France in the test set).
"""

from collections import Counter

import numpy as np
import pandas as pd

import sim

FEATURES = None  # filled on first call; the ordered list of model input columns


def _pairwise(a_list, b_list, fn):
    return np.fromiter((fn(a, b) for a, b in zip(a_list, b_list)), dtype=np.float32, count=len(a_list))


def _set_feats(a_list, b_list):
    """(both_present, jaccard, conflict) for space-separated token sets such as house numbers."""
    both, jac, conf = [], [], []
    for a, b in zip(a_list, b_list):
        ta, tb = set(a.split()), set(b.split())
        if ta and tb:
            both.append(1)
            j = len(ta & tb) / len(ta | tb)
            jac.append(j)
            conf.append(1 if not (ta & tb) else 0)
        else:
            both.append(0)
            jac.append(-1)
            conf.append(0)
    return (np.array(both, np.float32), np.array(jac, np.float32), np.array(conf, np.float32))


def _second_best(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """For every row, the 2nd-highest value in its group (0 if the group has one row)."""
    d = pd.DataFrame({"g": groups, "v": values})
    d = d.sort_values(["g", "v"], ascending=[True, False])
    d["r"] = d.groupby("g").cumcount()
    second = d[d.r == 1].set_index("g")["v"]
    return pd.Series(groups).map(second).fillna(0.0).values


COLS = ["name_core", "name_norm", "name_alt", "name_compact", "name_initials", "name_legal",
        "name_first", "name_digits", "name_web", "name_maxdf", "addr_norm", "addr_words",
        "addr_numbers", "addr_postal", "addr_landmark", "addr_tail", "addr_house", "addr_street",
        "country_norm", "source"]


def _name_maxdf(s1n: pd.DataFrame, pooln: pd.DataFrame):
    """log document frequency of the most common word of each name, counted over this split.
    Placeholder / brand names ("Halocira", "Zephsolpyra") have only rare words."""
    df_tok = Counter()
    for col in (s1n.name_core, pooln.name_core):
        for x in col:
            df_tok.update(set(x.split()))

    def maxdf(names):
        return np.array([np.log1p(max((df_tok[t] for t in x.split()), default=0)) for x in names], np.float32)
    return maxdf(s1n.name_core.values), maxdf(pooln.name_core.values)


def _pair_chunk(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """All features that depend only on the two records of a pair (run in parallel)."""
    f = pd.DataFrame(index=range(len(A)))
    # ---- name
    an, bn = A.name_core.tolist(), B.name_core.tolist()
    f["n_ratio"] = _pairwise(an, bn, sim.ratio)
    f["n_jw"] = _pairwise(an, bn, sim.jaro_winkler)
    f["n_tsort"] = _pairwise(an, bn, sim.token_sort_ratio)
    f["n_tset"] = _pairwise(an, bn, sim.token_set_ratio)
    f["n_partial"] = _pairwise(an, bn, sim.partial_ratio)
    f["n_jacc"] = _pairwise(an, bn, sim.jaccard)
    f["n_overlap"] = _pairwise(an, bn, sim.overlap)
    f["n_full_ratio"] = _pairwise(A.name_norm.tolist(), B.name_norm.tolist(), sim.ratio)
    f["n_compact_eq"] = (A.name_compact.values == B.name_compact.values).astype(np.float32)
    f["n_compact_jw"] = _pairwise(A.name_compact.tolist(), B.name_compact.tolist(), sim.jaro_winkler)
    f["n_first_eq"] = (A.name_first.values == B.name_first.values).astype(np.float32)
    # DBA / trade names: best similarity over all name variants
    best = f["n_tsort"].values.copy()
    for x, y in ((A.name_alt, B.name_core), (A.name_core, B.name_alt), (A.name_alt, B.name_alt)):
        s = _pairwise(x.tolist(), y.tolist(), lambda a, b: sim.token_sort_ratio(a, b) if a and b else 0.0)
        best = np.maximum(best, s)
    f["n_best_variant"] = best
    # acronyms: "IBM" vs "International Business Machines"
    acr = []
    for ca, cb, ia, ib in zip(A.name_compact, B.name_compact, A.name_initials, B.name_initials):
        acr.append(1.0 if (len(ia) >= 2 and ia == cb) or (len(ib) >= 2 and ib == ca) else 0.0)
    f["n_acronym"] = np.array(acr, np.float32)
    f["n_len_a"] = A.name_core.str.len().values.astype(np.float32)
    f["n_len_b"] = B.name_core.str.len().values.astype(np.float32)
    f["n_len_diff"] = np.abs(f.n_len_a - f.n_len_b)
    f["n_ntok_diff"] = np.abs(A.name_core.str.count(" ").values - B.name_core.str.count(" ").values).astype(np.float32)
    # digits inside names ("Shop 7" vs "Shop 9") are strong negative evidence
    f["n_dig_both"], f["n_dig_jacc"], f["n_dig_conflict"] = _set_feats(A.name_digits.tolist(), B.name_digits.tolist())
    # legal form
    la, lb = A.name_legal.tolist(), B.name_legal.tolist()
    f["legal_both"], f["legal_jacc"], f["legal_conflict"] = _set_feats(la, lb)

    # ---- address
    aa, ba = A.addr_norm.tolist(), B.addr_norm.tolist()
    f["a_ratio"] = _pairwise(aa, ba, sim.ratio)
    f["a_tset"] = _pairwise(aa, ba, sim.token_set_ratio)
    f["a_tsort"] = _pairwise(aa, ba, sim.token_sort_ratio)
    f["a_jacc"] = _pairwise(A.addr_words.tolist(), B.addr_words.tolist(), sim.jaccard)
    f["a_overlap"] = _pairwise(A.addr_words.tolist(), B.addr_words.tolist(), sim.overlap)
    f["a_tail_tset"] = _pairwise(A.addr_tail.tolist(), B.addr_tail.tolist(),
                                 lambda a, b: sim.token_set_ratio(a, b) if a and b else -1.0)
    f["a_num_both"], f["a_num_jacc"], f["a_num_conflict"] = _set_feats(A.addr_numbers.tolist(), B.addr_numbers.tolist())
    pa, pb = A.addr_postal.values, B.addr_postal.values
    both_p = (pa != "") & (pb != "")
    f["postal_both"] = both_p.astype(np.float32)
    f["postal_eq"] = np.where(both_p, (pa == pb).astype(np.float32), -1.0).astype(np.float32)
    f["postal_prefix3"] = np.where(both_p, np.array([x[:3] == y[:3] for x, y in zip(pa, pb)], np.float32), -1.0)
    f["landmark_tset"] = _pairwise(A.addr_landmark.tolist(), B.addr_landmark.tolist(),
                                   lambda a, b: sim.token_set_ratio(a, b) if a and b else -1.0)
    f["a_len_a"] = A.addr_norm.str.len().values.astype(np.float32)
    f["a_len_b"] = B.addr_norm.str.len().values.astype(np.float32)
    f["a_empty_any"] = ((A.addr_norm.values == "") | (B.addr_norm.values == "")).astype(np.float32)

    # ---- house number: branches of a chain sit at nearby numbers on the same street,
    #      while the same business is often written with a dropped / extra digit
    ha, hb = A.addr_house.values, B.addr_house.values
    both_h = (ha != "") & (hb != "")
    na = np.array([int(x[:9]) if x else 0 for x in ha], dtype=np.float64)
    nb = np.array([int(x[:9]) if x else 0 for x in hb], dtype=np.float64)
    f["hn_both"] = both_h.astype(np.float32)
    f["hn_eq"] = np.where(both_h, (ha == hb).astype(np.float32), -1.0).astype(np.float32)
    f["hn_diff"] = np.where(both_h, np.sign(nb - na) * np.log1p(np.abs(nb - na)), 0.0).astype(np.float32)
    f["hn_absdiff"] = np.where(both_h, np.log1p(np.abs(nb - na)), -1.0).astype(np.float32)
    f["hn_reldiff"] = np.where(both_h, np.abs(nb - na) / np.maximum(np.maximum(na, nb), 1), -1.0).astype(np.float32)
    f["hn_affix"] = np.array([1.0 if a and b and a != b and (a.endswith(b) or b.endswith(a) or a.startswith(b)
                                                           or b.startswith(a)) else 0.0
                              for a, b in zip(ha, hb)], np.float32)
    f["hn_ratio"] = _pairwise(ha.tolist(), hb.tolist(), lambda a, b: sim.ratio(a, b) if a and b else -1.0)
    sa, sb = A.addr_street.values, B.addr_street.values
    both_s = (sa != "") & (sb != "")
    f["street_eq"] = np.where(both_s, (sa == sb).astype(np.float32), -1.0).astype(np.float32)
    f["street_ratio"] = _pairwise(sa.tolist(), sb.tolist(), lambda a, b: sim.ratio(a, b) if a and b else -1.0)
    f["same_spot"] = (both_h & (ha == hb) & (sa == sb)).astype(np.float32)
    f["same_spot_namediff"] = f["same_spot"] * (100.0 - f["n_tsort"]) / 100.0

    # ---- placeholder / brand / website names ("Halocira", "dclake.com") at the same address:
    #      how common the name's words are across all records of this split
    f["a_name_maxdf"] = A.name_maxdf.values
    f["b_name_maxdf"] = B.name_maxdf.values
    f["b_name_ntok"] = (B.name_core.str.count(" ").values + (B.name_core.values != "")).astype(np.float32)
    wa, wb = A.name_web.values, B.name_web.values
    f["web_any"] = ((wa != "") | (wb != "")).astype(np.float32)
    f["web_sim"] = np.maximum(
        _pairwise(wb.tolist(), A.name_compact.tolist(), lambda w, c: sim.jaro_winkler(w, c) if w and c else -1.0),
        _pairwise(wa.tolist(), B.name_compact.tolist(), lambda w, c: sim.jaro_winkler(w, c) if w and c else -1.0))
    f["web_initials"] = np.array([1.0 if (w and len(ini) >= 2 and w.startswith(ini)) else 0.0
                                  for w, ini in zip(wb, A.name_initials.values)], np.float32)

    # ---- other
    f["country_eq"] = (A.country_norm.values == B.country_norm.values).astype(np.float32)
    f["is_s3"] = (B.source.values == "S3").astype(np.float32)

    return f


def build_features(pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame) -> pd.DataFrame:
    global FEATURES
    from joblib import Parallel, delayed

    import config as C
    from blocking import N_BITS

    i, j = pairs.i.values, pairs.j.values
    base = [c for c in COLS if c not in ("name_maxdf", "source")]
    S1, P = s1n[base].copy(), pooln[base].copy()
    S1["name_maxdf"], P["name_maxdf"] = _name_maxdf(s1n, pooln)
    S1["source"] = "S1"
    P["source"] = pooln["source"].values

    step = C.FEATURE_CHUNK
    jobs = (delayed(_pair_chunk)(S1.iloc[i[s:s + step]].reset_index(drop=True),
                                 P.iloc[j[s:s + step]].reset_index(drop=True))
            for s in range(0, len(pairs), step))
    n_jobs = 1 if len(pairs) <= step else C.N_JOBS
    parts = Parallel(n_jobs=n_jobs)(jobs)
    pf = pd.concat(parts, ignore_index=True)

    f = pd.DataFrame(index=range(len(pairs)))
    # ---- blocking cosines (TF-IDF + embedding)
    for c in ("cos_name", "cos_full", "cos_addr", "emb_name", "rev_rank", "blockers"):
        f[c] = pairs[c].values
    for b in range(N_BITS):
        f[f"blk_{b}"] = ((pairs.blockers.values >> b) & 1).astype(np.float32)
    f = pd.concat([f, pf], axis=1)
    f["emb_x_addr"] = f.emb_name * f.cos_addr

    # ---- context features: how this candidate compares with the others
    key_i = pd.Series(i)
    key_j = pd.Series(j)
    combo = (f.cos_name + f.cos_full + f.n_tsort / 100 + f.a_tset / 100
             + np.clip(f.emb_name.values, 0, 1)).values
    f["combo"] = combo
    g_i = pd.Series(combo).groupby(key_i)
    f["rank_in_s1"] = g_i.rank(ascending=False, method="min").values.astype(np.float32)
    f["gap_to_best_s1"] = (g_i.transform("max") - combo).values.astype(np.float32)
    f["n_cand_s1"] = g_i.transform("size").values.astype(np.float32)
    f["gap_to_2nd_s1"] = (combo - _second_best(combo, i)).astype(np.float32)
    g_j = pd.Series(combo).groupby(key_j)
    f["rank_in_pool"] = g_j.rank(ascending=False, method="min").values.astype(np.float32)
    f["gap_to_best_pool"] = (g_j.transform("max") - combo).values.astype(np.float32)
    f["n_s1_for_pool"] = g_j.transform("size").values.astype(np.float32)
    # how many pool records share this exact compact name (chains / franchises)
    name_freq = pooln.name_compact.map(pooln.name_compact.value_counts())
    f["pool_name_freq"] = name_freq.iloc[j].values.astype(np.float32)

    FEATURES = [c for c in f.columns]
    return f
