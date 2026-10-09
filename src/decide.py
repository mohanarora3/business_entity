"""Turning pair probabilities into final match lists, tuned for macro-averaged F0.5.

F0.5 weights precision 2x over recall, and singletons (Source 1 entities with no true
match) score 1.0 only when we predict an EMPTY list. So the decision rule is deliberately
conservative and is tuned directly on the metric, not on log-loss or accuracy.

Rule (all three knobs are searched on out-of-fold predictions):
  keep candidate if  p >= t                       (absolute threshold)
               and   p >= r * best_p_of_that_S1   (drop weak extras next to a strong match)
  if one_to_one: a Source 2/3 record may be assigned to at most ONE Source 1 entity —
                 the one with the highest probability. Source 1 is deduplicated, so a
                 record that matches two Source 1 entities is always at least one error.
"""

import itertools

import numpy as np
import pandas as pd

import config as C

BETA2 = 0.25  # beta = 0.5


def f05(pred: set, true: set) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: dict, truth: dict, s1_ids) -> float:
    return float(np.mean([f05(set(pred.get(s, ())), set(truth.get(s, ()))) for s in s1_ids]))


def apply_rule(scored: pd.DataFrame, t: float, r: float, one_to_one: bool) -> dict:
    """scored: columns s1_id, pool_id, p. Returns {s1_id: [pool ids]}."""
    d = scored[scored.p >= t]
    if r > 0 and len(d):
        best = d.groupby("s1_id").p.transform("max")
        d = d[d.p >= r * best]
    if one_to_one and len(d):
        d = d.sort_values("p", ascending=False).drop_duplicates("pool_id", keep="first")
    out = {}
    for s, pid in zip(d.s1_id.values, d.pool_id.values):
        out.setdefault(s, []).append(pid)
    return out


def tune(scored: pd.DataFrame, truth: dict, s1_ids, verbose=True) -> dict:
    """Grid-search (t, r, one_to_one) for the best macro F0.5."""
    best = None
    for t, r, o2o in itertools.product(C.THRESHOLDS, C.RELATIVE, (True, False)):
        score = macro_f05(apply_rule(scored, t, r, o2o), truth, s1_ids)
        if best is None or score > best["score"] + 1e-9:
            best = {"t": t, "r": r, "one_to_one": o2o, "score": score}
    if verbose:
        print(f"[decide] best rule: p >= {best['t']}, p >= {best['r']} x best, "
              f"one_to_one={best['one_to_one']}  ->  macro F0.5 = {best['score']:.4f}")
    return best


def detail_report(pred: dict, truth: dict, s1_ids, country_of: dict):
    """Precision/recall/F0.5 split into singletons vs matched entities and by country."""
    rows = []
    for s in s1_ids:
        p, t = set(pred.get(s, ())), set(truth.get(s, ()))
        rows.append((country_of.get(s, "?"), bool(t), f05(p, t), len(p & t), len(p), len(t)))
    df = pd.DataFrame(rows, columns=["country", "has_match", "f", "tp", "npred", "ntrue"])
    print(f"           macro F0.5 = {df.f.mean():.4f} | pair precision = "
          f"{df.tp.sum() / max(df.npred.sum(), 1):.4f} | pair recall = {df.tp.sum() / max(df.ntrue.sum(), 1):.4f}")
    for flag, g in df.groupby("has_match"):
        label = "entities with matches" if flag else "singletons"
        print(f"           {label:>22}: n={len(g):,}  F0.5={g.f.mean():.4f}")
    for c, g in df.groupby("country"):
        print(f"           {c:>22}: n={len(g):,}  F0.5={g.f.mean():.4f}")
    return df
