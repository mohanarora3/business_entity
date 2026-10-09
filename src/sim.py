"""String similarity functions.

Uses rapidfuzz (MIT, fast C++) when installed. If it is not installed, falls back to
pure-Python implementations with the same 0-100 scale so the pipeline still runs
(slower). Always install rapidfuzz for the real dataset: `pip install rapidfuzz`.
"""

from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

    HAVE_RAPIDFUZZ = True

    def ratio(a, b):
        return fuzz.ratio(a, b)

    def partial_ratio(a, b):
        return fuzz.partial_ratio(a, b)

    def token_sort_ratio(a, b):
        return fuzz.token_sort_ratio(a, b)

    def token_set_ratio(a, b):
        return fuzz.token_set_ratio(a, b)

    def jaro_winkler(a, b):
        return 100.0 * JaroWinkler.normalized_similarity(a, b)

except ImportError:  # ---------------------------------------------- pure-Python fallback
    HAVE_RAPIDFUZZ = False

    def ratio(a, b):
        if not a and not b:
            return 100.0
        if not a or not b:
            return 0.0
        return 100.0 * SequenceMatcher(None, a, b, autojunk=False).ratio()

    def partial_ratio(a, b):
        if not a or not b:
            return 0.0
        short, long_ = (a, b) if len(a) <= len(b) else (b, a)
        n = len(short)
        best = 0.0
        for i in range(0, len(long_) - n + 1):
            r = ratio(short, long_[i:i + n])
            if r > best:
                best = r
                if best == 100.0:
                    break
        return best

    def token_sort_ratio(a, b):
        return ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))

    def token_set_ratio(a, b):
        ta, tb = set(a.split()), set(b.split())
        if not ta or not tb:
            return 0.0
        inter = " ".join(sorted(ta & tb))
        da = " ".join(sorted(ta - tb))
        db = " ".join(sorted(tb - ta))
        c1 = (inter + " " + da).strip()
        c2 = (inter + " " + db).strip()
        if inter and (not da or not db):
            return 100.0
        return max(ratio(inter, c1), ratio(inter, c2), ratio(c1, c2))

    def jaro_winkler(a, b, p=0.1):
        if a == b:
            return 100.0 if a else 0.0
        la, lb = len(a), len(b)
        if not la or not lb:
            return 0.0
        rng = max(la, lb) // 2 - 1
        ma, mb = [False] * la, [False] * lb
        m = 0
        for i in range(la):
            lo, hi = max(0, i - rng), min(i + rng + 1, lb)
            for j in range(lo, hi):
                if not mb[j] and a[i] == b[j]:
                    ma[i] = mb[j] = True
                    m += 1
                    break
        if not m:
            return 0.0
        t, k = 0, 0
        for i in range(la):
            if ma[i]:
                while not mb[k]:
                    k += 1
                if a[i] != b[k]:
                    t += 1
                k += 1
        jaro = (m / la + m / lb + (m - t / 2) / m) / 3
        pre = 0
        for x, y in zip(a[:4], b[:4]):
            if x != y:
                break
            pre += 1
        return 100.0 * (jaro + pre * p * (1 - jaro))


def jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def overlap(a: str, b: str) -> float:
    """|A ∩ B| / min(|A|, |B|) — robust when one side is a partial address."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))
