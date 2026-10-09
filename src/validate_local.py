"""Our own copy of the submission rules, run automatically after prediction.
Still run the official utils/validate_submission.py before every upload."""

import sys
from pathlib import Path

from io_utils import read_tsv


def _check(path, col, s1_ids, valid_ids, label):
    problems = []
    df = read_tsv(path)
    if list(df.columns) != ["source1_entity_id", col]:
        problems.append(f"{label}: header must be source1_entity_id<TAB>{col}, got {list(df.columns)}")
        return problems, {}
    if df.source1_entity_id.duplicated().any():
        problems.append(f"{label}: duplicate source1_entity_id rows")
    missing = set(s1_ids) - set(df.source1_entity_id)
    extra = set(df.source1_entity_id) - set(s1_ids)
    if missing:
        problems.append(f"{label}: {len(missing)} Source 1 ids missing (e.g. {sorted(missing)[:3]})")
    if extra:
        problems.append(f"{label}: {len(extra)} unknown Source 1 ids")
    lists = {}
    for s, ids in zip(df.source1_entity_id, df[col]):
        items = [x for x in ids.split(",") if x] if ids else []
        if len(items) != len(set(items)):
            problems.append(f"{label}: duplicate ids in list for {s}")
        bad = [x for x in items if x not in valid_ids]
        if bad:
            problems.append(f"{label}: {s} lists ids not in test Source 2/3: {bad[:3]}")
        lists[s] = set(items)
    return problems, lists


def validate(matching, candidates, test_dir) -> bool:
    test_dir = Path(test_dir)
    s1_ids = read_tsv(test_dir / "test_source1.tsv").entity_id.tolist()
    valid = set(read_tsv(test_dir / "test_source2.tsv").entity_id) | set(read_tsv(test_dir / "test_source3.tsv").entity_id)
    p1, m = _check(matching, "matched_entity_ids", s1_ids, valid, "matching_results")
    p2, c = _check(candidates, "candidate_entity_ids", s1_ids, valid, "candidate_pairs")
    problems = p1 + p2
    not_cand = sum(len(m[s] - c.get(s, set())) for s in m)
    if not_cand:
        problems.append(f"{not_cand} matched ids are not in candidate_pairs (pipeline bug)")
    if problems:
        print("[validate] FAIL")
        for k, p in enumerate(problems, 1):
            print(f"  {k}. {p}")
        return False
    print("[validate] PASS — both files follow every rule we know of")
    return True


if __name__ == "__main__":
    ok = validate(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0 if ok else 1)
