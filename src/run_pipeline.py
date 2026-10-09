"""One command, end to end: train -> tune -> predict -> validate.

    python src/run_pipeline.py --data-dir ../../dataset --out-dir ../../output
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # faiss + lightgbm on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import time

import config as C
import predict
import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--out-dir", default=str(C.OUTPUT_DIR))
    ap.add_argument("--sample", type=float, default=None,
                    help="train on this fraction of regions only (e.g. 0.02) and skip test prediction")
    a = ap.parse_args()
    t0 = time.time()
    train.main(a.data_dir, a.sample)
    if a.sample:
        print("[sample] sampled run: skipped test prediction (run without --sample for a submission)")
    else:
        predict.main(a.data_dir, a.out_dir)
    print(f"[done] total {time.time() - t0:.0f}s")
