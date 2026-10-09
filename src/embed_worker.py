"""Runs the embedding model in its own process.

Kept separate on purpose: on macOS, PyTorch and FAISS / LightGBM each ship their own OpenMP
runtime and crash (segmentation fault) when loaded into the same process.

    python embed_worker.py <texts.txt> <out.npy> <model name> <batch size>
"""

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


def main(inp, out, model_name, batch):
    import torch
    from sentence_transformers import SentenceTransformer

    device = os.environ.get("EMB_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(inp, encoding="utf-8") as f:
        texts = f.read().split("\n")
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model.half()
    print(f"[embed] model {model_name} on {device}: {len(texts):,} names", flush=True)
    E = model.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=len(texts) > 50_000)
    np.save(out, E.astype(np.float16))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
