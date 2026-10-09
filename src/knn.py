"""Nearest-neighbour search.

* dense_topk  : FAISS inner-product search on L2-normalised embeddings (= cosine).
                Exact (IndexFlatIP) for small pools, IVF (approximate, much faster) for big ones.
                Falls back to chunked numpy if FAISS is not installed.
* sparse_topk : exact cosine top-k between two TF-IDF matrices (used inside one region,
                where the pool is small, so brute force is cheap).
"""

import numpy as np
import scipy.sparse as sp

import config as C

import importlib.util

HAVE_FAISS = importlib.util.find_spec("faiss") is not None   # imported lazily, when first used


def backend() -> str:
    return "FAISS" if HAVE_FAISS else "numpy (FAISS not installed)"


def dense_topk(Q: np.ndarray, X: np.ndarray, k: int):
    """For each row of Q, the k rows of X with the highest dot product. Returns (idx, score);
    idx is -1 where fewer than k results exist."""
    nq, nx = len(Q), len(X)
    k = min(k, nx)
    if nq == 0 or nx == 0 or k == 0:
        return np.zeros((nq, 0), np.int64), np.zeros((nq, 0), np.float32)
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    X = np.ascontiguousarray(X, dtype=np.float32)
    if not HAVE_FAISS:
        idx_all = np.empty((nq, k), np.int64)
        sc_all = np.empty((nq, k), np.float32)
        step = max(1, int(5e7 // nx))
        for s in range(0, nq, step):
            S = Q[s:s + step] @ X.T
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            sc = np.take_along_axis(S, idx, axis=1)
            order = np.argsort(-sc, axis=1)
            idx_all[s:s + step] = np.take_along_axis(idx, order, axis=1)
            sc_all[s:s + step] = np.take_along_axis(sc, order, axis=1)
        return idx_all, sc_all
    import faiss
    d = X.shape[1]
    if nx <= C.FAISS_EXACT_MAX:
        index = faiss.IndexFlatIP(d)
    else:
        nlist = int(min(65536, max(256, 4 * np.sqrt(nx))))
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(C.SEED)
        train = X[rng.choice(nx, size=min(nx, nlist * 50), replace=False)]
        index.train(train)
        index.nprobe = C.FAISS_NPROBE
    index.add(X)
    sc, idx = index.search(Q, k)
    return idx.astype(np.int64), sc.astype(np.float32)


def sparse_topk(A: sp.csr_matrix, B: sp.csr_matrix, k: int):
    """For each row of A, the k most cosine-similar rows of B (TF-IDF rows are L2-normalised)."""
    nA, nB = A.shape[0], B.shape[0]
    k = min(k, nB)
    if nA == 0 or k == 0:
        return np.zeros((nA, 0), np.int64), np.zeros((nA, 0), np.float32)
    chunk = max(16, min(C.CHUNK, int(5e7 // max(nB, 1))))
    BT = B.T.tocsc()
    idx_all = np.zeros((nA, k), dtype=np.int64)
    sc_all = np.zeros((nA, k), dtype=np.float32)
    for s in range(0, nA, chunk):
        S = (A[s:s + chunk] @ BT).toarray()
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sc = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-sc, axis=1)
        idx_all[s:s + chunk] = np.take_along_axis(idx, order, axis=1)
        sc_all[s:s + chunk] = np.take_along_axis(sc, order, axis=1)
    return idx_all, sc_all
