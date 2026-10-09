"""Central configuration. Edit values here, not inside the modules."""

from pathlib import Path

# ---------------------------------------------------------------- paths
# Root of the code package (business_entity_resolution/)
PKG_ROOT = Path(__file__).resolve().parents[1]

# Where the competition data lives. The default expects the official layout:
#   <DATA_DIR>/train/train_source1.tsv ... train_ground_truth.tsv
#   <DATA_DIR>/test/test_source1.tsv  ...
# Override on the command line with --data-dir.
DATA_DIR = PKG_ROOT.parent.parent / "dataset"

OUTPUT_DIR = PKG_ROOT.parent.parent / "output"      # matching_results.tsv, candidate_pairs.tsv
ARTIFACT_DIR = PKG_ROOT / "artifacts"                # trained model, decision params, reports

# ---------------------------------------------------------------- columns
ID, NAME, ADDR, COUNTRY = "entity_id", "business_name", "business_address", "country"

# ---------------------------------------------------------------- blocking
# Top-k neighbours taken from each blocker (per Source 1 record).
# Raise these if the blocking-recall report says recall < 0.98.
K_NAME_CHAR = 25      # char n-gram TF-IDF on the cleaned core name
K_FULL_WORD = 25      # word TF-IDF on name + address
K_ADDR_CHAR = 15      # char n-gram TF-IDF on the address
K_REVERSE = 10        # reverse direction (pool record -> its nearest Source 1 records)
POSTAL_BLOCK_CAP = 40 # max candidates added from the "same postal code + shared name token" block
SPOT_BLOCK_CAP = 15   # max candidates added from the "same house number + street" block
MAX_CANDIDATES = 60   # final cap per Source 1 record after union (ordered by pre-score)
CHUNK = 512           # rows per sparse-matmul chunk (lower it if you run out of RAM)
SEC_REGION_MAX = 50_000  # city-word regions bigger than this are skipped (generic words)
N_JOBS = -1           # CPU cores for blocking / features (-1 = all)

# ---------------------------------------------------------------- embeddings + FAISS
USE_EMBEDDINGS = True  # needs: pip install sentence-transformers faiss-cpu
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Apache-2.0, 118M params
EMB_BATCH = 512
K_EMB_NAME = 20        # FAISS neighbours per Source 1 record (searched within the same country)
EMB_MIN_SCORE = 0.6    # ignore embedding neighbours below this cosine
FAISS_EXACT_MAX = 300_000  # pools up to this size use exact search, bigger ones IVF (approximate)
FAISS_NPROBE = 32
FEATURE_CHUNK = 250_000    # pairs per parallel feature job

# ---------------------------------------------------------------- model
SEED = 42
N_FOLDS = 5

# ---------------------------------------------------------------- decision search grid
THRESHOLDS = [round(0.20 + 0.025 * i, 3) for i in range(30)]   # 0.20 .. 0.925
RELATIVE = [0.0, 0.5, 0.7, 0.85]                                # keep p >= r * best_p of that S1
