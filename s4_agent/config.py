"""
Stage 4 central configuration.
All other modules import paths and settings from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT     = Path(__file__).resolve().parent.parent

VECTORSTORE_INDEX = PROJECT_ROOT / "s4_agent" / "vectorstore" / "review_chunks.index"
VECTORSTORE_META  = PROJECT_ROOT / "s4_agent" / "vectorstore" / "review_chunks.pkl"

CLASSIFIER_DIR    = PROJECT_ROOT / "s4_agent" / "artifacts" / "roberta_5class_best"

DATA_PATH         = PROJECT_ROOT / "data" / "processed" / "yelp_reviews_sampled_50k.csv"

RESULTS_DIR       = PROJECT_ROOT / "s4_agent" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedding model  (must match what build_vectorstore.py used)
# ---------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Ollama settings  (used by summarizer_tool)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:7b"

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 8
