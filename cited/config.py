import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CITED_DATA_DIR", ROOT / "data" / "raw"))
CACHE_DIR = ROOT / "data" / "cache"

EMBED_MODEL = os.getenv("CITED_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("CITED_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODEL = os.getenv("CITED_LLM_MODEL", "claude-sonnet-5-5")
# Set CITED_BACKEND=tfidf to skip downloading embedding models (used in tests / offline).
FORCE_BACKEND = os.getenv("CITED_BACKEND")

TOP_K = 4          # passages shown to the LLM / user
CANDIDATES = 20    # candidates per retriever before fusion / reranking
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
RRF_K = 60

# Refusal gate: if the best dense cosine similarity is below this, we say "not found".
# Tune on your own eval set (see eval/run_eval.py --sweep).
MIN_CONF = {"st": 0.57, "tfidf": 0.15}
