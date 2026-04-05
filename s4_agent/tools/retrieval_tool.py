"""
Tool 1 — Retrieval

Exposes two LangChain tools for semantic search over the chunk-level
vector store built in Step 1:

  search_review_chunks_global(query, top_k)
      Full-corpus FAISS search. Use when no specific business is mentioned.

  search_review_chunks_by_business(business_id, query, top_k)
      Pre-filtered search restricted to one business's chunks.
      Implements the pre-filter design from stage4_plan.md:
        "先筛出该商家的 chunk 子集，再在子集内计算相似度"
      This avoids the global Top-K + post-filter anti-pattern.

Both tools return a list of result dicts (not a concatenated string) so
that the RAG pipeline and Agent can decide how to format the output.
"""

import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from langchain.tools import tool
from sentence_transformers import SentenceTransformer

# Allow running this file directly from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    VECTORSTORE_INDEX,
    VECTORSTORE_META,
    EMBED_MODEL,
    DEFAULT_TOP_K,
)


# ---------------------------------------------------------------------------
# Lazy singletons  (loaded once on first call, reused afterwards)
# ---------------------------------------------------------------------------

_store: Optional[dict] = None
_index: Optional[faiss.IndexFlatIP] = None
_embed_model: Optional[SentenceTransformer] = None


def _load_store() -> tuple[dict, faiss.IndexFlatIP, SentenceTransformer]:
    global _store, _index, _embed_model

    if _store is None:
        print("[retrieval_tool] Loading vector store …")
        with open(VECTORSTORE_META, "rb") as f:
            _store = pickle.load(f)
        _index = faiss.read_index(str(VECTORSTORE_INDEX))
        _embed_model = SentenceTransformer(EMBED_MODEL)
        print(f"[retrieval_tool] Ready — {_index.ntotal:,} chunks, "
              f"{len(_store['business_to_indices']):,} businesses")

    return _store, _index, _embed_model


def _encode_query(query: str) -> np.ndarray:
    """Return a (1, 384) float32 normalised query vector."""
    _, _, model = _load_store()
    vec = model.encode([query], normalize_embeddings=True).astype("float32")
    return vec


def _format_results(indices: list[int], scores: list[float], store: dict) -> list[dict]:
    """Convert raw indices + scores into structured result dicts."""
    results = []
    for idx, score in zip(indices, scores):
        chunk = store["chunks"][idx]
        results.append(
            {
                "chunk_idx"   : int(idx),
                "review_id"   : chunk["review_id"],
                "business_id" : chunk["business_id"],
                "stars"       : float(chunk["stars"]),
                "chunk_text"  : chunk["chunk_text"],
                "similarity"  : round(float(score), 4),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Tool 1a: global search
# ---------------------------------------------------------------------------

@tool
def search_review_chunks_global(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Search the full Yelp review corpus for chunks semantically similar to
    the query.  Use this when no specific business_id is mentioned.

    Args:
        query:  Natural language search query, e.g. "rude staff terrible service".
        top_k:  Number of results to return (default 8).

    Returns:
        List of dicts, each with keys:
            chunk_idx, review_id, business_id, stars, chunk_text, similarity
    """
    store, index, _ = _load_store()
    q_vec = _encode_query(query)

    scores, idxs = index.search(q_vec, top_k)
    return _format_results(idxs[0].tolist(), scores[0].tolist(), store)


# ---------------------------------------------------------------------------
# Tool 1b: business-filtered search
# ---------------------------------------------------------------------------

@tool
def search_review_chunks_by_business(
    business_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Search reviews belonging to a specific business for chunks semantically
    similar to the query.  Use this when a business_id is available.

    Pre-filtering approach: retrieves only the target business's chunk
    embeddings, then computes cosine similarity directly on that subset.
    This is more accurate than global Top-K + post-filter for businesses
    with few reviews.

    Args:
        business_id:  Yelp business ID string.
        query:        Natural language search query.
        top_k:        Number of results to return (default 8).

    Returns:
        List of dicts, each with keys:
            chunk_idx, review_id, business_id, stars, chunk_text, similarity
        Returns an empty list if the business_id is not found.
    """
    store, _, _ = _load_store()

    biz_indices = store["business_to_indices"].get(business_id)
    if not biz_indices:
        return []

    q_vec = _encode_query(query)                         # (1, 384)
    subset = store["embeddings"][biz_indices]            # (M, 384)
    sims   = (subset @ q_vec.T).squeeze()                # (M,)

    # Handle the edge case where a business has only one chunk
    if sims.ndim == 0:
        sims = np.array([float(sims)])

    k = min(top_k, len(biz_indices))
    top_pos  = np.argsort(sims)[::-1][:k]
    top_idxs = [biz_indices[p] for p in top_pos]
    top_sims = [float(sims[p]) for p in top_pos]

    return _format_results(top_idxs, top_sims, store)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Tool 1a: global search ===")
    results = search_review_chunks_global.invoke(
        {"query": "rude staff terrible service", "top_k": 3}
    )
    for r in results:
        print(f"  [{r['stars']}★  sim={r['similarity']}]  {r['chunk_text'][:100]}")

    print("\n=== Tool 1b: business-filtered search ===")
    # Use the first business in the store as a test target
    store, _, _ = _load_store()
    sample_biz = list(store["business_to_indices"].keys())[0]
    results = search_review_chunks_by_business.invoke(
        {"business_id": sample_biz, "query": "food quality taste", "top_k": 3}
    )
    print(f"  Business: {sample_biz}  (chunks in store: "
          f"{len(store['business_to_indices'][sample_biz])})")
    for r in results:
        print(f"  [{r['stars']}★  sim={r['similarity']}]  {r['chunk_text'][:100]}")
