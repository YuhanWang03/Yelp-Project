"""
Stage 4 - Step 3: RAG Baseline Pipeline

Fixed-flow retrieval-augmented generation pipeline.
No autonomous tool selection — the execution order is hardcoded per question type.

Two flows (from stage4_plan.md):

  Flow A — Business question  (business_id is provided)
    1. get_business_stats      → statistical overview
    2. search_review_chunks_by_business  → semantic evidence
    3. summarize_evidence      → structured analytical answer

  Flow B — Global question    (no business_id)
    1. search_review_chunks_global  → semantic evidence
    2. summarize_evidence           → structured analytical answer

Entry point:
    run_rag_pipeline(question, business_id=None, top_k=8) -> dict

Return schema:
    {
        "question"        : str,
        "mode"            : "business" | "global",
        "business_id"     : str | None,
        "business_stats"  : dict | None,
        "retrieved_chunks": list[dict],
        "synthesis"       : {
            "main_findings"      : list[str],
            "supporting_evidence": list[dict],
            "uncertainties"      : list[str]
        },
        "tools_called"    : list[str],
        "elapsed_seconds" : float
    }

Usage:
    python s4_agent/pipelines/rag_baseline.py
"""

import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_TOP_K
from tools.retrieval_tool import (
    search_review_chunks_global,
    search_review_chunks_by_business,
)
from tools.stats_tool import get_business_stats
from tools.summarizer_tool import summarize_evidence


# ---------------------------------------------------------------------------
# Flow A: business-specific question
# ---------------------------------------------------------------------------

def _run_flow_a(
    question: str,
    business_id: str,
    top_k: int,
) -> dict:
    """
    Flow A execution:
      get_business_stats  →  search_review_chunks_by_business  →  summarize_evidence
    """
    tools_called: list[str] = []

    # Step 1 — statistical overview
    print(f"  [Flow A / Step 1] get_business_stats({business_id[:16]}…)")
    stats = get_business_stats.invoke({"business_id": business_id})
    tools_called.append("get_business_stats")

    if stats["review_count"] == 0:
        return {
            "business_stats"  : stats,
            "retrieved_chunks": [],
            "synthesis"       : {
                "main_findings"      : [f"No reviews found for business_id={business_id}"],
                "supporting_evidence": [],
                "uncertainties"      : [],
            },
            "tools_called": tools_called,
        }

    # Step 2 — semantic retrieval (pre-filtered to this business)
    print(f"  [Flow A / Step 2] search_review_chunks_by_business(query='{question[:50]}…')")
    chunks = search_review_chunks_by_business.invoke(
        {"business_id": business_id, "query": question, "top_k": top_k}
    )
    tools_called.append("search_review_chunks_by_business")

    # Step 3 — synthesis
    print(f"  [Flow A / Step 3] summarize_evidence ({len(chunks)} chunks) …")
    synthesis = summarize_evidence.invoke(
        {"question": question, "evidence_chunks": chunks}
    )
    tools_called.append("summarize_evidence")

    return {
        "business_stats"  : stats,
        "retrieved_chunks": chunks,
        "synthesis"       : synthesis,
        "tools_called"    : tools_called,
    }


# ---------------------------------------------------------------------------
# Flow B: global / cross-business question
# ---------------------------------------------------------------------------

def _run_flow_b(question: str, top_k: int) -> dict:
    """
    Flow B execution:
      search_review_chunks_global  →  summarize_evidence
    """
    tools_called: list[str] = []

    # Step 1 — global semantic retrieval
    print(f"  [Flow B / Step 1] search_review_chunks_global(query='{question[:50]}…')")
    chunks = search_review_chunks_global.invoke(
        {"query": question, "top_k": top_k}
    )
    tools_called.append("search_review_chunks_global")

    # Step 2 — synthesis
    print(f"  [Flow B / Step 2] summarize_evidence ({len(chunks)} chunks) …")
    synthesis = summarize_evidence.invoke(
        {"question": question, "evidence_chunks": chunks}
    )
    tools_called.append("summarize_evidence")

    return {
        "business_stats"  : None,
        "retrieved_chunks": chunks,
        "synthesis"       : synthesis,
        "tools_called"    : tools_called,
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def run_rag_pipeline(
    question: str,
    business_id: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Run the RAG baseline pipeline.

    Args:
        question    : Natural language question from the user.
        business_id : Yelp business ID.  If provided, Flow A is used.
                      If None, Flow B (global search) is used.
        top_k       : Number of review chunks to retrieve.

    Returns:
        Structured result dict (see module docstring for schema).
    """
    mode = "business" if business_id else "global"
    print(f"\n{'='*60}")
    print(f"RAG Pipeline  |  mode={mode}  |  top_k={top_k}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    t0 = time.time()

    if business_id:
        flow_result = _run_flow_a(question, business_id, top_k)
    else:
        flow_result = _run_flow_b(question, top_k)

    elapsed = round(time.time() - t0, 2)

    return {
        "question"        : question,
        "mode"            : mode,
        "business_id"     : business_id,
        "business_stats"  : flow_result["business_stats"],
        "retrieved_chunks": flow_result["retrieved_chunks"],
        "synthesis"       : flow_result["synthesis"],
        "tools_called"    : flow_result["tools_called"],
        "elapsed_seconds" : elapsed,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    """Print a pipeline result in a human-readable format."""
    print(f"\n{'─'*60}")
    print(f"Mode            : {result['mode']}")

    if result["business_stats"]:
        s = result["business_stats"]
        dist = "  ".join(f"{k}★:{v}" for k, v in s["star_distribution"].items())
        print(f"Business        : {s['business_id']}")
        print(f"Review count    : {s['review_count']}   avg {s['avg_stars']}★")
        print(f"Distribution    : {dist}")

    chunks = result["retrieved_chunks"]
    print(f"\nRetrieved chunks: {len(chunks)}")
    for i, c in enumerate(chunks[:3], 1):
        print(f"  [{i}] {c['stars']}★  sim={c['similarity']}  "
              f"{c['chunk_text'][:90]}…")
    if len(chunks) > 3:
        print(f"  … and {len(chunks)-3} more")

    syn = result["synthesis"]
    print(f"\nMain findings ({len(syn['main_findings'])}):")
    for f in syn["main_findings"]:
        print(f"  • {f}")

    if syn["supporting_evidence"]:
        print(f"\nSupporting evidence ({len(syn['supporting_evidence'])} claims):")
        for item in syn["supporting_evidence"][:2]:
            print(f"  Claim: {item['claim']}")
            for e in item["evidence"][:2]:
                print(f"    – {e[:100]}")

    if syn["uncertainties"]:
        print(f"\nUncertainties:")
        for u in syn["uncertainties"]:
            print(f"  ? {u}")

    print(f"\nTools called    : {' → '.join(result['tools_called'])}")
    print(f"Elapsed         : {result['elapsed_seconds']}s")


# ---------------------------------------------------------------------------
# Standalone test — covers all 4 question types from stage4_plan.md
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Real business IDs from the dataset
    LOW_SCORE_BIZ  = "ORL4JE6tz3rJxVqkdKfegA"   # 182 reviews, avg 2.42★
    HIGH_SCORE_BIZ = "2KIDQyTh-HzLxOUEDqtDBg"   # 62 reviews,  avg 4.13★

    TEST_CASES = [
        # Flow A — Complaint Mining
        {
            "question"   : "What do customers complain about most at this business?",
            "business_id": LOW_SCORE_BIZ,
            "label"      : "Flow A | Complaint Mining",
        },
        # Flow A — Aspect Analysis
        {
            "question"   : "How do customers describe the service and staff at this business?",
            "business_id": HIGH_SCORE_BIZ,
            "label"      : "Flow A | Aspect Analysis",
        },
        # Flow A — Business Profiling
        {
            "question"   : "Give an overall profile of this business based on customer reviews.",
            "business_id": LOW_SCORE_BIZ,
            "label"      : "Flow A | Business Profiling",
        },
        # Flow B — Cross-Business Pattern Search
        {
            "question"   : "What service problems appear most often in low-rated reviews?",
            "business_id": None,
            "label"      : "Flow B | Cross-Business Pattern Search",
        },
    ]

    for case in TEST_CASES:
        print(f"\n{'#'*60}")
        print(f"TEST: {case['label']}")

        result = run_rag_pipeline(
            question    = case["question"],
            business_id = case.get("business_id"),
            top_k       = 8,
        )
        print_result(result)

    print(f"\n{'='*60}")
    print("All RAG baseline tests complete.")
    print("Next: implement s4_agent/pipelines/agent_runner.py")
