"""
Stage 4 - Step 5: Three-Way Evaluation

Runs all 20 questions against three systems and saves results to CSV.

Systems:
    direct_llm    — Qwen answers from memory, no retrieval, no tools
    rag_baseline  — Fixed pipeline from rag_baseline.py
    full_agent    — LangGraph ReAct agent from agent_runner.py

Two modes:
    --run        Generate answers for all questions and save to CSV
    --summarise  Read the scored CSV and compute aggregate statistics

Usage:
    # Generate answers (fills answer + auto-metrics columns)
    python s4_agent/evaluation/run_eval.py --run

    # After manually filling score columns, compute summary
    python s4_agent/evaluation/run_eval.py --summarise
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, RESULTS_DIR
from pipelines.rag_baseline import run_rag_pipeline
from pipelines.agent_runner import run_agent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"
OUTPUT_CSV     = RESULTS_DIR / "eval_results.csv"

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------
FIELDNAMES = [
    # Question metadata
    "question_id", "question_type", "business_id", "question",
    # System identifier
    "system",
    # Answer content
    "answer",
    # Auto-computed metrics
    "tools_called", "tool_count", "elapsed_seconds",
    "has_evidence",        # bool: answer contains a quoted excerpt
    "answer_length",       # character count
    # Manual scoring columns (left empty for human to fill)
    "score_correctness",   # 0-2
    "score_evidence",      # 0-2
    "score_groundedness",  # 0-2
    "score_tool_use",      # 0-2 (N/A for direct_llm)
    "score_efficiency",    # 0-2
    "notes",
]


# ---------------------------------------------------------------------------
# System 1: Direct LLM (no tools, no retrieval)
# ---------------------------------------------------------------------------

def run_direct_llm(question: str, business_id: Optional[str]) -> dict:
    """Call Qwen directly via Ollama with no context."""
    if business_id:
        prompt = (
            f"You are a Yelp review analyst.\n"
            f"Answer the following question about Yelp business ID: {business_id}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on your general knowledge."
        )
    else:
        prompt = (
            f"You are a Yelp review analyst.\n"
            f"Question: {question}\n\n"
            f"Answer based only on your general knowledge."
        )

    t0 = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model"  : OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream" : False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json()["message"]["content"]
    except Exception as e:
        answer = f"[ERROR] {e}"

    elapsed = round(time.time() - t0, 2)
    return {
        "answer"         : answer,
        "tools_called"   : "",
        "tool_count"     : 0,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# System 2: RAG Baseline
# ---------------------------------------------------------------------------

def run_rag(question: str, business_id: Optional[str]) -> dict:
    result = run_rag_pipeline(question=question, business_id=business_id, top_k=8)
    syn    = result["synthesis"]

    # Flatten synthesis into a readable answer string (include supporting evidence
    # so that quoted excerpts are visible for human scoring and evidence detection)
    parts = []

    findings = syn.get("main_findings", [])
    if findings:
        parts.append("\n".join(f"• {f}" for f in findings))

    evidence = syn.get("supporting_evidence", [])
    if evidence:
        ev_lines = []
        for item in evidence:
            claim = item.get("claim", "")
            quotes = item.get("evidence", [])
            ev_lines.append(f'  Claim: "{claim}"')
            for q in quotes[:2]:
                ev_lines.append(f'    – "{q}"')
        parts.append("Supporting evidence:\n" + "\n".join(ev_lines))

    uncertainties = syn.get("uncertainties", [])
    if uncertainties:
        parts.append("Uncertainties:\n" + "\n".join(f"? {u}" for u in uncertainties))

    answer = "\n".join(parts).strip()

    return {
        "answer"         : answer,
        "tools_called"   : " → ".join(result["tools_called"]),
        "tool_count"     : len(result["tools_called"]),
        "elapsed_seconds": result["elapsed_seconds"],
    }


# ---------------------------------------------------------------------------
# System 3: Full Agent
# ---------------------------------------------------------------------------

def run_full_agent(question: str, business_id: Optional[str]) -> dict:
    result = run_agent(question=question, business_id=business_id)
    return {
        "answer"         : result["final_answer"],
        "tools_called"   : " → ".join(tc["tool"] for tc in result["tool_calls"]),
        "tool_count"     : result["steps"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


# ---------------------------------------------------------------------------
# Auto-metric helpers
# ---------------------------------------------------------------------------

EVIDENCE_SIGNALS = ['"', "'", "review", "customer said", "one reviewer", "excerpt"]

def _has_evidence(answer: str) -> bool:
    """Heuristic: does the answer contain quoted text or explicit review references?"""
    lowered = answer.lower()
    return any(sig in lowered for sig in EVIDENCE_SIGNALS)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(resume: bool = True) -> None:
    """
    Run all 20 questions × 3 systems and write results to CSV.

    Args:
        resume: If True, skip questions already present in the output CSV.
    """
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    # Load already-completed rows to support resume
    completed: set[tuple] = set()
    if resume and OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed.add((row["question_id"], row["system"]))
        print(f"Resuming — {len(completed)} rows already completed.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists() or not resume

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        total = len(questions) * 3
        done  = 0

        for q in questions:
            qid        = q["id"]
            qtype      = q["type"]
            biz_id     = q.get("business_id")
            question   = q["question"]

            systems = [
                ("direct_llm",   run_direct_llm),
                ("rag_baseline", run_rag),
                ("full_agent",   run_full_agent),
            ]

            for sys_name, sys_fn in systems:
                done += 1

                if (qid, sys_name) in completed:
                    print(f"  [{done}/{total}] SKIP  {qid} | {sys_name}")
                    continue

                print(f"\n  [{done}/{total}] Running  {qid} | {sys_name}  —  {question[:55]}…")

                try:
                    sys_result = sys_fn(question, biz_id)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    sys_result = {
                        "answer"         : f"[ERROR] {e}",
                        "tools_called"   : "",
                        "tool_count"     : 0,
                        "elapsed_seconds": 0,
                    }

                row = {
                    "question_id"     : qid,
                    "question_type"   : qtype,
                    "business_id"     : biz_id or "",
                    "question"        : question,
                    "system"          : sys_name,
                    "answer"          : sys_result["answer"].replace("\n", " | "),
                    "tools_called"    : sys_result["tools_called"],
                    "tool_count"      : sys_result["tool_count"],
                    "elapsed_seconds" : sys_result["elapsed_seconds"],
                    "has_evidence"    : _has_evidence(sys_result["answer"]),
                    "answer_length"   : len(sys_result["answer"]),
                    # Manual score columns — left blank for human scoring
                    "score_correctness" : "",
                    "score_evidence"    : "",
                    "score_groundedness": "",
                    "score_tool_use"    : "",
                    "score_efficiency"  : "",
                    "notes"             : "",
                }
                writer.writerow(row)
                f.flush()  # write immediately so progress is not lost on crash

                print(f"    → {sys_name}: {len(sys_result['answer'])} chars, "
                      f"{sys_result['tool_count']} tools, {sys_result['elapsed_seconds']}s")

    print(f"\nEvaluation complete. Results saved to:\n  {OUTPUT_CSV}")
    print("Next: open the CSV, fill in score columns, then run --summarise.")


# ---------------------------------------------------------------------------
# Summary (after manual scoring)
# ---------------------------------------------------------------------------

def summarise() -> None:
    """Read the scored CSV and print aggregate statistics per system."""
    if not OUTPUT_CSV.exists():
        print(f"No results file found at {OUTPUT_CSV}")
        print("Run with --run first.")
        return

    import csv as _csv

    rows = []
    with open(OUTPUT_CSV, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("Results file is empty.")
        return

    score_cols = [
        "score_correctness", "score_evidence",
        "score_groundedness", "score_tool_use", "score_efficiency",
    ]
    systems = ["direct_llm", "rag_baseline", "full_agent"]

    print(f"\n{'='*70}")
    print("Stage 4 Evaluation Summary")
    print(f"{'='*70}")

    # Auto-metrics (no human scores needed)
    print("\n--- Auto Metrics ---")
    header = f"{'System':<18} {'Avg Tools':>10} {'Avg Time(s)':>12} {'Evidence%':>10} {'Avg Length':>11}"
    print(header)
    print("-" * len(header))
    for sys in systems:
        sys_rows = [r for r in rows if r["system"] == sys]
        if not sys_rows:
            continue
        avg_tools   = sum(int(r["tool_count"]) for r in sys_rows) / len(sys_rows)
        avg_elapsed = sum(float(r["elapsed_seconds"]) for r in sys_rows) / len(sys_rows)
        ev_rate     = sum(1 for r in sys_rows if r["has_evidence"] == "True") / len(sys_rows)
        avg_len     = sum(int(r["answer_length"]) for r in sys_rows) / len(sys_rows)
        print(f"{sys:<18} {avg_tools:>10.1f} {avg_elapsed:>12.1f} {ev_rate:>9.0%} {avg_len:>11.0f}")

    # Manual scores
    scored_rows = [r for r in rows if r["score_correctness"] != ""]
    if not scored_rows:
        print("\n[No manual scores found yet. Fill in score columns and re-run --summarise.]")
        return

    print(f"\n--- Manual Scores  ({len(scored_rows)} rows scored) ---")
    header2 = (f"{'System':<18} {'Correct':>8} {'Evidence':>9} {'Ground':>7} "
               f"{'Tool':>6} {'Effic':>6} {'TOTAL':>7}")
    print(header2)
    print("-" * len(header2))

    for sys in systems:
        sys_rows = [r for r in scored_rows if r["system"] == sys]
        if not sys_rows:
            continue
        avgs = {}
        for col in score_cols:
            vals = [float(r[col]) for r in sys_rows if r[col] != ""]
            avgs[col] = sum(vals) / len(vals) if vals else 0.0
        total = sum(avgs.values())
        print(
            f"{sys:<18} "
            f"{avgs['score_correctness']:>8.2f} "
            f"{avgs['score_evidence']:>9.2f} "
            f"{avgs['score_groundedness']:>7.2f} "
            f"{avgs['score_tool_use']:>6.2f} "
            f"{avgs['score_efficiency']:>6.2f} "
            f"{total:>7.2f}"
        )

    # Per-type breakdown
    qtypes = sorted(set(r["question_type"] for r in scored_rows))
    print(f"\n--- Correctness by Question Type ---")
    print(f"{'Type':<28} " + "  ".join(f"{s[:8]:>8}" for s in systems))
    print("-" * 60)
    for qt in qtypes:
        row_str = f"{qt:<28} "
        for sys in systems:
            vals = [
                float(r["score_correctness"])
                for r in scored_rows
                if r["system"] == sys and r["question_type"] == qt and r["score_correctness"] != ""
            ]
            avg = sum(vals) / len(vals) if vals else 0.0
            row_str += f"{avg:>9.2f} "
        print(row_str)

    # Hallucination rate
    print(f"\n--- Hallucination Rate (score_groundedness == 0) ---")
    for sys in systems:
        sys_rows = [r for r in scored_rows if r["system"] == sys and r["score_groundedness"] != ""]
        if not sys_rows:
            continue
        hall_rate = sum(1 for r in sys_rows if float(r["score_groundedness"]) == 0) / len(sys_rows)
        print(f"  {sys:<18}: {hall_rate:.0%}")

    print(f"\nFull results: {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4 three-way evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run",       action="store_true", help="Run all questions and save results")
    group.add_argument("--summarise", action="store_true", help="Summarise a scored CSV")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh even if results CSV exists")
    args = parser.parse_args()

    if args.run:
        run_evaluation(resume=not args.no_resume)
    elif args.summarise:
        summarise()
