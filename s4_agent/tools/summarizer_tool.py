"""
Tool 4 — Evidence Summarizer

Calls Qwen2.5-7B-Instruct (via Ollama) to synthesise a set of retrieved
review chunks into a structured analytical answer.

The LLM receives only the chunks that have already been retrieved — it does
NOT have direct access to the vector store or the classifier.  Its role is
purely synthesis and reasoning over provided evidence.

Output format (from stage4_plan.md):
{
    "main_findings": [
        "Customers frequently complain about long wait times.",
        "Service attitude is often described as rude."
    ],
    "supporting_evidence": [
        {
            "claim"    : "Long wait times are a recurring complaint.",
            "evidence" : ["Review excerpt 1 ...", "Review excerpt 2 ..."]
        }
    ],
    "uncertainties": [
        "Food quality complaints are present but less frequent."
    ]
}
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

import requests
from langchain.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

# How many evidence chunks to include in the prompt (keeps context manageable)
MAX_EVIDENCE_IN_PROMPT = 10
# Max characters per chunk shown in the prompt
MAX_CHUNK_CHARS        = 400


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, temperature: float = 0.1) -> str:
    """
    Send a chat request to the local Ollama server and return the response
    text.  Raises RuntimeError if the server is unreachable.
    """
    url  = f"{OLLAMA_BASE_URL}/api/chat"
    body = {
        "model"  : OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream" : False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Ensure Ollama is running and qwen2.5:7b is pulled."
        )


def _build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Construct the synthesis prompt.  Evidence is formatted as numbered
    excerpts to make it easy for the model to cite specific reviews.
    """
    evidence_lines = []
    for i, chunk in enumerate(chunks[:MAX_EVIDENCE_IN_PROMPT], 1):
        stars  = chunk.get("stars", "?")
        text   = chunk.get("chunk_text", chunk.get("text", ""))[:MAX_CHUNK_CHARS]
        biz_id = chunk.get("business_id", "unknown")[:12]
        evidence_lines.append(
            f"[{i}] ({stars}★, biz:{biz_id}…)\n{text}"
        )
    evidence_block = "\n\n".join(evidence_lines)

    return f"""You are an expert Yelp review analyst.
Your task: answer the question below using ONLY the provided review excerpts as evidence.
Do NOT use prior knowledge. Every claim must be supported by at least one excerpt.

QUESTION:
{question}

REVIEW EXCERPTS:
{evidence_block}

Respond with a JSON object in exactly this format (no extra text outside the JSON):
{{
  "main_findings": [
    "<finding 1>",
    "<finding 2>"
  ],
  "supporting_evidence": [
    {{
      "claim": "<restate a main finding>",
      "evidence": ["<direct quote or paraphrase from excerpt>", "..."]
    }}
  ],
  "uncertainties": [
    "<aspect that is mentioned but not well-supported by the evidence>"
  ]
}}"""


def _parse_response(raw: str) -> dict:
    """
    Extract the JSON object from the model response.
    Uses three successive strategies before falling back to plain text.

    Strategy 1 — direct parse:
        The model followed instructions perfectly; try json.loads immediately.

    Strategy 2 — regex extraction + unicode normalisation:
        The model added preamble/postamble text, or used curly quotes / smart
        apostrophes copied from review text.  Strip fences, find the {...}
        block, normalise common Unicode punctuation, then parse.

    Strategy 3 — key-level extraction:
        JSON is structurally broken (unescaped quotes inside string values).
        Use regex to extract just the main_findings array, which is almost
        always well-formed even when the rest of the JSON is not.
    """
    # ── Strategy 1: direct parse ──────────────────────────────────────────
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ── Strategy 2: find JSON block + normalise Unicode punctuation ───────
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group()
        # Replace curly / smart quotes with straight equivalents
        candidate = candidate.replace("\u2018", "'").replace("\u2019", "'")
        candidate = candidate.replace("\u201c", '"').replace("\u201d", '"')
        # Replace em-dash / en-dash with hyphen
        candidate = candidate.replace("\u2013", "-").replace("\u2014", "-")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # ── Strategy 3: extract main_findings array only ──────────────────────
    findings_match = re.search(
        r'"main_findings"\s*:\s*\[([^\]]*)\]', cleaned, re.DOTALL
    )
    if findings_match:
        findings_raw = findings_match.group(1)
        findings = re.findall(r'"([^"]+)"', findings_raw)
        if findings:
            return {
                "main_findings"      : findings,
                "supporting_evidence": [],
                "uncertainties"      : ["Partial parse: only main_findings extracted."],
            }

    # ── Final fallback ────────────────────────────────────────────────────
    return {
        "main_findings"      : [raw.strip()],
        "supporting_evidence": [],
        "uncertainties"      : ["JSON parsing failed; see main_findings for raw output."],
    }


# ---------------------------------------------------------------------------
# Tool 4
# ---------------------------------------------------------------------------

@tool
def summarize_evidence(question: str, evidence_chunks: list) -> dict:
    """
    Synthesise a list of retrieved Yelp review chunks into a structured
    analytical answer using Qwen2.5-7B-Instruct (via Ollama).

    This tool does NOT search the database itself — it only analyses the
    evidence passed to it.  Always call a retrieval tool first to gather
    relevant chunks, then pass the results here.

    Args:
        question:        The original user question, used as the analytical focus.
        evidence_chunks: List of chunk dicts returned by search_review_chunks_*
                         tools.  Each dict must contain at least 'chunk_text'
                         and 'stars'.

    Returns:
        Dict with keys:
            main_findings       — list of key conclusions (str)
            supporting_evidence — list of {claim, evidence} dicts
            uncertainties       — list of caveats or weakly-supported claims
    """
    if not evidence_chunks:
        return {
            "main_findings"      : ["No evidence provided."],
            "supporting_evidence": [],
            "uncertainties"      : [],
        }

    prompt = _build_prompt(question, evidence_chunks)
    raw    = _call_ollama(prompt)
    return _parse_response(raw)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal synthetic evidence to test without needing the full vector store
    fake_chunks = [
        {
            "stars"       : 1,
            "business_id" : "TEST001",
            "chunk_text"  : "The wait was over 90 minutes and nobody apologised. Food arrived cold.",
        },
        {
            "stars"       : 2,
            "business_id" : "TEST001",
            "chunk_text"  : "Staff were dismissive and the manager ignored our complaints entirely.",
        },
        {
            "stars"       : 4,
            "business_id" : "TEST001",
            "chunk_text"  : "On a quiet Tuesday the food was actually quite good and service was prompt.",
        },
    ]

    print("=== Tool 4: summarize_evidence ===")
    print("Calling Ollama …")
    result = summarize_evidence.invoke(
        {
            "question"       : "What are customers' main complaints about this business?",
            "evidence_chunks": fake_chunks,
        }
    )

    print("\nMain findings:")
    for f in result["main_findings"]:
        print(f"  • {f}")

    print("\nSupporting evidence:")
    for item in result["supporting_evidence"]:
        print(f"  Claim: {item['claim']}")
        for e in item["evidence"]:
            print(f"    – {e}")

    print("\nUncertainties:")
    for u in result["uncertainties"]:
        print(f"  ? {u}")
