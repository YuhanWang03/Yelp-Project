"""
Stage 4 — Yelp Business Intelligence Agent
Hugging Face Spaces Deployment Version

Differences from app.py (local):
  - Large assets downloaded from HF Hub at startup (no local paths needed)
  - Ollama replaced with Groq API  (llama-3.3-70b-versatile, free tier)
  - Business stats derived directly from the vector store pkl (no CSV upload needed)
  - Business catalogue loaded from a small JSON on HF Hub
  - All pipeline logic is inlined — no imports from local s4_agent packages

────────────────────────────────────────────────────────
SETUP INSTRUCTIONS
────────────────────────────────────────────────────────
1. Generate business_catalogue.json locally (one-time):
       python s4_agent/app_hf.py --build-assets

2. Upload the following files to a HF Hub DATASET repo
   (e.g. "YourName/yelp-agent-assets"):
       vectorstore/review_chunks.index     (89 MB)
       vectorstore/review_chunks.pkl       (136 MB)
       artifacts/roberta_5class_best/      (~475 MB, full directory)
       business_catalogue.json             (< 1 MB, generated in step 1)

3. Create a HF Spaces (Gradio SDK) repo and upload this file as app.py
   (rename app_hf.py → app.py in the Space).

4. In the Space → Settings → Repository secrets, add:
       GROQ_API_KEY    — free at console.groq.com
       HF_ASSET_REPO   — e.g. "YourName/yelp-agent-assets"
────────────────────────────────────────────────────────
"""

import os
import sys
import json
import re
import time
import pickle
import warnings
import tempfile

import numpy as np

# Heavy dependencies are imported lazily so that --build-assets works
# without requiring groq / langchain_groq / faiss to be installed.
_RUNTIME_IMPORTS_DONE = False

def _ensure_runtime_imports():
    """Import all runtime dependencies (not needed for --build-assets)."""
    global _RUNTIME_IMPORTS_DONE
    if _RUNTIME_IMPORTS_DONE:
        return
    global faiss, torch, F, gr, Groq, hf_hub_download, snapshot_download
    global SentenceTransformer, AutoModelForSequenceClassification, AutoTokenizer
    global tool, ChatGroq, HumanMessage, AIMessage, ToolMessage, SystemMessage
    global create_react_agent

    import faiss
    import torch
    import torch.nn.functional as F
    import gradio as gr
    from groq import Groq
    from huggingface_hub import hf_hub_download, snapshot_download
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from langchain.tools import tool
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.prebuilt import create_react_agent

    _RUNTIME_IMPORTS_DONE = True

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
HF_ASSET_REPO = os.environ.get("HF_ASSET_REPO", "")
GROQ_MODEL    = "llama-3.3-70b-versatile"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K  = 8
MAX_EVIDENCE   = 10
MAX_CHUNK_CHARS = 400

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "yelp_agent_assets")

# ─────────────────────────────────────────────────────────────────────────────
# Asset download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download(filename: str) -> str:
    """Download a single file from HF_ASSET_REPO and return the local path."""
    return hf_hub_download(
        repo_id=HF_ASSET_REPO,
        filename=filename,
        repo_type="dataset",
        local_dir=_CACHE_DIR,
    )


def _download_classifier() -> str:
    """Download the full RoBERTa directory and return the local path."""
    root = snapshot_download(
        repo_id=HF_ASSET_REPO,
        repo_type="dataset",
        local_dir=_CACHE_DIR,
        allow_patterns="artifacts/roberta_5class_best/*",
    )
    return os.path.join(root, "artifacts", "roberta_5class_best")


# ─────────────────────────────────────────────────────────────────────────────
# Load all assets at startup (singletons)
# ─────────────────────────────────────────────────────────────────────────────

if "--build-assets" not in sys.argv:
    _ensure_runtime_imports()

if "--build-assets" in sys.argv:
    # ── Build assets locally, then exit ─────────────────────────────────────
    # This block runs without needing groq/faiss/etc.
    import pathlib

    print("Building business_catalogue.json from local files…")

    _PKL_PATH = pathlib.Path(__file__).parent / "vectorstore" / "review_chunks.pkl"
    _BIZ_JSON = (pathlib.Path(__file__).parent.parent
                 / "data" / "raw" / "yelp_academic_dataset_business.json")
    _OUT_PATH = pathlib.Path(__file__).parent / "business_catalogue.json"

    with open(_PKL_PATH, "rb") as f:
        _store = pickle.load(f)

    _eligible = {
        bid: len(idxs)
        for bid, idxs in _store["business_to_indices"].items()
        if len(idxs) > 50
    }
    print(f"  Eligible businesses (>50 chunks): {len(_eligible)}")

    _catalogue = {}
    with open(_BIZ_JSON, encoding="utf-8") as f:
        for line in f:
            biz = json.loads(line)
            bid = biz.get("business_id", "")
            if bid in _eligible:
                cats = (biz.get("categories") or "").split(", ")
                _catalogue[bid] = {
                    "name"       : biz.get("name", bid),
                    "city"       : biz.get("city", ""),
                    "stars"      : biz.get("stars", 0),
                    "categories" : cats[0] if cats else "Business",
                    "chunk_count": _eligible[bid],
                }

    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(_catalogue, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(_catalogue)} businesses → {_OUT_PATH}")
    print("  Upload this file to your HF_ASSET_REPO and you're ready to deploy.")
    sys.exit(0)

print("=" * 60)
print("Yelp Business Intelligence Agent — loading assets…")
print("=" * 60)

# Vector store
_INDEX_PATH = _download("vectorstore/review_chunks.index")
_META_PATH  = _download("vectorstore/review_chunks.pkl")
_CAT_PATH   = _download("business_catalogue.json")
_CLF_DIR    = _download_classifier()

with open(_META_PATH, "rb") as _f:
    _STORE = pickle.load(_f)
_INDEX = faiss.read_index(_INDEX_PATH)
_EMBED = SentenceTransformer(EMBED_MODEL)
print(f"[retrieval]  {_INDEX.ntotal:,} chunks, "
      f"{len(_STORE['business_to_indices']):,} businesses")

# Business catalogue  {business_id: {name, city, stars, categories, chunk_count}}
with open(_CAT_PATH, encoding="utf-8") as _f:
    CATALOGUE = json.load(_f)

# Business stats derived from the pkl  {business_id: {review_count, avg_stars, star_distribution}}
# Deduplicate on review_id so multi-chunk reviews aren't double-counted.
print("[stats]  Computing business stats from vector store…")
_BIZ_STATS: dict = {}
_seen_reviews: dict[str, set] = {}   # business_id → set of review_ids

for _chunk in _STORE["chunks"]:
    _bid  = _chunk["business_id"]
    _rid  = _chunk["review_id"]
    _star = int(_chunk["stars"])
    if _bid not in _seen_reviews:
        _seen_reviews[_bid] = set()
        _BIZ_STATS[_bid] = {"business_id": _bid, "review_count": 0,
                             "avg_stars": 0.0, "_stars_sum": 0,
                             "star_distribution": {str(s): 0 for s in range(1, 6)}}
    if _rid not in _seen_reviews[_bid]:
        _seen_reviews[_bid].add(_rid)
        _BIZ_STATS[_bid]["review_count"] += 1
        _BIZ_STATS[_bid]["_stars_sum"]   += _star
        _BIZ_STATS[_bid]["star_distribution"][str(_star)] += 1

for _bid, _s in _BIZ_STATS.items():
    _n = _s["review_count"]
    _s["avg_stars"] = round(_s["_stars_sum"] / _n, 2) if _n else 0.0
    del _s["_stars_sum"]

print(f"[stats]  Stats ready for {len(_BIZ_STATS):,} businesses")

# RoBERTa classifier
_CLF_TOKENIZER = AutoTokenizer.from_pretrained(_CLF_DIR)
_CLF_MODEL = AutoModelForSequenceClassification.from_pretrained(_CLF_DIR)
_CLF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CLF_MODEL = _CLF_MODEL.to(_CLF_DEVICE).eval()
print(f"[classifier] RoBERTa ready on {_CLF_DEVICE.upper()}")

# Groq client
_GROQ = Groq(api_key=GROQ_API_KEY)

print("All assets loaded.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Business dropdown
# ─────────────────────────────────────────────────────────────────────────────

_sorted_bizs = sorted(CATALOGUE.items(), key=lambda x: -x[1]["chunk_count"])
DROPDOWN_CHOICES = ["(Global search — no specific business)"] + [
    f"{info['name']} — {info['city']} (★{info['stars']}, {info['chunk_count']} chunks)"
    for _, info in _sorted_bizs
]
LABEL_TO_ID = {
    f"{info['name']} — {info['city']} (★{info['stars']}, {info['chunk_count']} chunks)": bid
    for bid, info in _sorted_bizs
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def _encode_query(query: str) -> np.ndarray:
    return _EMBED.encode([query], normalize_embeddings=True).astype("float32")


def _format_results(indices: list, scores: list) -> list[dict]:
    results = []
    for idx, score in zip(indices, scores):
        chunk = _STORE["chunks"][idx]
        results.append({
            "chunk_idx"  : int(idx),
            "review_id"  : chunk["review_id"],
            "business_id": chunk["business_id"],
            "stars"      : float(chunk["stars"]),
            "chunk_text" : chunk["chunk_text"],
            "similarity" : round(float(score), 4),
        })
    return results


@tool
def search_review_chunks_global(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Search the full Yelp review corpus for chunks semantically similar to
    the query. Use this when no specific business_id is mentioned.
    """
    q_vec = _encode_query(query)
    scores, idxs = _INDEX.search(q_vec, top_k)
    return _format_results(idxs[0].tolist(), scores[0].tolist())


@tool
def search_review_chunks_by_business(
    business_id: str, query: str, top_k: int = DEFAULT_TOP_K
) -> list[dict]:
    """
    Search reviews belonging to a specific business for chunks semantically
    similar to the query. Use this when a business_id is available.
    """
    biz_indices = _STORE["business_to_indices"].get(business_id)
    if not biz_indices:
        return []
    q_vec  = _encode_query(query)
    subset = _STORE["embeddings"][biz_indices]
    sims   = (subset @ q_vec.T).squeeze()
    if sims.ndim == 0:
        sims = np.array([float(sims)])
    k        = min(top_k, len(biz_indices))
    top_pos  = np.argsort(sims)[::-1][:k]
    top_idxs = [biz_indices[p] for p in top_pos]
    top_sims = [float(sims[p]) for p in top_pos]
    return _format_results(top_idxs, top_sims)


@tool
def get_business_stats(business_id: str) -> dict:
    """
    Return review count, average star rating, and star distribution (1–5)
    for a given Yelp business.
    """
    if business_id in _BIZ_STATS:
        return _BIZ_STATS[business_id]
    return {
        "business_id": business_id, "review_count": 0,
        "avg_stars": None, "star_distribution": {},
    }


@tool
def classify_review(text: str) -> dict:
    """
    Classify a Yelp review text and return a predicted star rating (1–5)
    using the fine-tuned RoBERTa model from Stage 2.
    """
    inputs = _CLF_TOKENIZER(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    inputs = {k: v.to(_CLF_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _CLF_MODEL(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()
    predicted_idx = int(torch.argmax(logits, dim=-1).item())
    return {
        "predicted_stars"   : predicted_idx + 1,
        "confidence"        : round(probs[predicted_idx], 4),
        "score_distribution": {f"{i+1}_star": round(probs[i], 4) for i in range(len(probs))},
    }


def _call_groq(prompt: str, temperature: float = 0.1) -> str:
    """Call Groq API and return response text."""
    resp = _GROQ.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def _parse_llm_json(raw: str) -> dict:
    """
    Three-strategy JSON parser (same logic as original summarizer_tool.py).
    Strategy 1 — direct parse.
    Strategy 2 — strip fences + normalise Unicode punctuation.
    Strategy 3 — extract main_findings array only.
    """
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = (match.group()
                     .replace("\u2018", "'").replace("\u2019", "'")
                     .replace("\u201c", '"').replace("\u201d", '"')
                     .replace("\u2013", "-").replace("\u2014", "-"))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    findings_match = re.search(
        r'"main_findings"\s*:\s*\[([^\]]*)\]', cleaned, re.DOTALL
    )
    if findings_match:
        findings = re.findall(r'"([^"]+)"', findings_match.group(1))
        if findings:
            return {
                "main_findings"      : findings,
                "supporting_evidence": [],
                "uncertainties"      : ["Partial parse: only main_findings extracted."],
            }

    return {
        "main_findings"      : [raw.strip()],
        "supporting_evidence": [],
        "uncertainties"      : ["JSON parsing failed; see main_findings for raw output."],
    }


@tool
def summarize_evidence(question: str, evidence_chunks: list) -> dict:
    """
    Synthesise a list of retrieved Yelp review chunks into a structured
    analytical answer. Always call a retrieval tool first to gather chunks,
    then pass the results here as the final step.
    """
    if not evidence_chunks:
        return {
            "main_findings"      : ["No evidence provided."],
            "supporting_evidence": [],
            "uncertainties"      : [],
        }

    evidence_lines = []
    for i, chunk in enumerate(evidence_chunks[:MAX_EVIDENCE], 1):
        stars  = chunk.get("stars", "?")
        text   = chunk.get("chunk_text", chunk.get("text", ""))[:MAX_CHUNK_CHARS]
        biz_id = chunk.get("business_id", "unknown")[:12]
        evidence_lines.append(f"[{i}] ({stars}★, biz:{biz_id}…)\n{text}")

    prompt = f"""You are an expert Yelp review analyst.
Answer the question using ONLY the provided review excerpts. Do NOT use prior knowledge.
Every claim must be supported by at least one excerpt.

QUESTION:
{question}

REVIEW EXCERPTS:
{chr(10).join(evidence_lines)}

Respond with a JSON object in exactly this format (no extra text outside the JSON):
{{
  "main_findings": ["<finding 1>", "<finding 2>"],
  "supporting_evidence": [
    {{"claim": "<restate a main finding>", "evidence": ["<direct quote or paraphrase>"]}}
  ],
  "uncertainties": ["<aspect mentioned but not well-supported>"]
}}"""

    raw = _call_groq(prompt)
    return _parse_llm_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph ReAct agent (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Yelp Business Intelligence Agent with access to a \
database of 50,000 real Yelp reviews.

You have the following tools:
- search_review_chunks_global: search all reviews semantically
- search_review_chunks_by_business: search reviews for a specific business
- get_business_stats: get star distribution for a specific business
- classify_review: predict star rating for a piece of text using a fine-tuned model
- summarize_evidence: synthesise retrieved review chunks into a structured answer

Guidelines:
1. If a business_id is mentioned in the question, use the business-specific tools.
2. Always retrieve evidence BEFORE calling summarize_evidence.
3. Call summarize_evidence as your LAST step to produce the final structured answer.
4. Use classify_review when you want to verify the sentiment of a specific text snippet.
5. Be concise in tool inputs — avoid repeating the full question verbatim.
"""

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
        tools = [
            search_review_chunks_global,
            search_review_chunks_by_business,
            get_business_stats,
            classify_review,
            summarize_evidence,
        ]
        _agent = create_react_agent(llm, tools)
    return _agent


def _extract_trace(messages: list) -> tuple[str, list[dict]]:
    """Parse LangGraph message list → (final_answer, tool_calls)."""
    tool_calls: list[dict] = []
    final_answer = ""

    tool_msg_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            tool_msg_map[msg.tool_call_id] = (
                content[:600] + " …[truncated]" if len(content) > 600 else content
            )

    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "tool"  : tc["name"],
                        "input" : str(tc["args"]),
                        "output": tool_msg_map.get(tc["id"], "(no output)"),
                    })
            elif msg.content:
                final_answer = msg.content

    return final_answer, tool_calls


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runners
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_pipeline(question: str, business_id=None, top_k: int = DEFAULT_TOP_K) -> dict:
    tools_called: list[str] = []
    t0 = time.time()

    if business_id:
        stats = get_business_stats.invoke({"business_id": business_id})
        tools_called.append("get_business_stats")
        chunks = search_review_chunks_by_business.invoke(
            {"business_id": business_id, "query": question, "top_k": top_k}
        )
        tools_called.append("search_review_chunks_by_business")
    else:
        stats  = None
        chunks = search_review_chunks_global.invoke({"query": question, "top_k": top_k})
        tools_called.append("search_review_chunks_global")

    synthesis = summarize_evidence.invoke({"question": question, "evidence_chunks": chunks})
    tools_called.append("summarize_evidence")

    return {
        "question"        : question,
        "mode"            : "business" if business_id else "global",
        "business_id"     : business_id,
        "business_stats"  : stats,
        "retrieved_chunks": chunks,
        "synthesis"       : synthesis,
        "tools_called"    : tools_called,
        "elapsed_seconds" : round(time.time() - t0, 2),
    }


def run_agent(question: str, business_id=None, max_iterations: int = 6) -> dict:
    full_question = (
        f"{question}\n\n[Target business_id: {business_id}]" if business_id else question
    )
    agent = _get_agent()
    t0 = time.time()
    result = agent.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=full_question)]},
        config={"recursion_limit": max_iterations * 2},
    )
    elapsed = round(time.time() - t0, 2)
    final_answer, tool_calls = _extract_trace(result["messages"])
    return {
        "question"       : question,
        "business_id"    : business_id,
        "final_answer"   : final_answer,
        "tool_calls"     : tool_calls,
        "steps"          : len(tool_calls),
        "elapsed_seconds": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def on_business_select(label: str) -> str:
    if label == "(Global search — no specific business)" or not label:
        return ""
    return LABEL_TO_ID.get(label, "")


def _format_stats_dict(s: dict) -> str:
    if not s or s.get("review_count", 0) == 0:
        return ""
    bid      = s.get("business_id", "")
    dist     = s.get("star_distribution", {})
    dist_str = " &nbsp;\\|&nbsp; ".join(f"★{k}: **{v}**" for k, v in sorted(dist.items()))
    label    = ID_TO_LABEL.get(bid, "")
    name     = label.split(" — ")[0] if label else bid
    city     = label.split(" — ")[1].split(" (")[0] if " — " in label else ""
    return (
        f"### 📊 Business Stats\n\n"
        f"| Metric | Details |\n| :--- | :--- |\n"
        f"| **🏢 Name** | {name} |\n"
        f"| **🏙️ City** | {city} |\n"
        f"| **🆔 ID** | `{bid}` |\n"
        f"| **📝 Indexed Reviews** | {s['review_count']} chunks |\n"
        f"| **⭐ Avg Stars** | {s['avg_stars']} |\n"
        f"| **📈 Distribution** | {dist_str} |\n"
    )


def _format_stats_from_id(business_id) -> str:
    if not business_id or business_id not in CATALOGUE:
        return ""
    info = CATALOGUE[business_id]
    return (
        f"### 📊 Business Info\n\n"
        f"| Metric | Details |\n| :--- | :--- |\n"
        f"| **🏢 Name** | {info['name']} |\n"
        f"| **🏙️ City** | {info['city']} |\n"
        f"| **🏷️ Category** | {info['categories']} |\n"
        f"| **⭐ Yelp Stars** | {info['stars']} |\n"
        f"| **📝 Indexed Reviews** | {info['chunk_count']} chunks |\n"
    )


def run_query(question: str, business_id: str, system: str):
    question    = question.strip()
    business_id = business_id.strip() or None

    if not question:
        yield "Please enter a question.", "", "", ""
        return

    yield (
        f"⏳ **Processing… please wait.**\n\n"
        f"*(Running via {system} — this may take 20–60 s on first call)*",
        "⏳ *Waiting for tools…*",
        "⏳ *Waiting for retrieval…*",
        "⏳ *Fetching stats…*",
    )

    # ── Direct LLM ──────────────────────────────────────────────────────────
    if system == "Direct LLM":
        t0 = time.time()
        try:
            answer = _call_groq(question, temperature=0.3)
        except Exception as e:
            answer = f"Error: {e}"
        elapsed = round(time.time() - t0, 2)
        yield (
            f"### 🤖 Answer *(Direct LLM — no retrieval)*\n\n{answer}",
            f"No tools used.\n\n*⏱️ Elapsed: **{elapsed}s***",
            "*Direct LLM does not retrieve evidence.*",
            _format_stats_from_id(business_id),
        )
        return

    # ── RAG Baseline ─────────────────────────────────────────────────────────
    if system == "RAG Baseline":
        try:
            result = run_rag_pipeline(question, business_id=business_id)
        except Exception as e:
            yield f"Pipeline error: {e}", "", "", ""
            return

        synthesis = result.get("synthesis", {})
        findings  = synthesis.get("main_findings", [])
        evidence  = synthesis.get("supporting_evidence", [])
        uncertain = synthesis.get("uncertainties", [])
        elapsed   = result.get("elapsed_seconds", "?")
        tools     = result.get("tools_called", [])
        stats     = result.get("business_stats")
        chunks    = result.get("retrieved_chunks", [])

        answer_lines = ["### 🎯 Main Findings\n"] + [f"- {f}" for f in findings]
        if uncertain:
            answer_lines += ["\n### ❓ Uncertainties\n"] + [f"- {u}" for u in uncertain]

        evid_lines = [f"### 📑 Retrieved Evidence ({len(chunks)} total)\n"]
        for item in evidence[:5]:
            evid_lines.append(f"**📌 Claim:** {item.get('claim', '')}\n")
            for quote in item.get("evidence", [])[:2]:
                evid_lines.append(f"> ❝ *{quote}* ❞\n")
            evid_lines.append("---\n")
        if not evidence and chunks:
            for c in chunks[:3]:
                evid_lines.append(
                    f"> ❝ *{c['chunk_text'][:200]}…* ❞\n> \n> — *(★{c['stars']})*\n\n---\n"
                )

        tool_lines = (
            [f"### ⚙️ Tools Called ({len(tools)} steps)\n"]
            + [f"{i}. `{t}`" for i, t in enumerate(tools, 1)]
            + [f"\n*⏱️ Elapsed: **{elapsed}s***"]
        )
        yield (
            "\n".join(answer_lines),
            "\n".join(tool_lines),
            "\n".join(evid_lines),
            _format_stats_dict(stats) if stats else _format_stats_from_id(business_id),
        )
        return

    # ── Full Agent ────────────────────────────────────────────────────────────
    if system == "Full Agent":
        try:
            result = run_agent(question, business_id=business_id, max_iterations=6)
        except Exception as e:
            yield f"Agent error: {e}", "", "", ""
            return

        answer     = result.get("final_answer") or result.get("answer", "*(No answer)*")
        tool_calls = result.get("tool_calls", [])
        elapsed    = result.get("elapsed_seconds", "?")

        tool_lines = [f"### ⚙️ Action Trajectory ({len(tool_calls)} steps)\n"]
        for i, tc in enumerate(tool_calls, 1):
            tool_lines += [
                f"**Step {i}: `{tc.get('tool', '?')}`**",
                f"- **In:** `{str(tc.get('input', ''))[:120]}`",
                f"- **Out:** `{str(tc.get('output', ''))[:150]}…`",
                "",
            ]
        tool_lines.append(f"*⏱️ Elapsed: **{elapsed}s***")

        evid_lines   = ["### 📑 Retrieved Evidence *(from tools)*\n"]
        has_evidence = False
        for tc in tool_calls:
            if "search" in tc.get("tool", ""):
                try:
                    chunks_out = json.loads(tc["output"])
                    for c in chunks_out[:3]:
                        evid_lines.append(
                            f"> ❝ *{c['chunk_text'][:200]}…* ❞\n> \n"
                            f"> — *(★{c['stars']}, ID: `{c['business_id'][:8]}…`)*\n\n---\n"
                        )
                    has_evidence = True
                    break
                except Exception:
                    pass
        if not has_evidence:
            evid_lines.append("*No retrieval chunks found in context.*")

        stats_md = ""
        for tc in tool_calls:
            if tc.get("tool") == "get_business_stats":
                try:
                    stats_md = _format_stats_dict(json.loads(tc["output"]))
                except Exception:
                    pass
                break
        if not stats_md:
            stats_md = _format_stats_from_id(business_id)

        yield (
            f"### 🤖 Agent Answer\n\n{answer}",
            "\n".join(tool_lines),
            "\n".join(evid_lines),
            stats_md,
        )
        return

    yield "Unknown system.", "", "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI  (identical layout to app.py)
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["What do customers complain about most at this business?",
     "Gaylord Opryland Resort & Convention Center — Nashville (★3.0, 275 chunks)",
     "RAG Baseline"],
    ["Analyze the main weaknesses of this business based on reviews.",
     "Santa Barbara Shellfish Company — Santa Barbara (★4.0, 203 chunks)",
     "Full Agent"],
    ["What aspects do customers praise and criticize about food and service?",
     "(Global search — no specific business)", "RAG Baseline"],
    ["Give me an overall profile of this business based on customer reviews.",
     "Gaylord Opryland Resort & Convention Center — Nashville (★3.0, 275 chunks)",
     "Full Agent"],
    ["Do Yelp reviews show any patterns between wait time complaints and star ratings?",
     "(Global search — no specific business)", "Full Agent"],
    ["Find reviews that mention slow service or long wait times.",
     "(Global search — no specific business)", "RAG Baseline"],
]

_CSS = """
#input-row { align-items: stretch !important; }
#examples-col .dataset, #examples-col .table-wrap {
    max-height: 680px !important;
    overflow-y: auto !important;
}
#examples-col thead th {
    position: sticky !important; top: 0 !important;
    background-color: white !important; z-index: 1 !important;
}
"""


def build_ui():
    with gr.Blocks(
        title="Yelp Business Intelligence Agent",
        theme=gr.themes.Soft(),
        css=_CSS,
    ) as demo:

        with gr.Row(elem_id="input-row"):
            with gr.Column(scale=3):
                gr.Markdown(
                    f"""
# Yelp Business Intelligence Agent
**Stage 4 of CSE4601 Text Mining Project**

Ask questions about Yelp businesses using three different AI systems:
- **Direct LLM** — {GROQ_MODEL} with no retrieval (baseline)
- **RAG Baseline** — Fixed retrieval pipeline (Stats → Search → Summarize)
- **Full Agent** — LangGraph ReAct agent with autonomous tool selection

> Vector store: 60,823 review chunks · 160 businesses (>50 reviews each) · Embeddings: all-MiniLM-L6-v2
"""
                )

                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What do customers complain about most at this business?",
                    lines=2,
                )
                business_dropdown = gr.Dropdown(
                    choices=DROPDOWN_CHOICES,
                    value="(Global search — no specific business)",
                    label="Select Business (160 available, sorted by review count)",
                    filterable=True,
                )
                business_id_input = gr.Textbox(
                    label="Business ID (auto-filled from dropdown, or type manually)",
                    placeholder="e.g. ORL4JE6tz3rJxVqkdKfegA",
                )
                system_input = gr.Dropdown(
                    choices=["RAG Baseline", "Full Agent", "Direct LLM"],
                    value="RAG Baseline",
                    label="System",
                )
                submit_btn = gr.Button("Run Query", variant="primary")

                business_dropdown.change(
                    fn=on_business_select,
                    inputs=business_dropdown,
                    outputs=business_id_input,
                )

            with gr.Column(scale=2, elem_id="examples-col"):
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[question_input, business_dropdown, system_input],
                    label="Quick Examples",
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=2):
                answer_output   = gr.Markdown(label="Answer")
                evidence_output = gr.Markdown(label="Retrieved Evidence")
            with gr.Column(scale=1):
                stats_output = gr.Markdown(label="Business Info")
                tools_output = gr.Markdown(label="Tool Calls")

        submit_btn.click(
            fn=run_query,
            inputs=[question_input, business_id_input, system_input],
            outputs=[answer_output, tools_output, evidence_output, stats_output],
        )

        gr.Markdown(
            """
---
**Evaluation Results** (20 questions × 3 systems, human scored):

| System | Correctness | Evidence | Groundedness | Tool Use | Efficiency | **Total /10** |
|---|---|---|---|---|---|---|
| Direct LLM | 0.25 | 0.00 | 0.00 | 0.00 | 1.70 | **1.95** |
| RAG Baseline | 0.95 | 1.60 | 1.75 | 0.95 | 1.65 | **6.90** |
| Full Agent | 1.05 | 1.15 | 1.15 | 1.30 | 0.10 | **4.75** |

Hallucination Rate: Direct LLM **100%** → Full Agent **25%** → RAG Baseline **5%**
"""
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port)
