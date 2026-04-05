"""
Stage 4 - Step 4: Full Agent Pipeline (LangGraph ReAct)

Gives Qwen2.5-7B-Instruct (via Ollama) autonomous control over all 5 tools
and lets it decide the call order based on the question.

Registered tools:
    search_review_chunks_global        — full-corpus semantic search
    search_review_chunks_by_business   — pre-filtered search for one business
    get_business_stats                 — star distribution for one business
    classify_review                    — RoBERTa 5-class prediction
    summarize_evidence                 — Qwen synthesis over retrieved chunks

Compared to the RAG baseline, the Agent can:
    - Call tools in any order
    - Call the same tool multiple times with different queries
    - Skip tools that are not relevant to the question
    - Call classify_review on individual chunks for deeper analysis

Return schema:
    {
        "question"        : str,
        "business_id"     : str | None,
        "final_answer"    : str,
        "tool_calls"      : [{"tool": str, "input": str, "output": str}],
        "steps"           : int,
        "elapsed_seconds" : float
    }

Usage:
    python s4_agent/pipelines/agent_runner.py
"""

import sys
import time
from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from tools.retrieval_tool import (
    search_review_chunks_global,
    search_review_chunks_by_business,
)
from tools.stats_tool import get_business_stats
from tools.classifier_tool import classify_review
from tools.summarizer_tool import summarize_evidence

# ---------------------------------------------------------------------------
# System prompt — gives the agent context and guides tool selection
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Build agent (lazy singleton)
# ---------------------------------------------------------------------------

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        print("[agent_runner] Initialising LangGraph ReAct agent …")
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0,
        )
        tools = [
            search_review_chunks_global,
            search_review_chunks_by_business,
            get_business_stats,
            classify_review,
            summarize_evidence,
        ]
        _agent = create_react_agent(llm, tools)
        print("[agent_runner] Agent ready.")
    return _agent


# ---------------------------------------------------------------------------
# Trace extraction from LangGraph message history
# ---------------------------------------------------------------------------

def _extract_trace(messages: list) -> tuple[str, list[dict]]:
    """
    Parse the LangGraph message list and return:
        final_answer : str   — last AIMessage content with no tool_calls
        tool_calls   : list  — [{tool, input, output}] in order of execution
    """
    tool_calls: list[dict] = []
    final_answer = ""

    # Build a map from tool_call_id -> ToolMessage for quick lookup
    tool_msg_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Truncate very long tool outputs for readability
            content = str(msg.content)
            tool_msg_map[msg.tool_call_id] = (
                content[:600] + " …[truncated]" if len(content) > 600 else content
            )

    # Walk AIMessages to collect tool calls in execution order
    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        {
                            "tool"  : tc["name"],
                            "input" : str(tc["args"]),
                            "output": tool_msg_map.get(tc["id"], "(no output)"),
                        }
                    )
            elif msg.content:
                # An AIMessage with content and no tool_calls is the final answer
                final_answer = msg.content

    return final_answer, tool_calls


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_agent(
    question: str,
    business_id: Optional[str] = None,
    max_iterations: int = 10,
) -> dict:
    """
    Run the full ReAct agent on a question.

    Args:
        question       : Natural language question from the user.
        business_id    : Optional Yelp business ID.  If provided, it is
                         appended to the question so the agent knows to use
                         business-specific tools.
        max_iterations : Hard cap on tool-call rounds to prevent loops.

    Returns:
        Structured result dict (see module docstring for schema).
    """
    # Inject business_id into the question text if provided
    full_question = question
    if business_id:
        full_question = f"{question}\n\n[Target business_id: {business_id}]"

    print(f"\n{'='*60}")
    print(f"Agent Pipeline  |  model={OLLAMA_MODEL}")
    print(f"Question: {question}")
    if business_id:
        print(f"Business ID: {business_id}")
    print(f"{'='*60}")

    agent = _get_agent()

    t0 = time.time()
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=full_question),
            ]
        },
        config={"recursion_limit": max_iterations * 2},
    )
    elapsed = round(time.time() - t0, 2)

    messages = result["messages"]
    final_answer, tool_calls = _extract_trace(messages)

    print(f"\n  Tool calls ({len(tool_calls)}):")
    for i, tc in enumerate(tool_calls, 1):
        print(f"    [{i}] {tc['tool']}({tc['input'][:60]}…)")
    print(f"  Elapsed: {elapsed}s")

    return {
        "question"       : question,
        "business_id"    : business_id,
        "final_answer"   : final_answer,
        "tool_calls"     : tool_calls,
        "steps"          : len(tool_calls),
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"Steps taken     : {result['steps']}")
    print(f"Elapsed         : {result['elapsed_seconds']}s")

    print(f"\nTool call trace:")
    for i, tc in enumerate(result["tool_calls"], 1):
        print(f"  [{i}] {tc['tool']}")
        print(f"       Input  : {tc['input'][:100]}")
        print(f"       Output : {tc['output'][:150]}…")

    print(f"\nFinal answer:")
    # Indent each line for readability
    for line in result["final_answer"].splitlines():
        print(f"  {line}")


# ---------------------------------------------------------------------------
# Standalone test — same 4 question types as RAG baseline for comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    LOW_SCORE_BIZ  = "ORL4JE6tz3rJxVqkdKfegA"   # 182 reviews, avg 2.42★
    HIGH_SCORE_BIZ = "2KIDQyTh-HzLxOUEDqtDBg"   # 62 reviews,  avg 4.13★

    TEST_CASES = [
        {
            "question"   : "What do customers complain about most at this business?",
            "business_id": LOW_SCORE_BIZ,
            "label"      : "Complaint Mining",
        },
        {
            "question"   : "How do customers describe the service and staff at this business?",
            "business_id": HIGH_SCORE_BIZ,
            "label"      : "Aspect Analysis",
        },
        {
            "question"   : "Give an overall profile of this business based on customer reviews.",
            "business_id": LOW_SCORE_BIZ,
            "label"      : "Business Profiling",
        },
        {
            "question"   : "What service problems appear most often in low-rated reviews?",
            "business_id": None,
            "label"      : "Cross-Business Pattern Search",
        },
    ]

    for case in TEST_CASES:
        print(f"\n{'#'*60}")
        print(f"TEST: {case['label']}")

        result = run_agent(
            question    = case["question"],
            business_id = case.get("business_id"),
        )
        print_result(result)

    print(f"\n{'='*60}")
    print("Agent tests complete.")
    print("Next: build the evaluation suite  (s4_agent/evaluation/)")
