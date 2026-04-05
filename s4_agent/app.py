"""
Stage 4 — Yelp Business Intelligence Agent
Gradio Demo Interface

Usage:
    python s4_agent/app.py
    python s4_agent/app.py --share      # public link
"""

import sys, os, time, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from s4_agent.config import VECTORSTORE_META
from s4_agent.pipelines.rag_baseline import run_rag_pipeline
from s4_agent.pipelines.agent_runner import run_agent

# ---------------------------------------------------------------------------
# Load business catalogue at startup
# ---------------------------------------------------------------------------

def _load_business_catalogue() -> dict:
    """
    Returns {business_id: {name, city, stars, categories, chunk_count}}
    for every business with >50 review chunks in the vector store.
    """
    BIZ_JSON = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "raw", "yelp_academic_dataset_business.json",
    )

    # Step 1 — eligible IDs from vector store
    with open(VECTORSTORE_META, "rb") as f:
        store = pickle.load(f)
    b2i = store["business_to_indices"]
    eligible = {bid: len(idxs) for bid, idxs in b2i.items() if len(idxs) > 50}

    # Step 2 — enrich with names from business JSON
    catalogue = {}
    with open(BIZ_JSON, encoding="utf-8") as f:
        for line in f:
            biz = json.loads(line)
            bid = biz.get("business_id", "")
            if bid in eligible:
                cats = (biz.get("categories") or "").split(", ")
                primary_cat = cats[0] if cats else "Business"
                catalogue[bid] = {
                    "name"       : biz.get("name", bid),
                    "city"       : biz.get("city", ""),
                    "stars"      : biz.get("stars", 0),
                    "categories" : primary_cat,
                    "chunk_count": eligible[bid],
                }
    return catalogue


print("Loading business catalogue…")
CATALOGUE = _load_business_catalogue()

# Build sorted dropdown choices: "Name — City (★X.X, N chunks)"
# Sorted by chunk count descending so busiest businesses appear first
_sorted_bizs = sorted(CATALOGUE.items(), key=lambda x: -x[1]["chunk_count"])
DROPDOWN_CHOICES = ["(Global search — no specific business)"] + [
    f"{info['name']} — {info['city']} (★{info['stars']}, {info['chunk_count']} chunks)"
    for _, info in _sorted_bizs
]
# Map label → business_id
LABEL_TO_ID = {
    f"{info['name']} — {info['city']} (★{info['stars']}, {info['chunk_count']} chunks)": bid
    for bid, info in _sorted_bizs
}
# Map business_id → label (for stats panel)
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

print(f"Catalogue loaded: {len(CATALOGUE)} businesses available.")

# ---------------------------------------------------------------------------
# Example questions
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["What do customers complain about most at this business?",
     "Gaylord Opryland Resort & Convention Center — Nashville (★3.0, 275 chunks)",
     "RAG Baseline"],
    ["Analyze the main weaknesses of this business based on reviews.",
     "Santa Barbara Shellfish Company — Santa Barbara (★4.0, 203 chunks)",
     "Full Agent"],
    ["What aspects do customers praise and criticize about food and service?",
     "(Global search — no specific business)",
     "RAG Baseline"],
    ["Give me an overall profile of this business based on customer reviews.",
     "Gaylord Opryland Resort & Convention Center — Nashville (★3.0, 275 chunks)",
     "Full Agent"],
    ["Do Yelp reviews show any patterns between wait time complaints and star ratings?",
     "(Global search — no specific business)",
     "Full Agent"],
    ["Find reviews that mention slow service or long wait times.",
     "(Global search — no specific business)",
     "RAG Baseline"],
]

# ---------------------------------------------------------------------------
# Handler: dropdown selection → fill business_id text box
# ---------------------------------------------------------------------------

def on_business_select(label: str) -> str:
    if label == "(Global search — no specific business)" or not label:
        return ""
    return LABEL_TO_ID.get(label, "")


# ---------------------------------------------------------------------------
# Core query handler
# ---------------------------------------------------------------------------

def run_query(question: str, business_id: str, system: str):
    """
    Dispatch to the selected pipeline and return formatted outputs.
    Returns: (answer_md, tools_md, evidence_md, stats_md)
    """
    question = question.strip()
    business_id = business_id.strip() or None

    if not question:
        yield "Please enter a question.", "", "", ""
        return

    # ==========================================
    # 第一步：瞬间更新 UI，显示进度提示
    # ==========================================
    yield (
        f"⏳ **Processing query... Please wait.**\n\n*(Running via {system}... This may take a few seconds)*",
        "⏳ *Waiting for tools...*",
        "⏳ *Waiting for retrieval...*",
        "⏳ *Fetching stats...*"
    )

    # ---- Direct LLM -------------------------------------------------------
    if system == "Direct LLM":
        import requests
        t0 = time.time()
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "qwen2.5:7b",
                    "messages": [{"role": "user", "content": question}],
                    "stream": False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"]
        except Exception as e:
            answer = f"Error: {e}"
        elapsed = round(time.time() - t0, 2)

        answer_md = f"### 🤖 Answer *(Direct LLM — no retrieval)*\n\n{answer}"
        tools_md = f"No tools used.\n\n*⏱️ Elapsed: **{elapsed}s***"
        evid_md = "*Direct LLM does not retrieve evidence.*"
        stats_md = _format_stats_from_id(business_id)
        yield answer_md, tools_md, evid_md, stats_md
        return

    # ---- RAG Baseline -----------------------------------------------------
    if system == "RAG Baseline":
        try:
            result = run_rag_pipeline(question, business_id=business_id)
        except Exception as e:
            yield f"Pipeline error: {e}", "", "", ""
            return

        synthesis = result.get("synthesis", {})
        findings = synthesis.get("main_findings", [])
        evidence = synthesis.get("supporting_evidence", [])
        uncertain = synthesis.get("uncertainties", [])
        elapsed = result.get("elapsed_seconds", "?")
        tools = result.get("tools_called", [])
        stats = result.get("business_stats")
        chunks = result.get("retrieved_chunks", [])

        # ---- Answer (RAG) ----
        answer_lines = ["### 🎯 Main Findings\n"]
        for f in findings:
            answer_lines.append(f"- {f}")
        if uncertain:
            answer_lines.append("\n### ❓ Uncertainties\n")
            for u in uncertain:
                answer_lines.append(f"- {u}")
        answer_md = "\n".join(answer_lines)

        # ---- Tools (RAG) ----
        tool_lines = [f"### ⚙️ Tools Called ({len(tools)} steps)\n"]
        for i, t in enumerate(tools, 1):
            tool_lines.append(f"{i}. `{t}`")
        tool_lines.append(f"\n*⏱️ Elapsed: **{elapsed}s***")
        tools_md = "\n".join(tool_lines)

        # ---- Evidence (RAG) ----
        evid_lines = [f"### 📑 Retrieved Evidence ({len(chunks)} total)\n"]
        for item in evidence[:5]:
            claim = item.get("claim", "")
            evid_lines.append(f"**📌 Claim:** {claim}\n")
            for quote in item.get("evidence", [])[:2]:
                evid_lines.append(f"> ❝ *{quote}* ❞\n")
            evid_lines.append("---\n")  # 水平分割线

        if not evidence and chunks:
            for c in chunks[:3]:
                evid_lines.append(
                    f"> ❝ *{c['chunk_text'][:200]}…* ❞\n> \n> — *(★{c['stars']})*\n\n---\n"
                )
        evid_md = "\n".join(evid_lines)

        # Stats
        stats_md = _format_stats_dict(stats) if stats else _format_stats_from_id(business_id)

        yield answer_md, tools_md, evid_md, stats_md
        return

    # ---- Full Agent -------------------------------------------------------
    if system == "Full Agent":
        try:
            result = run_agent(question, business_id=business_id, max_iterations=6)
        except Exception as e:
            yield f"Agent error: {e}", "", "", ""
            return

        answer = result.get("final_answer") or result.get("answer", "*(No answer)*")
        tool_calls = result.get("tool_calls", [])
        elapsed = result.get("elapsed_seconds", "?")

        # ---- Answer (Agent) ----
        answer_md = f"### 🤖 Agent Answer\n\n{answer}"

        # ---- Tools (Agent) ----
        tool_lines = [f"### ⚙️ Action Trajectory ({len(tool_calls)} steps)\n"]
        for i, tc in enumerate(tool_calls, 1):
            name = tc.get("tool", "?")
            inp = tc.get("input", "")
            out = tc.get("output", "")
            tool_lines.append(f"**Step {i}: `{name}`**")
            tool_lines.append(f"- **In:** `{str(inp)[:120]}`")
            if out:
                tool_lines.append(f"- **Out:** `{str(out)[:150]}…`")
            tool_lines.append("")
        tool_lines.append(f"*⏱️ Elapsed: **{elapsed}s***")
        tools_md = "\n".join(tool_lines)

        # ---- Evidence (Agent) ----
        evid_lines = ["### 📑 Retrieved Evidence *(from tools)*\n"]
        has_evidence = False
        for tc in tool_calls:
            if "search" in tc.get("tool", ""):
                try:
                    chunks = json.loads(tc["output"])
                    for c in chunks[:3]:
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
        evid_md = "\n".join(evid_lines)

        # Stats — prefer tool output, fall back to catalogue
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

        yield answer_md, tools_md, evid_md, stats_md
        return

    yield "Unknown system.", "", "", ""


# ---------------------------------------------------------------------------
# Stats formatting helpers
# ---------------------------------------------------------------------------

def _format_stats_dict(s: dict) -> str:
    if not s or s.get("review_count", 0) == 0:
        return ""
    bid = s.get("business_id", "")
    dist = s.get("star_distribution", {})
    # 优化星级分布的显示
    dist_str = " &nbsp;\|&nbsp; ".join(f"★{k}: **{v}**" for k, v in sorted(dist.items()))
    label = ID_TO_LABEL.get(bid, "")

    city = ""
    name = bid
    if label:
        name = label.split(' — ')[0]
        if ' — ' in label:
            city = label.split(' — ')[1].split(' (')[0]

    md = (
        f"### 📊 Business Stats\n\n"
        f"| Metric | Details |\n"
        f"| :--- | :--- |\n"
        f"| **🏢 Name** | {name} |\n"
        f"| **🏙️ City** | {city} |\n"
        f"| **🆔 ID** | `{bid}` |\n"
        f"| **📝 Indexed Reviews** | {s['review_count']} chunks |\n"
        f"| **⭐ Avg Stars** | {s['avg_stars']} |\n"
        f"| **📈 Distribution** | {dist_str} |\n"
    )
    return md


def _format_stats_from_id(business_id: str | None) -> str:
    if not business_id or business_id not in CATALOGUE:
        return ""
    info = CATALOGUE[business_id]
    return (
        f"### 📊 Business Info *(Catalogue)*\n\n"
        f"| Metric | Details |\n"
        f"| :--- | :--- |\n"
        f"| **🏢 Name** | {info['name']} |\n"
        f"| **🏙️ City** | {info['city']} |\n"
        f"| **🏷️ Category** | {info['categories']} |\n"
        f"| **⭐ Yelp Stars** | {info['stars']} |\n"
        f"| **📝 Indexed Reviews** | {info['chunk_count']} chunks |\n"
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CSS = """
/* 让右侧 Column 的外层容器对齐 */
#input-row {
    align-items: stretch !important;
}

/* 核心修复：限制表格区域的最大高度，并开启垂直滚动 */
#examples-col .dataset,
#examples-col .table-wrap {
    max-height: 680px !important; /* 这个高度大约等于左侧所有输入框+按钮的总高度 */
    overflow-y: auto !important;  /* 强制内容过多时出现上下滚动条 */
}

/* 优化体验：让表格的表头在滚动时固定在顶部（吸顶效果） */
#examples-col thead th {
    position: sticky !important;
    top: 0 !important;
    background-color: white !important;
    z-index: 1 !important;
}
"""


def build_ui():
    with gr.Blocks(
        title="Yelp Business Intelligence Agent",
        theme=gr.themes.Soft(),
        css=_CSS,
    ) as demo:

        # gr.Markdown(
        #     """
        #     # Yelp Business Intelligence Agent
        #     **Stage 4 of CSE4601 Text Mining Project**
        #
        #     Ask questions about Yelp businesses using three different AI systems:
        #     - **Direct LLM** — Qwen2.5-7B with no retrieval (baseline)
        #     - **RAG Baseline** — Fixed retrieval pipeline (Stats → Search → Summarize)
        #     - **Full Agent** — LangGraph ReAct agent with autonomous tool selection
        #
        #     > Vector store: 60,823 review chunks · 160 businesses (>50 reviews each) · Embeddings: all-MiniLM-L6-v2
        #     """
        # )

        with gr.Row(elem_id="input-row"):
            # ---- Left: inputs ----
            with gr.Column(scale=3):
                # 【修改点】将大标题直接放在左侧列的第一个位置
                gr.Markdown(
                    """
                    # Yelp Business Intelligence Agent
                    **Stage 4 of CSE4601 Text Mining Project**

                    Ask questions about Yelp businesses using three different AI systems:
                    - **Direct LLM** — Qwen2.5-7B with no retrieval (baseline)
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

                # Wire dropdown → business_id text box
                business_dropdown.change(
                    fn=on_business_select,
                    inputs=business_dropdown,
                    outputs=business_id_input,
                )

            # # ---- Right: quick examples ----
            # with gr.Column(scale=2, elem_id="examples-col"):
            #     gr.Markdown("### Quick Examples")
            #     gr.Examples(
            #         examples=EXAMPLES,
            #         inputs=[question_input, business_dropdown, system_input],
            #         label="",
            #     )

            # ---- Right: quick examples ----
            with gr.Column(scale=2, elem_id="examples-col"):
                # 【修改点 1】删除了单独的 gr.Markdown("### Quick Examples")
                # 【修改点 2】将标题直接写在 label 参数里
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[question_input, business_dropdown, system_input],
                    label="Quick Examples",
                )

        gr.Markdown("---")

        # ---- Output panels ----
        with gr.Row():
            # 左侧列：占 2/3 宽度 (scale=2)
            with gr.Column(scale=2):
                answer_output = gr.Markdown(label="Answer")
                evidence_output = gr.Markdown(label="Retrieved Evidence")

            # 右侧列：占 1/3 宽度 (scale=1)
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Starting Yelp Business Intelligence Agent demo...")
    print(f"  Port  : {args.port}")
    print(f"  Share : {args.share}")
    print(f"  Businesses in dropdown: {len(CATALOGUE)}")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
