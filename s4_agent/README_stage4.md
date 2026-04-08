# Stage 4: Yelp Business Intelligence Agent

**CSE4601 Text Mining Project — Stage 4**

A production-style AI agent that answers natural language questions about Yelp businesses by autonomously selecting and invoking tools over a 60,823-chunk vector knowledge base. Evaluated against a RAG baseline and a no-retrieval LLM through a structured three-way benchmark.

---

## System Architecture

Stage 4 implements a two-layer architecture:

```
User Question
      │
      ├──► [Layer 1] RAG Baseline (fixed flow)
      │         get_business_stats
      │              │
      │         search_by_business / search_global
      │              │
      │         summarize_evidence
      │              │
      │         Structured Answer
      │
      └──► [Layer 2] Full Agent (autonomous)
                LangGraph ReAct Loop
                ┌─────────────────────────────┐
                │  Qwen2.5-7B (via Ollama)    │
                │  ┌─────────────────────┐    │
                │  │ Decide which tool   │    │
                │  └──────────┬──────────┘    │
                │             │               │
                │    ┌────────▼────────┐      │
                │    │   Tool Call     │      │
                │    └────────┬────────┘      │
                │             │               │
                │    ┌────────▼────────┐      │
                │    │ Observe Result  │      │
                │    └────────┬────────┘      │
                │             │ (repeat)      │
                └─────────────┼───────────────┘
                              │
                         Final Answer
```

---

## Demo

🚀 **Live Demo:** [huggingface.co/spaces/YUHAN03/yelp-agent](https://huggingface.co/spaces/YUHAN03/yelp-agent)

**RAG Baseline** — Fixed retrieval pipeline (Stats → Search → Summarize):

![RAG Baseline Demo](RAG_Baseline.gif)

**Full Agent** — LangGraph ReAct agent with autonomous tool selection:

![Full Agent Demo](Full_Agent.gif)

---

## Technical Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph `create_react_agent` |
| Agent Brain | Qwen2.5-7B-Instruct via Ollama |
| Vector Store | FAISS `IndexFlatIP` (cosine similarity) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Review Classifier | RoBERTa-base fine-tuned 5-class (Stage 2 best run) |
| LLM Orchestration | LangChain + `langchain-ollama` |
| Demo Interface | Gradio 5 |
| Data | Yelp Academic Dataset — 50k reviews sampled |

---

## Quick Start

### Prerequisites

```bash
# 1. Activate the project environment
conda activate yelp_nlp

# 2. Ensure Ollama is running with Qwen2.5-7B
ollama serve
ollama pull qwen2.5:7b
```

### Build the Vector Store (first time only)

```bash
python s4_agent/vectorstore/build_vectorstore.py
```

This encodes 50k reviews into 60,823 chunks and saves the FAISS index to `s4_agent/vectorstore/`.

### Train and Save the Classifier (first time only)

```bash
python s4_agent/step0_train_and_save.py
```

Re-trains the Stage 2 best-run RoBERTa config and saves the checkpoint to `s4_agent/artifacts/roberta_5class_best/`.

### Launch the Demo

```bash
python s4_agent/app.py
# Open http://localhost:7860
```

Add `--share` for a temporary public link:

```bash
python s4_agent/app.py --share
```

### Run the Evaluation

```bash
# Generate answers for all 60 combinations (20 questions × 3 systems)
python s4_agent/evaluation/run_eval.py --run

# Print the summary table (after manual scoring)
python s4_agent/evaluation/run_eval.py --summarise
```

---

## Tools

The agent has access to four tools:

| Tool | Description |
|---|---|
| `get_business_stats` | Returns review count, average stars, and star distribution for a business |
| `search_review_chunks_by_business` | Semantic search restricted to a single business (pre-filtered FAISS) |
| `search_review_chunks_global` | Global semantic search across all 60,823 chunks |
| `summarize_evidence` | Calls Qwen via Ollama to synthesize retrieved chunks into structured findings |
| `classify_review` | Runs RoBERTa 5-class classifier on a review text |

The two retrieval tools are split intentionally: giving a 7B model one focused tool per scope significantly improves tool selection reliability.

---

## Evaluation Results

**Setup:** 20 questions × 4 types × 3 systems = 60 evaluated responses.
Human scored on 5 dimensions (0–2 scale each, max 10 points total).

### Score Breakdown

| System | Correctness | Evidence | Groundedness | Tool Use | Efficiency | **Total** |
|---|---|---|---|---|---|---|
| Direct LLM | 0.25 | 0.00 | 0.00 | 0.00 | 1.70 | **1.95** |
| RAG Baseline | 0.95 | 1.60 | 1.75 | 0.95 | 1.65 | **6.90** |
| Full Agent | 1.05 | 1.15 | 1.15 | 1.30 | 0.10 | **4.75** |

### Hallucination Rate (Groundedness = 0)

| System | Rate |
|---|---|
| Direct LLM | 100% |
| Full Agent | 25% |
| RAG Baseline | 5% |

### Correctness by Question Type

| Question Type | Direct LLM | RAG Baseline | Full Agent |
|---|---|---|---|
| Complaint Mining | 0.00 | 1.00 | 0.80 |
| Aspect Analysis | 0.00 | 0.80 | 1.20 |
| Business Profiling | 0.00 | 1.00 | 1.00 |
| Cross-Business Pattern | 1.00 | 1.00 | 1.20 |

### Key Findings

- **RAG Baseline** achieves the highest total score (6.90/10) due to reliable tool sequencing, rich evidence citation, and the lowest hallucination rate (5%). Its fixed flow eliminates the instability introduced by autonomous decision-making in small LLMs.
- **Full Agent** scores highest on correctness for complex question types (Aspect Analysis, Cross-Business Pattern) where free tool selection helps, but suffers from high latency (~45s avg) and a 25% hallucination rate — a known limitation of 7B models under ReAct prompting.
- **Direct LLM** cannot access review data at all, producing generic responses with 100% hallucination rate. Its only advantage is speed (9s avg).

---

## Scoring Rubric

Five dimensions, 0–2 scale:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Correctness | Off-topic / wrong | Partially correct | Accurate and complete |
| Evidence Support | No citations | Vague reference | 2+ specific review quotes |
| Groundedness | Multiple fabrications | Mostly supported | Every claim traceable |
| Tool Use | Wrong tool / none | Correct but incomplete | Right tools in right order |
| Efficiency | Highly redundant | Acceptable | Minimal calls, fast response |

Full rubric: [`s4_agent/evaluation/rubric.md`](evaluation/rubric.md)

---

## Project Structure

```
s4_agent/
├── app.py                          # Gradio demo interface
├── config.py                       # Centralized path and model config
├── step0_train_and_save.py         # Re-train and save RoBERTa checkpoint
├── test_classifier_load.py         # Verify classifier loads correctly
│
├── artifacts/
│   └── roberta_5class_best/        # Saved RoBERTa model (Stage 2 best run)
│
├── vectorstore/
│   ├── build_vectorstore.py        # Build FAISS index from 50k reviews
│   ├── review_chunks.index         # FAISS IndexFlatIP (384-dim)
│   └── review_chunks.pkl           # Chunk metadata + business_to_indices map
│
├── tools/
│   ├── retrieval_tool.py           # Global and business-filtered search tools
│   ├── stats_tool.py               # Business statistics tool
│   ├── classifier_tool.py          # RoBERTa inference tool
│   └── summarizer_tool.py          # LLM evidence synthesis tool
│
├── pipelines/
│   ├── rag_baseline.py             # Fixed-flow RAG pipeline
│   └── agent_runner.py             # LangGraph ReAct agent
│
├── evaluation/
│   ├── test_questions.json         # 20 evaluation questions (4 types × 5)
│   ├── rubric.md                   # Scoring rubric (5 dimensions, 0-2 scale)
│   └── run_eval.py                 # Evaluation runner + summariser
│
└── results/
    └── eval_results.csv            # 60 evaluated responses with human scores
```

---

## Notable Design Decisions

**Chunk-level pre-filtering over global Top-K**
When a `business_id` is provided, the retrieval tool loads only that business's embedding subset and runs matrix multiplication locally, rather than doing a global FAISS search followed by post-filtering. This avoids result dilution for businesses with few reviews.

**Tool splitting by retrieval scope**
Giving the 7B agent one tool for business-scoped search and one for global search (instead of a single multi-purpose tool) reduces ambiguous tool selection errors under ReAct prompting.

**Three-layer JSON parsing in the summarizer**
Qwen occasionally produces malformed JSON (curly quotes from review text, truncated responses). The `summarize_evidence` tool handles this with three successive fallback strategies before returning a safe default.

**Resume mechanism in the evaluator**
The evaluation runner skips already-completed rows when restarted, preventing duplicate work after Ollama crashes during long batch runs.

---

## Portfolio Highlights

This stage demonstrates the following skills relevant to AI Agent engineering roles:

- **RAG pipeline design**: chunk-level indexing, pre-filtered semantic retrieval, evidence-grounded synthesis
- **LangGraph agent implementation**: ReAct loop, tool binding, message trace extraction
- **Tool design for small LLMs**: decomposing tool scope to improve 7B model reliability
- **Structured evaluation**: rubric design, three-way system comparison, hallucination rate measurement
- **Production-readiness patterns**: resume on failure, multi-layer error handling, config centralisation
- **Interactive demo**: Gradio app with 160-business searchable dropdown and live tool call logs
