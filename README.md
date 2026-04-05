# Yelp Review Rating Prediction: Baselines, BERT Fine-Tuning, LLM Evaluation, and Business Intelligence Agent

This repository contains a multi-stage NLP project for **Yelp review analysis**. The project is organized as a progressive pipeline:

1. **Stage 1 — Baseline comparison**
   Compare traditional sparse features, frozen sentence embeddings, and a transformer baseline.
2. **Stage 2 — BERT fine-tuning**
   Systematically explore backbone choices, task settings, and hyperparameters.
3. **Stage 3 — LLM evaluation**
   Benchmark instruction-tuned large language models on the same review rating task.
4. **Stage 4 — Business Intelligence Agent**
   Build a production-style RAG + LangGraph ReAct agent that answers natural language questions about Yelp businesses, evaluated through a three-way benchmark.

The overall goal is to understand how performance and capability evolve as we move from:

- feature engineering,
- to fixed pretrained semantic representations,
- to end-to-end transformer fine-tuning,
- to direct prompting with modern LLMs,
- and finally to agentic retrieval-augmented systems.

---

## Project Structure

```text
Yelp_Project/
├── .idea/
├── data/
├── s1_baseline_models/
├── s1_baseline_results/
├── s1_baseline_scripts/
│   ├── baseline_bert.ipynb
│   ├── baseline_embedding.ipynb
│   ├── baseline_tfidf.ipynb
│   ├── preprocess_and_split.ipynb
│   └── sample_data.ipynb
├── s2_bert_models/
├── s2_bert_results/
├── s2_bert_scripts/
│   ├── config.py
│   ├── data_loader.py
│   ├── run_colab.sh
│   ├── run_experiments.py
│   ├── run_local.sh
│   ├── train.py
│   └── utils.py
├── s3_LLM/
│   ├── stage3_deepseek_2class_direct_benchmark.ipynb
│   ├── stage3_deepseek_5class_direct_benchmark.ipynb
│   ├── stage3_deepseek_2class_improved_benchmark.ipynb
│   ├── stage3_deepseek_5class_improved_benchmark.ipynb
│   ├── stage3_deepseek_generate_thinking_chains.ipynb
│   ├── stage3_deepseek_2class_finetune_bf16.ipynb
│   ├── stage3_deepseek_5class_finetune_bf16.ipynb
│   ├── stage3_qwen_2class_direct_benchmark.ipynb
│   ├── stage3_qwen_5class_direct_benchmark.ipynb
│   ├── stage3_qwen_2class_finetune.ipynb
│   ├── stage3_qwen_5class_finetune.ipynb
│   ├── stage3_qwen_2class_finetune_bf16.ipynb
│   ├── stage3_qwen_5class_finetune_bf16.ipynb
│   └── experiment output folders
└── s4_agent/
    ├── app.py
    ├── config.py
    ├── step0_train_and_save.py
    ├── artifacts/roberta_5class_best/
    ├── vectorstore/
    ├── tools/
    ├── pipelines/
    ├── evaluation/
    └── results/
```

---

## Stage 1: Baseline Comparison

Stage 1 establishes the initial comparison among three modeling paradigms.

### 1. TF-IDF baseline
**File:** `s1_baseline_scripts/baseline_tfidf.ipynb`

- Text representation: TF-IDF features
- Classifier: Logistic Regression
- Purpose: provide a traditional sparse-feature baseline

### 2. Frozen sentence embedding baseline
**File:** `s1_baseline_scripts/baseline_embedding.ipynb`

- Text representation: Sentence-BERT embeddings
- Classifier: Logistic Regression
- Purpose: test a fixed pretrained semantic representation without end-to-end fine-tuning

### 3. Transformer baseline
**File:** `s1_baseline_scripts/baseline_bert.ipynb`

- Backbone: transformer classifier baseline
- Training style: end-to-end fine-tuning
- Purpose: show whether a task-adaptive transformer is stronger than the two earlier baselines

### Supporting notebooks
- `preprocess_and_split.ipynb`: data cleaning and train/validation/test split
- `sample_data.ipynb`: sample extraction or quick dataset inspection

### Stage 1 outputs
Stage 1 stores artifacts in:

- `s1_baseline_models/`
- `s1_baseline_results/`

The baseline notebooks are configured to save both **validation** and **test** results, including metrics, classification reports, confusion matrices, and prediction files.

### Stage 1 Key Results (5-class, test set)

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 56.92% | 56.56% |
| Sentence-BERT + Logistic Regression | 53.25% | 52.89% |
| BERT (fine-tuned) | 62.50% | 62.50% |

---

## Stage 2: BERT Fine-Tuning Experiments

Stage 2 expands the transformer experiments into a more systematic fine-tuning framework.

**Folder:** `s2_bert_scripts/`

### Main components
- `config.py` — default experiment configuration
- `data_loader.py` — dataset loading, task mapping, and tokenization helpers
- `train.py` — single-run training pipeline
- `utils.py` — path setup, metrics, and result saving utilities
- `run_experiments.py` — CLI entry point for launching experiments
- `run_local.sh` — local batch runs
- `run_colab.sh` — Google Colab batch runs

### Experiment dimensions and hyperparameter search space

| Dimension | Values explored |
|---|---|
| Backbone model | `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base` |
| Task setting | binary (2-class), 3-class, 5-class |
| Mode | Classification, Regression |
| Learning rate | 1e-5, 2e-5 |
| Batch size | 4, 8, 16 |
| Epochs | 3, 4 |

Total runs: 216 across all combinations.

### Best model hyperparameters

**Best binary classifier** (RoBERTa-base, Accuracy 97.82%)

| Hyperparameter | Value |
|---|---|
| Model | `roberta-base` |
| Mode | Classification |
| Learning rate | 1e-5 |
| Batch size | 4 |
| Epochs | 3 |
| AUC | 0.9968 |

**Best 5-class classifier** (RoBERTa-base, Accuracy 68.56%)

| Hyperparameter | Value |
|---|---|
| Model | `roberta-base` |
| Mode | Classification |
| Learning rate | 1e-5 |
| Batch size | 4 |
| Epochs | 4 |
| Off-by-1 Accuracy | 98.19% |

### Stage 2 outputs
- `s2_bert_models/` — experiment-specific model/checkpoint directories
- `s2_bert_results/` — metrics, logs, reports, and evaluation artifacts

This stage is intended to identify stronger transformer settings than the Stage 1 baseline.

Due to memory limitations on the computing cluster, model checkpoints are not saved during Stage 2 experiments. As a result, this stage only preserves evaluation outputs such as metrics, logs, and reports. If checkpoint saving or model reloading is required, please modify the corresponding code files before running the experiments.

### Stage 2 Key Results (best run per model, test set)

**Binary (2-class):**

| Model | Best Accuracy | Best Macro F1 |
|---|---|---|
| DistilBERT-base-uncased | 96.54% | 96.53% |
| BERT-base-uncased | 97.07% | 97.06% |
| RoBERTa-base | **97.82%** | **97.81%** |

**3-class:**

| Model | Best Accuracy | Best Macro F1 |
|---|---|---|
| DistilBERT-base-uncased | 82.61% | 78.36% |
| BERT-base-uncased | 83.16% | 78.75% |
| RoBERTa-base | **84.52%** | **80.90%** |

**5-class:**

| Model | Best Accuracy | Best Macro F1 | Off-by-1 Accuracy |
|---|---|---|---|
| DistilBERT-base-uncased | 65.43% | 65.36% | 97.35% |
| BERT-base-uncased | 66.75% | 66.36% | 97.49% |
| RoBERTa-base | **68.56%** | **68.49%** | **98.19%** |

---

## Stage 3: LLM Evaluation and Fine-Tuning

Stage 3 evaluates and fine-tunes instruction-tuned LLMs on the review rating task. It follows a three-tier progression for each model: zero-shot inference → improved prompting → LoRA fine-tuning.

**Folder:** `s3_LLM/`

**Models:**
- `Qwen/Qwen2.5-7B-Instruct`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

---

### Tier 1 — Zero-Shot Direct Benchmarking

Evaluates raw out-of-the-box performance with a minimal prompt and no examples.

- `stage3_qwen_2class_direct_benchmark.ipynb`
- `stage3_qwen_5class_direct_benchmark.ipynb`
- `stage3_deepseek_2class_direct_benchmark.ipynb`
- `stage3_deepseek_5class_direct_benchmark.ipynb`

---

### Tier 2 — Improved Prompting (DeepSeek)

Addresses DeepSeek-R1's low zero-shot accuracy by increasing `max_tokens` and adding few-shot examples. The reasoning model's chain-of-thought is now given enough token budget to complete.

- `stage3_deepseek_2class_improved_benchmark.ipynb`
- `stage3_deepseek_5class_improved_benchmark.ipynb`

---

### Tier 3 — LoRA Fine-Tuning

Fine-tunes both models using LoRA adapters on the training set. Two hardware configurations are provided.

**Qwen2.5-7B-Instruct (QLoRA — 4-bit NF4 quantization):**
- `stage3_qwen_2class_finetune.ipynb`
- `stage3_qwen_5class_finetune.ipynb`

**Qwen2.5-7B-Instruct (bf16 full precision, for A100 80GB):**
- `stage3_qwen_2class_finetune_bf16.ipynb`
- `stage3_qwen_5class_finetune_bf16.ipynb`

**DeepSeek-R1-Distill-Qwen-14B (bf16 + thinking-chain training data):**

Fine-tuning DeepSeek requires thinking-chain training data generated in a separate step:
1. Run `stage3_deepseek_generate_thinking_chains.ipynb` — runs the model on the training set, extracts `<think>...</think>` reasoning chains from correct predictions, and saves them as structured CSV files.
2. Run `stage3_deepseek_2class_finetune_bf16.ipynb` — fine-tunes on 2-class thinking-chain data.
3. Run `stage3_deepseek_5class_finetune_bf16.ipynb` — fine-tunes on 5-class thinking-chain data.

The assistant turn format used during fine-tuning:
```
<think>
{model-generated reasoning chain}
</think>
Final label: {label}
```

---

### Stage 3 Key Results

**2-class (binary):**

| Model | Method | Accuracy | Macro F1 |
|---|---|---|---|
| Qwen2.5-7B-Instruct | Zero-shot | 98.16% | 98.16% |
| Qwen2.5-7B-Instruct | LoRA fine-tune (bf16) | **98.55%** | **98.55%** |
| DeepSeek-R1-14B | Zero-shot | 80.04% | 79.43% |
| DeepSeek-R1-14B | Few-shot (improved) | 97.33% | 97.33% |
| DeepSeek-R1-14B | LoRA fine-tune (bf16) | 98.42% | 98.42% |

**5-class:**

| Model | Method | Accuracy | Macro F1 | Off-by-1 Accuracy |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | Zero-shot | 67.36% | 66.91% | 98.51% |
| Qwen2.5-7B-Instruct | LoRA fine-tune (bf16) | **69.59%** | **69.27%** | 98.34% |
| DeepSeek-R1-14B | Zero-shot | 55.60% | 54.48% | 89.93% |
| DeepSeek-R1-14B | Few-shot (improved) | 61.56% | 60.90% | 96.99% |
| DeepSeek-R1-14B | LoRA fine-tune (bf16) | 67.70% | 66.61% | 98.44% |

---

## Stage 4: Business Intelligence Agent

Stage 4 extends the project from classification to open-ended question answering. A LangGraph ReAct agent answers natural language questions about Yelp businesses by autonomously retrieving and synthesising evidence from a 60,823-chunk vector knowledge base.

**Folder:** `s4_agent/`  
**Full documentation:** [`s4_agent/README_stage4.md`](s4_agent/README_stage4.md)

### Demo

**RAG Baseline** — Fixed retrieval pipeline (Stats → Search → Summarize):

![RAG Baseline Demo](s4_agent/RAG_Baseline.gif)

**Full Agent** — LangGraph ReAct agent with autonomous tool selection:

![Full Agent Demo](s4_agent/Full_Agent.gif)

### Architecture

Two systems are implemented and compared:

| System | Description |
|---|---|
| **RAG Baseline** | Fixed flow: `get_business_stats` → `search_by_business / search_global` → `summarize_evidence` |
| **Full Agent** | LangGraph ReAct loop — Qwen2.5-7B autonomously selects tools and iterates until a final answer is reached |

### Tools

| Tool | Description |
|---|---|
| `get_business_stats` | Review count, average stars, and star distribution |
| `search_review_chunks_by_business` | Semantic search within a single business (pre-filtered FAISS) |
| `search_review_chunks_global` | Global semantic search across all 60,823 chunks |
| `summarize_evidence` | LLM synthesis of retrieved chunks into structured findings |
| `classify_review` | RoBERTa 5-class star rating prediction |

### Stage 4 Evaluation Results

**Setup:** 20 questions × 4 types × 3 systems = 60 evaluated responses.  
Human scored on 5 dimensions (0–2 scale each, max 10 points total).

| System | Correctness | Evidence | Groundedness | Tool Use | Efficiency | **Total** |
|---|---|---|---|---|---|---|
| Direct LLM | 0.25 | 0.00 | 0.00 | 0.00 | 1.70 | **1.95** |
| RAG Baseline | 0.95 | 1.60 | 1.75 | 0.95 | 1.65 | **6.90** |
| Full Agent | 1.05 | 1.15 | 1.15 | 1.30 | 0.10 | **4.75** |

**Hallucination Rate:** Direct LLM **100%** → Full Agent **25%** → RAG Baseline **5%**

### Quick Start (Stage 4)

```bash
conda activate yelp_nlp

# First-time setup
python s4_agent/vectorstore/build_vectorstore.py
python s4_agent/step0_train_and_save.py

# Launch demo
python s4_agent/app.py
# Open http://localhost:7860
```

---

## Data

The project expects processed CSV files under a structure similar to:

```text
data/
└── processed/
    ├── train_data.csv
    ├── val_data.csv
    └── test_data.csv
```

The processed files should contain at least:
- a text column such as `text`
- a label source column such as `stars`

If you are reproducing the pipeline from scratch, run the preprocessing notebook first.

---

## Environment and Dependencies

A typical Python environment for this project includes:

- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- torch
- transformers
- datasets
- accelerate
- evaluate
- scipy
- sentence-transformers

You can install the common dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers datasets accelerate evaluate scipy sentence-transformers
```

For GPU training, make sure your PyTorch installation matches your CUDA version.

**Stage 4 additional dependencies:**

```bash
pip install faiss-cpu langchain langchain-ollama langgraph gradio
```

Stage 4 also requires [Ollama](https://ollama.com) with the `qwen2.5:7b` model pulled locally:

```bash
ollama pull qwen2.5:7b
```

---

## How to Run

## Stage 1
Run the notebooks in `s1_baseline_scripts/` in roughly this order:

1. `sample_data.ipynb`
2. `preprocess_and_split.ipynb`
3. `baseline_tfidf.ipynb`
4. `baseline_embedding.ipynb`
5. `baseline_bert.ipynb`

## Stage 2
Run scripted experiments from the project root.

### Local
```bash
bash s2_bert_scripts/run_local.sh
```

### Colab
Update the project path inside `s2_bert_scripts/run_colab.sh`, then run:

```bash
bash s2_bert_scripts/run_colab.sh
```

You can also launch a single experiment manually:

```bash
python s2_bert_scripts/run_experiments.py \
  --project_root "/path/to/Yelp_Project" \
  --model_name "distilbert-base-uncased" \
  --task_type "5_class" \
  --use_regression "False"
```

## Stage 3
Run the notebooks inside `s3_LLM/` in the following order for each model:

**Qwen:** run the direct benchmark notebooks first, then the finetune notebooks.

**DeepSeek:**
1. `stage3_deepseek_generate_thinking_chains.ipynb` — generates thinking-chain training data (prerequisite)
2. `stage3_deepseek_2class_improved_benchmark.ipynb` / `stage3_deepseek_5class_improved_benchmark.ipynb` — improved zero-shot evaluation
3. `stage3_deepseek_2class_finetune_bf16.ipynb` / `stage3_deepseek_5class_finetune_bf16.ipynb` — LoRA fine-tuning (requires step 1)

## Stage 4

```bash
conda activate yelp_nlp

# First-time setup (run once)
python s4_agent/vectorstore/build_vectorstore.py   # build FAISS index
python s4_agent/step0_train_and_save.py            # save RoBERTa classifier

# Launch interactive demo
python s4_agent/app.py

# Run evaluation (generates 60 answers, then summarise after manual scoring)
python s4_agent/evaluation/run_eval.py --run
python s4_agent/evaluation/run_eval.py --summarise
```

---

## Result Organization

### Stage 1
Saved under `s1_baseline_results/`, typically including:
- validation metrics
- test metrics
- classification reports
- confusion matrices
- prediction CSV files

### Stage 2
Saved under `s2_bert_results/`, typically including:
- per-run configuration files
- training logs
- validation/test evaluation artifacts
- experiment summaries

### Stage 3
Saved inside `s3_LLM/` experiment output folders, typically including:
- `*_predictions.csv` — per-sample predictions with raw model output
- `*_metrics.csv` — accuracy, F1, MAE, MSE
- `*_summary.json` — full run configuration and metrics
- `*_confusion_matrix.png` — confusion matrix plot
- `lora_adapter/` — saved LoRA adapter weights (fine-tuning runs only)
- `deepseek_thinking_chains/` — thinking-chain training data generated for DeepSeek fine-tuning

### Stage 4
Saved inside `s4_agent/`:
- `vectorstore/review_chunks.index` — FAISS index (60,823 chunks, 384-dim)
- `vectorstore/review_chunks.pkl` — chunk metadata and `business_to_indices` map
- `artifacts/roberta_5class_best/` — saved RoBERTa classifier checkpoint
- `results/eval_results.csv` — 60 evaluated responses with human scores

---

## Recommended Reading Order on GitHub

If you are browsing this repository for the first time, the most natural order is:

1. Read this `README.md`
2. Inspect `s1_baseline_scripts/` to understand the initial baselines
3. Review `s2_bert_scripts/` for the systematic fine-tuning framework
4. Open `s3_LLM/` for the LLM evaluation and fine-tuning stage
5. Read `s4_agent/README_stage4.md` for the full agent architecture and evaluation
6. Check `s1_baseline_results/`, `s2_bert_results/`, Stage 3 output folders, and `s4_agent/results/` for saved artifacts

---

## Notes

- Stage 1 is designed to establish a clean baseline comparison.
- Stage 2 is designed to explore stronger transformer settings without changing the overall task.
- Stage 3 follows a three-tier approach: zero-shot inference → improved prompting → LoRA fine-tuning, demonstrating progressive accuracy gains for both models and both task types.
- Stage 4 shifts the task from classification to open-ended QA, introducing RAG, tool use, and agentic reasoning. The three-way evaluation (Direct LLM vs RAG Baseline vs Full Agent) quantifies the value of retrieval and autonomous tool selection.
- Paths in shell scripts may need to be updated for your local machine or cloud environment.
- If you do not include raw Yelp data in the repository, add a short note in `data/` describing how to obtain and preprocess it.
- Stage 4 requires Ollama running locally with `qwen2.5:7b` pulled.

