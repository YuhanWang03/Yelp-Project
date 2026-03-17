# Yelp Review Rating Prediction: Baselines, BERT Fine-Tuning, and LLM Evaluation

This repository contains a multi-stage NLP project for **Yelp review rating prediction**. The project is organized as a progressive pipeline:

1. **Stage 1 — Baseline comparison**
   Compare traditional sparse features, frozen sentence embeddings, and a transformer baseline.
2. **Stage 2 — BERT fine-tuning**
   Systematically explore backbone choices, task settings, and hyperparameters.
3. **Stage 3 — LLM evaluation**
   Benchmark instruction-tuned large language models on the same review rating task.

The overall goal is to understand how performance changes as we move from:

- feature engineering,
- to fixed pretrained semantic representations,
- to end-to-end transformer fine-tuning,
- and finally to direct prompting with modern LLMs.

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
└── s3_LLM/
    ├── stage3_deepseek_2class_direct_benchmark.ipynb
    ├── stage3_deepseek_5class_direct_benchmark.ipynb
    ├── stage3_qwen_2class_direct_benchmark.ipynb
    ├── stage3_qwen_5class_direct_benchmark.ipynb
    └── experiment output folders
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

### Typical experiment dimensions
- backbone model choice
- task setting: binary / 3-class / 5-class
- classification vs regression mode
- max sequence length
- learning rate
- batch size
- number of epochs
- scheduler and regularization settings

### Stage 2 outputs
- `s2_bert_models/` — experiment-specific model/checkpoint directories
- `s2_bert_results/` — metrics, logs, reports, and evaluation artifacts

This stage is intended to identify stronger transformer settings than the Stage 1 baseline.

Due to memory limitations on the computing cluster, model checkpoints are not saved during Stage 2 experiments. As a result, this stage only preserves evaluation outputs such as metrics, logs, and reports. If checkpoint saving or model reloading is required, please modify the corresponding code files before running the experiments. 

---

## Stage 3: LLM Evaluation

Stage 3 evaluates instruction-tuned LLMs directly on the review rating task.

**Folder:** `s3_LLM/`

This stage includes experiments with models such as:
- Qwen/Qwen2.5-7B-Instruct
- deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

Representative notebooks include:
- `stage3_deepseek_2class_direct_benchmark.ipynb`
- `stage3_deepseek_5class_direct_benchmark.ipynb`
- `stage3_qwen_2class_direct_benchmark.ipynb`
- `stage3_qwen_5class_direct_benchmark.ipynb`

The goal here is to compare direct LLM performance with the supervised baselines and fine-tuned transformer models from Stages 1 and 2.

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
Run the benchmark notebooks inside `s3_LLM/` according to the target model and task setup.

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
Saved inside `s3_LLM/` experiment folders, typically including:
- prompts or benchmark notebooks
- raw model outputs
- parsed predictions
- evaluation summaries

---

## Recommended Reading Order on GitHub

If you are browsing this repository for the first time, the most natural order is:

1. Read this `README.md`
2. Inspect `s1_baseline_scripts/` to understand the initial baselines
3. Review `s2_bert_scripts/` for the systematic fine-tuning framework
4. Open `s3_LLM/` for the direct LLM benchmarking stage
5. Check `s1_baseline_results/`, `s2_bert_results/`, and the Stage 3 output folders for saved experiment artifacts

---

## Notes

- Stage 1 is designed to establish a clean baseline comparison.
- Stage 2 is designed to explore stronger transformer settings without changing the overall task.
- Stage 3 is designed to test whether direct LLM inference can outperform or complement supervised models.
- Paths in shell scripts may need to be updated for your local machine or cloud environment.
- If you do not include raw Yelp data in the repository, add a short note in `data/` describing how to obtain and preprocess it.

---

## License

Add your preferred license here if you plan to make the repository public.
