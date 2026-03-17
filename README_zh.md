# Yelp 评论评分预测：Baseline 对比、BERT 微调与大语言模型评测

本仓库包含一个多阶段的自然语言处理项目，主题是 **Yelp 评论评分预测（Yelp review rating prediction）**。整个项目按照逐步推进的实验流程组织：

1. **第一阶段（Stage 1）—— Baseline 对比**  
   比较传统稀疏特征、冻结句向量表示，以及 transformer baseline。
2. **第二阶段（Stage 2）—— BERT 微调**  
   系统探索不同 backbone、任务设定和超参数配置。
3. **第三阶段（Stage 3）—— 大语言模型评测**  
   在同一评论评分任务上评测指令微调的大语言模型。

整个项目的目标是观察：当方法从以下几类逐步升级时，性能会如何变化：

- 特征工程方法；
- 固定的预训练语义表示；
- 端到端的 transformer 微调；
- 以及基于现代大语言模型的直接提示推理。

---

## 项目结构

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
    └── 各类实验输出文件夹
```

---

## 第一阶段：Baseline 对比

第一阶段用于建立三类建模范式之间的初步对比。

### 1. TF-IDF baseline
**文件：** `s1_baseline_scripts/baseline_tfidf.ipynb`

- 文本表示：TF-IDF 特征
- 分类器：Logistic Regression
- 目的：提供一个传统稀疏特征 baseline

### 2. 冻结句向量 baseline
**文件：** `s1_baseline_scripts/baseline_embedding.ipynb`

- 文本表示：Sentence-BERT embeddings
- 分类器：Logistic Regression
- 目的：测试不进行端到端微调时，固定预训练语义表示的效果

### 3. Transformer baseline
**文件：** `s1_baseline_scripts/baseline_bert.ipynb`

- Backbone：transformer 分类模型 baseline
- 训练方式：端到端微调
- 目的：验证任务自适应的 transformer 是否优于前两个 baseline

### 辅助 notebook
- `preprocess_and_split.ipynb`：数据清洗与 train/validation/test 划分
- `sample_data.ipynb`：样本抽取或数据快速检查

### 第一阶段输出
第一阶段的结果保存在：

- `s1_baseline_models/`
- `s1_baseline_results/`

这些 baseline notebook 会同时保存 **validation** 和 **test** 的结果，包括指标、classification report、confusion matrix 以及 prediction 文件。

---

## 第二阶段：BERT 微调实验

第二阶段将 transformer 实验扩展为一个更系统的微调框架。

**文件夹：** `s2_bert_scripts/`

### 主要组成部分
- `config.py` —— 默认实验配置
- `data_loader.py` —— 数据加载、任务映射和分词辅助函数
- `train.py` —— 单次训练流程
- `utils.py` —— 路径设置、指标计算与结果保存工具
- `run_experiments.py` —— 命令行实验入口
- `run_local.sh` —— 本地批量运行脚本
- `run_colab.sh` —— Google Colab 批量运行脚本

### 常见实验维度
- backbone 模型选择
- 任务设定：binary / 3-class / 5-class
- classification 与 regression 模式
- 最大序列长度
- 学习率
- batch size
- 训练轮数
- scheduler 与正则化相关设置

### 第二阶段输出
- `s2_bert_models/` —— 各实验对应的模型或 checkpoint 目录
- `s2_bert_results/` —— 指标、日志、报告与评测结果文件

这一阶段的目标是找到比第一阶段 baseline 更强的 transformer 设置。

由于算力集群的内存限制，第二阶段实验过程中不会保存模型 checkpoint 文件。因此，本阶段仅保留测评输出结果，例如指标、日志和报告等。如果需要保存 checkpoint 或重新加载模型，请先在对应的代码文件中进行修改后再运行实验

---

## 第三阶段：大语言模型评测

第三阶段直接评测指令微调的大语言模型在评论评分任务上的表现。

**文件夹：** `s3_LLM/`

这一阶段包含了如下模型的实验：
- Qwen/Qwen2.5-7B-Instruct
- deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

代表性 notebook 包括：
- `stage3_deepseek_2class_direct_benchmark.ipynb`
- `stage3_deepseek_5class_direct_benchmark.ipynb`
- `stage3_qwen_2class_direct_benchmark.ipynb`
- `stage3_qwen_5class_direct_benchmark.ipynb`

这一阶段的目标是将大语言模型的直接推理表现，与第一、二阶段中的监督学习模型进行比较。

---

## 数据

项目默认使用如下结构下的处理后 CSV 文件：

```text
data/
└── processed/
    ├── train_data.csv
    ├── val_data.csv
    └── test_data.csv
```

这些处理后的文件至少应包含：
- 文本列，例如 `text`
- 标签来源列，例如 `stars`

如果你需要从头复现实验流程，请先运行预处理 notebook。

---

## 环境与依赖

一个典型的 Python 环境应包含：

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

可以使用下面的命令安装常用依赖：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers datasets accelerate evaluate scipy sentence-transformers
```

如果使用 GPU 训练，请确保你的 PyTorch 安装版本与你的 CUDA 版本匹配。

---

## 运行方式

## 第一阶段
按大致如下顺序运行 `s1_baseline_scripts/` 中的 notebook：

1. `sample_data.ipynb`
2. `preprocess_and_split.ipynb`
3. `baseline_tfidf.ipynb`
4. `baseline_embedding.ipynb`
5. `baseline_bert.ipynb`

## 第二阶段
在项目根目录下运行脚本实验。

### 本地运行
```bash
bash s2_bert_scripts/run_local.sh
```

### Colab 运行
先修改 `s2_bert_scripts/run_colab.sh` 中的项目路径，然后运行：

```bash
bash s2_bert_scripts/run_colab.sh
```

你也可以手动启动单个实验：

```bash
python s2_bert_scripts/run_experiments.py \
  --project_root "/path/to/Yelp_Project" \
  --model_name "distilbert-base-uncased" \
  --task_type "5_class" \
  --use_regression "False"
```

## 第三阶段
根据目标模型和任务设定，运行 `s3_LLM/` 中对应的 benchmark notebook。

---

## 结果组织方式

### 第一阶段
保存在 `s1_baseline_results/` 下，通常包括：
- validation 指标
- test 指标
- classification report
- confusion matrix
- prediction CSV 文件

### 第二阶段
保存在 `s2_bert_results/` 下，通常包括：
- 单次实验配置文件
- 训练日志
- validation/test 评测结果
- 实验汇总结果

### 第三阶段
保存在 `s3_LLM/` 各实验输出文件夹中，通常包括：
- prompts 或 benchmark notebook
- 原始模型输出
- 解析后的预测结果
- 评测汇总结果

---

## GitHub 推荐阅读顺序

如果你第一次浏览这个仓库，推荐按以下顺序阅读：

1. 先阅读本 `README.md`
2. 查看 `s1_baseline_scripts/`，理解初始 baseline
3. 查看 `s2_bert_scripts/`，理解系统化微调框架
4. 打开 `s3_LLM/`，查看大语言模型 benchmark 阶段
5. 最后查看 `s1_baseline_results/`、`s2_bert_results/` 和第三阶段的输出文件夹，了解保存下来的实验结果

---

## 说明

- 第一阶段用于建立清晰的 baseline 对比。
- 第二阶段用于在不改变整体任务设定的前提下，探索更强的 transformer 配置。
- 第三阶段用于检验大语言模型的直接推理是否能够超越或补充监督学习模型。
- shell 脚本中的路径可能需要根据本地环境或云端环境进行修改。
- 如果仓库中不包含原始 Yelp 数据，建议在 `data/` 中补充一个简短说明，介绍如何获取并预处理数据。

