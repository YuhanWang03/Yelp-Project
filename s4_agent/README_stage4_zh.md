# Stage 4：Yelp 商业智能 Agent

**CSE4601 文本挖掘项目 — 第四阶段**

一个生产级 AI Agent，能够对 Yelp 商家相关的自然语言问题进行自主工具调用，检索基于 60,823 个文本块构建的向量知识库，并给出有据可查的结构化回答。通过三方基准测试，与 RAG 固定流水线和无检索 LLM 进行了系统性对比评测。

---

## 系统架构

Stage 4 采用两层架构：

```
用户问题
    │
    ├──► [第一层] RAG Baseline（固定流程）
    │         get_business_stats（获取商家统计）
    │              │
    │         search_by_business / search_global（语义检索）
    │              │
    │         summarize_evidence（证据综合）
    │              │
    │         结构化回答
    │
    └──► [第二层] Full Agent（自主决策）
              LangGraph ReAct 循环
              ┌─────────────────────────────┐
              │  Qwen2.5-7B（通过 Ollama）  │
              │  ┌─────────────────────┐    │
              │  │  决定调用哪个工具    │    │
              │  └──────────┬──────────┘    │
              │             │               │
              │    ┌────────▼────────┐      │
              │    │   调用工具      │      │
              │    └────────┬────────┘      │
              │             │               │
              │    ┌────────▼────────┐      │
              │    │  观察工具结果   │      │
              │    └────────┬────────┘      │
              │             │（循环迭代）    │
              └─────────────┼───────────────┘
                            │
                         最终回答
```

---

## Demo 演示

**RAG Baseline** — 固定流程：统计 → 检索 → 综合：

![RAG Baseline Demo](RAG_Baseline.gif)

**Full Agent** — LangGraph ReAct Agent 自主工具调用：

![Full Agent Demo](Full_Agent.gif)

---

## 技术栈

| 组件 | 技术选型 |
|---|---|
| Agent 框架 | LangGraph `create_react_agent` |
| Agent 大脑 | Qwen2.5-7B-Instruct（通过 Ollama 本地运行） |
| 向量存储 | FAISS `IndexFlatIP`（余弦相似度） |
| 文本嵌入 | `all-MiniLM-L6-v2`（sentence-transformers） |
| 评论分类器 | RoBERTa-base 5分类微调（Stage 2 最优配置） |
| LLM 编排 | LangChain + `langchain-ollama` |
| Demo 界面 | Gradio 5 |
| 数据集 | Yelp Academic Dataset — 5 万条评论采样 |

---

## 快速启动

### 环境准备

```bash
# 1. 激活项目环境
conda activate yelp_nlp

# 2. 确保 Ollama 已启动并加载 Qwen2.5-7B
ollama serve
ollama pull qwen2.5:7b
```

### 构建向量库（首次运行）

```bash
python s4_agent/vectorstore/build_vectorstore.py
```

将 5 万条评论编码为 60,823 个文本块，保存 FAISS 索引至 `s4_agent/vectorstore/`。

### 训练并保存分类器（首次运行）

```bash
python s4_agent/step0_train_and_save.py
```

按 Stage 2 最优配置重新训练 RoBERTa，保存 checkpoint 至 `s4_agent/artifacts/roberta_5class_best/`。

### 启动 Demo

```bash
python s4_agent/app.py
# 浏览器打开 http://localhost:7860
```

加 `--share` 参数可生成临时公网链接：

```bash
python s4_agent/app.py --share
```

### 运行评测

```bash
# 生成全部 60 组回答（20 题 × 3 系统）
python s4_agent/evaluation/run_eval.py --run

# 人工打分完成后，输出汇总表
python s4_agent/evaluation/run_eval.py --summarise
```

---

## 工具说明

Agent 可调用以下五个工具：

| 工具名 | 功能 |
|---|---|
| `get_business_stats` | 返回指定商家的评论数量、平均星级和星级分布 |
| `search_review_chunks_by_business` | 在单个商家的评论范围内进行语义检索（预过滤 FAISS） |
| `search_review_chunks_global` | 在全部 60,823 个文本块中进行全局语义检索 |
| `summarize_evidence` | 调用 Qwen 将检索结果综合为结构化发现（含证据引文） |
| `classify_review` | 调用 RoBERTa 5分类器对单条评论进行星级预测 |

将检索工具拆分为"商家内检索"和"全局检索"两个独立工具，是有意为之的设计决策：对 7B 小模型而言，工具职责越明确，自主选择的准确率越高。

---

## 评测结果

**评测设置：** 20 道题 × 4 种题型 × 3 个系统 = 60 组回答，由人工在 5 个维度上评分（每维 0–2 分，满分 10 分）。

### 总分对比

| 系统 | 正确性 | 证据引用 | 可溯源性 | 工具使用 | 效率 | **总分** |
|---|---|---|---|---|---|---|
| Direct LLM | 0.25 | 0.00 | 0.00 | 0.00 | 1.70 | **1.95** |
| RAG Baseline | 0.95 | 1.60 | 1.75 | 0.95 | 1.65 | **6.90** |
| Full Agent | 1.05 | 1.15 | 1.15 | 1.30 | 0.10 | **4.75** |

### 幻觉率（可溯源性评分为 0 的比例）

| 系统 | 幻觉率 |
|---|---|
| Direct LLM | 100% |
| Full Agent | 25% |
| RAG Baseline | 5% |

### 按题型的正确性得分

| 题型 | Direct LLM | RAG Baseline | Full Agent |
|---|---|---|---|
| 投诉挖掘（Complaint Mining） | 0.00 | 1.00 | 0.80 |
| 方面分析（Aspect Analysis） | 0.00 | 0.80 | 1.20 |
| 商家画像（Business Profiling） | 0.00 | 1.00 | 1.00 |
| 跨商家规律（Cross-Business Pattern） | 1.00 | 1.00 | 1.20 |

### 核心结论

- **RAG Baseline 总分最高（6.90/10）**：固定的工具调用顺序保证了稳定性，证据引用充分，幻觉率仅 5%。固定流程避免了小模型自主决策带来的不稳定性。
- **Full Agent 在复杂题型上正确性更高**：对于方面分析和跨商家规律等开放性问题，自由工具选择有优势，但平均响应时间约 45 秒，幻觉率达 25%——这是 7B 模型在 ReAct 提示下的已知局限。
- **Direct LLM 完全无法访问评论数据**：100% 幻觉率，回答均为泛泛而谈，唯一优势是响应速度（平均 9 秒）。

---

## 评分标准

五个维度，每维 0–2 分：

| 维度 | 0 分 | 1 分 | 2 分 |
|---|---|---|---|
| 正确性 | 完全偏题或错误 | 部分正确 | 准确且完整 |
| 证据引用 | 无引用 | 笼统提及评论 | 2+ 条具体引文 |
| 可溯源性 | 多处无中生有 | 大体有据 | 每句话均可追溯 |
| 工具使用 | 用错工具/未使用 | 使用正确但不完整 | 工具选择和顺序均正确 |
| 效率 | 明显冗余/极慢 | 可接受 | 调用最少、响应最快 |

完整评分标准见：[`s4_agent/evaluation/rubric.md`](evaluation/rubric.md)

---

## 目录结构

```
s4_agent/
├── app.py                          # Gradio Demo 界面
├── config.py                       # 路径与模型参数集中配置
├── step0_train_and_save.py         # 重新训练并保存 RoBERTa checkpoint
├── test_classifier_load.py         # 验证分类器可正常加载
│
├── artifacts/
│   └── roberta_5class_best/        # 已保存的 RoBERTa 模型（Stage 2 最优配置）
│
├── vectorstore/
│   ├── build_vectorstore.py        # 从 5 万条评论构建 FAISS 索引
│   ├── review_chunks.index         # FAISS IndexFlatIP（384 维）
│   └── review_chunks.pkl           # 文本块元数据 + business_to_indices 映射
│
├── tools/
│   ├── retrieval_tool.py           # 全局检索与商家内检索工具
│   ├── stats_tool.py               # 商家统计工具
│   ├── classifier_tool.py          # RoBERTa 推理工具
│   └── summarizer_tool.py          # LLM 证据综合工具
│
├── pipelines/
│   ├── rag_baseline.py             # 固定流程 RAG 流水线
│   └── agent_runner.py             # LangGraph ReAct Agent
│
├── evaluation/
│   ├── test_questions.json         # 20 道评测问题（4 种题型 × 5 题）
│   ├── rubric.md                   # 评分标准（5 维度，0-2 分制）
│   └── run_eval.py                 # 评测运行器 + 结果汇总
│
└── results/
    └── eval_results.csv            # 60 组回答及人工评分结果
```

---

## 关键设计决策

**chunk 级预过滤，而非全局 Top-K 后过滤**
当提供 `business_id` 时，检索工具仅加载该商家的嵌入子集并在本地做矩阵乘法，而不是先做全局 FAISS 检索再按商家过滤。这避免了评论较少的商家在全局检索中被稀释的问题。

**按检索范围拆分工具**
将检索工具分为"商家内检索"和"全局检索"两个独立工具（而非一个多用途工具），显著降低了 7B 模型在 ReAct 提示下的工具选择歧义错误率。

**综合工具的三层 JSON 解析**
Qwen 偶尔会生成格式错误的 JSON（评论文本中的弯引号、截断的响应等）。`summarize_evidence` 工具通过三层降级策略处理此类问题，在所有策略失败时返回安全默认值。

**评测运行器的断点续跑机制**
评测脚本在重启时会跳过已完成的行，防止在 Ollama 崩溃后重跑时产生重复数据。

---

## 项目亮点（面向 AI Agent 岗位）

本阶段展示了以下与 AI Agent 工程岗位相关的核心技能：

- **RAG 流水线设计**：chunk 级索引、预过滤语义检索、基于证据的结构化综合
- **LangGraph Agent 实现**：ReAct 循环、工具绑定、消息轨迹提取
- **面向小模型的工具设计**：通过拆分工具职责提升 7B 模型工具选择可靠性
- **结构化评测体系**：评分标准设计、三方系统对比、幻觉率量化
- **生产级工程实践**：崩溃续跑、多层错误处理、配置集中管理
- **可交互 Demo**：Gradio 应用，支持 160 个商家的可搜索下拉菜单和实时工具调用日志展示
