<!-- <h1 align="center">Inferential Question Answering (Inferential QA)</h1> -->

<p align="center">
  <img src="asset/quit_logo.svg" alt="Inferential QA Logo" width="400"/>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/JamshidJDMY/InferentialQA"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow&logo=huggingface"></a>
  <a href=""><img src="https://img.shields.io/static/v1?label=Paper&message=Unpublished&color=green&logo=arXiv"></a>
  <a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/static/v1?label=License&message=MIT&color=red"></a>
</p>

# Inferential Question Answering (Inferential QA)

**Inferential Question Answering (Inferential QA)** introduces a new class of reasoning QA tasks that challenge models to infer answers from indirect textual evidence rather than extracting them directly from answer-containing passages.

We present **QUIT (QUestions requiring Inference from Texts)** — a large-scale benchmark of **7,401 questions** and **2.4 million passages**, designed to evaluate how well modern retrieval-augmented systems and large language models (LLMs) can perform inference-based reasoning.

---

## 🧠 Motivation

Most existing QA datasets assume *answer containment* — that the answer explicitly appears in a retrieved passage.  
However, many real-world questions (e.g., educational reasoning, knowledge-based inference) require deriving answers from **clues and context** instead.

Inferential QA bridges this gap by focusing on **answer-supporting passages** — those that provide **evidence for inference**, not the answer itself.

---

## 📘 QUIT: A Benchmark for Inferential QA

**QUIT (QUestions requiring Inference from Texts)** is a **large-scale benchmark** designed to test whether modern QA systems can solve questions where:

✅ the evidence is present  
❌ but the answer is *not explicitly stated*  

Unlike traditional QA datasets, QUIT focuses on **answer-supporting passages**: passages that contain **clues**, not spans.

### 🔥 Benchmark Highlights

- 🧠 **7,401 inference-heavy questions**
- 📚 **2.4M passages** built from compositional hint combinations
- 🧩 Each question has **325 candidate passages**
- 🎯 Multi-level relevance labels:
  - **2**: fully relevant (enables inference)
  - **1**: partially relevant (weak or indirect evidence)
  - **0**: irrelevant

### 📊 Benchmark Statistics

| Split     | # Questions |    # Passages |
| :-------- | ----------: | ------------: |
| Train     |       4,811 |     1,563,575 |
| Dev       |         862 |       280,150 |
| Test      |       1,728 |       561,600 |
| **Total** |   **7,401** | **2,405,325** |

---

## 📦 Dataset Access (Download QUIT)

✅ The full QUIT benchmark is publicly available on HuggingFace:

👉 **HuggingFace Dataset:** https://huggingface.co/datasets/JamshidJDMY/InferentialQA

### 🚀 Quick Downloads

- **📥 Corpus (2.4M passages)**  
  https://huggingface.co/datasets/JamshidJDMY/InferentialQA/resolve/main/corpus/corpus.jsonl?download=true

- **📥 Train Set (4,811 questions)**  
  https://huggingface.co/datasets/JamshidJDMY/InferentialQA/resolve/main/train.json?download=true

- **📥 Dev Set (862 questions)**  
  https://huggingface.co/datasets/JamshidJDMY/InferentialQA/resolve/main/dev.json?download=true

- **📥 Test Set (1,728 questions)**  
  https://huggingface.co/datasets/JamshidJDMY/InferentialQA/resolve/main/test.json?download=true


### ⚡ Recommended Usage

- Use the **Corpus** for indexing (retrievers / rerankers)
- Use **Train** for fine-tuning retrievers/rerankers
- Use **Dev/Test** for *fair comparison* and reporting benchmark numbers

---

## ⚙️ Methodology

**QUIT** is constructed in two stages:

### 1. Question Sampling

- Source datasets: **TriviaHG** (machine-authored hints) & **WikiHint** (human-authored hints)
- Filtered using **BEM** to remove answer leakage
- Question type and difficulty estimated via **HintEval**
- Removed questions that LLMs could answer *parametrically* (without context)

### 2. Dataset Preparation

- Generated all subsets and permutations of top-5 hints per question → **325 passages per question**
- Labeled using **Gemma 3 1B**, **Qwen 3 4B**, **LLaMA 3.1 8B** with GPT-Eval
- Dev/Test verified by human annotators and relabeled for leakage

---

## 🧩 Experimental Setup

We evaluate a **Retriever–Reranker–Reader** pipeline across multiple models:

| Component          | Models                              |
| :----------------- | :---------------------------------- |
| **Retrievers**     | BM25, DPR, ColBERT, Contriever, BGE |
| **Rerankers**      | LiT5, MonoT5, RankGPT, RankT5, UPR  |
| **Readers (LLMs)** | LLaMA 3.2 1B, Gemma 3 4B, Qwen 3 8B |

Evaluation metrics: **Hit@K**, **Recall@K**, **MRR**, **NDCG@K**, and **Exact Match (EM)**.

### 📌 Key Observation

If retrieval and reranking were perfect, LLMs could achieve **≈ 90% EM (oracle)**.  
However, current pipelines reach only **~10–15% EM**.

General-purpose LLMs (**Gemma 3 4B**) outperform reasoning-oriented ones (**Qwen 3 8B**), showing that scaling or reasoning orientation alone does not solve inference-based QA.

---

## 🔍 Overall Insights

- 🧭 **Retrieval** is the dominant bottleneck — current retrievers cannot locate answer-supporting passages.
- 🔁 **Reranking** helps little; fine-tuning retrievers and rerankers gives inconsistent gains.
- 🧠 **General-purpose LLMs** (e.g., Gemma 3 4B) handle inferential reasoning better than reasoning-specialized ones.
- 🚨 The gap between **Oracle (~90% EM)** and **real pipelines (~10%)** exposes the core limitation of today’s RAG systems in inference-based reasoning.

---

## 💻 Reproducibility & Evaluation

We release QUIT together with **full reproducibility scripts** and **pre-computed results**, so anyone can:

✅ reproduce all benchmark numbers  
✅ evaluate new retrievers / rerankers / readers  
✅ compare against strong baselines  

---

### 🛠️ Option A — Reproduce Everything From Scratch

> ⚠️ Recommended: **Python 3.10** (some dependencies are not fully compatible with newer versions)

```bash
git clone https://github.com/DataScienceUIBK/InferentialQA.git
cd InferentialQA
pip install -r requirements.txt
```

All experiments are organized inside `experiments/`.  
To reproduce any experiment:

1. go to its folder  
2. run the provided `run.sh`  

✅ Suggested order (end-to-end benchmark reproduction):

- `experiments/dataset`  
  Download QUIT from HuggingFace

- `experiments/index`  
  Build indexes and preprocess corpus

- `experiments/baseline`  
  Wikipedia / MSMARCO baselines

- `experiments/vanilla/oracle-rerankers`  
  Oracle reranker experiments (upper-bound analysis)

- `experiments/vanilla/retrievers`  
  Retriever-only benchmark runs

- `experiments/vanilla/rerankers`  
  Retriever + reranker

- `experiments/vanilla/rag`  
  Full Retriever → Reranker → Reader pipeline

---

### 🔥 Fine-tuning Experiments (Optional)

We also provide scripts to fine-tune components on QUIT:

- `experiments/finetuning/colbert`
- `experiments/finetuning/dpr`
- `experiments/finetuning/monot5`

And complete pipeline evaluations:

- `experiments/finetuning_pipeline/ft-retriever/reranker`
- `experiments/finetuning_pipeline/ft-retriever/rag`
- `experiments/finetuning_pipeline/ft-reranker/retriever`
- `experiments/finetuning_pipeline/ft-reranker/rag`
- `experiments/finetuning_pipeline/ft-reranker/retriever_reranker`

> ⚡ Note: some fine-tuning experiments require serious compute  
> e.g., **≥ 1× NVIDIA A100 GPU**, and can take **multiple days**.

---

### ✅ Option B — Use Our Precomputed Results (No GPU Needed)

No powerful resources? No problem.

We provide **precomputed outputs** for all benchmark experiments.
To reproduce tables and analysis from the paper:

1. go to the `results/` directory  
2. run the Python scripts

They will automatically download the needed files from HuggingFace and display the final results.

🎉 This option makes QUIT easy to use for:
- quick benchmarking
- ablation studies
- comparing new models
- classroom/educational usage

---

## 🏆 Leaderboard (Coming Soon)

| Rank | Model | Retriever | Reranker | Reader |  EM | NDCG@10 |
| :--: | :---- | :-------- | :------- | :----- | :-: | :-----: |
|  🥇  | –     | –         | –        | –      |  –  |    –    |
|  🥈  | –     | –         | –        | –      |  –  |    –    |
|  🥉  | –     | –         | –        | –      |  –  |    –    |

Stay tuned for the **official leaderboard** and evaluation scripts once the dataset is released.

---

## 🚀 Key Takeaways

- 🔍 **Inferential QA** requires reasoning from clues — not explicit spans
- ⚙️ **Current retrievers and rerankers** fail to identify sufficient evidence
- 🧩 **Fine-tuning** is insufficient; new paradigms for *retrieval-augmented reasoning* are needed
- 📈 **QUIT** exposes a fundamental limitation in today’s QA pipelines and opens a new research direction

---

## 🚀 Contribution Summary

✅ Introduce **Inferential QA**, a new reasoning-based QA task  
✅ Construct **QUIT**, the first large-scale dataset for inferential question answering  
✅ Evaluate **retrievers**, **rerankers**, and **LLM readers** extensively  
✅ Show that current QA pipelines fail under inference-based reasoning  
