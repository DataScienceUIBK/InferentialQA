<!-- <h1 align="center">Inferential Question Answering (Inferential QA)</h1> -->

<p align="center">
  <img src="asset/quit_logo.svg" alt="Inferential QA Logo" width="400"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Dataset-7.4k%20Questions-blue?style=flat-square"></a>
  <a href="#"><img src="https://img.shields.io/badge/Corpus-2.4M-green?style=flat-square"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-TBA-lightgrey?style=flat-square"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Under%20Review-orange?style=flat-square"></a>
</p>

**Inferential Question Answering (Inferential QA)** introduces a new class of QA tasks that challenge models to infer answers from indirect textual evidence rather than extracting them directly from answer-containing passages.

We present **QUIT (QUestions requiring Inference from Texts)** — a large-scale benchmark of **7,401 questions** and **2.4 million passages**, designed to evaluate how well modern retrieval-augmented systems and large language models (LLMs) can perform inference-based reasoning.

## 🧠 Motivation

Most existing QA datasets assume *answer containment* — that the answer explicitly appears in a retrieved passage.
However, many real-world questions (e.g., educational reasoning, knowledge-based inference) require deriving answers from **clues and context** instead.

Inferential QA bridges this gap by focusing on **answer-supporting** passages — those that provide **evidence for inference**, not the answer itself.

## 📘 The QUIT Dataset

The **QUIT** dataset consists of passages built from *hints* — short, human- or machine-authored clues describing entities without revealing their names.

| Split     | # Questions |    # Passages |
| :-------- | ----------: | ------------: |
| Train     |       4,811 |     1,563,575 |
| Dev       |         862 |       280,150 |
| Test      |       1,728 |       561,600 |
| **Total** |   **7,401** | **2,405,325** |

Each passage is labeled at **three relevance levels**:

* **2 – Fully relevant:** enables an LLM to infer the correct answer
* **1 – Partially relevant:** indirectly describes the answer
* **0 – Irrelevant:** unrelated to the answer

## 📦 Dataset Access

You can download the QUIT dataset from the following links:

* [📥 Corpus](https://drive.google.com/file/d/1evXedpnDdaSwPKRgVCL2hw8I0R0SGHB4/view?usp=sharing)
* [📥 Train Set](https://drive.google.com/file/d/18ig8pmvSCq9M6MftBHnOicdCIqXVhCHd/view?usp=sharing)
* [📥 Dev Set](https://drive.google.com/file/d/1CqYRu2yfjpycaEGZrQfgA0Cxu3bMWmgQ/view?usp=sharing)
* [📥 Test Set](https://drive.google.com/file/d/1m1YTB07af2ptuK0u3qVBK3RznVG9bufH/view?usp=sharing)

**⚠️ Attention:**
Following the official publication of the paper, the dataset will be made publicly available on Hugging Face.

## ⚙️ Methodology

**QUIT** is constructed in two stages:

### 1. Question Sampling

* Source datasets: **TriviaHG** (machine-authored hints) & **WikiHint** (human-authored hints).
* Filtered using **BEM** to remove answer leakage.
* Question type and difficulty estimated via **HintEval**.
* Removed questions that LLMs could answer *parametrically* (without context).

### 2. Dataset Preparation

* Generated all subsets and permutations of top-5 hints per question → 325 passages per question.
* Labeled using **Gemma 3 1B**, **Qwen 3 4B**, **LLaMA 3.1 8B** with GPT-Eval.
* Dev/Test verified by human annotators and relabeled for leakage.

## 🧩 Experimental Setup

We evaluate a **Retriever–Reranker–Reader** pipeline across multiple models:

| Component          | Models                              |
| :----------------- | :---------------------------------- |
| **Retrievers**     | BM25, DPR, ColBERT, Contriever, BGE |
| **Rerankers**      | LiT5, MonoT5, RankGPT, RankT5, UPR  |
| **Readers (LLMs)** | LLaMA 3.2 1B, Gemma 3 4B, Qwen 3 8B |

Evaluation metrics: **Hit@K**, **Recall@K**, **MRR**, **NDCG@K**, and **Exact Match (EM)**.

## 📊 Results

### **Table 2 – Retriever Performance**

| Retriever      | Corpus   |       Hit@1 |      Hit@10 |      Hit@50 |     Hit@100 |         MRR |
| :------------- | :------- | ----------: | ----------: | ----------: | ----------: | ----------: |
| **BM25**       | QUIT     |      0.00 % |      0.25 % |      0.44 % |      0.57 % |      0.04 % |
| **DPR**        | QUIT     |      9.89 % |     16.62 % |     21.22 % |     23.74 % |     11.28 % |
| **ColBERT**    | QUIT     |     12.41 % |     16.44 % |     19.40 % |     20.21 % |     12.62 % |
| **Contriever** | QUIT     |      6.49 % |     13.29 % |     18.95 % |     22.54 % |      8.15 % |
| **BGE**        | **QUIT** | **12.85 %** | **21.98 %** | **27.96 %** | **30.23 %** | **14.68 %** |

**🧩 Observation:**
Retrieval on **QUIT** is far harder than on MS MARCO or Wikipedia. Even strong neural retrievers struggle, showing that locating **answer-supporting (not answer-containing)** passages is substantially more difficult.


### **Table 3 – Reranker Comparison**

| Reranker   | Corpus   |     Hit@1 |    Hit@10 |    Hit@50 |       MRR |
| :--------- | :------- | --------: | --------: | --------: | --------: |
| LiT5       | QUIT     |     26.03 |     29.21 |     33.80 |     28.10 |
| **MonoT5** | **QUIT** | **27.60** | **29.98** | **32.35** | **28.54** |
| RankGPT    | QUIT     |     24.02 |     29.05 |     33.33 |     25.70 |
| RankT5     | QUIT     |     26.62 |     30.44 |     32.52 |     27.80 |
| UPR        | QUIT     |     26.85 |     29.86 |     32.70 |     27.89 |

**🧩 Observation:**
Reranking brings only **minor gains**. MonoT5 slightly leads, but the difference is small — indicating current rerankers cannot reliably surface the truly inferential passages.


### **Table 4 – Vanilla vs Fine-tuned Retrievers**

| Retriever         |       Hit@1 |       Hit@5 | Recall@10 | Recall@100 |     MRR |     nDCG@10 | nDCG@100 |
| :---------------- | ----------: | ----------: | --------: | ---------: | ------: | ----------: | -------: |
| **BGE (vanilla)** | **23.73 %** | **27.37 %** |    0.75 % |    25.45 % | 18.95 % | **21.14 %** |        – |
| FT-DPR            |     20.91 % |     28.07 % |    0.63 % |    23.56 % | 14.98 % |     16.69 % |        – |

**🧩 Observation:**
Fine-tuning offers **only marginal or inconsistent improvements**. BGE remains strongest despite no task-specific tuning — suggesting that Inferential QA requires **new retrieval paradigms** rather than more training.


### **Table 5 – Reranker on Top of Retrievers**

| Retriever           | Reranker |       Hit@1 |       Hit@5 |  Recall@10 | Recall@100 |         MRR |     nDCG@10 |    nDCG@100 |
| :------------------ | :------- | ----------: | ----------: | ---------: | ---------: | ----------: | ----------: | ----------: |
| **BGE**             |**MonoT5**| **27.60 %** | **29.46 %** | **0.84 %** | **4.01 %** | **28.54 %** | **22.36 %** | **21.81 %** |
| FT-DPR              |  MonoT5  |     28.01 % |     31.89 % |     0.78 % |     3.39 % |     30.24 % |     20.34 % |     16.63 % |
| FT-ColBERT          |  MonoT5  |     22.69 % |     25.58 % |     0.65 % |     2.86 % |     24.16 % |     17.07 % |     14.80 % |

**🧩 Observation:**
Even when stacked on fine-tuned retrievers, rerankers **cannot overcome retrieval errors**. The challenge lies deeper — understanding **indirect textual clues**.


### **Table 6 – Fine-tuned MonoT5 Reranker**

| Retriever              |   Reranker  |       Hit@1 |       Hit@5 | Recall@10 |     MRR | nDCG@10 |
| :--------------------- | :---------- | ----------: | ----------: | --------: | ------: | ------: |
| **BGE**                |**FT-MonoT5**| **23.44 %** | **26.98 %** |    0.74 % | 18.67 % | 20.77 % |
| FT-DPR                 |  FT-MonoT5  |     19.91 % |     28.07 % |    0.60 % | 13.96 % | 15.36 % |
| FT-ColBERT             |  FT-MonoT5  |     18.11 % |     23.84 % |    0.53 % | 13.00 % | 13.98 % |

**🧩 Observation:**
Fine-tuning MonoT5 **reduces performance** compared to the vanilla version — showing that rerankers fail to adapt to Inferential QA even with additional supervision.


### **Table 7 – Oracle Reranking**

| Reranker             |      nDCG@5 |     nDCG@10 |     nDCG@50 |    nDCG@100 |
| :------------------- | ----------: | ----------: | ----------: | ----------: |
| LiT5                 |     72.94 % |     75.49 % |     79.34 % |     82.99 % |
| RankGPT              |     65.02 % |     69.74 % |     78.09 % |     82.24 % |
| RankT5               |     78.96 % |     80.18 % |     84.69 % |     87.49 % |
| UPR                  |     78.56 % |     79.72 % |     84.30 % |     87.25 % |
| **MonoT5 (vanilla)** | **82.01 %** | **82.95 %** | **86.46 %** | **88.71 %** |
| **FT-MonoT5**        | **83.56 %** | **84.24 %** | **87.08 %** | **89.17 %** |

**🧩 Observation:**
Even assuming perfect retrieval, **fine-tuned MonoT5** only slightly outperforms the vanilla one — the true bottleneck remains **retrieval**, not reranking.


### **Table 8 – Reader (LLM) Results**

| Retriever – Reranker   | Strategy | LLaMA 3.2 1B |  Gemma 3 4B | Qwen 3 8B |
| :--------------------- | :------- | -----------: | ----------: | --------: |
| **Oracle (perfect)**   | –        |      40.68 % | **90.16 %** |   62.50 % |
| **Oracle + MonoT5**    | UF       |      20.25 % |     50.41 % |   34.32 % |
| **BGE + MonoT5**       | UN       |       4.98 % |     15.34 % |   12.38 % |
| **FT-DPR + FT-MonoT5** | UN       |       4.17 % |     12.44 % |    8.80 % |

**🧩 Observation:**
If retrieval and reranking were perfect, LLMs could achieve ≈ 90 % EM (oracle).
Current pipelines reach only ~10–15 %. General-purpose LLMs (**Gemma 3 4B**) outperform reasoning-oriented ones (**Qwen 3 8B**), showing that scaling or reasoning orientation alone does not solve inference-based QA.


### **Overall Insights**

* 🧭 **Retrieval** is the dominant bottleneck — current retrievers cannot locate answer-supporting passages.
* 🔁 **Reranking** helps little; fine-tuning retrievers and rerankers gives inconsistent gains.
* 🧠 **General-purpose LLMs** (e.g., Gemma 3 4B) handle inferential reasoning better than reasoning-specialized ones.
* 🚨 The gap between **Oracle (~90 % EM)** and **real pipelines (~10 %)** exposes the core limitation of today’s RAG systems in inference-based reasoning.


## 🏆 Leaderboard (Coming Soon)

| Rank | Model | Retriever | Reranker | Reader |  EM | NDCG@10 |
| :--: | :---- | :-------- | :------- | :----- | :-: | :-----: |
|  🥇  | –     | –         | –        | –      |  –  |    –    |
|  🥈  | –     | –         | –        | –      |  –  |    –    |
|  🥉  | –     | –         | –        | –      |  –  |    –    |

Stay tuned for the **official leaderboard** and evaluation scripts once the dataset is released.

## 💻 Code & Evaluation (Coming Soon)

To reproduce results and evaluate on QUIT:

```bash
git clone https://github.com/yourusername/inferential-qa.git
cd inferential-qa
pip install -r requirements.txt
python evaluate.py --model bge --reranker monot5 --reader gemma
```

Evaluation script supports:

* Custom retrievers, rerankers, or LLM readers
* Both zero-shot and fine-tuned evaluation
* Metrics: *Hit@K, Recall@K, MRR, NDCG@K, EM*

## 🚀 Key Takeaways

* 🔍 **Inferential QA** requires reasoning from clues — not explicit spans.
* ⚙️ **Current retrievers and rerankers** fail to identify sufficient evidence.
* 🧩 **Fine-tuning** is insufficient; new paradigms for *retrieval-augmented reasoning* are needed.
* 📈 **QUIT** exposes a fundamental limitation in today’s QA pipelines and opens a new research direction.

## 🚀 Contribution Summary

✅ Introduce **Inferential QA**, a new reasoning-based QA task.  
✅ Construct **QUIT**, the first large-scale dataset for inferential question answering.  
✅ Evaluate **retrievers**, **rerankers**, and **LLM readers** extensively.  
✅ Show that current QA pipelines fail under inference-based reasoning.  

