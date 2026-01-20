<!-- <h1 align="center">Inferential Question Answering (Inferential QA)</h1> -->

<p align="center">
  <img src="asset/quit_logo.svg" alt="Inferential QA Logo" width="400"/>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/JamshidJDMY/InferentialQA"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow&logo=huggingface"></a>
  <a href=""><img src="https://img.shields.io/static/v1?label=Paper&message=Unpublished&color=green&logo=arXiv"></a>
  <a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/static/v1?label=License&message=MIT&color=red"></a>
</p>

**Inferential Question Answering (Inferential QA)** introduces a new class of reasoning QA tasks that challenge models to infer answers from indirect textual evidence rather than extracting them directly from answer-containing passages.

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

* [📥 Corpus](https://huggingface.co/datasets/JamshidJDMY/InferentialQA/resolve/main/corpus/corpus.jsonl?download=true)
* [📥 Train Set](https://huggingface.co/datasets/JamshidJDMY/InferentialQA/resolve/main/train.json?download=true)
* [📥 Dev Set](https://huggingface.co/datasets/JamshidJDMY/InferentialQA/resolve/main/dev.json?download=true)
* [📥 Test Set](https://huggingface.co/datasets/JamshidJDMY/InferentialQA/resolve/main/test.json?download=true)

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

**🧩 Observation:**
If retrieval and reranking were perfect, LLMs could achieve ≈ 90 % EM (oracle).
Current pipelines reach only ~10–15 %. General-purpose LLMs (**Gemma 3 4B**) outperform reasoning-oriented ones (**Qwen 3 8B**), showing that scaling or reasoning orientation alone does not solve inference-based QA.


### **Overall Insights**

* 🧭 **Retrieval** is the dominant bottleneck — current retrievers cannot locate answer-supporting passages.
* 🔁 **Reranking** helps little; fine-tuning retrievers and rerankers gives inconsistent gains.
* 🧠 **General-purpose LLMs** (e.g., Gemma 3 4B) handle inferential reasoning better than reasoning-specialized ones.
* 🚨 The gap between **Oracle (~90 % EM)** and **real pipelines (~10 %)** exposes the core limitation of today’s RAG systems in inference-based reasoning.

## 💻 Reproducibility & Evaluation

To reproduce results and evaluate on QUIT, there are two options you can use:

### **Reproducing results from scratch**

To do it, you should run the command lines below to install the suitable environment. Note that use Python 3.10 to have a good environment because some of the libraries are compatible with Python 3.10 only.

```bash
git clone https://github.com/DataScienceUIBK/InferentialQA.git
cd InferentialQA
pip install -r requirements.txt
```
To reproduce from scratch, you should run codes in the experiments directory. To rerun and reproduce each experiment, you should go to the related directory and run run.sh file. To have a correct results, you should follow the following order of running experiments:

* experiments/dataset: it downloads the dataset from our HuggingFace repository to be used in other experiments.
* experiments/index: it prepares the index files and corpus for other experiments such as retriever, reranker, and reader experiments.
* experiment/baseline: it reproduces the experiment regarding the baseline methods such as Wikipedia and MSMARCO
* experiments/vanilla/oracle-rerankers: it reproduces the experiments regarding results of using rerankers as oracle situation.
* experiments/vanilla/retrievers: it reproduces the experiments regarding results of using retrievers without using rerankers.
* experiments/vanilla/rerankers: it reproduces the experiments regarding results of using retrievers and rerankers.
* experiments/vanilla/rag: it reproduces the experiments regarding results of using the reader on outputs of retrievers and rerankers.
* experiments\finetuning\colbert: it finetunes colbert retriever on QUIT benchmark to have a finetuned version of colbert retriever.
* experiments\finetuning\dpr: it finetunes dpr retriever on QUIT benchmark to have a finetuned version of dpr retriever.
* experiments\finetuning\monot5: it finetunes monot5 reranker on QUIT benchmark to have a finetuned version of monot5 reranker.
* experiments\finetuning_pipeline\ft-retriever\reranker: it reproduces the results of the reranker experiment while using finetuned retrievers as the retriever.
* experiments\finetuning_pipeline\ft-retriever\rag: it reproduces the results of the rag experiment while using finetuned retrievers as the retriever and vanilla rerankers.
* experiments\finetuning_pipeline\ft-reranker\retriever: it reproduces the results of the reranker experiment while using finetuned reranker as the reranker.
* experiments\finetuning_pipeline\ft-reranker\rag: it reproduces the results of the rag experiment while using finetuned reranker as the reranker and vanilla retrievers.
* experiments\finetuning_pipeline\ft-reranker\retriever_reranker: it reproduces the results of the reranker experiment while using finetuned reranker and finetuned retrievers.
* experiments\finetuning_pipeline\ft-reranker\rag: it reproduces the results of the rag experiment while using finetuned reranker as the reranker and finetuned retrievers as the retriever.

Note that for running some of experiments, you will need to have powerful resources such as at least one Nvidia A100 gpu and some of these experiments need around 10 days to get finished.

Additionally, for using some experiments, you should create new environments such as:
* To use colbert and finetuning colbert model, you should follow their [github repository](https://github.com/stanford-futuredata/ColBERT).
* To finetune dpr, you should use [Tevratron](https://github.com/texttron/tevatron) toolkit and follow their instructions.
* To finetune monot5, you should use [pygaggle](https://github.com/castorini/pygaggle) library and follow their instructions.

Evaluation script supports:

* Custom retrievers, rerankers, or LLM readers
* Both zero-shot and fine-tuned evaluation
* Metrics: *Hit@K, Recall@K, MRR, NDCG@K, EM*


## 🏆 Leaderboard (Coming Soon)

| Rank | Model | Retriever | Reranker | Reader |  EM | NDCG@10 |
| :--: | :---- | :-------- | :------- | :----- | :-: | :-----: |
|  🥇  | –     | –         | –        | –      |  –  |    –    |
|  🥈  | –     | –         | –        | –      |  –  |    –    |
|  🥉  | –     | –         | –        | –      |  –  |    –    |

Stay tuned for the **official leaderboard** and evaluation scripts once the dataset is released.


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

