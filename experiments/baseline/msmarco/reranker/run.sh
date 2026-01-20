#!/bin/bash
clear

# Define retrievers
retrievers=("bm25" "dpr" "contriever" "colbert" "bge")

# Define rerankers and their models
declare -A rerankers

# monot5
rerankers[monot5]="monot5-base-msmarco-10k monot5-large-msmarco-10k monot5-3b-msmarco-10k"

# lit5dist
rerankers[lit5dist]="LiT5-Distill-base-v2 LiT5-Distill-large-v2 LiT5-Distill-xl-v2"

# rankt5
rerankers[rankt5]="rankt5-base rankt5-large rankt5-3b"

# upr
rerankers[upr]=" t5-small t5-large gpt2-large t0-3b "

# rankgpt
rerankers[rankgpt]="meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B"

# Explicit order
reranker_order=("monot5" "lit5dist" "rankt5" "upr" "rankgpt")

# Loop through retrievers and rerankers in the given order
for retriever in "${retrievers[@]}"; do
  for reranker_method in "${reranker_order[@]}"; do
    for reranker_model in ${rerankers[$reranker_method]}; do
      echo "Running: retriever=$retriever, reranker_method=$reranker_method, reranker_model=$reranker_model"
      python reranker.py --retriever "$retriever" --reranker_method "$reranker_method" --reranker_model "$reranker_model"
    done
  done
done
