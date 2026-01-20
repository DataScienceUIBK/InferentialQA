#!/bin/bash

# Clear the terminal screen
clear

python make_rag_inputs.py

models=(
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
)

retrievers=(
    "colbert"
    "dpr"
)

rerankers=(
    ""
    "lit5dist-LiT5-Distill-base-v2"
    "lit5dist-LiT5-Distill-large-v2"
    "lit5dist-LiT5-Distill-xl-v2"
    "monot5-monot5-3b-msmarco-10k"
    "monot5-monot5-base-msmarco-10k"
    "monot5-monot5-large-msmarco-10k"
    "rankgpt-Llama-3.1-8B-Instruct"
    "rankgpt-Qwen2.5-7B"
    "rankt5-rankt5-3b"
    "rankt5-rankt5-base"
    "rankt5-rankt5-large"
    "upr-gpt2-large"
    "upr-t0-3b"
    "upr-t5-large"
    "upr-t5-small"
)

for model in "${models[@]}"; do
    for retriever in "${retrievers[@]}"; do
        for reranker in "${rerankers[@]}"; do
            cmd="python rag.py --model \"$model\" --retriever \"$retriever\""
            if [ -n "$reranker" ]; then
                cmd+=" --reranker \"$reranker\""
            fi
            echo "Running: $cmd"
            eval $cmd
        done
    done
done

python merge_answers.py