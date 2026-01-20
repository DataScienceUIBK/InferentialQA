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
    "ft_monot5"
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