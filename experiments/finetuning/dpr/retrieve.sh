#!/bin/sh

# === Java environment setup ===
JAVA_HOME="/home/c703/c7031431/java/jdk-21.0.5"
export JAVA_HOME
export JDK_HOME="$JAVA_HOME"
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$JAVA_HOME/lib:$LD_LIBRARY_PATH"

# === Training loop ===
PAIRS="1:10 5:5 10:2 50:1 100:1 200:1"

for pair in $PAIRS; do
  KEY=${pair%%:*}
  EPOCHS=${pair#*:}   # not needed here, but kept in case you want consistency

  echo "==> Running encoding & retrieval when number of passages is ${KEY}"

  # dev encoding
  mkdir -p ./${KEY}_positive/dev_dir
  python -m tevatron.driver.encode \
    --output_dir temp \
    --model_name_or_path "/gpfs/gpfs1/scratch/c7031431/Projects/Quit-Final/finetuning/dpr/${KEY}_positive/checkpoints" \
    --fp16 \
    --per_device_eval_batch_size 256 \
    --dataset_name dev_dir \
    --encode_in_path dev_data.jsonl \
    --encoded_save_path ./${KEY}_positive/dev_dir/encoded_queries_${KEY}_positive.pkl \
    --encode_is_qry

  # corpus encoding
  mkdir -p ./${KEY}_positive/corpus_dir
  python -m tevatron.driver.encode \
    --output_dir temp \
    --model_name_or_path "/gpfs/gpfs1/scratch/c7031431/Projects/Quit-Final/finetuning/dpr/${KEY}_positive/checkpoints" \
    --fp16 \
    --per_device_eval_batch_size 256 \
    --p_max_len 256 \
    --dataset_name corpus_dir \
    --encode_in_path corpus.jsonl \
    --encoded_save_path ./${KEY}_positive/corpus_dir/encoded_passages_${KEY}_positive.pkl

  # retrieval
  python -m tevatron.faiss_retriever \
    --query_reps ./${KEY}_positive/dev_dir/encoded_queries_${KEY}_positive.pkl \
    --passage_reps ./${KEY}_positive/corpus_dir/encoded_passages_${KEY}_positive.pkl \
    --depth 1000 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to ./${KEY}_positive/run.txt

  # convert to json
  python ./to_json.py --num_of_passages ${KEY}

done

echo "All encoding & retrieval runs finished."
