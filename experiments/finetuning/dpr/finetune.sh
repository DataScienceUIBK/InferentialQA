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
  KEY=${pair%%:*}   # before :
  EPOCHS=${pair#*:} # after :

  OUTPUT_DIR="./${KEY}_positive/checkpoints"
  TRAIN_DIR="/gpfs/gpfs1/scratch/c7031431/Projects/Quit-Final/finetuning/dpr/${KEY}_positive/train_dir"

  echo "==> Train on ${KEY} positive passages with ${EPOCHS} epochs"
  python -m tevatron.retriever.driver.train \
    --output_dir "${OUTPUT_DIR}" \
    --corpus_path "${TRAIN_DIR}" \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --save_steps 20000 \
    --fp16 \
    --per_device_train_batch_size 64 \
    --train_group_size 2 \
    --learning_rate 1e-5 \
    --query_max_len 64 \
    --passage_max_len 256 \
    --num_train_epochs "${EPOCHS}" \
    --attn_implementation sdpa \
    --overwrite_output_dir
done

echo "All runs finished."
