#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "glue_baseline.sh <task> <batch_size>"
  exit 1
fi
TASK_NAME=$1
BATCH_SIZE=$2
tstr=$(date +"%FT%H%M")

GLUE_DIR="../data"

python ../train.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ${GLUE_DIR}/${TASK_NAME} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=${BATCH_SIZE}   \
    --per_gpu_train_batch_size=${BATCH_SIZE}   \
    --learning_rate 2e-5 \
    --num_train_epochs 20.0 \
    --output_dir "tmp/${TASK_NAME}_baseline_${tstr}/" \
    --evaluate_during_training \
    --freeze_encoder
