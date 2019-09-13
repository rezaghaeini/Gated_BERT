#!/bin/bash
if [[ $# -ne 4 ]]; then
  echo "glue_player_norm.sh <task> <batch_size> <do_gate_dropout> <do_gate_weight_normalization>"
  exit 1
fi
tstr=$(date +"%FT%H%M")

GLUE_DIR="../data"
TASK_NAME=$1
BATCH_SIZE=$2
GATE_DROPOUT=$3
GATE_NORMALIZE=$4

SUB_CFG=""
if [[ $GATE_DROPOUT == "1" ]] && [[ $GATE_NORMALIZE == "1" ]]; then
  SUB_CFG="_dn"
elif [[ $GATE_DROPOUT == "1" ]]; then
  SUB_CFG="_d"
elif [[ $GATE_NORMALIZE == "1" ]]; then
  SUB_CFG="_n"
fi

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
    --output_dir "tmp/${TASK_NAME}_player_norm${SUB_CFG}_${tstr}/" \
    --gate_dropout ${GATE_DROPOUT} \
    --gate_normalize ${GATE_NORMALIZE} \
    --evaluate_during_training \
    --freeze_encoder \
    --smart_pooling
