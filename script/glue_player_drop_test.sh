#!/bin/bash
if [[ $# -ne 5 ]]; then
  echo "glue_player_drop_test.sh <task> <batch_size> <drop> <do_gate_dropout> <do_gate_weight_normalization>"
  exit 1
fi
tstr=$(date +"%FT%H%M")

GLUE_DIR="../data"
TASK_NAME=$1
BATCH_SIZE=$2
PRUNE_LAYER=$3
GATE_DROPOUT=$4
GATE_NORMALIZE=$5

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
    --output_dir "tmp/${TASK_NAME}_player_drop_${PRUNE_LAYER}${SUB_CFG}_${tstr}/" \
    --gate_dropout ${GATE_DROPOUT} \
    --gate_normalize ${GATE_NORMALIZE} \
    --lp_init_method "norm" \
    --prune_layer_count ${PRUNE_LAYER} \
    --evaluate_during_training \
    --freeze_encoder \
    --smart_pooling
