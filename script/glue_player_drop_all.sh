#!/bin/bash
if [[ $# -ne 3 ]]; then
  echo "glue_player_drop_all.sh <drop> <do_gate_dropout> <do_gate_weight_normalization>"
  exit 1
fi
PRUNE_LAYER=$1
GATE_DROPOUT=$2
GATE_NORMALIZE=$3

./glue_player_drop_test.sh MRPC 8 ${PRUNE_LAYER} ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_drop_test.sh RTE 8 ${PRUNE_LAYER} ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_drop_test.sh STS-B 8 ${PRUNE_LAYER} ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_drop_test.sh CoLA 8 ${PRUNE_LAYER} ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_drop_test.sh SST-2 8 ${PRUNE_LAYER} ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_drop_test.sh QNLI 8 ${PRUNE_LAYER} ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_drop_test.sh MNLI 8 ${PRUNE_LAYER} ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_drop_test.sh QQP 8 ${PRUNE_LAYER} ${GATE_DROPOUT} ${GATE_NORMALIZE}
