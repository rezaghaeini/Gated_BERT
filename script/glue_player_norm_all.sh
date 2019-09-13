#!/bin/bash
GATE_DROPOUT=$1
GATE_NORMALIZE=$2
./glue_player_norm_test.sh MRPC 8 ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_norm_test.sh RTE 8 ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_norm_test.sh STS-B 8 ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_norm_test.sh CoLA 8 ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_norm_test.sh SST-2 8 ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_norm_test.sh QNLI 8 ${GATE_DROPOUT} ${GATE_NORMALIZE}
./glue_player_norm_test.sh MNLI 8 ${GATE_DROPOUT} ${GATE_NORMALIZE}
