#!/bin/bash

print_usage() {
    echo "Usage: train_des.sh [option1] [option2] [option3] [option4] [option5]"
    echo "   option1 - EXP_NAME: name of this experiment."
    echo "   option2 - PRETRAIN: dir of pretrain model, if this parameter does not end with '.tar' or '.pth', then the code will load checkpoint from 'outputs/{PRETRAIN}/train/checkpoint.pth.tar'."
    echo "   option3 - CONFIG: path of config file of this experiment, the code will load config from 'core/configs/train/descriptor/{CONFIG}'."
    echo "   option4 - EVAL_CONFIG: path of eval config file, the code will load config from 'core/configs/eval/descriptor/{EVAL_CONFIG}'."
    echo "   option5 - OPTS: extra options, default is ''."
}

# help command
if [ "$#" -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_usage
    exit 1
fi

# set EXP_NAME
EXP_NAME=${1:-"debug"}
echo "Set EXP_NAME: ${EXP_NAME}"

# set PRETRAIN
suffix=$(echo $2 | awk -F'.' '{print $NF}')
if [ $suffix != 'tar' ] && [ $suffix != 'pth' ]; then
  PRETRAIN="outputs/$2/train/checkpoint.pth.tar"
else
  PRETRAIN=$2
fi
echo "Set PRETRAIN: ${PRETRAIN}"

# set CONFIG
CONFIG="core/configs/train/descriptor/$3"
echo "Set CONFIG: ${CONFIG}"

# set EVAL_CONFIG
EVAL_CONFIG="core/configs/eval/descriptor/$4"
echo "Set EVAL_CONFIG: ${EVAL_CONFIG}"

# set OPTS
OPTS=${5:-""}
echo "Set OPTS: ${OPTS}"

PORT=$(python3 -c "from core.distributed import _get_available_port; print(_get_available_port())")
echo "Launch on MASTER_PORT ${PORT}"

torchrun -m \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port $PORT \
  core.run.train.train \
  --arch descriptor \
  --output-dir "outputs/${EXP_NAME}/train" \
  --config-file "${CONFIG}" \
  opts "train.pretrain=${PRETRAIN} ${OPTS}"


EVAL_DIR=$(echo $EVAL_CONFIG | awk -F'.' '{print $1}' | awk -F'/' '{print $NF}')
PORT=$(python3 -c "from core.distributed import _get_available_port; print(_get_available_port())")
echo "Launch Caching on MASTER_PORT ${PORT}"

torchrun -m \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port $PORT \
  core.run.eval.eval_des \
  --config-file "${EVAL_CONFIG}" \
  --weights "outputs/${EXP_NAME}/train/checkpoint.pth.tar" \
  --output-dir "outputs/${EXP_NAME}/eval/${EVAL_DIR}"
