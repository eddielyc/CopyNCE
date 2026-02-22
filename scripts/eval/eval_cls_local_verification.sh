#!/bin/bash

print_usage() {
    echo "Usage: eval_cls_local_verification.sh [option1] [option2] [option3]"
    echo "   option1 - MODEL: name of this experiment or checkpoint path. If this parameter does not end with '.tar' or '.pth', then the code will load checkpoint from 'outputs/{MODEL}/train/checkpoint.pth.tar'."
    echo "   option2 - EVAL_CONFIG: path of eval config file, the code will load config from 'core/configs/eval/matching/{EVAL_CONFIG}'."
    echo "   option3 - OPTS: extra options, default is ''."
}

# help command
if [ "$#" -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_usage
    exit 1
fi

# set WEIGHTS
suffix=$(echo $1 | awk -F'.' '{print $NF}')
if [ $suffix != 'tar' ] && [ $suffix != 'pth' ] && [ $suffix != 'pt' ]; then
  WEIGHTS="outputs/$1/train/checkpoint.pth.tar"
  EXP_NAME="$1"
else
  WEIGHTS=$1
  if [[ "$WEIGHTS" == outputs/* ]]; then
      EXP_NAME=$(echo "$WEIGHTS" | cut -d'/' -f2)
  else
      BASENAME=$(basename "$WEIGHTS")
      EXP_NAME="${BASENAME%%.*}"
  fi
fi
echo "Set WEIGHTS: ${WEIGHTS}"
echo "Set EXP_NAME: ${EXP_NAME}"

# set EVAL_CONFIG
EVAL_CONFIG="core/configs/eval/matching/$2"
echo "Set EVAL_CONFIG: ${EVAL_CONFIG}"

# set OPTS
OPTS=${3:-""}
echo "Set OPTS: ${OPTS}"

EVAL_DIR_BASENAME=$(basename "$EVAL_CONFIG")
EVAL_DIR_STEM="${EVAL_DIR_BASENAME%%.*}"
EVAL_DIR="outputs/${EXP_NAME}/eval/${EVAL_DIR_STEM}"

PORT=$(python3 -c "from core.distributed import _get_available_port; print(_get_available_port())")
echo "Launch Caching on MASTER_PORT ${PORT}"

torchrun -m \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port $PORT \
  core.run.eval.eval_cls_local_verification \
  --config-file "${EVAL_CONFIG}" \
  --weights "${WEIGHTS}" \
  --output-dir "${EVAL_DIR}" \
  opts "${OPTS}"
