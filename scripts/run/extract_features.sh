#!/bin/bash

print_usage() {
    echo "Usage: extract_features.sh [WEIGHTS_OR_EXP] [DATASET] [CONFIG_FILE] [OUTPUT_NAME]"
    echo ""
    echo "Arguments:"
    echo "  WEIGHTS_OR_EXP   - Either a checkpoint path (ending with .tar or .pth or .pt),"
    echo "                     or an experiment name. If it's an experiment name,"
    echo "                     the script loads 'outputs/{name}/train/checkpoint.pth.tar'."
    echo "  DATASET          - The dataset which acts as the input."
    echo "                     Reference to 'core/data/data_paths.py' for dataset name."
    echo "  CONFIG_FILE      - Path relative to 'core/configs/eval/descriptor/',"
    echo "                     e.g., 'dev/vits_lin.yaml'. The full path becomes"
    echo "                     'core/configs/eval/descriptor/{CONFIG_FILE}'."
    echo "  OUTPUT_NAME      - Output filename (e.g., 'train_feat_ep-30.pth')."
    echo "                     Will be saved under 'outputs/{EXP_NAME}/eval/{CONFIG_FILE_STEM}'."
}

# Help handling
if [ "$#" -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_usage
    exit 0
fi

# Parse WEIGHTS and EXP_NAME
WEIGHTS_ARG="$1"
suffix=$(echo "$WEIGHTS_ARG" | awk -F'.' '{print $NF}')
if [[ "$suffix" != "tar" && "$suffix" != "pth" ]]; then
    WEIGHTS="outputs/$WEIGHTS_ARG/train/checkpoint.pth.tar"
    EXP_NAME="$WEIGHTS_ARG"
else
    WEIGHTS="$WEIGHTS_ARG"
    # Extract meaningful experiment name from path (e.g., from 'outputs/exp123/...')
    # Fallback: use basename if no 'outputs/' prefix
    if [[ "$WEIGHTS" == outputs/* ]]; then
        EXP_NAME=$(echo "$WEIGHTS" | cut -d'/' -f2)
    else
        BASENAME=$(basename "$WEIGHTS")
        EXP_NAME="${BASENAME%%.*}"
    fi
fi

# Parse dataset name
DATASET="$2"

# Parse config file
CONFIG_FILE="$3"
FULL_CONFIG="core/configs/eval/descriptor/$CONFIG_FILE"

CONFIG_FILE_BASENAME=$(basename "$CONFIG_FILE")
CONFIG_FILE_STEM="${CONFIG_FILE_BASENAME%%.*}"

# Parse output name
OUTPUT_NAME="$4"
OUTPUT_DIR="outputs/${EXP_NAME}/eval/${CONFIG_FILE_STEM}"
OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"

# Get available port
PORT=$(python3 -c "from core.distributed import _get_available_port; print(_get_available_port())")
echo "Launching feature extraction on MASTER_PORT ${PORT}"

# Run torchrun
torchrun -m \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port $PORT \
  scripts.run.extract_features \
  --weights "${WEIGHTS}" \
  --dataset "${DATASET}" \
  --config-file "${FULL_CONFIG}" \
  --output-path "${OUTPUT_PATH}"