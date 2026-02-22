#!/bin/bash

print_usage() {
    echo "Usage: extract_features.sh [DATASET] [CONFIG_FILE] [OUTPUT_NAME]"
    echo ""
    echo "Arguments:"
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

# Parse dataset name
DATASET="$1"

# Parse config file
CONFIG_FILE="$2"
FULL_CONFIG="core/configs/eval/descriptor/$CONFIG_FILE"

CONFIG_FILE_BASENAME=$(basename "$CONFIG_FILE")
CONFIG_FILE_STEM="${CONFIG_FILE_BASENAME%%.*}"

# Parse output name
OUTPUT_PATH="outputs/dino/eval/$3"

# Get available port
PORT=$(python3 -c "from core.distributed import _get_available_port; print(_get_available_port())")
echo "Launching feature extraction on MASTER_PORT ${PORT}"

# Run torchrun
torchrun -m \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port $PORT \
  scripts.run.extract_dino_features \
  --weights "${WEIGHTS}" \
  --dataset "${DATASET}" \
  --config-file "${FULL_CONFIG}" \
  --output-path "${OUTPUT_PATH}"