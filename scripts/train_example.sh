#!/usr/bin/env bash

# ==========================================================================
# DreamBooth Training Launch Script
# ==========================================================================
#
# This script launches DreamBooth training using Accelerate for multi-GPU
# distribution and YAML configuration files.
#
# ==========================================================================

# Set configuration paths
# train_dog_dreambooth.yaml
# train_duck_toy_dreambooth.yaml
TRAIN_CONFIG_PATH="configs/train/train_duck_toy_dreambooth.yaml" 
ACCELERATE_CONFIG_PATH="configs/accelerate_config.yaml"

echo "=== Starting DreamBooth Training ==="
echo "→ Training config: ${TRAIN_CONFIG_PATH}"
echo "→ Accelerate config: ${ACCELERATE_CONFIG_PATH}"

# Create timestamp for run identification
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "→ Run timestamp: ${TIMESTAMP}"

# Launch training with Accelerate
accelerate launch --config_file ${ACCELERATE_CONFIG_PATH} scripts/train.py --config ${TRAIN_CONFIG_PATH}

echo "=== Training Complete ==="
echo "Check output directory specified in config for results"