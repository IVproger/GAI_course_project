#!/usr/bin/env bash

# ==========================================================================
# DreamBooth Inference Script with YAML Configuration
# ==========================================================================
#
# This script generates images using a trained DreamBooth model
# using a YAML configuration file.
#
# ==========================================================================

# Set config path
CONFIG_PATH="configs/inference_config.yaml"

# Create timestamp for unique filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Print configuration
echo "=== DreamBooth Inference ==="
echo "Config Path: ${CONFIG_PATH}"
echo "Timestamp: ${TIMESTAMP}"
echo "=== Starting Generation ==="

# Pre-create output directory (safer approach)
mkdir -p "inference_outputs/dog_dreambooth/inference_results"

# Run inference with YAML config
python scripts/inference.py --config "${CONFIG_PATH}"

echo "=== Generation Complete ==="
echo "Check inference_outputs directory for results"