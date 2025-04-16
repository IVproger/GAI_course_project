#!/usr/bin/env bash

# ==========================================================================
# Rare Token Selection Script with YAML Config for DreamBooth
# ==========================================================================
#
# This script helps find rare tokens for DreamBooth training using
# YAML configuration and records selections for reproducibility.
#
# ==========================================================================

# Default config path
CONFIG_PATH="configs/token_selection_config.yaml"

# Help message
function show_help {
    echo "Usage: ./select_token.sh --config <config_path>"
    echo ""
    echo "Required options:"
    echo "  --config <path>      Path to YAML configuration file"
    echo ""
    echo "Example configuration file format:"
    echo "  task_name: \"dog_dreambooth\""
    echo "  concept_type: \"dog\""
    echo "  model_name: \"runwayml/stable-diffusion-v1-5\""
    echo "  num_suggestions: 10"
    echo "  token_json_path: \"configs/rare_tokens.json\""
    echo ""
    exit 1
}

echo "=== Rare Token Selection for DreamBooth ==="
echo "→ Using config file: ${CONFIG_PATH}"

# Run the Python script
python scripts/find_rare_token.py --config "${CONFIG_PATH}"

echo "=== Token Selection Complete ==="
echo "You can use the selected token in your DreamBooth config."