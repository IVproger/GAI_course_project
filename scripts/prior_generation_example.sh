#!/usr/bin/env bash

# ==========================================================================
# Prior Preservation Data Generation Script
# ==========================================================================

CONFIG_PATH="configs/prior_generation_config.yaml"

echo "=== Generating Prior Preservation Data ==="
echo "→ Using config file: ${CONFIG_PATH}"

# Create output directory structure
mkdir -p "data/prior_preservation"

# Run the generation script
python scripts/generate_priors.py --config "${CONFIG_PATH}"

echo "=== Prior Data Generation Complete ==="