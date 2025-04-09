# ⚙️ Configuration Files (`configs/`)

This directory contains configuration files for the project, primarily in YAML format (`.yaml`). These files define the parameters and settings used by the different scripts, especially for training and data generation.

## Purpose

-   **Parameter Management:** Centralizes hyperparameters (learning rate, batch size, epochs), model identifiers, paths, and other settings.
-   **Experiment Tracking:** Allows for easy definition and tracking of different experimental setups by creating separate config files.
-   **Reproducibility:** Ensures that runs can be reproduced by using the exact configuration file associated with a specific experiment.

## Contents

-   `*.yaml`: Configuration files for specific tasks or experiments.
    -   Example: `inference_config.yaml` defines settings for inference a DreamBooth model on a specific dog dataset.

## Usage

Configuration files are typically loaded by scripts in the `scripts/` directory using utilities found in `src/config_loader.py`. The path to the desired configuration file is usually passed as a command-line argument to the script.

Example script usage:
```bash
# Run training using a specific config
accelerate launch scripts/train.py --config configs/accelerate.yaml
