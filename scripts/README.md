# ▶️ Executable Scripts (`scripts/`)

This directory contains the main Python scripts that serve as entry points for running different stages of the DreamBooth project workflow.

## Purpose

-   **Workflow Orchestration:** Provides user-facing scripts to execute specific tasks like generating prior data, training a model, or performing inference.
-   **Integration:** These scripts integrate the various components defined in the `src/` directory (data loading, model setup, training logic) and utilize configurations from the `configs/` directory.

## Key Scripts

-   **`generate_priors.py`**:
    -   **Goal:** Generates and saves the necessary data (latents, embeddings) for prior preservation loss.
    -   **Usage:** Run this script before training if `use_prior_preservation` is enabled in the config and the prior data doesn't exist yet.
    -   Example: `python -m scripts.generate_priors --config configs/train_dog_dreambooth.yaml`

-   **`train.py`**:
    -   **Goal:** Executes the main model fine-tuning process using DreamBooth and prior preservation (if enabled).
    -   **Usage:** This script loads data, sets up models, runs the training loop defined in `src/trainer.py`, saves checkpoints to `outputs/`, and logs metrics (e.g., to ClearML). It should be launched using `accelerate`.
    -   Example: `accelerate launch scripts/train.py --config configs/train_dog_dreambooth.yaml`

-   **`inference.py`**:
    -   **Goal:** Loads a trained DreamBooth model (specifically, the fine-tuned UNet and text encoder weights saved during training) and generates images based on text prompts.
    -   **Usage:** Used to test and utilize the fine-tuned model after training is complete. Requires paths to the saved model checkpoints from the `outputs/` directory.
    -   Example: `python -m scripts.inference --model_base "CompVis/stable-diffusion-v1-4" --unet_path "outputs/dog_dreambooth_XYZ/unet_final.pt" --text_encoder_path "outputs/dog_dreambooth_XYZ/text_encoder_final.pt" --prompt "a photo of xon dog"`

## Running Scripts

It is recommended to run these scripts from the **project root directory** (the directory containing `src/`, `scripts/`, `configs/`, etc.) using the `python -m <module_path>` syntax or `accelerate launch` for the training script. This ensures that Python can correctly resolve imports from the `src/` package.

`.sh` scripts will demonstrate you examples of how to run python script
