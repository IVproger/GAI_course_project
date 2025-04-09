# 🐍 Source Code (`src/`)

This directory contains the core Python source code for the DreamBooth fine-tuning project. It is structured as a Python package (due to the presence of `__init__.py`) and holds reusable modules implementing the project's logic.

## Purpose

-   **Modularity:** Encapsulates distinct functionalities into separate modules (data handling, model loading, training logic, utilities).
-   **Reusability:** Allows code components to be imported and used across different scripts (e.g., training, inference, data generation).
-   **Maintainability:** Organizes the codebase logically, making it easier to understand, debug, and extend.

## Key Modules

-   **`__init__.py`**: Makes the `src` directory recognizable as a Python package.
-   **`config_loader.py`**: Utility functions for loading and parsing configuration files (e.g., `.yaml`) from the `configs/` directory.
-   **`data_handling.py`**: Contains PyTorch `Dataset` classes (like `PriorClassDataset`), data transformations (`get_train_transforms`), collate functions (`collate_fn`), and functions for loading data (like `load_prior_data`). Handles preparation of data from the `data/` directory.
-   **`model_setup.py`**: Provides functions to load pre-trained model components (Tokenizer, Text Encoder, VAE, UNet, Scheduler) from Hugging Face `diffusers` and `transformers` libraries, and to create inference pipelines.
-   **`prior_generation.py`**: Implements the logic for generating the images/latents required for prior preservation loss, typically invoked by `scripts/generate_priors.py`.
-   **`trainer.py`**: Contains the main training loop logic (`DreamBoothTrainer` class), including optimizer/scheduler setup, loss calculation, gradient updates, integration with `accelerate`, checkpoint saving, and periodic evaluation/logging (using ClearML).
-   **`utils.py`**: Holds miscellaneous utility functions used across the project (e.g., `get_free_gpu` for device selection, `set_seed` for reproducibility, visualization helpers).

## Usage

Modules within `src/` are imported and utilized by the executable scripts located in the `scripts/` directory to perform tasks like data preparation, model training, and inference.
