# Jupyter Notebooks

This directory contains Jupyter notebooks used for experimentation, analysis, and development of the DreamBooth project.

## Notebooks Overview

### Main Notebooks

- `POC_DreamBooth.ipynb` - Proof of Concept implementation of DreamBooth
  - Contains the initial implementation and testing of the DreamBooth approach
  - Demonstrates the fine-tuning process on reference images
  - Shows example generations and results

- `diffusion_text-img.ipynb` - Text-to-Image Diffusion Model Experiments
  - Exploratory notebook for text-to-image diffusion models
  - Contains experiments with different model architectures
  - Includes visualization and analysis tools

- `clearml.ipynb` - Experiment Tracking with ClearML
  - Integration with ClearML for experiment tracking
  - Demonstrates logging and monitoring capabilities
  - Contains examples of metric tracking and visualization

### Additional Files

- `config.txt` - Configuration settings for notebooks
- `experiments_and_raw_code/` - Directory containing experimental code and raw implementations

## Usage

1. Ensure you have Jupyter installed in your environment
2. Install required dependencies from the main project's requirements.txt
3. Launch Jupyter Notebook or Jupyter Lab
4. Open the desired notebook

## Notebook Dependencies

Each notebook may require specific dependencies. Make sure to:
1. Run all cells in order
2. Install any additional requirements mentioned in the notebook
3. Check the notebook's first cell for specific setup instructions

## Experiment Tracking

The `clearml.ipynb` notebook demonstrates how to:
- Track experiments using ClearML
- Log metrics and parameters
- Visualize results
- Compare different runs

## Note

These notebooks are primarily for experimentation and development. For production use, refer to the scripts in the `scripts/` directory.