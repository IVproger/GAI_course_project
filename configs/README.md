# Configuration Files

This directory contains all configuration files used in the DreamBooth project for training, inference, and model deployment.

## Directory Structure

- `prior/` - Configuration files for prior preservation training
- `inference/` - Configuration files for model inference
- `train/` - Configuration files for model training

## Configuration Files

### Main Configuration Files

- `accelerate_config.yaml` - Configuration for distributed training using Accelerate
- `inference_config.yaml` - Settings for model inference and generation
- `prior_generation_config.yaml` - Configuration for prior preservation generation
- `token_selection_config.yaml` - Settings for token selection and management
- `rare_tokens.json` - List of rare tokens used for subject binding

## Usage

These configuration files are used to:
1. Configure training parameters and hyperparameters
2. Set up inference settings
3. Manage model deployment configurations
4. Control token selection and management
5. Configure distributed training settings

## File Descriptions

### accelerate_config.yaml
Contains settings for distributed training using the Accelerate library, including:
- Mixed precision settings
- Distributed training configuration
- Hardware utilization parameters

### inference_config.yaml
Defines parameters for model inference:
- Generation settings
- Model loading configurations
- Output formatting options

### prior_generation_config.yaml
Settings for prior preservation:
- Prior preservation loss parameters
- Training configuration for prior generation
- Model checkpoint settings

### token_selection_config.yaml
Configuration for token selection:
- Token selection criteria
- Token management parameters
- Vocabulary settings

### rare_tokens.json
Contains the list of rare tokens used for subject binding in the DreamBooth approach.

## DiffusionLens Visualization Configuration

Training configurations now support DiffusionLens for visualizing the diffusion process during training. 
To enable this feature, add the following parameters to your training config:

```yaml
# DiffusionLens Visualization Settings
use_diffusion_lens: true  # Enable DiffusionLens visualization
diffusion_lens_epochs: 50  # Generate visualizations every N epochs
diffusion_lens_params:
  start_layer: 0  # Start visualization from this layer/step
  end_layer: 20  # End visualization at this layer/step
  step_layer: 1  # Visualize every N steps/layers
  num_inference_steps: 100  # Number of inference steps
  guidance_scale: 7.5  # Guidance scale
```

The DiffusionLens visualizations will be saved in `{output_dir}/{task_name}/diffusion_lens_images/`.
This feature requires the DiffusionLens library to be installed.
