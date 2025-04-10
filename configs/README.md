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
