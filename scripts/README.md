# Scripts Directory

This directory contains all executable scripts for training, inference, and utility functions in the DreamBooth project.

## Main Scripts

### Training and Model Management

- `train.py` - Main training script for DreamBooth fine-tuning
  - Handles model training configuration
  - Manages training loop
  - Saves checkpoints and logs

- `generate_priors.py` - Generates prior preservation data
  - Creates prior preservation images
  - Processes reference images
  - Saves generated data

- `find_rare_token.py` - Utility for finding rare tokens
  - Analyzes token frequency
  - Selects appropriate tokens
  - Generates token reports

### Inference and Generation

- `inference.py` - Handles model inference
  - Loads trained models
  - Generates images from prompts
  - Saves generated outputs

### Example Scripts

- `train_example.sh` - Example training script
  - Shows training configuration
  - Demonstrates parameter usage
  - Provides usage examples

- `prior_generation_example.sh` - Example prior generation
  - Shows prior generation setup
  - Demonstrates parameter usage
  - Provides usage examples

- `inference_example.sh` - Example inference script
  - Shows inference configuration
  - Demonstrates prompt usage
  - Provides usage examples

- `select_token_example.sh` - Example token selection
  - Shows token selection process
  - Demonstrates parameter usage
  - Provides usage examples

### Utility Scripts

- `path_setup.sh` - Sets up environment paths
  - Configures Python path
  - Sets up environment variables
  - Initializes workspace

## Usage

### Training

```bash
# Run training with example configuration
./train_example.sh

# Or run directly with Python
python train.py --config configs/train/config.yaml
```

### Prior Generation

```bash
# Generate prior preservation data
./prior_generation_example.sh

# Or run directly
python generate_priors.py --config configs/prior/config.yaml
```

### Inference

```bash
# Run inference with example configuration
./inference_example.sh

# Or run directly
python inference.py --config configs/inference/config.yaml
```

### Token Selection

```bash
# Find and select rare tokens
./select_token_example.sh

# Or run directly
python find_rare_token.py --config configs/token_selection_config.yaml
```

## Best Practices

1. **Script Usage**
   - Always check configuration files
   - Use example scripts as templates
   - Follow parameter guidelines

2. **Environment Setup**
   - Run path_setup.sh first
   - Verify dependencies
   - Check GPU availability

3. **Error Handling**
   - Check logs for errors
   - Verify input data
   - Monitor resource usage

## Note

- All scripts require proper configuration files
- Check GPU requirements before running
- Monitor memory usage during execution
- Keep logs for debugging purposes
