# scripts/generate_priors.py
import argparse
import torch
from accelerate.utils import set_seed
import os

# Adjust import paths relative to the project root if running from root
from src.config_loader import load_config
from src.prior_generation import generate_prior_preservation_data
from src.utils import get_free_gpu # Optional: keep device selection simple


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Prior Preservation Data for DreamBooth")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration YAML file")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    # --- Setup ---
    set_seed(config.get('seed', 42)) # Use seed from config if available
    device = torch.device(f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu")

    weight_dtype = torch.float32
    if config.get('mixed_precision') == "fp16":
        weight_dtype = torch.float16
    elif config.get('mixed_precision') == "bf16":
        weight_dtype = torch.bfloat16

    print(f"Using device: {device}, dtype: {weight_dtype}")

    # Ensure prior generation is specifically requested via config or script purpose
    if not os.path.exists(config['prior_data_dir']):
         print(f"Prior data directory {config['prior_data_dir']} does not exist. Creating...")
    else:
         print(f"Prior data directory {config['prior_data_dir']} already exists. Files might be overwritten.")


    # --- Generate Data ---
    generate_prior_preservation_data(
        model_name=config['model_name'],
        revision=config.get('revision'),
        variant=config.get('variant'),
        device=device,
        weight_dtype=weight_dtype,
        num_prior_images=config['num_prior_images'],
        class_prompt=config['class_prompt'],
        image_resolution=config['image_resolution'],
        batch_size=config['prior_generation_batch_size'],
        num_inference_steps=config.get('prior_num_inference_steps', 50),
        guidance_scale=config.get('prior_guidance_scale', 7.5),
        save_path=config['prior_data_dir']
    )

    print("Prior data generation finished.")

if __name__ == "__main__":
    main()
    