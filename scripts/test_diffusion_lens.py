#!/usr/bin/env python
# scripts/test_diffusion_lens.py
"""
Test script to verify DiffusionLens integration in the DreamBooth trainer.
This script loads a trained model and generates DiffusionLens visualizations.
"""

import argparse
import os
import torch
from accelerate import Accelerator
from clearml import Logger, Task

from src.config_loader import load_config
from src.model_setup import load_tokenizer, load_text_encoder, load_vae, load_unet, load_scheduler
from src.utils import set_seed
from src.trainer import DreamBoothTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Test DiffusionLens integration")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration file")
    parser.add_argument("--unet_path", type=str, required=True, help="Path to the fine-tuned UNet weights")
    parser.add_argument("--text_encoder_path", type=str, required=True, help="Path to the fine-tuned text encoder weights")
    parser.add_argument("--prompt", type=str, help="Optional prompt to use (defaults to subject_prompt in config)")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override prompt if provided
    if args.prompt:
        config['subject_prompt'] = args.prompt
    
    # Setup Accelerator
    accelerator = Accelerator(
        mixed_precision=config.get('mixed_precision', 'no'),
    )
    
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Initialize ClearML logger
    task = Task.init(
        project_name=config.get('project_name', 'DreamBooth Testing'),
        task_name=f"test_diffusion_lens_{os.path.basename(args.config).split('.')[0]}",
        reuse_last_task_id=False
    )
    logger = task.get_logger()
    
    # Load model components
    print("Loading model components...")
    tokenizer = load_tokenizer(config.get('tokenizer_name', config['model_name']))
    noise_scheduler = load_scheduler(config['model_name'], config.get('ddpm_num_train_timesteps', 1000))
    vae = load_vae(config['model_name'])
    
    # Load UNet and text encoder
    unet = load_unet(config['model_name'])
    text_encoder = load_text_encoder(config['model_name'])
    
    # Load fine-tuned weights
    print(f"Loading fine-tuned UNet weights from {args.unet_path}")
    unet_state_dict = torch.load(args.unet_path, map_location="cpu")
    unet.load_state_dict(unet_state_dict)
    
    print(f"Loading fine-tuned text encoder weights from {args.text_encoder_path}")
    text_encoder_state_dict = torch.load(args.text_encoder_path, map_location="cpu")
    text_encoder.load_state_dict(text_encoder_state_dict)
    
    # Move models to device
    weight_dtype = torch.float16 if config['mixed_precision'] == 'fp16' else torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = DreamBoothTrainer(config, accelerator, logger)
    
    # Force enable DiffusionLens for testing
    if not trainer.use_diffusion_lens:
        print("Warning: DiffusionLens is not available or not enabled in the config.")
        print("This test will not generate any visualizations.")
    else:
        # Generate DiffusionLens visualizations
        print("Generating DiffusionLens visualizations...")
        trainer.log_diffusion_lens_images(
            unet=unet, 
            vae=vae, 
            text_encoder=text_encoder, 
            tokenizer=tokenizer, 
            noise_scheduler=noise_scheduler, 
            config=config, 
            epoch=0,
            final=True
        )
        
        print("Test completed!")
        print(f"Check {os.path.join(config['output_dir'], config['task_name'], 'diffusion_lens_images')} for results.")

if __name__ == "__main__":
    main() 