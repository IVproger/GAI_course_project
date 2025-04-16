# scripts/inference.py
import argparse
import torch
import yaml
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import os
from PIL import Image
from accelerate.utils import set_seed
# Adjust imports if running from project root
from src.model_setup import load_unet
from src.utils import get_free_gpu

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using a trained DreamBooth model")
    parser.add_argument("--config", type=str, required=True, help="Path to the inference configuration YAML file")
    return parser.parse_args()

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # --- Setup ---
    if config.get('seed'):
        set_seed(config['seed'])

    device = torch.device(f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, dtype: {weight_dtype}")

    # --- Load Base Components ---
    print("Loading base model components...")
    tokenizer = CLIPTokenizer.from_pretrained(config['model_base'], subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(config['model_base'], subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(config['model_base'], subfolder="vae", torch_dtype=weight_dtype)

    # --- Load Trained Models ---
    print("Loading trained UNet and Text Encoder...")
    unet = load_unet(config['model_base'])
    text_encoder = CLIPTextModel.from_pretrained(config['model_base'], subfolder="text_encoder")

    try:
        unet_state_dict = torch.load(config['unet_path'], map_location="cpu")
        if not any("module." in k for k in unet_state_dict.keys()):
            unet.load_state_dict(unet_state_dict)
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in unet_state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            unet.load_state_dict(new_state_dict)
        print(f"Loaded UNet weights from: {config['unet_path']}")

        text_encoder_state_dict = torch.load(config['text_encoder_path'], map_location="cpu")
        if not any("module." in k for k in text_encoder_state_dict.keys()):
            text_encoder.load_state_dict(text_encoder_state_dict)
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in text_encoder_state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            text_encoder.load_state_dict(new_state_dict)
        print(f"Loaded Text Encoder weights from: {config['text_encoder_path']}")

    except FileNotFoundError as e:
        print(f"Error: Could not find state dict file: {e}")
        return
    except Exception as e:
        print(f"Error loading state dicts: {e}")
        return

    # Move models to device and set dtype
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.eval()
    text_encoder.eval()
    vae.eval()

    # --- Create Pipeline ---
    print("Creating inference pipeline...")
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline.set_progress_bar_config(disable=False)

    # --- Generate Image ---
    print(f"Generating image for prompt: '{config['prompt']}'...")
    generator = torch.Generator(device=device)
    if config.get('seed'):
        generator.manual_seed(config['seed'])

    with torch.autocast(device.type, dtype=weight_dtype):
        image = pipeline(
            config['prompt'],
            num_inference_steps=config.get('num_steps', 50),
            guidance_scale=config.get('guidance_scale', 7.5),
            generator=generator
        ).images[0]

    # Make sure output directory exists
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    
    # --- Save Image ---
    image.save(config['output_path'])
    print(f"Image saved to: {config['output_path']}")

if __name__ == "__main__":
    main()