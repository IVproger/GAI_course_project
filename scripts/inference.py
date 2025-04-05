# scripts/inference.py
import argparse
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import os
from PIL import Image
from accelerate.utils import set_seed
# Adjust imports if running from project root
from src.model_setup import load_unet # Need to load UNet structure first
from src.utils import get_free_gpu

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using a trained DreamBooth model")
    parser.add_argument("--model_base", type=str, required=True, help="Base model identifier (e.g., 'CompVis/stable-diffusion-v1-4')")
    parser.add_argument("--unet_path", type=str, required=True, help="Path to the trained UNet state dict (.pt file)")
    parser.add_argument("--text_encoder_path", type=str, required=True, help="Path to the trained text encoder state dict (.pt file)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output_path", type=str, default="generated_image.png", help="Path to save the generated image")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Setup ---
    if args.seed:
        set_seed(args.seed)

    device = torch.device(f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu")
    # Assume fp16 for inference if available, adjust if needed
    weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, dtype: {weight_dtype}")


    # --- Load Base Components ---
    print("Loading base model components...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_base, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(args.model_base, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.model_base, subfolder="vae", torch_dtype=weight_dtype)

    # --- Load Trained Models ---
    print("Loading trained UNet and Text Encoder...")
    # Load the model architecture first
    unet = load_unet(args.model_base) # Loads the base UNet architecture
    text_encoder = CLIPTextModel.from_pretrained(args.model_base, subfolder="text_encoder") # Base text encoder architecture

    # Load the trained weights
    try:
        unet_state_dict = torch.load(args.unet_path, map_location="cpu")
        # Handle potential keys mismatch if saved with Accelerate/DDP ('module.' prefix)
        if not any("module." in k for k in unet_state_dict.keys()): # Check if no prefix exists
             unet.load_state_dict(unet_state_dict)
        else: # Handle DDP/Accelerate prefix
             from collections import OrderedDict
             new_state_dict = OrderedDict()
             for k, v in unet_state_dict.items():
                  name = k[7:] if k.startswith("module.") else k # remove `module.`
                  new_state_dict[name] = v
             unet.load_state_dict(new_state_dict)
        print(f"Loaded UNet weights from: {args.unet_path}")

        text_encoder_state_dict = torch.load(args.text_encoder_path, map_location="cpu")
        if not any("module." in k for k in text_encoder_state_dict.keys()):
             text_encoder.load_state_dict(text_encoder_state_dict)
        else:
             from collections import OrderedDict
             new_state_dict = OrderedDict()
             for k, v in text_encoder_state_dict.items():
                  name = k[7:] if k.startswith("module.") else k # remove `module.`
                  new_state_dict[name] = v
             text_encoder.load_state_dict(new_state_dict)
        print(f"Loaded Text Encoder weights from: {args.text_encoder_path}")

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
    print(f"Generating image for prompt: '{args.prompt}'...")
    generator = torch.Generator(device=device)
    if args.seed:
       generator.manual_seed(args.seed)

    with torch.autocast(device.type, dtype=weight_dtype):
         image = pipeline(
             args.prompt,
             num_inference_steps=args.num_steps,
             guidance_scale=args.guidance_scale,
             generator=generator
         ).images[0]

    # --- Save Image ---
    image.save(args.output_path)
    print(f"Image saved to: {args.output_path}")

if __name__ == "__main__":
    main()
    
    