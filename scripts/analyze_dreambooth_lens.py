import torch
import os
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from DiffusionLens.pipeline_stable_diffusion import StableDiffusionPipeline as StableDiffusionGlassPipeline
from PIL import Image
import argparse
from collections import OrderedDict
# Adjust import based on your project structure - assuming src is accessible
try:
    from src.model_setup import load_unet
    from src.utils import get_free_gpu # If needed for device selection, otherwise remove
except ImportError:
    print("Warning: Could not import from src. Ensure 'src' is in the Python path or adjust imports.")
    # Define dummy functions or raise error if essential
    def load_unet(base_model):
        print("Error: src.model_setup.load_unet could not be imported!")
        print("Loading UNet using standard from_pretrained method as fallback.")
        # Fallback or raise error - using standard loading here
        return UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    def get_free_gpu():
         # Fallback device selection
         return 0


# --- Configuration ---
# Model Paths - provide paths from your inference config
MODEL_BASE = "runwayml/stable-diffusion-v1-5" # <<< EDIT: Your base model ID
UNET_PATH = "outputs/dreambooth_dog/unet_final.pt" # <<< EDIT: Path to your fine-tuned UNet weights file
TEXT_ENCODER_PATH = "outputs/dreambooth_dog/text_encoder_final.pt" # <<< EDIT: Path to your fine-tuned text encoder weights file

# Analysis Parameters
PROMPT = "a photo of xon dog in the car"  # Replace [V] with your Dreambooth trigger word if needed
NEGATIVE_PROMPT = "low quality, blurry, deformed"
OUTPUT_DIR = "outputs/dreambooth_lens_analysis"
SEED = 42
NUM_IMAGES_PER_PROMPT = 1 # Generate 1 image per prompt for simplicity
NUM_INFERENCE_STEPS = 100 # Standard number of steps
GUIDANCE_SCALE = 7.5

# Diffusion Lens Parameters (Control which intermediate steps are visualized)
# These might need tuning based on the paper/original repo examples.
# 'Layers' in this context likely refers to specific UNet blocks or timesteps.
# The pipeline code needs inspection to be sure. Let's assume it maps to timesteps for now.
START_LAYER = 0             # Start visualization from this step/layer
END_LAYER = 20 # End visualization at this step/layer (inclusive?)
STEP_LAYER = 1              # Visualize every N steps/layers

# Determine device dynamically or keep fixed
DEVICE = torch.device(f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Simpler fixed device selection
WEIGHT_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
# --- End Configuration ---

def main(args):
    # print(f"Loading Dreambooth model from: {args.model_path}") # Removed old message
    print(f"Using base model: {args.model_base}")
    print(f"Loading UNet weights from: {args.unet_path}")
    print(f"Loading Text Encoder weights from: {args.text_encoder_path}")
    print(f"Using device: {DEVICE}, dtype: {WEIGHT_DTYPE}")


    # 1. Load Model Components (following scripts/inference.py logic)
    try:
        # Load base components
        print("Loading base tokenizer, scheduler, vae...")
        tokenizer = CLIPTokenizer.from_pretrained(args.model_base, subfolder="tokenizer")
        # Using DDPMScheduler as in inference.py, or keep DPMSolver? Let's stick to inference.py's scheduler for consistency.
        # If DPMSolver is needed for DiffusionLens pipeline, we might need to load it separately or modify the pipeline init.
        # For now, using DDPMScheduler.
        scheduler = DDPMScheduler.from_pretrained(args.model_base, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(args.model_base, subfolder="vae", torch_dtype=WEIGHT_DTYPE)
        print("Base components loaded.")

        # Load structure for UNet and Text Encoder
        print("Loading UNet and Text Encoder structures...")
        unet = load_unet(args.model_base) # Using the function from src
        text_encoder = CLIPTextModel.from_pretrained(args.model_base, subfolder="text_encoder")
        print("Structures loaded.")

        # Load fine-tuned weights
        print("Loading fine-tuned UNet weights...")
        unet_state_dict = torch.load(args.unet_path, map_location="cpu")
        if not any("module." in k for k in unet_state_dict.keys()):
            unet.load_state_dict(unet_state_dict)
        else:
            new_state_dict = OrderedDict()
            for k, v in unet_state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            unet.load_state_dict(new_state_dict)
        print("UNet weights loaded.")

        print("Loading fine-tuned Text Encoder weights...")
        text_encoder_state_dict = torch.load(args.text_encoder_path, map_location="cpu")
        if not any("module." in k for k in text_encoder_state_dict.keys()):
            text_encoder.load_state_dict(text_encoder_state_dict)
        else:
            new_state_dict = OrderedDict()
            for k, v in text_encoder_state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            text_encoder.load_state_dict(new_state_dict)
        print("Text Encoder weights loaded.")

    except FileNotFoundError as e:
        print(f"Error: Could not find state dict file: {e}")
        print(f"Please ensure --unet_path ('{args.unet_path}') and --text_encoder_path ('{args.text_encoder_path}') are correct.")
        return
    except ImportError as e:
         print(f"Import Error: {e}. Could not load model structure. Make sure 'src' is accessible.")
         return
    except Exception as e:
        print(f"Error loading model components: {e}")
        # print("Please ensure the path is correct and contains the necessary subfolders (tokenizer, text_encoder, vae, unet, scheduler).") # Old message
        return

    # Move models to device and set dtype
    unet.to(DEVICE, dtype=WEIGHT_DTYPE)
    text_encoder.to(DEVICE, dtype=WEIGHT_DTYPE)
    vae.to(DEVICE, dtype=WEIGHT_DTYPE)
    unet.eval()
    text_encoder.eval()
    vae.eval()

    print("Model components loaded and configured successfully.")

    # 2. Instantiate Diffusion Lens Pipeline
    # NOTE: DiffusionLens pipeline might expect a specific scheduler type (e.g., DPMSolverMultistepScheduler).
    # We loaded DDPMScheduler from inference.py. If this causes issues, we might need to:
    # a) Modify DiffusionLens pipeline to accept DDPMScheduler.
    # b) Load DPMSolverMultistepScheduler separately and pass it here instead of the loaded 'scheduler'.
    # Let's try with the loaded DDPMScheduler first.
    pipe = StableDiffusionGlassPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None, # Assuming no safety checker for analysis
        feature_extractor=None,
    )
    pipe = pipe.to(DEVICE)
    print("Diffusion Lens pipeline instantiated.")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # Set seed for reproducibility
    generator = torch.Generator(DEVICE).manual_seed(args.seed)

    # 3. Run the Pipeline
    print(f"Running pipeline for prompt: '{args.prompt}'")
    print(f"Visualizing from layer {args.start_layer} to {args.end_layer} with step {args.step_layer}")

    # The StableDiffusionGlassPipeline returns a list of outputs, one per visualized layer/step
    try:
        layer_outputs = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            num_images_per_prompt=args.num_images_per_prompt,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            step_layer=args.step_layer,
            output_type="pil" # Get PIL images directly
        )
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        # This might happen if start/end/step_layer arguments are incompatible
        # with how the custom pipeline expects them.
        # You might need to inspect the __call__ method of StableDiffusionGlassPipeline more closely.
        return

    # 4. Process and Save Results
    print(f"Pipeline finished. Processing {len(layer_outputs)} layer outputs.")
    
    # Determine the layer index based on start/step
    current_layer = args.start_layer
    
    for i, output in enumerate(layer_outputs):
        if not hasattr(output, 'images') or not output.images:
            print(f"Warning: No images found in output for layer index {i}")
            continue

        # Usually, num_images_per_prompt is 1 for this kind of analysis
        img = output.images[0] 
        
        # Save the image
        layer_filename = f"layer_{current_layer:03d}_step_{i:03d}.png"
        output_path = os.path.join(args.output_dir, layer_filename)
        img.save(output_path)
        
        print(f"Saved image for layer {current_layer} to {output_path}")
        
        current_layer += args.step_layer

    print("Analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Diffusion Lens analysis on a Dreambooth model.")
    
    # parser.add_argument("--model_path", type=str, default=DREAMBOOTH_MODEL_PATH, help="Path to the Dreambooth model directory.") # Removed
    parser.add_argument("--model_base", type=str, default=MODEL_BASE, help="Base model identifier (e.g., 'stabilityai/stable-diffusion-2-1-base').")
    parser.add_argument("--unet_path", type=str, default=UNET_PATH, help="Path to the fine-tuned UNet weights file (.pt, .bin, .safetensors).")
    parser.add_argument("--text_encoder_path", type=str, default=TEXT_ENCODER_PATH, help="Path to the fine-tuned text encoder weights file (.pt, .bin, .safetensors).")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Text prompt for image generation.")
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT, help="Negative text prompt.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save the output images.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for generation.")
    parser.add_argument("--num_images_per_prompt", type=int, default=NUM_IMAGES_PER_PROMPT, help="Number of images to generate for the prompt.")
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE, help="Classifier-free guidance scale.")
    parser.add_argument("--start_layer", type=int, default=START_LAYER, help="Starting layer/step for visualization.")
    parser.add_argument("--end_layer", type=int, default=END_LAYER, help="Ending layer/step for visualization.")
    parser.add_argument("--step_layer", type=int, default=STEP_LAYER, help="Step size for layer/step visualization.")

    args = parser.parse_args()
    
    # if args.model_path == "YOUR_DREAMBOOTH_MODEL_PATH": # Removed old check
    #     print("Error: Please edit 'analyze_dreambooth_lens.py' and set the DREAMBOOTH_MODEL_PATH variable.")
    if args.unet_path == "path/to/your/trained_unet.pt" or args.text_encoder_path == "path/to/your/trained_text_encoder.pt":
        print("Error: Please edit 'analyze_dreambooth_lens.py' and set the UNET_PATH and TEXT_ENCODER_PATH variables,")
        print("or provide them using --unet_path and --text_encoder_path arguments.")
    else:
        main(args) 