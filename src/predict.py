import os
import torch
import yaml
from datetime import datetime
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate.utils import set_seed
from collections import OrderedDict

from src.model_setup import load_unet
from src.utils import get_free_gpu
from src.enums import AnimalType


# --- Helper: Load config ---
def load_config_by_animal(animal_type: AnimalType) -> dict:
    config_path = f"configs/inference/{animal_type.value}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config found for animal type '{animal_type.value}' at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    


# --- Helper: Load all components ---
def load_model_components(config, device, weight_dtype):
    tokenizer = CLIPTokenizer.from_pretrained(config['model_base'], subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(config['model_base'], subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(config['model_base'], subfolder="vae", torch_dtype=weight_dtype)
    unet = load_unet(config['model_base'])
    text_encoder = CLIPTextModel.from_pretrained(config['model_base'], subfolder="text_encoder")

    # Load UNet weights
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

    # Move to device and eval
    for model in [unet, vae, text_encoder]:
        model.to(device, dtype=weight_dtype)
        model.eval()

    return tokenizer, scheduler, vae, unet, text_encoder


# --- Main function ---
def predict(prompt: str, animal_type: AnimalType) -> str:
    """
    Generate an image using a fine-tuned Stable Diffusion model based on the given prompt and animal type.

    Args:
        prompt (str): The text prompt describing the image to generate.
        animal_type (AnimalType): The type of animal to use for inference. Determines which model/config to load.

    Returns:
        str: The file path where the generated image is saved.

    Raises:
        FileNotFoundError: If the config file for the given animal type does not exist.
        RuntimeError: If model weights fail to load properly.
    
    Example:
        >>> predict("A duck in a space suit on the moon", AnimalType.DUCK)
        'outputs/duck_20250406_143000.png'
    """

    config = load_config_by_animal(animal_type)

    if config.get('seed'):
        set_seed(config['seed'])

    device = torch.device(f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer, scheduler, vae, unet, text_encoder = load_model_components(config, device, weight_dtype)

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device)
    if config.get('seed'):
        generator.manual_seed(config['seed'])

    with torch.autocast(device.type, dtype=weight_dtype):
        image = pipeline(
            prompt,
            num_inference_steps=config.get('num_steps', 50),
            guidance_scale=config.get('guidance_scale', 7.5),
            generator=generator
        ).images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.get('output_dir', 'outputs/')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{animal_type.value}_{timestamp}.png")
    image.save(output_path)

    return output_path