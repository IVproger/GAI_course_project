# src/model_setup.py
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline

def load_tokenizer(name_or_path):
    print(f"Loading tokenizer: {name_or_path}")
    return CLIPTokenizer.from_pretrained(name_or_path)

def load_text_encoder(name_or_path, revision=None, variant=None, subfolder="text_encoder"):
    print(f"Loading text encoder: {name_or_path}/{subfolder}")
    return CLIPTextModel.from_pretrained(name_or_path, subfolder=subfolder, revision=revision, variant=variant)

def load_vae(name_or_path, revision=None, variant=None, subfolder="vae"):
    print(f"Loading VAE: {name_or_path}/{subfolder}")
    return AutoencoderKL.from_pretrained(name_or_path, subfolder=subfolder, revision=revision, variant=variant)

def load_unet(name_or_path, revision=None, variant=None, subfolder="unet"):
    print(f"Loading UNet: {name_or_path}/{subfolder}")
    return UNet2DConditionModel.from_pretrained(name_or_path, subfolder=subfolder, revision=revision, variant=variant)

def load_scheduler(name_or_path, num_train_timesteps, subfolder="scheduler"):
    print(f"Loading scheduler: {name_or_path}/{subfolder}")
    return DDPMScheduler.from_pretrained(name_or_path, subfolder=subfolder, num_train_timesteps=num_train_timesteps)

def create_pipeline(vae, text_encoder, tokenizer, unet, scheduler, device):
     """Creates the Stable Diffusion pipeline for inference/evaluation."""
     pipeline = StableDiffusionPipeline(
         vae=vae,
         text_encoder=text_encoder,
         tokenizer=tokenizer,
         unet=unet,
         scheduler=scheduler,
         safety_checker=None, # Disable safety checker for fine-tuning usually
         feature_extractor=None,
     )
     pipeline = pipeline.to(device)
     print("Stable Diffusion pipeline created.")
     return pipeline
 