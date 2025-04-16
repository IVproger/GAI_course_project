# src/prior_generation.py
import torch
import os
from tqdm.auto import tqdm
from math import ceil
from diffusers import StableDiffusionPipeline # Need pipeline just for generation part
from .model_setup import load_vae, load_text_encoder, load_tokenizer # Reuse loaders
from .data_handling import get_train_transforms # Reuse transforms


@torch.no_grad()
def generate_prior_preservation_data(
    model_name,
    revision,
    variant,
    device,
    weight_dtype,
    num_prior_images,
    class_prompt,
    image_resolution,
    batch_size, # Generation batch size
    num_inference_steps,
    guidance_scale,
    save_path
):
    """Generates and saves prior preservation latents and embeddings."""

    print("\n--- Starting Prior Preservation Data Generation ---")
    os.makedirs(save_path, exist_ok=True)

    # Load necessary components specifically for generation
    # Using a pipeline temporarily simplifies text encoding and image generation
    # Alternatively, manually load components and implement the generation loop
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name, revision=revision, variant=variant, torch_dtype=weight_dtype
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        print(f"Loaded generation pipeline: {model_name}")
    except Exception as e:
         print(f"Could not load full pipeline for prior generation: {e}")
         print("Attempting to load individual components...")
         # Fallback: Load components manually if pipeline fails (less convenient)
         tokenizer = load_tokenizer(model_name)
         text_encoder = load_text_encoder(model_name, revision, variant).to(device, dtype=weight_dtype)
         vae = load_vae(model_name, revision, variant).to(device, dtype=weight_dtype)
         # Need UNet too for generation loop - this makes prior generation heavy
         # Consider if just encoding prompts and saving images is better, then encode latents later
         # For now, stick with pipeline for simplicity if possible. Abort if pipeline fails.
         raise RuntimeError("Failed to load necessary components for prior generation.") from e


    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer

    # Get class prompt embeddings (once)
    text_inputs = tokenizer(
        class_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    class_embeddings = text_encoder(text_input_ids, attention_mask=attention_mask)[0].detach() # shape: (1, seq_len, embed_dim)
    # No need to expand here, will do during training load

    # Ensure VAE is in eval mode for stable encoding
    vae.eval()

    # Prepare transforms (only needed if saving images first, not latents directly)
    # We will generate images and then encode them
    # Note: Generating latents directly in the generation loop is more efficient but complex

    prior_latents_list = []
    num_batches = ceil(num_prior_images / batch_size)
    print(f"Generating {num_prior_images} prior images for prompt: '{class_prompt}' in {num_batches} batches...")

    generator = torch.Generator(device=device).manual_seed(pipeline.config.seed if hasattr(pipeline.config, 'seed') else 42) # Set generator seed

    for i in tqdm(range(num_batches), desc="Generating Prior Images"):
        num_images_in_batch = min(batch_size, num_prior_images - len(prior_latents_list) * batch_size) # Handle last batch size
        if num_images_in_batch <= 0: break

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', dtype=weight_dtype if weight_dtype != torch.float32 else None):
            images = pipeline(
                prompt=class_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_in_batch, # Generate batch size images
                generator=generator,
            ).images

        # Encode generated images to latents
        # Prepare images for VAE
        processed_images = []
        img_transforms = get_train_transforms(image_resolution) # Use same transforms as training
        for img in images:
             processed_images.append(img_transforms(img))

        image_batch = torch.stack(processed_images).to(device=device, dtype=vae.dtype) # Match VAE dtype

        # Encode to latents
        batch_latents = vae.encode(image_batch).latent_dist.sample() * vae.config.scaling_factor
        prior_latents_list.append(batch_latents.detach().cpu()) # Move to CPU to save memory


    # Concatenate all latent batches
    prior_latents_tensor = torch.cat(prior_latents_list, dim=0)

    # Save the generated data
    latent_path = os.path.join(save_path, 'prior_latents_tensor.pt')
    embedding_path = os.path.join(save_path, 'prior_embeddings.pt')
    mask_path = os.path.join(save_path, 'prior_attention_mask.pt')

    torch.save(prior_latents_tensor, latent_path)
    # Save the single class embedding and mask (they will be expanded during training)
    torch.save(class_embeddings.cpu(), embedding_path) # Save on CPU
    torch.save(attention_mask.cpu(), mask_path) # Save on CPU

    print(f"Saved prior latents to: {latent_path} (Shape: {prior_latents_tensor.shape})")
    print(f"Saved prior embeddings to: {embedding_path} (Shape: {class_embeddings.shape})")
    print(f"Saved prior attention mask to: {mask_path} (Shape: {attention_mask.shape})")
    print("--- Prior Preservation Data Generation Complete ---")
    