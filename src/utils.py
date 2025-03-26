import random
import matplotlib.pyplot as plt
import torch
import subprocess
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
import os
from pathlib import Path

def create_rare_token(tokenizer, num_tokens=1, token_range=(5000, 10000), max_length=3, seed=None):
    """
    Generates a rare identifier by sampling tokens from the tokenizer's vocabulary.
    
    Args:
        tokenizer: Hugging Face tokenizer (e.g. T5Tokenizer).
        num_tokens: Number of tokens to sample (concatenated in order).
        token_range: Tuple indicating the range of token IDs to consider.
        max_length: Maximum length (in characters) for each decoded token.
        seed: Seed for the random number generator.
    
    Returns:
        A string identifier that is the concatenation of sampled rare tokens.
    """
    if seed is not None:
        random.seed(seed)
    
    candidate_tokens = []
    
    # Iterate through the specified token ID range.
    for token_id in range(token_range[0], token_range[1]):
        token_str = tokenizer.decode([token_id]).strip()
        # Check that the token is non-empty, doesn't contain spaces, and is short.
        if token_str and (" " not in token_str) and (len(token_str) <= max_length):
            candidate_tokens.append(token_str)
    
    if not candidate_tokens:
        raise ValueError("No suitable rare tokens found in the given range.")
    
    # Randomly sample the requested number of tokens and concatenate them.
    selected_tokens = random.choices(candidate_tokens, k=num_tokens)
    rare_identifier = " ".join(selected_tokens)
    return rare_identifier

def visualize_image(dataset, idx):
    """
    Visualizes an image from the dataset.

    Args:
        dataset: The dataset containing the image.
        idx: The index of the image in the dataset.
    """
    try:
        image, _, _ = dataset[idx]
    except (IndexError, TypeError) as e:
        raise ValueError(f"Invalid dataset or index: {e}") from e
    
    image = image.permute(1, 2, 0)  # Change the order of dimensions for visualization
    image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Unnormalize the image
    image = image.numpy()
    
    plt.imshow(image)
    plt.title(f"Prompt: {dataset.subject_prompt if dataset.use_dreambooth_prompts else dataset.class_prompt}")
    plt.axis('off')
    plt.show()

def collate_fn(examples):
    """
    Collates a list of examples into a batch.

    Args:
        examples: A list of examples, where each example is a tuple containing pixel values, input IDs, and attention mask.

    Returns:
        A dictionary containing the batched data: pixel values, input IDs, and attention mask.
    """
    pixel_values = torch.stack([example[0] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example[1] for example in examples])
    attention_mask = torch.stack([example[2] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

def get_free_gpu():
    """
    Identifies the GPU with the most free memory.

    Returns:
        The index of the GPU with the most free memory.
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']
    )
    free_memory = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return free_memory.index(max(free_memory))

def generate_prior_preservation_samples(
    pretrained_model_name_or_path,
    revision,
    variant,
    device,
    weight_dtype,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    num_prior_images,
    class_prompt,
    train_transforms,
    batch_size=64,
    num_inference_steps=50,
    guidance_scale=7.5,
    save_path=None
):
    """
    Generate prior preservation samples for Stable Diffusion fine-tuning.
    
    Args:
        pretrained_model_name_or_path: Path to pretrained model
        revision: Model revision
        variant: Model variant
        device: Device to run on
        weight_dtype: Data type for model weights
        vae: VAE model
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        noise_scheduler: Noise scheduler
        num_prior_images: Number of prior images to generate
        class_prompt: Prompt for the class
        train_transforms: Image transformations
        batch_size: Batch size for generation
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        save_path: Directory path to save prior images and tensors (optional)
    
    Returns:
        tuple: (prior_latents_tensor, prior_embeddings, prior_attention_mask, generated_images)
    """
    # Create a frozen copy of the original UNet for generating prior samples
    unet_pretrained = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=revision, 
        variant=variant
    )
    unet_pretrained.to(device, dtype=weight_dtype)
    unet_pretrained.eval()
    
    # Build a separate pipeline using the frozen UNet
    pipeline_pretrained = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet_pretrained,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)
    
    # Pre-generate prior samples
    prior_latents_list = []
    print("Generating prior preservation latents...")
    total_generated = 0
    
    while total_generated < num_prior_images:
        current_batch_size = min(batch_size, num_prior_images - total_generated)
        print(f"Generating images {total_generated} to {total_generated + current_batch_size - 1}")
        with torch.no_grad(), torch.autocast("cuda"):
            output = pipeline_pretrained(
                [class_prompt] * current_batch_size, 
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale
            )
        
        for j in range(current_batch_size):
            gen_image = output.images[j].convert("RGB")      
                  
            gen_image_tensor = train_transforms(gen_image).unsqueeze(0).to(device, dtype=weight_dtype)
            with torch.no_grad():
                latent = vae.encode(gen_image_tensor).latent_dist.sample() * vae.config.scaling_factor
            prior_latents_list.append(latent)
        
        total_generated += current_batch_size
    
    # Prepare prior prompt embeddings
    prior_inputs = tokenizer(
        class_prompt, 
        return_tensors="pt", 
        max_length=tokenizer.model_max_length,
        padding="max_length", 
        truncation=True
    )
    prior_input_ids = prior_inputs["input_ids"].to(device)
    prior_attention_mask = prior_inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        prior_embeddings = text_encoder(
            input_ids=prior_input_ids, 
            attention_mask=prior_attention_mask
        ).last_hidden_state
    
    # Concatenate all latents into a single tensor
    prior_latents_tensor = torch.cat(prior_latents_list, dim=0)
    
    # Save tensors if save path is provided
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        torch.save(prior_latents_tensor, os.path.join(save_path, 'prior_latents_tensor.pt'))
        torch.save(prior_embeddings, os.path.join(save_path, 'prior_embeddings.pt'))
        torch.save(prior_attention_mask, os.path.join(save_path, 'prior_attention_mask.pt'))
    
    return prior_latents_tensor, prior_embeddings, prior_attention_mask

def generate_and_save_images(
    pipeline,
    prompt,
    num_inference_steps=100,
    guidance_scale=7.5,
    device=None,
    output_dir=".",
    prefix="target_generated_image",
    display=True, 
    save=False,
):
    """
    Generate images using a diffusion pipeline, display them, and save them to disk.
    
    Args:
        pipeline: Diffusion pipeline
        prompt: Text prompt for image generation
        num_inference_steps: Number of inference steps for the diffusion process
        guidance_scale: Guidance scale for classifier-free guidance
        device: Device for automatic mixed precision (if None, uses pipeline's device)
        output_dir: Directory to save the generated images
        prefix: Prefix for the saved image filenames
        display: Whether to display the images using matplotlib
        
    Returns:
        list: Generated PIL images
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    # Generate images
    with torch.autocast(device.type):
        output = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    
    images = output.images
    
    # Display and save images
    for idx, img in enumerate(images):
        if display:
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Generated image {idx+1}")
            plt.show()
        
        if save:
            save_path = os.path.join(output_dir, f"{prefix}_{idx}.png")
            img.save(save_path)
            print(f"Saved image to {save_path}")
    
    return images