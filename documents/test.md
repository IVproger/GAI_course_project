**Phase 1: Project Structure Design**

Let's define a clear directory structure:

```
dreambooth-project/
├── .github/                      # (Optional) For GitHub Actions CI/CD
│   └── workflows/
│       └── python-ci.yml
├── .gitignore                    # Git ignore file
├── README.md                     # Main project README (Update this!)
├── requirements.txt              # Project dependencies
├── configs/                      # Configuration files
│   └── train_dog_dreambooth.yaml # Example config for this specific run
├── data/                         # Raw and processed data
│   ├── paper_dataset/            # Your input subject images
│   │   └── dog/
│   │       └── ... (image files)
│   └── prior_imgs/               # Generated prior preservation images/latents
│       └── dog_updated/
│           ├── prior_latents_tensor.pt
│           ├── prior_embeddings.pt
│           └── prior_attention_mask.pt
├── notebooks/                    # Your existing notebooks for experimentation
│   ├── 1. diffusion_text-img.ipynb
│   ├── 2. POC_DreamBooth.ipynb   # Your original notebook (keep for reference)
│   ├── experiments_and_raw_code/ # Keep this as is
│   └── styled_generated_image_0.png # Keep this as is
│   └── README.md                 # The README you provided for notebooks/
├── outputs/                      # Saved models, logs, generated images (if not using ClearML exclusively)
│   └── dog_dreambooth_run_XYZ/   # Example output dir for a specific run
│       ├── unet_state_dict.pt
│       ├── text_encoder_state_dict.pt
│       └── generated_images/
│           └── epoch_100_img_0.png
├── src/                          # Source code for the project
│   ├── __init__.py
│   ├── config_loader.py          # Utility to load YAML configurations
│   ├── data_handling.py          # Dataset classes, transformations, collate_fn
│   ├── model_setup.py            # Functions to load Diffusers models/components
│   ├── prior_generation.py       # Logic for generating prior preservation data
│   ├── trainer.py                # The main training loop logic (as a class or function)
│   └── utils.py                  # Utility functions (GPU selection, visualization helpers etc.)
└── scripts/                      # Executable scripts
    ├── generate_priors.py        # Script to run prior image generation
    ├── train.py                  # Main script to start a training run
    └── inference.py              # Script to load a trained model and generate images
```

**Phase 2: Code Decomposition and Refactoring**

Let's move the code from `POC_DreamBooth.ipynb` into the structure above.

1.  **Configuration (`configs/train_dog_dreambooth.yaml`)**
    *   Move all hyperparameters from the Python dictionary into a YAML file. This makes experiments easier to track and modify.
    *   Use libraries like `PyYAML` to load this config.

    ```yaml
    # configs/train_dog_dreambooth.yaml
    project_name: "DreamBooth Training"
    task_name: "dreambooth_dog"
    reuse_last_task_id: false

    # Model details
    model_name: "CompVis/stable-diffusion-v1-4"
    revision: null
    variant: null
    tokenizer_name: "openai/clip-vit-base-patch32" # Often same as model_name or derived

    # Prompts & Identifiers
    subject_tokens: "xon"         # Unique identifier token(s)
    subject_prompt: "a photo of xon dog" # Instance prompt
    class_prompt: "a photo of a dog"       # Class prompt for prior preservation

    # Data paths
    subject_image_dir: "data/paper_dataset/dog"
    prior_data_dir: "data/prior_imgs/dog_updated" # Directory containing pre-generated priors

    # Prior Preservation Settings
    use_prior_preservation: true
    generate_prior_images: false # Set to true only when running generate_priors.py
    num_prior_images: 1000        # Number of images to generate/use
    prior_loss_weight: 1.0
    prior_generation_batch_size: 64     # Batch size for generating priors
    prior_num_inference_steps: 50     # Inference steps for prior generation
    prior_guidance_scale: 7.5       # Guidance scale for prior generation

    # Training Settings
    num_train_epochs: 101
    learning_rate: 2.0e-6
    lr_scheduler_type: "constant" # e.g., constant, linear, cosine
    lr_warmup_steps: 500
    optimizer:
      type: "AdamW"
      betas: [0.9, 0.999]
      weight_decay: 0.01
      eps: 1.0e-8
    train_batch_size: 5
    prior_batch_size_train: 5 # Batch size for prior latents during training
    use_dreambooth_prompts: true # Whether to use standard DB prompt format
    ddpm_num_train_timesteps: 1000
    image_resolution: 256 # Target resolution
    mixed_precision: "fp16" # Or "bf16" or "no" for float32
    gradient_accumulation_steps: 1 # Accumulate grads for larger effective batch size

    # Logging & Saving
    save_model_epochs: 100 # Frequency to save checkpoints
    save_model_steps: null # Alternative: save every N steps
    output_dir: "outputs" # Base directory for local saving
    log_image_epochs: 100 # Frequency to log sample images

    # Misc
    seed: 42 # For reproducibility
    ```

2.  **Config Loader (`src/config_loader.py`)**
    *   A simple function to load the YAML file.

    ```python
    # src/config_loader.py
    import yaml
    from types import SimpleNamespace
    import os

    def load_config(config_path):
        """Loads YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Optionally convert to SimpleNamespace for attribute access (config.model_name)
        # Or just return the dictionary
        return config_dict # Or SimpleNamespace(**config_dict)
    ```

3.  **Utilities (`src/utils.py`)**
    *   Keep `get_free_gpu`.
    *   `visualize_image` might be more suitable for notebooks, but you can keep it here if used elsewhere.
    *   Add helper functions if needed (e.g., setting random seeds).

    ```python
    # src/utils.py
    import torch
    import GPUtil
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    def get_free_gpu(min_memory=1000):
        """Finds the GPU with the most free memory."""
        try:
            gpus = GPUtil.getGPUs()
            best_gpu = -1
            max_mem = -1
            for gpu in gpus:
                if gpu.memoryFree > max_mem and gpu.memoryFree >= min_memory:
                    max_mem = gpu.memoryFree
                    best_gpu = gpu.id
            if best_gpu == -1:
                print("Warning: No suitable GPU found or GPUtil not available. Falling back.")
                return 0 # Default to GPU 0 if none suitable or GPUtil fails
            print(f"Selected GPU {best_gpu} with {max_mem}MB free memory.")
            return best_gpu
        except Exception as e:
            print(f"GPU selection failed: {e}. Defaulting to GPU 0 or CPU.")
            return 0

    def set_seed(seed: int):
        """Sets random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def visualize_image(dataset, index=0):
        """Visualizes an image and its prompt from the dataset."""
        item = dataset[index]
        img_tensor = item['pixel_values']
        prompt = item['text']

        # De-normalize and convert tensor to PIL image for display
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_tensor = inv_normalize(img_tensor)
        img = transforms.ToPILImage()(img_tensor).convert("RGB")

        plt.imshow(img)
        plt.title(f"Prompt: {prompt}")
        plt.axis("off")
        plt.show()

    # Add other utility functions as needed...
    ```

4.  **Data Handling (`src/data_handling.py`)**
    *   Move the `PriorClassDataset` class here.
    *   Move `collate_fn` here.
    *   Define image transformations here.

    ```python
    # src/data_handling.py
    import os
    import glob
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from torchvision import transforms
    import torch

    class PriorClassDataset(Dataset):
        """
        Dataset for DreamBooth training.
        It prepares pairs of (image, prompt) for the subject class.
        """
        def __init__(self, image_dir, subject_prompt, class_prompt, tokenizer, transform, use_dreambooth_prompts=True):
            self.image_dir = image_dir
            self.tokenizer = tokenizer
            self.transform = transform
            self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                               glob.glob(os.path.join(image_dir, "*.png")) + \
                               glob.glob(os.path.join(image_dir, "*.jpeg"))

            self.subject_prompt = subject_prompt
            self.class_prompt = class_prompt # Needed if not using DB prompts
            self.use_dreambooth_prompts = use_dreambooth_prompts

            # Standard DreamBooth prompts (modify if needed)
            self.prompts = [
                f"a photo of {subject_prompt}",
                f"a rendering of {subject_prompt}",
                f"a cropped photo of the {subject_prompt}",
                f"the photo of a {subject_prompt}",
                f"a photo of a {subject_prompt}",
                f"a close-up photo of {subject_prompt}",
                f"a bright photo of the {subject_prompt}",
                f"a photo of my {subject_prompt}",
                # Add more variations as needed
            ]
            print(f"Found {len(self.image_paths)} images in {image_dir}")
            print(f"Using subject prompt base: '{subject_prompt}'")


        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load image {img_path}. Skipping or using placeholder. Error: {e}")
                # Handle error: return a placeholder or skip (might require changes in collate_fn)
                # For now, let's raise error to be explicit
                raise IOError(f"Failed to load image: {img_path}") from e

            if self.transform:
                image = self.transform(image)

            if self.use_dreambooth_prompts:
                # Cycle through predefined prompts for variety
                text = self.prompts[idx % len(self.prompts)]
            else:
                text = self.subject_prompt # Use the single subject prompt

            # Tokenize prompt
            # Max length should probably be a config parameter
            inputs = self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            input_ids = inputs.input_ids[0] # Remove batch dim
            attention_mask = inputs.attention_mask[0] # Remove batch dim

            return {"pixel_values": image, "input_ids": input_ids, "attention_mask": attention_mask, "text": text}


    def get_train_transforms(resolution):
        """Returns standard image transformations for training."""
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.CenterCrop(resolution), # Added for safety
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]), # Normalize to [-1, 1] common for diffusers
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Original Imagenet norm
        ])

    def collate_fn(examples):
        """Collates batches of data correctly."""
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])

        # Include text for potential debugging or reference, though not used in training loop directly
        texts = [example["text"] for example in examples]

        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "texts": texts}

    def load_prior_data(prior_data_dir, batch_size, device):
        """Loads pre-generated prior latents, embeddings, and mask."""
        try:
            prior_latents_tensor = torch.load(os.path.join(prior_data_dir, 'prior_latents_tensor.pt'), map_location='cpu')
            prior_embeddings = torch.load(os.path.join(prior_data_dir, 'prior_embeddings.pt'), map_location='cpu')
            prior_attention_mask = torch.load(os.path.join(prior_data_dir, 'prior_attention_mask.pt'), map_location='cpu')
            print(f"Loaded prior data from {prior_data_dir}:")
            print(f"  Latents shape: {prior_latents_tensor.shape}")
            print(f"  Embeddings shape: {prior_embeddings.shape}")
            print(f"  Attention Mask shape: {prior_attention_mask.shape}")

        except FileNotFoundError as e:
            print(f"Error: Prior data files not found in {prior_data_dir}.")
            print("Please run the 'generate_priors.py' script first.")
            raise e

        prior_dataset = TensorDataset(prior_latents_tensor)
        prior_dataloader = DataLoader(prior_dataset, batch_size=batch_size, shuffle=True)

        # Move embeddings and mask to the target device once
        prior_embeddings = prior_embeddings.to(device)
        prior_attention_mask = prior_attention_mask.to(device)

        return prior_dataloader, prior_embeddings, prior_attention_mask

    ```

5.  **Model Setup (`src/model_setup.py`)**
    *   Functions to load the tokenizer, text encoder, VAE, UNet, and scheduler.

    ```python
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
    ```

6.  **Prior Generation Logic (`src/prior_generation.py`)**
    *   Encapsulate the logic for generating and saving prior latents/embeddings.
    *   This should be callable from a separate script.

    ```python
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
    ```

7.  **Trainer Logic (`src/trainer.py`)**
    *   Create a class or function to handle the training loop, optimizer setup, loss calculation, backpropagation, logging, and periodic evaluation/saving.
    *   This makes `train.py` cleaner.

    ```python
    # src/trainer.py
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    from diffusers.optimization import get_scheduler
    from accelerate import Accelerator # Use Accelerate for easier device placement, mixed precision, etc.
    from clearml import Logger
    import os
    from .model_setup import create_pipeline # For evaluation pipeline
    from PIL import Image

    class DreamBoothTrainer:
        def __init__(self, config, accelerator, logger):
            self.config = config
            self.accelerator = accelerator
            self.logger = logger

        def train(self, unet, vae, text_encoder, tokenizer, noise_scheduler, train_dataloader, prior_data):
            """Main training loop."""

            cfg = self.config # Shorthand

            # Prepare optimizer
            optimizer_cfg = cfg['optimizer']
            optimizer = torch.optim.AdamW(
                # Only train UNet and text_encoder parameters
                list(unet.parameters()) + list(text_encoder.parameters()),
                lr=cfg['learning_rate'],
                betas=tuple(optimizer_cfg['betas']),
                weight_decay=optimizer_cfg['weight_decay'],
                eps=optimizer_cfg['eps'],
            )

            # Prepare LR Scheduler
            num_update_steps_per_epoch = len(train_dataloader) // cfg['gradient_accumulation_steps']
            max_train_steps = cfg['num_train_epochs'] * num_update_steps_per_epoch

            lr_scheduler = get_scheduler(
                cfg['lr_scheduler_type'],
                optimizer=optimizer,
                num_warmup_steps=cfg['lr_warmup_steps'] * cfg['gradient_accumulation_steps'],
                num_training_steps=max_train_steps,
            )

            # Prepare models, optimizer, dataloader, scheduler with Accelerate
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
            # VAE should be in eval mode and doesn't need gradients, move manually
            vae.to(self.accelerator.device, dtype=torch.float16 if cfg['mixed_precision'] == 'fp16' else torch.bfloat16 if cfg['mixed_precision'] == 'bf16' else torch.float32)
            vae.eval()


            # Handle prior data loading if needed
            prior_dataloader, prior_embeddings, prior_attention_mask = None, None, None
            if cfg['use_prior_preservation']:
                prior_dataloader, prior_embeddings, prior_attention_mask = prior_data
                # Prepare prior dataloader if not already done (Accelerate usually handles this)
                prior_dataloader = self.accelerator.prepare(prior_dataloader)
                # Move embeddings/mask to device (already done in load_prior_data, but double check)
                prior_embeddings = prior_embeddings.to(self.accelerator.device)
                prior_attention_mask = prior_attention_mask.to(self.accelerator.device)


            weight_dtype = torch.float32
            if cfg['mixed_precision'] == "fp16":
                 weight_dtype = torch.float16
            elif cfg['mixed_precision'] == "bf16":
                 weight_dtype = torch.bfloat16


            print("\n--- Starting Training ---")
            print(f"  Num Epochs: {cfg['num_train_epochs']}")
            print(f"  Train Batch Size (per device): {cfg['train_batch_size']}")
            print(f"  Prior Batch Size (per device): {cfg['prior_batch_size_train']}")
            print(f"  Gradient Accumulation Steps: {cfg['gradient_accumulation_steps']}")
            print(f"  Total optimization steps: {max_train_steps}")
            print(f"  Using prior preservation: {cfg['use_prior_preservation']}")
            print(f"  Mixed Precision: {cfg['mixed_precision']}")


            global_step = 0
            for epoch in range(cfg['num_train_epochs']):
                unet.train()
                text_encoder.train()
                epoch_losses = []

                # Setup prior iterator if using prior preservation
                prior_iter = iter(prior_dataloader) if prior_dataloader else None

                progress_bar = tqdm(
                    range(num_update_steps_per_epoch),
                    disable=not self.accelerator.is_local_main_process,
                    desc=f"Epoch {epoch+1}/{cfg['num_train_epochs']}"
                )

                for step, batch in enumerate(train_dataloader):
                    with self.accelerator.accumulate(unet), self.accelerator.accumulate(text_encoder): # Accumulate gradients
                        # --- Subject Loss ---
                        pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                        input_ids = batch["input_ids"]
                        attention_mask = batch["attention_mask"]

                        # Convert images to latent space
                        with torch.no_grad():
                             latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                        # Sample noise and timesteps
                        noise = torch.randn_like(latents)
                        batch_size = latents.shape[0]
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device).long()
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get text embeddings
                        # Need to unwrap the model for direct call if using Accelerate and DDP
                        encoder_hidden_states = self.accelerator.unwrap_model(text_encoder)(input_ids=input_ids, attention_mask=attention_mask)[0]


                        # Predict noise residual (subject)
                        model_pred_subject = self.accelerator.unwrap_model(unet)(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                        # Calculate subject loss
                        loss_subject = F.mse_loss(model_pred_subject.float(), noise.float(), reduction="mean")


                        # --- Prior Preservation Loss ---
                        loss_prior = torch.tensor(0.0, device=self.accelerator.device, dtype=loss_subject.dtype)
                        if cfg['use_prior_preservation']:
                           try:
                               prior_batch_latents = next(prior_iter)[0] # Shape: (prior_bs, C, H, W)
                           except StopIteration:
                                prior_iter = iter(prior_dataloader) # Reset iterator
                                prior_batch_latents = next(prior_iter)[0]

                           # Ensure batch size matches (last batch might be smaller)
                           num_prior_samples = prior_batch_latents.shape[0]

                           # Sample noise and timesteps for prior latents
                           noise_prior = torch.randn_like(prior_batch_latents)
                           t_prior = torch.randint(0, noise_scheduler.config.num_train_timesteps, (num_prior_samples,), device=prior_batch_latents.device).long()
                           noisy_prior_latents = noise_scheduler.add_noise(prior_batch_latents, noise_prior, t_prior)


                           # Expand prior embeddings/mask to match the prior batch size
                           prior_embeddings_expanded = prior_embeddings.expand(num_prior_samples, -1, -1)
                           prior_attention_mask_expanded = prior_attention_mask.expand(num_prior_samples, -1)

                           # Predict noise residual (prior)
                           model_pred_prior = self.accelerator.unwrap_model(unet)(noisy_prior_latents, t_prior, encoder_hidden_states=prior_embeddings_expanded).sample


                           # Calculate prior loss
                           loss_prior = F.mse_loss(model_pred_prior.float(), noise_prior.float(), reduction="mean")


                        # --- Total Loss ---
                        total_loss = loss_subject + cfg['prior_loss_weight'] * loss_prior

                        # Backpropagate
                        self.accelerator.backward(total_loss)

                        # Gradient clipping (optional but recommended)
                        if self.accelerator.sync_gradients:
                             self.accelerator.clip_grad_norm_(list(unet.parameters()) + list(text_encoder.parameters()), 1.0) # Max grad norm = 1.0


                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    # Checks should be done on the main process only after gradient synchronization
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        avg_loss = self.accelerator.gather(total_loss.repeat(cfg['train_batch_size'])).mean().item()
                        epoch_losses.append(avg_loss)
                        progress_bar.set_postfix(Loss=avg_loss, LR=lr_scheduler.get_last_lr()[0])

                        # Log metrics (main process only)
                        if self.accelerator.is_main_process and self.logger:
                            self.logger.report_scalar(title="Loss/Train", series="Total Loss", value=avg_loss, iteration=global_step)
                            self.logger.report_scalar(title="Loss/Train", series="Subject Loss", value=loss_subject.item(), iteration=global_step)
                            if cfg['use_prior_preservation']:
                                self.logger.report_scalar(title="Loss/Train", series="Prior Loss", value=loss_prior.item(), iteration=global_step)
                            self.logger.report_scalar(title="Learning Rate", series="LR", value=lr_scheduler.get_last_lr()[0], iteration=global_step)

                        # Periodic saving (main process only)
                        if self.accelerator.is_main_process:
                            if cfg.get('save_model_steps') and global_step % cfg['save_model_steps'] == 0:
                                self.save_checkpoint(unet, text_encoder, cfg, global_step)
                            # Periodic image generation/logging
                            if cfg.get('log_image_epochs') and global_step % (num_update_steps_per_epoch * cfg['log_image_epochs']) == 0:
                                self.log_sample_images(unet, vae, text_encoder, tokenizer, noise_scheduler, cfg, epoch)


            # End of training
            if self.accelerator.is_main_process:
                print("\n--- Training Complete ---")
                # Final save
                self.save_checkpoint(unet, text_encoder, cfg, global_step, final=True)
                # Final image generation
                self.log_sample_images(unet, vae, text_encoder, tokenizer, noise_scheduler, cfg, epoch, final=True)


        def save_checkpoint(self, unet, text_encoder, config, step_or_epoch, final=False):
            """Saves the UNet and text_encoder state dicts."""
            unet_out_path = os.path.join(config['output_dir'], config['task_name'], f"unet_{'final' if final else step_or_epoch}.pt")
            text_encoder_out_path = os.path.join(config['output_dir'], config['task_name'], f"text_encoder_{'final' if final else step_or_epoch}.pt")
            os.makedirs(os.path.dirname(unet_out_path), exist_ok=True)

            # Unwrap models before saving state_dict
            unet_state_dict = self.accelerator.unwrap_model(unet).state_dict()
            text_encoder_state_dict = self.accelerator.unwrap_model(text_encoder).state_dict()

            self.accelerator.save(unet_state_dict, unet_out_path)
            self.accelerator.save(text_encoder_state_dict, text_encoder_out_path)
            print(f"Saved checkpoint at step/epoch {step_or_epoch} to {os.path.dirname(unet_out_path)}")

            # Optionally, upload to ClearML as artifacts
            if self.logger:
                self.logger.upload_artifact(name=f"unet_state_dict_{'final' if final else step_or_epoch}", artifact_object=unet_out_path)
                self.logger.upload_artifact(name=f"text_encoder_state_dict_{'final' if final else step_or_epoch}", artifact_object=text_encoder_out_path)


        @torch.no_grad()
        def log_sample_images(self, unet, vae, text_encoder, tokenizer, noise_scheduler, config, epoch, final=False):
            """Generates and logs sample images."""
            print(f"\nGenerating sample images for epoch {epoch+1}{' (Final)' if final else ''}...")
            # Create evaluation pipeline using the trained components
            # Important: Use unwrapped models for pipeline creation
            unet_eval = self.accelerator.unwrap_model(unet)
            text_encoder_eval = self.accelerator.unwrap_model(text_encoder)

            eval_pipeline = create_pipeline(vae, text_encoder_eval, tokenizer, unet_eval, noise_scheduler, self.accelerator.device)
            eval_pipeline.set_progress_bar_config(disable=True)

            generator = torch.Generator(device=self.accelerator.device).manual_seed(config.get('seed', 42) + epoch) # Use different seed per epoch

            # Use the core subject prompt for evaluation
            prompt = config['subject_prompt']
            print(f"  Using prompt: '{prompt}'")

            # Generate images
            with torch.autocast(self.accelerator.device.type, dtype=(torch.float16 if config['mixed_precision'] == 'fp16' else torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float32)):
                 images = eval_pipeline(
                     prompt,
                     num_inference_steps=100, # Use a reasonable number of steps for eval
                     guidance_scale=7.5,       # Standard guidance scale
                     num_images_per_prompt=4, # Generate a few samples
                     generator=generator
                 ).images


            # Log images to ClearML (and optionally save locally)
            series_name = f"Epoch_{epoch+1}" if not final else "Final_Output"
            output_image_dir = os.path.join(config['output_dir'], config['task_name'], "generated_images")
            os.makedirs(output_image_dir, exist_ok=True)

            for idx, img in enumerate(images):
                 if self.logger:
                     self.logger.report_image(
                         title="Sample Outputs",
                         series=series_name,
                         iteration=epoch+1 if not final else config['num_train_epochs'],
                         image=img
                     )
                 # Save locally
                 img_path = os.path.join(output_image_dir, f"{series_name}_img_{idx}.png")
                 img.save(img_path)

            print(f"Logged/Saved {len(images)} sample images.")
            del eval_pipeline # Free up memory
            torch.cuda.empty_cache()

    ```
    *Code Improvements in Trainer:*
        *   Introduced **Hugging Face Accelerate**: Simplifies handling device placement (CPU/GPU/multi-GPU), mixed precision (`fp16`/`bf16`), and gradient accumulation across multiple devices. This makes the code cleaner and more portable.
        *   Clear Separation: The trainer focuses solely on the training loop mechanics.
        *   Checkpointing: Added basic model saving logic.
        *   Evaluation: Integrated periodic sample image generation and logging.

8.  **Scripts (`scripts/generate_priors.py`, `scripts/train.py`, `scripts/inference.py`)**

    *   **`scripts/generate_priors.py`**:
        *   Parses arguments (e.g., config path).
        *   Loads config using `src.config_loader`.
        *   Sets up device and dtype based on config.
        *   Calls `src.prior_generation.generate_prior_preservation_data`.

    ```python
    # scripts/generate_priors.py
    import argparse
    import torch
    from accelerate.utils import set_seed

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
    ```

    *   **`scripts/train.py`**:
        *   Parses arguments (config path).
        *   Loads config.
        *   Initializes ClearML.
        *   Initializes `Accelerator`.
        *   Sets seed.
        *   Loads models, tokenizer, scheduler using `src.model_setup`.
        *   Loads subject dataset using `src.data_handling`.
        *   Loads *pre-generated* prior data using `src.data_handling`.
        *   Instantiates `DreamBoothTrainer` from `src.trainer`.
        *   Calls the trainer's `train` method.
        *   Handles final ClearML task closure.

    ```python
    # scripts/train.py
    import argparse
    import torch
    import os
    from clearml import Task, Logger
    from accelerate import Accelerator
    from accelerate.utils import set_seed, ProjectConfiguration

    # Adjust imports if running from project root
    from src.config_loader import load_config
    from src.utils import set_seed # Use Accelerate's set_seed primarily
    from src.data_handling import PriorClassDataset, collate_fn, load_prior_data, get_train_transforms
    from src.model_setup import load_tokenizer, load_text_encoder, load_vae, load_unet, load_scheduler
    from src.trainer import DreamBoothTrainer


    def parse_args():
        parser = argparse.ArgumentParser(description="Train DreamBooth Model")
        parser.add_argument("--config", type=str, required=True, help="Path to the training configuration YAML file")
        return parser.parse_args()

    def main():
        args = parse_args()
        config = load_config(args.config)

        # --- Accelerator and Logging Setup ---
        # Define where Accelerate stores its cache/config if needed
        project_config = ProjectConfiguration(project_dir=config.get("output_dir", "outputs"), logging_dir=os.path.join(config.get("output_dir", "outputs"), "logs"))
        accelerator = Accelerator(
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            mixed_precision=config.get('mixed_precision', 'no'),
            log_with="clearml", # Integrate Accelerate with ClearML
            project_config=project_config,
        )

        # Initialize ClearML Task (only on main process)
        if accelerator.is_main_process:
            task = Task.init(
                project_name=config['project_name'],
                task_name=config['task_name'],
                reuse_last_task_id=config.get('reuse_last_task_id', False),
                output_uri=True # Automatically upload artifacts saved to output_dir
            )
            # Connect config dictionary
            task.connect(config)
            logger = task.get_logger() # Get ClearML logger instance
            print("ClearML Task initialized.")
        else:
            logger = None # No logger on non-main processes
            # Ensure non-main processes don't init ClearML or log excessively
            import logging
            logging.basicConfig(level=logging.WARNING) # Suppress info logs from libraries


        # Make one log on all processes to ensure synchronization
        accelerator.wait_for_everyone()
        accelerator.print(f"Accelerator setup complete. Device: {accelerator.device}")


        # --- Reproducibility ---
        set_seed(config.get('seed', 42))
        accelerator.print(f"Set random seed to {config.get('seed', 42)}")

        # --- Load Components ---
        accelerator.print("Loading model components...")
        tokenizer = load_tokenizer(config.get('tokenizer_name', config['model_name']))
        noise_scheduler = load_scheduler(config['model_name'], config['ddpm_num_train_timesteps'])
        # Load models onto CPU first, Accelerate will handle placement
        text_encoder = load_text_encoder(config['model_name'], config.get('revision'), config.get('variant'))
        vae = load_vae(config['model_name'], config.get('revision'), config.get('variant'))
        unet = load_unet(config['model_name'], config.get('revision'), config.get('variant'))
        accelerator.print("Model components loaded.")

        # --- Prepare Data ---
        accelerator.print("Loading datasets...")
        train_transforms = get_train_transforms(config['image_resolution'])
        subject_dataset = PriorClassDataset(
            image_dir=config['subject_image_dir'],
            subject_prompt=config['subject_prompt'],
            class_prompt=config['class_prompt'], # Needed for reference even if DB prompts used
            tokenizer=tokenizer,
            transform=train_transforms,
            use_dreambooth_prompts=config.get('use_dreambooth_prompts', True)
        )
        # Check dataset length
        if len(subject_dataset) == 0:
             raise ValueError(f"No images found in {config['subject_image_dir']}. Please check the path and image formats.")

        train_dataloader = torch.utils.data.DataLoader(
            subject_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=config['train_batch_size'],
            num_workers=config.get('dataloader_num_workers', 4) # Add num_workers config
        )

        prior_data = None
        if config.get('use_prior_preservation', False):
            accelerator.print("Loading prior preservation data...")
            try:
                # Load prior data and move embeddings/mask to the accelerator device later in trainer
                prior_dataloader, prior_embeddings, prior_attention_mask = load_prior_data(
                    config['prior_data_dir'], config['prior_batch_size_train'], accelerator.device
                )
                prior_data = (prior_dataloader, prior_embeddings, prior_attention_mask)
                accelerator.print("Prior data loaded.")
            except Exception as e:
                accelerator.print(f"Error loading prior data: {e}")
                accelerator.print("Ensure prior data exists and 'prior_data_dir' is correct in config.")
                # Decide whether to proceed without prior preservation or raise error
                if accelerator.is_main_process:
                     logger.report_text("Warning: Failed to load prior data. Proceeding without prior preservation.")
                config['use_prior_preservation'] = False # Disable it if loading failed
                prior_data = None #(None, None, None) - safer to just set prior_data to None
        else:
             accelerator.print("Prior preservation is disabled.")


        # --- Initialize Trainer ---
        trainer = DreamBoothTrainer(
            config=config,
            accelerator=accelerator,
            logger=logger  # Pass the ClearML logger instance
        )

        # --- Start Training ---
        try:
            trainer.train(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                train_dataloader=train_dataloader,
                prior_data=prior_data,
            )
        except Exception as e:
             accelerator.print(f"\n--- Training Failed ---")
             accelerator.print(f"Error: {e}")
             import traceback
             accelerator.print(traceback.format_exc())
             # Ensure task is closed even on failure
             if accelerator.is_main_process and task:
                 task.close()
             raise # Re-raise the exception


        # --- Finalize ---
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and task:
            accelerator.print("Closing ClearML Task.")
            task.close()

        accelerator.print("\n--- Script Finished ---")


    if __name__ == "__main__":
        main()

    ```

    *   **`scripts/inference.py`**: (Optional but Recommended)
        *   Loads a *trained* model (UNet/Text Encoder state dicts saved during training).
        *   Loads the base VAE, tokenizer, scheduler.
        *   Creates a `StableDiffusionPipeline`.
        *   Takes prompts as input and generates images.

    ```python
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
    ```

9.  **Dependencies (`requirements.txt`)**
    *   List all necessary libraries. Use `pip freeze > requirements.txt` in your environment *after* installing everything, then clean it up to include only direct dependencies.

    ```txt
    # requirements.txt
    torch
    torchvision
    torchaudio
    diffusers>=0.20.0 # Check for specific version compatibility if needed
    transformers>=4.25.0
    accelerate>=0.20.0
    xformers # Optional: for memory efficient attention
    clearml
    PyYAML
    tqdm
    Pillow
    matplotlib
    GPUtil # For GPU selection utility
    # Add specific versions if necessary (e.g., torch==2.0.1)
    ```

10. **`.gitignore`**
    *   Add standard Python ignores, plus data, outputs, environment folders etc.

    ```
    # .gitignore
    __pycache__/
    *.py[cod]
    *$py.class

    # Environments
    .env
    .venv
    env/
    venv/
    ENV/
    env.bak/
    venv.bak/

    # Data and Outputs (if large or generated, consider Git LFS or not committing)
    # data/prior_imgs/ # Maybe ignore generated data if large
    # outputs/
    # models/

    # IDE specific
    .vscode/
    .idea/
    *.swp
    *.swo

    # OS specific
    .DS_Store
    Thumbs.db

    # Logs
    *.log
    logs/

    # ClearML cache (optional)
    .clearml_cache/
    ```

**Phase 3: Running the Code**

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

1.  **Generate Priors (if needed):**
    ```bash
    python scripts.generate_priors --config configs/train_dog_dreambooth.yaml
    ```
2.  **Train:**
    ```bash
    # Single GPU/CPU
    accelerate launch scripts/train.py --config configs/train_dog_dreambooth.yaml

    # Multi-GPU (Configure accelerate first with `accelerate config`)
    # accelerate launch scripts/train.py --config configs/train_dog_dreambooth.yaml
    ```
3.  **Inference:**
    ```bash
    python scripts/inference.py \
        --model_base "CompVis/stable-diffusion-v1-4" \
        --unet_path "outputs/dog_dreambooth_run_XYZ/unet_final.pt" \
        --text_encoder_path "outputs/dog_dreambooth_run_XYZ/text_encoder_final.pt" \
        --prompt "a photo of xon dog swimming in a pool" \
        --output_path "xon_dog_swimming.png"
    ```

**Benefits of this Structure:**

1.  **Modularity:** Each component (data, model, training) is separate and can be modified independently.
2.  **Reusability:** Utility functions, data loaders, and model setup logic can be reused across different scripts (training, inference, evaluation).
3.  **Configuration Management:** Using YAML files makes managing hyperparameters and experiments much easier than hardcoding.
4.  **Testability:** Smaller, focused functions/classes are easier to unit test.
5.  **Collaboration:** Clear structure makes it easier for others (and your future self) to understand and contribute.
6.  **Scalability:** Using `accelerate` makes it trivial to scale from single GPU to multi-GPU or TPUs.
7.  **Maintainability:** Easier to debug and update the codebase.
8.  **Professionalism:** Aligns with standard practices for ML project development.

This refactoring takes your working notebook code and elevates it into a robust, professional project structure. Remember to update your main `README.md` to explain this new structure and how to run the scripts.