import argparse
import torch
from pathlib import Path
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader

# Define hyperparameters
hyperparameters = {
    "pretrained_model_name": "CompVis/stable-diffusion-v1-4",
    "instance_prompt": "a photo of sks dog",  # Use a unique token
    "class_prompt": "a photo of a dog",       # Class-level prompt
    "instance_data_dir": "path/to/instance/images",
    "output_dir": "dreambooth-model",
    "resolution": 512,
    "train_batch_size": 4,
    "learning_rate": 2e-6,
    "max_train_steps": 1000,
    "num_train_epochs": 100,
    "gradient_accumulation_steps": 1,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 500,
    "with_prior_preservation": True,
    "prior_loss_weight": 1.0,
    "num_class_images": 100,
    "class_data_dir": "path/to/class/images",
    "train_text_encoder": True,
    "seed": 42,
}

###### Model Loading
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
tokenizer = CLIPTokenizer.from_pretrained(hyperparameters["pretrained_model_name"], subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(hyperparameters["pretrained_model_name"], subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(hyperparameters["pretrained_model_name"], subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(hyperparameters["pretrained_model_name"], subfolder="unet")
noise_scheduler = DDPMScheduler.from_pretrained(hyperparameters["pretrained_model_name"], subfolder="scheduler")

# Move models to device
vae.to(device)
text_encoder.to(device)
unet.to(device)

# Freeze VAE and optionally text encoder
vae.requires_grad_(False)
if not hyperparameters["train_text_encoder"]:
    text_encoder.requires_grad_(False)
    
#### Dataset Implementation
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exist.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        
        # Tokenize prompt
        text_inputs = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            
            # Tokenize class prompt
            class_text_inputs = self.tokenizer(
                self.class_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            example["class_prompt_ids"] = class_text_inputs.input_ids
            example["class_attention_mask"] = class_text_inputs.attention_mask

        return example

##### Data Collation Function
def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    attention_mask = [example["instance_attention_mask"] for example in examples]
    
    # Concat class and instance examples for prior preservation
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        attention_mask += [example["class_attention_mask"] for example in examples]
    
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    
    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
    }
    
    return batch

#####  Prior Preservation Generation
def generate_prior_images(class_prompt, num_class_images, class_data_dir, pipeline):
    class_images_dir = Path(class_data_dir)
    class_images_dir.mkdir(parents=True, exist_ok=True)
    
    cur_class_images = len(list(class_images_dir.iterdir()))
    
    if cur_class_images < num_class_images:
        num_new_images = num_class_images - cur_class_images
        print(f"Generating {num_new_images} class images...")
        
        pipeline.safety_checker = None  # Disable safety checker for faster generation
        
        for i in range(num_new_images):
            image = pipeline(class_prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            image.save(f"{class_images_dir}/{i+cur_class_images}.jpg")
            
    print(f"Class images available: {len(list(class_images_dir.iterdir()))}")
    
####  Optimizer and Scheduler Setup
# Set up optimizer
params_to_optimize = (
    list(unet.parameters()) + list(text_encoder.parameters()) 
    if hyperparameters["train_text_encoder"] 
    else unet.parameters()
)

optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=hyperparameters["learning_rate"],
    betas=(0.9, 0.999),
    weight_decay=1e-2,
)

# Create dataloader
train_dataset = DreamBoothDataset(
    instance_data_root=hyperparameters["instance_data_dir"],
    instance_prompt=hyperparameters["instance_prompt"],
    class_data_root=hyperparameters["class_data_dir"] if hyperparameters["with_prior_preservation"] else None,
    class_prompt=hyperparameters["class_prompt"] if hyperparameters["with_prior_preservation"] else None,
    size=hyperparameters["resolution"],
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=hyperparameters["train_batch_size"],
    shuffle=True,
    collate_fn=lambda examples: collate_fn(examples, hyperparameters["with_prior_preservation"]),
    num_workers=2,
)

# Learning rate scheduler
from diffusers.optimization import get_scheduler

lr_scheduler = get_scheduler(
    hyperparameters["lr_scheduler"],
    optimizer=optimizer,
    num_warmup_steps=hyperparameters["lr_warmup_steps"] * hyperparameters["gradient_accumulation_steps"],
    num_training_steps=hyperparameters["max_train_steps"] * hyperparameters["gradient_accumulation_steps"],
)

#### Training Loop with Loss Computation
# Initialize accelerator
accelerator = Accelerator(
    gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
)

# Prepare everything with accelerator
unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler
)

# Training loop
global_step = 0
progress_bar = tqdm(range(hyperparameters["max_train_steps"]))

for epoch in range(hyperparameters["num_train_epochs"]):
    unet.train()
    if hyperparameters["train_text_encoder"]:
        text_encoder.train()
        
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            
            # Sample timesteps
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=latents.device)
            
            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(
                batch["input_ids"], attention_mask=batch["attention_mask"]
            )[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # DreamBooth loss calculation
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise  # predict the noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            # Separate instance and class predictions for prior preservation loss
            if hyperparameters["with_prior_preservation"]:
                # Split the output into the instance and class components
                model_pred_instance, model_pred_class = torch.chunk(model_pred, 2, dim=0)
                target_instance, target_class = torch.chunk(target, 2, dim=0)
                
                # Instance loss - for the specific object
                loss_instance = F.mse_loss(model_pred_instance.float(), target_instance.float(), reduction="mean")
                
                # Prior preservation loss - for class consistency
                loss_class = F.mse_loss(model_pred_class.float(), target_class.float(), reduction="mean")
                
                # Total loss is weighted sum
                loss = loss_instance + hyperparameters["prior_loss_weight"] * loss_class
            else:
                # Only instance loss if no prior preservation
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Backpropagate
            accelerator.backward(loss)
            
            # Clip gradients
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_optimize, 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # Increment and log
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
        
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        
        # Stop if max steps reached
        if global_step >= hyperparameters["max_train_steps"]:
            break

# Save the fine-tuned pipeline
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    pipeline = StableDiffusionPipeline.from_pretrained(
        hyperparameters["pretrained_model_name"],
        unet=accelerator.unwrap_model(unet),
        text_encoder=accelerator.unwrap_model(text_encoder),
    )
    pipeline.save_pretrained(hyperparameters["output_dir"])
    print(f"Model saved to {hyperparameters['output_dir']}")
    
### Inference with the Trained Model
# Load and use the trained model
def generate_samples(prompt, num_samples=4, num_inference_steps=50, guidance_scale=7.5):
    pipeline = StableDiffusionPipeline.from_pretrained(
        hyperparameters["output_dir"],
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Generate images
    images = []
    for _ in range(num_samples):
        image = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        images.append(image)
    
    return images

# Generate using the instance prompt
instance_images = generate_samples(hyperparameters["instance_prompt"])

# Generate with creative prompts using the learned concept
creative_prompt = hyperparameters["instance_prompt"] + " wearing a hat, highly detailed"
creative_images = generate_samples(creative_prompt)