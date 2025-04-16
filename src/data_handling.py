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
