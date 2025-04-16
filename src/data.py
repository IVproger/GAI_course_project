from torch.utils.data import Dataset
import os
from PIL import Image

class PriorClassDataset(Dataset):
    '''
    A dataset that loads images from a directory and tokenizes prompts.
    '''
    def __init__(self, image_dir, subject_prompt, class_prompt, tokenizer, transforms, use_dreambooth_prompts=True):
        '''
        Initialize dataset parameters and prepare data transforms.

        Args:
            image_dir (str): Path to directory where images are stored.
            subject_prompt (str): Prompt used when use_dreambooth_prompts is True.
            class_prompt (str): Prompt used when use_dreambooth_prompts is False.
            tokenizer: A tokenizer object that processes the prompts.
            transforms: A set of transformations applied to each image.
            use_dreambooth_prompts (bool): Flag to determine which prompt to use.
        '''
        # Store dataset configuration
        self.image_dir = image_dir
        self.subject_prompt = subject_prompt
        self.class_prompt = class_prompt
        self.tokenizer = tokenizer
        
        # Gather all image file paths from the directory
        self.image_paths = [
            os.path.join(image_dir, img) 
            for img in os.listdir(image_dir) 
            if img.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # Transform for images (e.g., resizing, normalization)
        self.transform = transforms
        
        # Whether to use subject prompt or class prompt
        self.use_dreambooth_prompts = use_dreambooth_prompts

    def __len__(self):
        '''
        Return the total number of images in the dataset.
        '''
        return len(self.image_paths)

    def __getitem__(self, idx):
        '''
        Return a transformed image and tokenized prompt at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (image (Tensor), input_ids (Tensor), attention_mask (Tensor))
        '''
        # Get the image path
        image_path = self.image_paths[idx]
        
        # Open and convert image to RGB
        image = Image.open(image_path).convert("RGB")
        
        # Apply the defined transformations
        image = self.transform(image)
        
        # Select which prompt to use
        prompt = self.subject_prompt if self.use_dreambooth_prompts else self.class_prompt
        
        # Tokenize the prompt
        encoding = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        )
        
        # Extract input IDs and attention mask from the encoding
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Return the processed image, prompt IDs, and attention mask
        return image, input_ids, attention_mask