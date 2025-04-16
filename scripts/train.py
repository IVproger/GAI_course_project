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
            output_uri = config.get('output_uri', False) 
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
    
    