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
try:
    from DiffusionLens.pipeline_stable_diffusion import StableDiffusionPipeline as StableDiffusionGlassPipeline
    DIFFUSION_LENS_AVAILABLE = True
except ImportError:
    StableDiffusionGlassPipeline = None # Or a dummy class
    DIFFUSION_LENS_AVAILABLE = False

class DreamBoothTrainer:
    def __init__(self, config, accelerator, logger):
        self.config = config
        self.accelerator = accelerator
        self.logger = logger
        
        # Check if DiffusionLens visualization is enabled
        self.use_diffusion_lens = config.get('use_diffusion_lens', False) and DIFFUSION_LENS_AVAILABLE
        if config.get('use_diffusion_lens', False) and not DIFFUSION_LENS_AVAILABLE:
            print("Warning: DiffusionLens visualization is enabled in config but DiffusionLens is not available. Disabling feature.")
        

    def train(self, unet, vae, text_encoder, tokenizer, noise_scheduler, train_dataloader, prior_data):
        """Main training loop."""

        cfg = self.config # Shorthand

        # Prepare optimizer
        optimizer_cfg = cfg['optimizer']
        train_text_encoder = cfg['train_text_encoder']
        optimizer = torch.optim.AdamW(
            # Only train UNet and text_encoder parameters
            list(unet.parameters()) + list(text_encoder.parameters()) if train_text_encoder else list(unet.parameters()),
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
                    # Original (incorrect): compares predicted noise to actual noise
                    # loss_subject = F.mse_loss(model_pred_subject.float(), noise.float(), reduction="mean")

                    # Corrected (DreamBooth paper): compares predicted x0 to actual x0 (latents)
                    # 1. Get scheduler alphas
                    alphas_cumprod_subject = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=model_pred_subject.dtype)
                    sqrt_alpha_t_subject = alphas_cumprod_subject[timesteps] ** 0.5
                    sqrt_one_minus_alpha_t_subject = (1.0 - alphas_cumprod_subject[timesteps]) ** 0.5

                    # 2. Reshape alphas and sigmas for broadcasting
                    sqrt_alpha_t_subject = sqrt_alpha_t_subject.flatten()
                    while len(sqrt_alpha_t_subject.shape) < noisy_latents.ndim:
                        sqrt_alpha_t_subject = sqrt_alpha_t_subject.unsqueeze(-1)
                    sqrt_one_minus_alpha_t_subject = sqrt_one_minus_alpha_t_subject.flatten()
                    while len(sqrt_one_minus_alpha_t_subject.shape) < noisy_latents.ndim:
                        sqrt_one_minus_alpha_t_subject = sqrt_one_minus_alpha_t_subject.unsqueeze(-1)

                    # 3. Denoise the prediction (get predicted x0)
                    pred_x0_subject = (noisy_latents - sqrt_one_minus_alpha_t_subject * model_pred_subject) / sqrt_alpha_t_subject

                    # 4. Calculate MSE loss against the original subject latents
                    loss_subject = F.mse_loss(pred_x0_subject.float(), latents.float(), reduction="mean")


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
                       # Original (incorrect): compares predicted noise to actual noise
                       # loss_prior = F.mse_loss(model_pred_prior.float(), noise_prior.float(), reduction="mean")

                       # Corrected (DreamBooth paper): compares predicted x0 to actual x0 (prior_batch_latents)
                       # We need to derive the predicted x0 from the model's noise prediction.
                       # 1. Get scheduler alphas
                       alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_prior_latents.device, dtype=model_pred_prior.dtype)
                       sqrt_alpha_t = alphas_cumprod[t_prior] ** 0.5
                       sqrt_one_minus_alpha_t = (1.0 - alphas_cumprod[t_prior]) ** 0.5

                       # 2. Reshape alphas and sigmas for broadcasting
                       sqrt_alpha_t = sqrt_alpha_t.flatten()
                       while len(sqrt_alpha_t.shape) < noisy_prior_latents.ndim:
                            sqrt_alpha_t = sqrt_alpha_t.unsqueeze(-1)
                       sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.flatten()
                       while len(sqrt_one_minus_alpha_t.shape) < noisy_prior_latents.ndim:
                            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.unsqueeze(-1)

                       # 3. Denoise the prediction (get predicted x0)
                       pred_x0_prior = (noisy_prior_latents - sqrt_one_minus_alpha_t * model_pred_prior) / sqrt_alpha_t

                       # 4. Calculate MSE loss against the original prior latents
                       # Cast both to float() like the subject loss calculation for numerical stability
                       loss_prior = F.mse_loss(pred_x0_prior.float(), prior_batch_latents.float(), reduction="mean")


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
                        if cfg.get('save_model_steps') and (global_step % (num_update_steps_per_epoch *cfg['save_model_steps']) == 0):
                            self.save_checkpoint(unet, text_encoder, cfg, epoch)
                        # Periodic image generation/logging
                        if cfg.get('log_image_epochs') and global_step % (num_update_steps_per_epoch * cfg['log_image_epochs']) == 0:
                            self.log_sample_images(unet, vae, text_encoder, tokenizer, noise_scheduler, cfg, epoch)
                        # Periodic DiffusionLens visualization if enabled
                        if self.use_diffusion_lens and cfg.get('diffusion_lens_epochs') and global_step % (num_update_steps_per_epoch * cfg['diffusion_lens_epochs']) == 0:
                            self.log_diffusion_lens_images(unet, vae, text_encoder, tokenizer, noise_scheduler, cfg, epoch)


        # End of training
        if self.accelerator.is_main_process:
            print("\n--- Training Complete ---")
            # Final save
            self.save_checkpoint(unet, text_encoder, cfg, global_step, final=True)
            # Final image generation
            self.log_sample_images(unet, vae, text_encoder, tokenizer, noise_scheduler, cfg, epoch, final=True)
            # Final DiffusionLens visualization if enabled
            if self.use_diffusion_lens:
                self.log_diffusion_lens_images(unet, vae, text_encoder, tokenizer, noise_scheduler, cfg, epoch, final=True)


    def save_checkpoint(self, unet, text_encoder, config, epoch, final=False):
        """Saves the UNet and text_encoder state dicts."""
        unet_out_path = os.path.join(config['output_dir'], config['task_name'], f"unet_{'final' if final else epoch+1}.pt")
        text_encoder_out_path = os.path.join(config['output_dir'], config['task_name'], f"text_encoder_{'final' if final else epoch+1}.pt")
        os.makedirs(os.path.dirname(unet_out_path), exist_ok=True)

        # Unwrap models before saving state_dict
        unet_state_dict = self.accelerator.unwrap_model(unet).state_dict()
        text_encoder_state_dict = self.accelerator.unwrap_model(text_encoder).state_dict()

        self.accelerator.save(unet_state_dict, unet_out_path)
        self.accelerator.save(text_encoder_state_dict, text_encoder_out_path)
        print(f"Saved checkpoint at step/epoch {epoch+1} to {os.path.dirname(unet_out_path)}")

        # # Optionally, upload to ClearML as artifacts
        # if self.logger:
        #     self.logger.upload_artifact(name=f"unet_state_dict_{'final' if final else step_or_epoch}", artifact_object=unet_out_path)
        #     self.logger.upload_artifact(name=f"text_encoder_state_dict_{'final' if final else step_or_epoch}", artifact_object=text_encoder_out_path)


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
                 num_inference_steps = 100, # Use a reasonable number of steps for eval
                 guidance_scale = 7.5,       # Standard guidance scale
                 num_images_per_prompt = 3, # Generate a few samples
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
        
    @torch.no_grad()
    def log_diffusion_lens_images(self, unet, vae, text_encoder, tokenizer, noise_scheduler, config, epoch, final=False):
        """Generates and logs images with DiffusionLens to visualize the diffusion process."""
        if not self.use_diffusion_lens:
            return
            
        print(f"\nGenerating DiffusionLens visualizations for epoch {epoch+1}{' (Final)' if final else ''}...")
        
        # Get DiffusionLens parameters from config
        lens_params = config.get('diffusion_lens_params', {})
        start_layer = lens_params.get('start_layer', 0)
        end_layer = lens_params.get('end_layer', 20)
        step_layer = lens_params.get('step_layer', 1)
        num_inference_steps = lens_params.get('num_inference_steps', 100)
        guidance_scale = lens_params.get('guidance_scale', 7.5)
        
        # Important: Use unwrapped models for pipeline creation
        unet_eval = self.accelerator.unwrap_model(unet)
        text_encoder_eval = self.accelerator.unwrap_model(text_encoder)
        
        # Set up DiffusionLens pipeline
        lens_pipeline = StableDiffusionGlassPipeline(
            vae=vae,
            text_encoder=text_encoder_eval,
            tokenizer=tokenizer,
            unet=unet_eval,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        lens_pipeline = lens_pipeline.to(self.accelerator.device)
        lens_pipeline.set_progress_bar_config(disable=True)
        
        # Set up generator for reproducibility
        generator = torch.Generator(device=self.accelerator.device).manual_seed(config.get('seed', 42) + epoch)
        
        # Use the core subject prompt for evaluation
        prompt = config['lens_prompt']
        negative_prompt = config.get('negative_prompt', "")
        print(f"  Using prompt: '{prompt}'")
        
        # Create output directory
        series_name = f"DiffusionLens_Epoch_{epoch+1}" if not final else "DiffusionLens_Final"
        output_dir = os.path.join(config['output_dir'], config['task_name'], "diffusion_lens_images", series_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the DiffusionLens pipeline
        weight_dtype = torch.float16 if config['mixed_precision'] == 'fp16' else torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float32
        with torch.autocast(self.accelerator.device.type, dtype=weight_dtype):
            try:
                layer_outputs = lens_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=1,  # Just one image per layer for visualization
                    start_layer=start_layer,
                    end_layer=end_layer,
                    step_layer=step_layer,
                    output_type="pil"
                )
                
                # Process and save the outputs
                current_layer = start_layer
                all_images = []
                
                for i, output in enumerate(layer_outputs):
                    if not hasattr(output, 'images') or not output.images:
                        print(f"Warning: No images found in output for layer index {i}")
                        continue
                        
                    img = output.images[0]  # Get the first (and only) image
                    all_images.append(img)
                    
                    # Save the image
                    layer_filename = f"layer_{current_layer:03d}_step_{i:03d}.png"
                    output_path = os.path.join(output_dir, layer_filename)
                    img.save(output_path)
                    
                    current_layer += step_layer
                
                # Log to ClearML if available
                if self.logger and all_images:
                    # Create a grid of images for easier viewing
                    # Log a few representative images
                    for idx, img in enumerate(all_images[::max(1, len(all_images)//5)]):  # Log ~5 images
                        self.logger.report_image(
                            title="DiffusionLens Visualizations",
                            series=series_name,
                            iteration=epoch+1 if not final else config['num_train_epochs'],
                            image=img
                        )
                
                print(f"Saved {len(all_images)} DiffusionLens visualization images to {output_dir}")
                
            except Exception as e:
                print(f"Error during DiffusionLens pipeline execution: {e}")
        
        # Clean up
        del lens_pipeline
        torch.cuda.empty_cache()
        
        