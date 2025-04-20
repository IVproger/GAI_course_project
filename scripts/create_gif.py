import os
import imageio.v2 as imageio
import re
import sys # Import sys to check imageio version

def create_layer_evolution_gifs(base_image_folder, output_folder):
    """
    Generates GIFs showing the evolution of each layer across epochs.

    Args:
        base_image_folder (str): Path to the folder containing epoch subfolders
                                 (e.g., 'diffusion lens images').
        output_folder (str): Path to the folder where GIFs will be saved.
    """
    print(f"Using imageio version: {imageio.__version__}") # Print imageio version

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    epoch_folders = sorted(
        [d for d in os.listdir(base_image_folder) if d.startswith("DiffusionLens_Epoch_") or d == "DiffusionLens_Final"],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf') # Sort numerically, 'Final' last
    )

    if not epoch_folders:
        print(f"Error: No epoch folders found in {base_image_folder}")
        return

    print(f"Found epoch folders: {epoch_folders}")

    # Determine layers by looking into the first epoch folder
    first_epoch_path = os.path.join(base_image_folder, epoch_folders[0])
    try:
        layer_files = [f for f in os.listdir(first_epoch_path) if f.startswith("layer_") and f.endswith(".png")]
        layers = sorted(list(set(re.match(r"layer_(\d+)_step_\d+\.png", f).group(1) for f in layer_files if re.match(r"layer_(\d+)_step_\d+\.png", f))), key=int)
    except FileNotFoundError:
        print(f"Error: Could not find or access files in {first_epoch_path}")
        return
    except Exception as e:
        print(f"Error processing files in {first_epoch_path}: {e}")
        return

    if not layers:
        print(f"Error: No layer image files found in {first_epoch_path}")
        return

    print(f"Found layers: {layers}")

    for layer_num_str in layers:
        layer_images = []
        print(f"\nProcessing Layer {layer_num_str}...")
        for epoch_folder in epoch_folders:
            image_filename = f"layer_{layer_num_str}_step_{layer_num_str}.png"
            image_path = os.path.join(base_image_folder, epoch_folder, image_filename)

            if os.path.exists(image_path):
                try:
                    layer_images.append(imageio.imread(image_path))
                    print(f"  Added image: {image_path}")
                except Exception as e:
                    print(f"  Warning: Could not read image {image_path}. Skipping. Error: {e}")
            else:
                print(f"  Warning: Image not found: {image_path}. Skipping.")

        if layer_images:
            gif_filename = f"layer_{layer_num_str}_evolution.gif"
            gif_path = os.path.join(output_folder, gif_filename)
            try:
                # Use fps instead of duration. fps=0.5 means 0.5 frames per second (2 seconds per frame)
                imageio.mimsave(gif_path, layer_images, fps=0.5)
                print(f"Successfully created GIF: {gif_path} (using fps=0.5)")
            except Exception as e:
                 print(f"Error creating GIF for layer {layer_num_str}: {e}")
        else:
            print(f"No images found for layer {layer_num_str}. Skipping GIF creation.")

# --- Configuration ---
# Set the path to the folder containing your 'DiffusionLens_Epoch_XXX' folders
IMAGE_SOURCE_DIR = '../lens_output/dreambooth_dog_with_lens/diffusion_lens_images'

# Set the path where you want to save the generated GIFs
GIF_OUTPUT_DIR = 'layer_evolution_gifs'
# --- End Configuration ---

if __name__ == "__main__":
    if not os.path.isdir(IMAGE_SOURCE_DIR):
        print(f"Error: Source directory '{IMAGE_SOURCE_DIR}' not found.")
        print("Please ensure the IMAGE_SOURCE_DIR variable points to the correct location.")
    else:
        print("Starting GIF generation...")
        create_layer_evolution_gifs(IMAGE_SOURCE_DIR, GIF_OUTPUT_DIR)
        print("\nGIF generation process finished.")
