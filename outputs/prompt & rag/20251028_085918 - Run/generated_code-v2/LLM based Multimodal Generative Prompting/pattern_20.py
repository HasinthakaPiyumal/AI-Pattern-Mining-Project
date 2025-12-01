from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

def generate_visual_content(base_prompt: str, modifiers: dict) -> Image.Image | None:
    """
    Generates an image based on a base prompt and modifiers.
    This function simulates the output of a video generation system for demonstration 
    purposes, focusing on the effect of prompt modifiers on visual content.
    """
    # Construct the full prompt by appending modifiers
    full_prompt_parts = [base_prompt]
    if "style" in modifiers and modifiers["style"]:
        full_prompt_parts.append(f", {modifiers['style']} style")
    if "lighting" in modifiers and modifiers["lighting"]:
        full_prompt_parts.append(f", {modifiers['lighting']}")
    if "camera" in modifiers and modifiers["camera"]:
        full_prompt_parts.append(f", {modifiers['camera']}")
    if "setting" in modifiers and modifiers["setting"]:
        full_prompt_parts.append(f", {modifiers['setting']}")
    if "artist" in modifiers and modifiers["artist"]:
        full_prompt_parts.append(f", by {modifiers['artist']}") # Example of another modifier

    final_prompt = ", ".join([p for p in full_prompt_parts if p.strip()])
    print(f"\n--- Final Prompt for Generation ---\n{final_prompt}\n-----------------------------------")

    try:
        # Load a pre-trained Stable Diffusion model
        # Using a widely accessible model for demonstration.
        # This requires `diffusers` and `torch` to be installed.
        # For optimal performance, a GPU is recommended. 
        # If no GPU is available, it will fall back to CPU, which can be slow.
        model_id = "runwayml/stable-diffusion-v1-5"
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(device)
        print(f"Loaded Stable Diffusion model on {device}.")

        # Generate image
        image = pipeline(final_prompt).images[0]
        return image
    except Exception as e:
        print(f"\nError during image generation: {e}")
        print("Please ensure you have `diffusers` and `torch` installed.")
        print(f"You can install them via: pip install diffusers torch {'--extra-index-url https://download.pytorch.org/whl/cu118' if torch.cuda.is_available() else ''} transformers safetensors")
        print("If running on CPU, generation might be very slow or require more RAM.")
        print("For a quick test without actual generation, you can comment out the `try-except` block ")
        print("and uncomment the placeholder image line below.")
        # return Image.new('RGB', (768, 512), color = 'lightgray') # Placeholder image for testing without GPU/model
        return None

# --- Main execution block ---
if __name__ == "__main__":
    print("\nWelcome to CineCraft AI: Prompt Modifier Demo!")
    print("This tool demonstrates how adding specific modifiers to a base prompt can influence")
    print("the generated visual content (simulated with an image for this demo).")

    # Get base prompt from user
    base_prompt_input = input("\nEnter your base prompt (e.g., 'A futuristic city street'): ")

    # Get modifiers from user
    print("\nNow, enter specific modifiers. Leave any field blank to skip it.")
    style_input = input("  Visual Style (e.g., 'vintage film', 'anime style', 'oil painting'): ")
    lighting_input = input("  Lighting (e.g., 'cinematic lighting', 'golden hour', 'neon glow'): ")
    camera_input = input("  Camera Angle (e.g., 'wide shot', 'close-up', 'dutch angle'): ")
    setting_input = input("  Setting Details (e.g., 'dystopian future', 'enchanted forest', 'bustling marketplace'): ")
    artist_input = input("  Inspired by Artist (e.g., 'Van Gogh', 'Studio Ghibli'): ")

    modifiers_dict = {
        "style": style_input,
        "lighting": lighting_input,
        "camera": camera_input,
        "setting": setting_input,
        "artist": artist_input,
    }

    # Generate content
    generated_image = generate_visual_content(base_prompt_input, modifiers_dict)

    if generated_image:
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, "cinecraft_output.png")
        generated_image.save(output_filename)
        print(f"\nSuccess! Image generated and saved to '{output_filename}'")
        print("Check the 'generated_images' folder to see your creation.")
    else:
        print("\nImage generation failed. Please review the error messages above.")