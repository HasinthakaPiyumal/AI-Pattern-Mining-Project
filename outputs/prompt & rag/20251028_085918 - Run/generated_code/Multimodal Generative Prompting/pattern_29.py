from diffusers import StableDiffusionPipeline
import torch

def generate_design_from_text(
    prompt: str,
    negative_prompt: str = None,
    style_modifier: str = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None,
):
    """
    Generates a fashion design image from a textual prompt, incorporating
    negative prompts and style modifiers using a Stable Diffusion model.

    Args:
        prompt (str): The main description of the desired fashion item.
        negative_prompt (str, optional): Elements to exclude from the design.
        style_modifier (str, optional): Stylistic or environmental modifiers.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Classifier-free guidance scale.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        PIL.Image.Image: The generated fashion design image.
    """
    # Load the Stable Diffusion model
    # Using a smaller, faster model for demonstration. For production, a larger model or fine-tuned model would be preferred.
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda") # Move model to GPU if available

    # Construct the full prompt with modifiers
    full_prompt = f"{prompt}, {style_modifier}" if style_modifier else prompt

    # Set random seed if provided for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Generate the image
    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return image

if __name__ == "__main__":
    print("--- Text-to-Design Generation Demo ---")

    # Example 1: Basic design with negative prompt
    print("\nExample 1: Basic summer dress, excluding polka dots")
    design_1 = generate_design_from_text(
        prompt="a bohemian style summer dress with floral patterns",
        negative_prompt="polka dots, too much lace, blurry, low quality",
        seed=42,
    )
    design_1.save("summer_dress_floral_no_polka_dots.png")
    print("Generated 'summer_dress_floral_no_polka_dots.png'")

    # Example 2: Dress with style modifier
    print("\nExample 2: Vintage aesthetic evening gown")
    design_2 = generate_design_from_text(
        prompt="an elegant evening gown",
        style_modifier="vintage aesthetic, cinematic lighting, high fashion photography",
        negative_prompt="modern, casual, sportswear",
        seed=123,
    )
    design_2.save("vintage_evening_gown.png")
    print("Generated 'vintage_evening_gown.png'")

    # Example 3: Sportswear with specific rendering
    print("\nExample 3: Futuristic tracksuit, watercolor style")
    design_3 = generate_design_from_text(
        prompt="a sleek futuristic tracksuit",
        negative_prompt="old, messy, traditional",
        style_modifier="rendered in a watercolor style, concept art",
        seed=789,
    )
    design_3.save("futuristic_tracksuit_watercolor.png")
    print("Generated 'futuristic_tracksuit_watercolor.png'")