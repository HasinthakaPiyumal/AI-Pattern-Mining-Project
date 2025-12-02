
import gradio as gr
from PIL import Image
import random
import os

# --- Placeholder for actual diffusion model interaction ---
def generate_image_with_negative_prompt_mock(positive_prompt: str, negative_prompt: str) -> Image.Image:
    """
    A mock function to simulate image generation using positive and negative prompts.
    In a real application, this would interact with a library like `diffusers`
    and a pre-trained model (e.g., Stable Diffusion).
    """
    print(f"[MOCK GENERATION] Positive Prompt: {positive_prompt}")
    print(f"[MOCK GENERATION] Negative Prompt: {negative_prompt}")

    # Simulate generating a placeholder image based on the prompts
    # For a real implementation, you would load your diffusion pipeline here:
    # from diffusers import StableDiffusionPipeline
    # import torch
    # pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    # pipeline.to("cuda") # if GPU is available
    # image = pipeline(prompt=positive_prompt, negative_prompt=negative_prompt).images[0]

    # Create a dummy image for demonstration
    width, height = 512, 512
    img = Image.new("RGB", (width, height), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)))

    # Add some text to indicate the prompts were used
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 10), f"Positive: {positive_prompt[:50]}...", (255, 255, 255), font=font)
    draw.text((10, 50), f"Negative: {negative_prompt[:50]}...", (255, 255, 255), font=font)
    draw.text((10, 90), "(Placeholder Image)", (255, 255, 255), font=font)

    return img

# --- Gradio Interface ---
iface = gr.Interface(
    fn=generate_image_with_negative_prompt_mock,
    inputs=[
        gr.Textbox(label="Positive Prompt (Describe your desired product image)",
                   placeholder="e.g., A sleek silver smartwatch on a person's wrist, outdoor setting, sunny day"),
        gr.Textbox(label="Negative Prompt (Describe elements to avoid)",
                   placeholder="e.g., blurry background, unnatural lighting, distorted product, irrelevant objects, bad shadows")
    ],
    outputs=gr.Image(label="Generated Product Image"),
    title="E-commerce Product Image Generator with Quality Control",
    description="Generate high-quality product images for e-commerce with positive and negative prompting to control output."
)

if __name__ == "__main__":
    iface.launch()
