import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the Stable Diffusion model
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda") # Move model to GPU if available

def generate_asset(base_prompt: str, prompt_modifier: str) -> Image.Image:
    full_prompt = f"{base_prompt}, {prompt_modifier}" if prompt_modifier else base_prompt
    image = pipeline(full_prompt).images[0]
    return image

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_asset,
    inputs=[
        gr.Textbox(label="Base Prompt", placeholder="e.g., a knight in shining armor"),
        gr.Textbox(label="Prompt Modifier", placeholder="e.g., oil painting style, vibrant colors, sunset lighting")
    ],
    outputs=gr.Image(label="Generated Asset"),
    title="Virtual World Creator for Game Developers",
    description="Generate game assets using AI with prompt modifiers. The more descriptive your modifier, the better!"
)

# Launch the Gradio interface
iface.launch()