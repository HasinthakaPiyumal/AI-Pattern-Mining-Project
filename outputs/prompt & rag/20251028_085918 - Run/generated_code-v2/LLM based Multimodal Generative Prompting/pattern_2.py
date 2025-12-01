import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline.to("cuda")

def generate_product_image(positive_prompt, negative_prompt):
    image = pipeline(prompt=positive_prompt, negative_prompt=negative_prompt).images[0]
    return image

iface = gr.Interface(
    fn=generate_product_image,
    inputs=[
        gr.Textbox(label="Positive Prompt", placeholder="e.g., a sleek silver smartphone, professional studio lighting"),
        gr.Textbox(label="Negative Prompt", placeholder="e.g., blurry, low resolution, watermark, bad quality, distorted")
    ],
    outputs=gr.Image(label="Generated Product Image"),
    title="E-commerce Product Image Generator with Negative Prompting",
    description="Generate high-quality product images by specifying desired features and preventing undesired elements using negative prompts."
)

iface.launch()