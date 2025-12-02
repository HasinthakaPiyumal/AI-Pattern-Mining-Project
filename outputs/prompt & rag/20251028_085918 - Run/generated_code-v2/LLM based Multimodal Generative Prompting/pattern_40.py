import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

# 1. Model Loading
# Using a small float16 model for potentially faster loading and lower VRAM usage
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

# 2. Backend Logic: Prompt Construction and Image Generation
def generate_scene(base_description: str, modifiers: list, custom_modifiers: str):
    full_prompt_parts = [base_description]

    if modifiers:
        full_prompt_parts.extend(modifiers)
    
    if custom_modifiers:
        full_prompt_parts.append(custom_modifiers)

    full_prompt = ", ".join(filter(None, full_prompt_parts))
    
    # Generate image
    with torch.autocast("cuda"):
        image = pipeline(full_prompt).images[0]
    
    return image

# 3. User Interface (UI) with Gradio
common_modifiers = [
    "cinematic lighting", 
    "vintage film grain", 
    "dramatic close-up", 
    "wide-angle shot", 
    "on a rainy street", 
    "sci-fi futuristic setting",
    "noir style",
    "moody atmosphere",
    "high contrast",
    "soft focus",
    "anamorphic lens flare"
]

iface = gr.Interface(
    fn=generate_scene,
    inputs=[
        gr.Textbox(label="Base Scene Description", placeholder="A detective in a dimly lit office"),
        gr.CheckboxGroup(choices=common_modifiers, label="Common Prompt Modifiers"),
        gr.Textbox(label="Custom Prompt Modifiers (comma-separated)", placeholder="e.g., volumetric fog, intense shadows")
    ],
    outputs=gr.Image(label="Generated Movie Scene", type="pil"),
    title="AI-Powered Movie Scene Generator",
    description="Generate movie scenes with precise control using prompt modifiers. Select common modifiers or add custom ones."
)

iface.launch()