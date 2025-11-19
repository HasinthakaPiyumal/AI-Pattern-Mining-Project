import gradio as gr
from PIL import Image
import io
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from rembg import remove, new_session
import torch


pipe_text2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe_text2img.to("cuda")

pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe_img2img.to("cuda")

rembg_session = new_session()

def text_to_image_customization(prompt: str, negative_prompt: str):
    image = pipe_text2img(prompt, negative_prompt=negative_prompt).images[0]
    return image

def visual_style_transfer(base_image: Image.Image, style_prompt: str):
    init_image = base_image.convert("RGB")
    image = pipe_img2img(prompt=style_prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
    return image

def generate_3d_model_placeholder(image_input: Image.Image = None, text_input: str = None):
    output_message = "3D model generation simulated. A GLB/USDZ file would be generated for AR preview.\n"
    if image_input:
        output_message += "Based on provided image."
    if text_input:
        output_message += " Based on provided text: " + text_input
    return output_message

def remove_background(input_image: Image.Image):
    output_image = remove(input_image, session=rembg_session)
    return output_image

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Dynamic Product Customizer for E-commerce
        Customize your products using advanced AI prompting techniques!
        """
    )

    with gr.Tab("Text-to-Image Customization"):
        with gr.Row():
            text2img_prompt = gr.Textbox(label="Product Description (Positive Prompt)", placeholder="A modern sofa with velvet upholstery")
            text2img_negative_prompt = gr.Textbox(label="Negative Prompt (What to exclude)", placeholder="not in blue, no visible seams")
        text2img_btn = gr.Button("Generate Product Image")
        text2img_output = gr.Image(label="Generated Product Image")
        text2img_btn.click(text_to_image_customization, inputs=[text2img_prompt, text2img_negative_prompt], outputs=text2img_output)

    with gr.Tab("Visual Style Transfer"):
        with gr.Row():
            style_base_image = gr.Image(type="pil", label="Upload Base Product Image")
            style_prompt = gr.Textbox(label="Style Description", placeholder="A specific wood grain, a floral print")
        style_transfer_btn = gr.Button("Apply Style")
        style_transfer_output = gr.Image(label="Styled Product Image")
        style_transfer_btn.click(visual_style_transfer, inputs=[style_base_image, style_prompt], outputs=style_transfer_output)

    with gr.Tab("3D Generation & AR Preview (Placeholder)"):
        gr.Markdown("This section simulates 3D model generation and AR preview functionality.")
        with gr.Row():
            ar_image_input = gr.Image(type="pil", label="Optional: Upload 2D Image for 3D Gen")
            ar_text_input = gr.Textbox(label="Optional: Text Prompt for 3D Gen", placeholder="make the legs slightly shorter, add a chrome finish")
        ar_gen_btn = gr.Button("Simulate 3D Model Generation")
        ar_output = gr.Textbox(label="3D Generation Status")
        ar_gen_btn.click(generate_3d_model_placeholder, inputs=[ar_image_input, ar_text_input], outputs=ar_output)

    with gr.Tab("Automatic Background Removal"):
        bg_removal_input = gr.Image(type="pil", label="Upload Image to Remove Background")
        bg_removal_btn = gr.Button("Remove Background")
        bg_removal_output = gr.Image(label="Image with Background Removed")
        bg_removal_btn.click(remove_background, inputs=bg_removal_input, outputs=bg_removal_output)

demo.launch()