from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import gradio as gr
from PIL import Image

from multimodal_prompter import MultimodalPrompter
from utils import decode_image, encode_image, create_dummy_image

app = FastAPI()
prompter = MultimodalPrompter()

class ProductGenerationRequest(BaseModel):
    text_description: str
    base_image_b64: Optional[str] = None
    annotations: Optional[Dict[str, Any]] = None
    negative_prompt: Optional[str] = None
    transformation_style: Optional[str] = None

@app.post("/generate_product")
async def generate_product_endpoint(request: ProductGenerationRequest):
    try:
        base_image = None
        if request.base_image_b64:
            base_image = decode_image(request.base_image_b64)

        prompt_components = prompter.construct_prompt(
            text_description=request.text_description,
            base_image=base_image,
            annotations=request.annotations,
            negative_prompt=request.negative_prompt,
            transformation_style=request.transformation_style
        )

        # --- Placeholder for actual AI model call ---
        # In a real application, you would integrate with a generative AI model (e.g., Stable Diffusion, DALL-E)
        # The model would take prompt_components['prompt'], prompt_components['negative_prompt'],
        # and potentially prompt_components['image_input'] for image-to-image tasks.
        # For this example, we'll return a dummy image and the constructed prompts.
        print(f"Simulating AI generation with prompt: {prompt_components['prompt']}")
        print(f"And negative prompt: {prompt_components['negative_prompt']}")

        generated_image = create_dummy_image() # Replace with actual AI generated image
        generated_image_b64 = encode_image(generated_image)
        # --- End Placeholder ---

        return {
            "generated_image_b64": generated_image_b64,
            "constructed_prompt": prompt_components['prompt'],
            "negative_prompt_used": prompt_components['negative_prompt']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Gradio Interface
def gradio_interface(
    text_description: str,
    base_image: Image.Image,
    annotations_str: str,
    negative_prompt: str,
    transformation_style: str
):
    annotations = None
    if annotations_str:
        try:
            import json
            annotations = json.loads(annotations_str)
        except json.JSONDecodeError:
            return None, "Invalid JSON for annotations", ""

    base_image_b64 = None
    if base_image:
        base_image_b64 = encode_image(base_image)

    request_payload = {
        "text_description": text_description,
        "base_image_b64": base_image_b64,
        "annotations": annotations,
        "negative_prompt": negative_prompt,
        "transformation_style": transformation_style
    }

    # In a real scenario, you'd make an HTTP request to your FastAPI endpoint
    # For simplicity in a single file, we'll directly call the prompter logic
    # and simulate the AI response.
    prompt_components = prompter.construct_prompt(
        text_description=text_description,
        base_image=base_image,
        annotations=annotations,
        negative_prompt=negative_prompt,
        transformation_style=transformation_style
    )

    # Simulate AI generation
    generated_image = create_dummy_image() # This would be the output from your actual AI model

    return (
        generated_image,
        prompt_components['prompt'],
        prompt_components['negative_prompt']
    )


with gr.Blocks() as demo:
    gr.Markdown("# E-commerce Product Customization and Virtual Try-On")
    with gr.Row():
        with gr.Column():
            text_desc_input = gr.Textbox(label="Text Description", placeholder="e.g., A vintage-style leather jacket")
            base_image_input = gr.Image(type="pil", label="Base Image (Optional)", interactive=True)
            annotations_input = gr.Textbox(label="Annotations (JSON, Optional)", placeholder='e.g., {"type": "bbox", "coordinates": [100, 50, 200, 150], "label": "logo"}')
            negative_prompt_input = gr.Textbox(label="Negative Prompt (Optional)", placeholder="e.g., low quality, blurry, modern style")
            transformation_style_input = gr.Dropdown(
                list(prompter.in_context_examples.keys()),
                label="Transformation Style (Optional)",
                allow_custom_value=True
            )
            generate_button = gr.Button("Generate Product")
        with gr.Column():
            output_image = gr.Image(label="Generated Product")
            output_prompt = gr.Textbox(label="Constructed Prompt")
            output_negative_prompt = gr.Textbox(label="Negative Prompt Used")

    generate_button.click(
        gradio_interface,
        inputs=[
            text_desc_input,
            base_image_input,
            annotations_input,
            negative_prompt_input,
            transformation_style_input
        ],
        outputs=[
            output_image,
            output_prompt,
            output_negative_prompt
        ]
    )

app = gr.mount_gradio_app(app, demo, path="/gradio")
