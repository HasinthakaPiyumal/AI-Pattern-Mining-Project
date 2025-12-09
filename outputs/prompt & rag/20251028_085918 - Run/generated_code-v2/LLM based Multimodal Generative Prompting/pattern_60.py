import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

def mock_generate_image_from_prompt(prompt: str) -> str:
    return f"Generated Image Concept: '{prompt}' - A photorealistic representation with nuanced details based on the prompt modifiers."

def construct_prompt(product_description: str, modifiers: dict) -> str:
    modifier_strings = []
    if "medium" in modifiers and modifiers["medium"]:
        modifier_strings.append(modifiers["medium"])
    if "lighting" in modifiers and modifiers["lighting"]:
        modifier_strings.append(modifiers["lighting"])
    if "background" in modifiers and modifiers["background"]:
        modifier_strings.append(modifiers["background"])
    if "style" in modifiers and modifiers["style"]:
        modifier_strings.append(modifiers["style"])

    full_prompt = product_description
    if modifier_strings:
        full_prompt += ", " + ", ".join(modifier_strings)

    return full_prompt

app = FastAPI(title="E-commerce Product Image Generator API")

class ImageGenerationRequest(BaseModel):
    product_description: str
    modifiers: dict = {}

@app.post("/generate-image")
async def generate_image_endpoint(request: ImageGenerationRequest):
    full_prompt = construct_prompt(request.product_description, request.modifiers)
    generated_image_output = mock_generate_image_from_prompt(full_prompt)
    return {"prompt": full_prompt, "generated_output": generated_image_output}

def gradio_generate_image(
    product_description: str,
    medium: str,
    lighting: str,
    background: str,
    style: str
):
    modifiers = {}
    if medium:
        modifiers["medium"] = medium
    if lighting:
        modifiers["lighting"] = lighting
    if background:
        modifiers["background"] = background
    if style:
        modifiers["style"] = style

    full_prompt = construct_prompt(product_description, modifiers)
    generated_image_concept = mock_generate_image_from_prompt(full_prompt)
    return full_prompt, generated_image_concept

with gr.Blocks() as demo:
    gr.Markdown("# E-commerce Product Image Generator")
    gr.Markdown("Generate and modify product images using prompt modifiers.")

    with gr.Row():
        with gr.Column():
            product_desc_input = gr.Textbox(label="Product Description", placeholder="e.g., A sleek black smartphone")
            medium_input = gr.Dropdown(
                label="Medium",
                choices=["", "on canvas", "digital art", "photorealistic", "oil painting"],
                value=""
            )
            lighting_input = gr.Dropdown(
                label="Lighting",
                choices=["", "well lit scene", "dramatic lighting", "soft studio light", "golden hour"],
                value=""
            )
            background_input = gr.Dropdown(
                label="Background",
                choices=["", "minimalist white background", "outdoor cafe setting", "abstract blur", "cityscape at night"],
                value=""
            )
            style_input = gr.Dropdown(
                label="Style",
                choices=["", "vintage aesthetic", "modern flat design", "fantasy art style", "cyberpunk"],
                value=""
            )
            generate_button = gr.Button("Generate Image")
        with gr.Column():
            output_prompt = gr.Textbox(label="Generated Prompt", interactive=False)
            output_image_concept = gr.Textbox(label="Generated Image Concept", interactive=False)

    generate_button.click(
        fn=gradio_generate_image,
        inputs=[
            product_desc_input,
            medium_input,
            lighting_input,
            background_input,
            style_input
        ],
        outputs=[output_prompt, output_image_concept]
    )

if __name__ == "__main__":
    demo.launch(share=False)