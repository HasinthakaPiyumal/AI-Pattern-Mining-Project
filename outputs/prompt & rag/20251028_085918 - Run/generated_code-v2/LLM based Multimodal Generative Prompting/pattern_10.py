from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import base64
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import gradio as gr
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the Stable Diffusion Img2Img pipeline
# Using a smaller model for faster execution in an example, replace with desired model
# For better quality, use "runwayml/stable-diffusion-v1-5" or a similar model
model_id = "stabilityai/stable-diffusion-v1-5"
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/generate_image")
async def generate_image_api(
    image_file: UploadFile = File(...),
    prompt: str = "",
    strength: float = 0.8,
    guidance_scale: float = 7.5
):
    # Read the uploaded image file
    image_bytes = await image_file.read()
    init_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Generate the image using the diffusion pipeline
    generated_image = pipeline(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale
    ).images[0]

    # Encode the generated image to base64
    buffered = BytesIO()
    generated_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"generated_image": img_str}


def gradio_predict(input_image, prompt_modifiers, strength_val, guidance_scale_val):
    if input_image is None:
        return None

    # Convert Gradio image to bytes
    buffered = BytesIO()
    input_image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Mock calling the FastAPI endpoint locally for Gradio interface
    # In a real deployed scenario, you would use requests.post to call the external API
    # For this combined file, we can directly call the pipeline for simplicity
    # and avoid the http overhead within the same process.

    # Construct the full prompt (can be more sophisticated for real products)
    full_prompt = f"product shot, high quality, professional, {prompt_modifiers}"

    generated_image = pipeline(
        prompt=full_prompt,
        image=input_image,
        strength=strength_val,
        guidance_scale=guidance_scale_val
    ).images[0]
    
    return generated_image

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # AI-powered Product Image Customizer
        Upload a product image and use text prompts to modify its appearance.
        """
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Product Image")
            prompt_modifiers = gr.Textbox(label="Prompt Modifiers", placeholder="e.g., 'on canvas, well-lit scene, serene beach background, watercolor painting style'")
            strength_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.8, label="Strength (for Img2Img)")
            guidance_scale_slider = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, label="Guidance Scale")
            generate_button = gr.Button("Generate Image")
        with gr.Column():
            output_image = gr.Image(label="Generated Image")

    generate_button.click(
        gradio_predict,
        inputs=[input_image, prompt_modifiers, strength_slider, guidance_scale_slider],
        outputs=output_image
    )

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

# To run this: uvicorn main:app --reload
# Or for direct execution if not using uvicorn cli:
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)