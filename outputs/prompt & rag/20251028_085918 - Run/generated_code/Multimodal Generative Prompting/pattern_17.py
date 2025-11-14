"""
This Python script implements an 'Intelligent Product Customizer' application.
It uses Gradio for the frontend, FastAPI for the backend API, and leverages
Hugging Face's Transformers (BLIP) and Diffusers (Stable Diffusion) for
multimodal AI capabilities. Users can provide text prompts, optional reference images,
and negative prompts to generate customized product designs.

Key features:
- Multimodal input: Accepts text prompts and image uploads.
- Image-to-text conversion: Uses BLIP to describe uploaded images, enriching text prompts.
- Generative AI: Employs Stable Diffusion for text-to-image generation.
- Prompt control: Supports negative prompts to exclude undesired elements and guidance scale.
- Web interface: Powered by Gradio for an interactive user experience.
- Backend API: FastAPI handles AI model inference requests.

To run this application, you need to install the required libraries:
`pip install gradio fastapi uvicorn pillow torch transformers diffusers accelerate`
(Note: 'accelerate' is listed in the architecture but not explicitly used in this simplified demo for direct inference calls, but it's a common dependency for diffusers/transformers environments.)

Ensure you have a GPU for faster inference with Stable Diffusion; otherwise, it will run on CPU.
"""

import gradio as gr
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import io
import base64
import uvicorn
import threading
import os

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Global AI Model Storage ---
blip_processor = None
blip_model = None
sd_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Loading Function ---
def load_models():
    global blip_processor, blip_model, sd_pipeline, device

    print(f"Loading BLIP model on {device}...")
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        print("BLIP model loaded successfully.")
    except Exception as e:
        print(f"Error loading BLIP model: {e}")
        blip_processor = None
        blip_model = None

    print(f"Loading Stable Diffusion pipeline on {device}...")
    try:
        # Using a smaller model for quicker loading/inference in a demo
        # For better quality, consider larger models like 'stabilityai/stable-diffusion-xl-base-1.0'
        sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        sd_pipeline = sd_pipeline.to(device)
        print("Stable Diffusion pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading Stable Diffusion pipeline: {e}")
        sd_pipeline = None

# --- Helper Functions for Image Encoding/Decoding ---
def image_to_base64(img: Image.Image) -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_string: str) -> Image.Image:
    """Converts a base64 encoded string to a PIL Image."""
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))

# --- FastAPI Endpoint ---
@app.post("/generate_design/")
async def generate_design_api(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    input_image_b64: str = Form(None), # Base64 encoded image
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(50)
):
    """
    FastAPI endpoint to generate product designs based on multimodal prompts.
    """
    global blip_processor, blip_model, sd_pipeline, device

    if sd_pipeline is None:
        return {"error": "AI models not loaded or failed to load. Please try again later."}

    generated_image_b64 = None
    full_prompt = prompt

    # 1. Multimodal Input Processing: Image to text if an image is provided
    if input_image_b64 and blip_processor and blip_model:
        try:
            input_image = base64_to_image(input_image_b64).convert("RGB")
            # Generate text description from image
            inputs = blip_processor(images=input_image, text=prompt, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs, max_new_tokens=50)
            img_description = blip_processor.decode(out[0], skip_special_tokens=True)
            print(f"BLIP generated description: '{img_description}'")
            # Combine original prompt with image description for richer context
            full_prompt = f"{prompt}, a {img_description}"
        except Exception as e:
            print(f"Error processing image with BLIP: {e}")
            # Fallback to original text prompt if BLIP fails
            full_prompt = prompt
    elif input_image_b64 and (blip_processor is None or blip_model is None):
        print("BLIP model not loaded, skipping image-to-text conversion.")

    # 2. Generative AI Engine: Stable Diffusion
    print(f"Generating design with prompt: '{full_prompt}'")
    print(f"Negative prompt: '{negative_prompt}'")

    try:
        # Stable Diffusion Text-to-Image Generation
        # For a full img2img capability, a different pipeline (e.g., StableDiffusionImg2ImgPipeline)
        # or custom logic would be needed. This example focuses on text2img with image conditioning via BLIP.
        with torch.no_grad():
            generated_image = sd_pipeline(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]

        generated_image_b64 = image_to_base64(generated_image)

    except Exception as e:
        print(f"Error during Stable Diffusion generation: {e}")
        # Create a placeholder error image
        error_img = Image.new("RGB", (512, 512), color='red')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(error_img)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, 10), f"Error: {str(e)[:100]}...", (0,0,0), font=font)
        generated_image_b64 = image_to_base64(error_img)

    return {"generated_image_b64": generated_image_b64}

# --- Gradio Interface --- 
def customizer_interface(
    text_prompt: str,
    negative_prompt: str,
    input_image: Image.Image,
    guidance_scale: float,
    num_inference_steps: int
):
    """
    Gradio interface function to handle user inputs and call the FastAPI backend.
    """
    # Convert image to base64 if provided
    input_image_b64 = None
    if input_image:
        input_image_b64 = image_to_base64(input_image)

    try:
        import requests
        # Call FastAPI backend
        response = requests.post(
            "http://127.0.0.1:8000/generate_design/",
            data={
                "prompt": text_prompt,
                "negative_prompt": negative_prompt,
                "input_image_b64": input_image_b64,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps
            }
        )
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()

        if "error" in response_data:
            return None, response_data["error"]
        
        generated_image_b64 = response_data["generated_image_b64"]
        generated_image = base64_to_image(generated_image_b64)
        return generated_image, "Design generated successfully!"

    except requests.exceptions.ConnectionError:
        return None, "Error: Could not connect to the FastAPI backend. Is it running?"
    except requests.exceptions.HTTPError as e:
        return None, f"Error from backend: {e} - {e.response.text}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# Intelligent Product Customizer")
    gr.Markdown("Design and customize products using advanced multimodal AI.")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Main Design Prompt", 
                                    placeholder="e.g., A modern chair with a wooden frame and velvet upholstery", 
                                    lines=3)
            negative_input = gr.Textbox(label="Negative Prompt", 
                                      placeholder="e.g., ugly, broken, blurry, low resolution", 
                                      lines=2)
            image_input = gr.Image(type="pil", label="Upload Reference Image (Optional)", image_mode="RGB")
            
            with gr.Accordion("Advanced Settings", open=False):
                guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Guidance Scale (CFG)")
                steps_slider = gr.Slider(minimum=10, maximum=150, value=50, step=5, label="Number of Inference Steps")
            
            generate_button = gr.Button("Generate Design", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Generated Product Design", height=512, width=512)
            status_text = gr.Textbox(label="Status", interactive=False, lines=2)

    generate_button.click(
        customizer_interface,
        inputs=[text_input, negative_input, image_input, guidance_slider, steps_slider],
        outputs=[output_image, status_text]
    )

# --- Main Execution Block ---
def run_fastapi_server():
    """Runs the FastAPI server using Uvicorn."""
    # Ensure models are loaded before starting the server
    load_models()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    print("Starting Intelligent Product Customizer...")
    
    # Start FastAPI in a separate thread
    # This allows Gradio to launch in the main thread.
    fastapi_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    fastapi_thread.start()

    # Wait a moment for FastAPI to start (optional, but good practice)
    import time
    time.sleep(5) 
    print("FastAPI server started in background thread.")

    # Launch Gradio interface
    print("Launching Gradio interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    print("Gradio interface stopped.")

