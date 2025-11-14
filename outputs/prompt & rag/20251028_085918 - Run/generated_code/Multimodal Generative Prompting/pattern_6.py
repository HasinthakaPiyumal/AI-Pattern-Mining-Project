import base64
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# Define the FastAPI application
app = FastAPI(
    title="Intelligent Product Customization and Design Assistant",
    description="Leveraging Advanced Multimodal Prompting for E-commerce Product Design."
)

# Load the Stable Diffusion models
# NOTE: This will download a large model the first time it's run.
# Ensure you have logged into Hugging Face and accepted the terms for the model.
# from huggingface_hub import login
# login() # Run this once if you haven't logged in via CLI

# Using a standard v1-5 model for broader compatibility
try:
    # Text-to-Image pipeline
    text2image_pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        text2image_pipeline.to("cuda")

    # Image-to-Image pipeline
    img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        img2img_pipeline.to("cuda")

except Exception as e:
    print(f"Failed to load Stable Diffusion models: {e}")
    print("Please ensure you have authenticated with Hugging Face (huggingface-cli login) and accepted the model license.")
    text2image_pipeline = None
    img2img_pipeline = None


class DesignInput(BaseModel):
    text_prompt: str
    negative_prompt: Optional[str] = None
    image_input_base64: Optional[str] = None  # Base64 encoded image for img2img
    three_d_annotation_text: Optional[str] = None # Textual guidance for 3D elements (e.g., "centered", "perspective")
    guidance_scale: float = 7.5 # How strongly the prompt influences the generation
    strength: float = 0.8 # For img2img, how much to transform the input image


@app.post("/generate_product_design")
async def generate_product_design(design_input: DesignInput):
    """
    Generates a product design based on multimodal input.
    Supports text prompts, negative prompts, image-to-image transformation,
    and textual guidance for 3D characteristics.
    """
    if not text2image_pipeline or not img2img_pipeline:
        raise HTTPException(status_code=500, detail="AI models not loaded. Server might be initializing or encountered an error.")

    full_prompt = design_input.text_prompt
    if design_input.three_d_annotation_text:
        full_prompt = f"{full_prompt}, {design_input.three_d_annotation_text}"

    generated_image_pil = None
    explanation = ""

    if design_input.image_input_base64:
        # Perform Image-to-Image generation
        try:
            image_data = base64.b64decode(design_input.image_input_base64)
            init_image = Image.open(BytesIO(image_data)).convert("RGB")
            init_image = init_image.resize((512, 512)) # Resize for better performance and consistency

            generated_image_pil = img2img_pipeline(
                prompt=full_prompt,
                image=init_image,
                negative_prompt=design_input.negative_prompt,
                guidance_scale=design_input.guidance_scale,
                strength=design_input.strength
            ).images[0]
            explanation = "Product design generated using image-to-image transformation based on your input image and text prompts."
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing input image for img2img: {e}")
    else:
        # Perform Text-to-Image generation
        generated_image_pil = text2image_pipeline(
            prompt=full_prompt,
            negative_prompt=design_input.negative_prompt,
            guidance_scale=design_input.guidance_scale
        ).images[0]
        explanation = "Product design generated from your text prompts, including any 3D annotations."

    # Encode the generated image to base64
    buffered = BytesIO()
    generated_image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"filename": "generated_product_design.png", "code": img_str, "explanation": explanation}
