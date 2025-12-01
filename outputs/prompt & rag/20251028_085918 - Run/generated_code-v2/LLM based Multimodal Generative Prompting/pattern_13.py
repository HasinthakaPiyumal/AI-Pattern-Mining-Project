import base64
from io import BytesIO
from typing import List

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from PIL import Image
from pydantic import BaseModel

# --- AI Model Service ---
class ImageTransformationModel:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU. This may be slow.")
            device = "cpu"
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        self.pipeline.to(device)

    def transform_image(self, init_image: Image.Image, prompt: str) -> Image.Image:
        # Ensure the image is in RGB format if it's not already
        if init_image.mode != "RGB":
            init_image = init_image.convert("RGB")

        # Resize the image to a common size suitable for the model, if necessary
        # The StableDiffusionImg2ImgPipeline often works well with square images
        # For simplicity, we'll just use the original size for now, but in a real app
        # you might want to preprocess for optimal results.
        # For img2img, the model expects an init_image of a certain size (e.g., 512x512, 768x768)
        # Let's resize it to 512x512 for demonstration purposes.
        # Note: Depending on the model, different sizes might be optimal.
        width, height = init_image.size
        if width != 512 or height != 512:
            init_image = init_image.resize((512, 512), Image.LANCZOS)

        transformed_images = self.pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
        return transformed_images[0] if transformed_images else init_image

# --- FastAPI Application ---
app = FastAPI(
    title="E-commerce Product Image Editor",
    description="AI-powered service for transforming product images using PairedImage Prompting concept."
)

# Initialize the AI model service
try:
    image_transformer = ImageTransformationModel()
except Exception as e:
    print(f"Error initializing ImageTransformationModel: {e}")
    print("Ensure 'runwayml/stable-diffusion-v1-5' model is downloadable and torch is configured correctly.")
    # Optionally, you might want to raise the exception or have a fallback
    image_transformer = None # For graceful handling if model fails to load

class TransformRequest(BaseModel):
    transformation_description: str

class TransformedImage(BaseModel):
    filename: str
    image_base64: str

class TransformResponse(BaseModel):
    transformed_images: List[TransformedImage]

@app.post("/transform_product_images", response_model=TransformResponse)
async def transform_product_images(
    transformation_description: str,
    new_images: List[UploadFile] = File(..., description="List of raw product images to transform")
):
    if image_transformer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI image transformation model is not loaded or available."
        )

    if not new_images:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No new images provided for transformation.")

    transformed_results = []
    for img_file in new_images:
        try:
            # Read image content
            content = await img_file.read()
            # Open image with PIL
            init_image = Image.open(BytesIO(content))

            # Perform transformation
            transformed_image = image_transformer.transform_image(init_image, transformation_description)

            # Save transformed image to bytes buffer and encode to base64
            buffered = BytesIO()
            transformed_image.save(buffered, format="PNG") # Using PNG for general quality
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            transformed_results.append(TransformedImage(filename=img_file.filename, image_base64=img_base64))
        except Exception as e:
            print(f"Error processing image {img_file.filename}: {e}")
            # Optionally, return an error for this specific image or skip it
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process image {img_file.filename}: {str(e)}"
            )
    
    return TransformResponse(transformed_images=transformed_results)

# To run the application:
# 1. Save this code as main.py
# 2. Install dependencies: pip install "fastapi[all]" torch diffusers accelerate transformers pillow
# 3. Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
