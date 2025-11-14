from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import io
from PIL import Image
import base64

from design_generator import DesignGenerator
from virtual_tryon import VirtualTryOn
from image_to_text import ImageDescriber

app = FastAPI(
    title="AI Product Customization & Virtual Try-on",
    description="Platform for generating custom product designs and virtual try-ons using multimodal AI."
)

# Initialize AI modules
try:
    design_generator = DesignGenerator()
    virtual_tryon = VirtualTryOn()
    image_describer = ImageDescriber()
except Exception as e:
    print(f"Error initializing AI models: {e}")
    # In a production environment, you might want to handle this more gracefully
    # For now, we'll let the app potentially fail if models can't load.

class DesignRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    strength: Optional[float] = 0.8  # For image-to-image transformations

class TryOnRequest(BaseModel):
    product_image_b64: str # Base64 encoded product image
    user_image_b64: str # Base64 encoded user image

class ImageDescriptionResponse(BaseModel):
    description: str

@app.post("/generate-design")
async def generate_product_design(
    request: DesignRequest,
    input_image: Optional[UploadFile] = File(None)
):
    """Generates a custom product design based on text and optional input image."""
    try:
        output_image = None
        if input_image:
            image_bytes = await input_image.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            output_image = design_generator.transform_image(
                pil_image, request.prompt, request.negative_prompt, request.strength
            )
        else:
            output_image = design_generator.generate_from_text(
                request.prompt, request.negative_prompt
            )

        if output_image:
            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            encoded_image = base64.b64encode(img_byte_arr.read()).decode("utf-8")
            return {"design_image_b64": encoded_image}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate design.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Design generation failed: {e}")

@app.post("/virtual-tryon")
async def perform_virtual_tryon(request: TryOnRequest):
    """Performs a 2D virtual try-on of a product on a user's image."""
    try:
        product_image_bytes = base64.b64decode(request.product_image_b64)
        user_image_bytes = base64.b64decode(request.user_image_b64)

        product_image = Image.open(io.BytesIO(product_image_bytes)).convert("RGBA")
        user_image = Image.open(io.BytesIO(user_image_bytes)).convert("RGB")

        # Segment the person from the user image
        person_mask = virtual_tryon.segment_person(user_image)
        
        # Overlay the product onto the user image
        tryon_image = virtual_tryon.overlay_product_2d(user_image, product_image, person_mask)

        img_byte_arr = io.BytesIO()
        tryon_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        encoded_image = base64.b64encode(img_byte_arr.read()).decode("utf-8")

        return {"tryon_image_b64": encoded_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {e}")

@app.post("/describe-image", response_model=ImageDescriptionResponse)
async def describe_uploaded_image(image: UploadFile = File(...)):
    """Generates a textual description for an uploaded image."""
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        description = image_describer.describe_image(pil_image)
        return ImageDescriptionResponse(description=description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image description failed: {e}")

if __name__ == "__main__":
    # To run the API, use: uvicorn main:app --reload
    # For direct execution (e.g., in a container): uvicorn main:app --host 0.0.0.0 --port 8000
    print("To run the API, use: uvicorn main:app --reload or uvicorn main:app --host 0.0.0.0 --port 8000")
