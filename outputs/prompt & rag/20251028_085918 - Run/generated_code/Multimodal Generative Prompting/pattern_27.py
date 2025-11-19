from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, HttpUrl
import base64
from typing import Optional, List

app = FastAPI()

# --- Pydantic Models for Request Bodies ---

class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    modifiers: Optional[List[str]] = None

class PairedImagePromptRequest(BaseModel):
    base_image_url: HttpUrl
    reference_image_url: HttpUrl

class ImageAsTextPromptRequest(BaseModel):
    image_base64: str  # Base64 encoded image
    additional_text_prompt: Optional[str] = None

class ThreeDGenerationRequest(BaseModel):
    text_prompt: str
    reference_image_url: Optional[HttpUrl] = None

class SegmentationPromptRequest(BaseModel):
    image_base64: str # Base64 encoded image
    segmentation_prompt: str # e.g., "shirt", "collar", "left sleeve"

# --- Mock AI Model Interactions ---

def mock_image_generator(prompt: str, negative_prompt: Optional[str], modifiers: Optional[List[str]]) -> str:
    print(f"Mock: Generating image with prompt='{prompt}', negative='{negative_prompt}', modifiers='{modifiers}'")
    # In a real app, this would call diffusers/Stable Diffusion
    return f"http://mock-cdn.com/generated-image-{hash(prompt + str(negative_prompt) + str(modifiers))}.png"

def mock_paired_image_processor(base_image_url: str, reference_image_url: str) -> str:
    print(f"Mock: Processing paired images for style transfer/embedding search. Base: {base_image_url}, Ref: {reference_image_url}")
    # In a real app, this would use CLIP for embeddings and then guide generation
    return f"http://mock-cdn.com/styled-image-{hash(base_image_url + reference_image_url)}.png"

def mock_image_captioner(image_base64: str) -> str:
    print(f"Mock: Captioning image (first 50 chars of base64): {image_base64[:50]}...")
    # In a real app, this would use BLIP/BLIP2
    return "a highly detailed product image of a modern minimalist chair"

def mock_3d_generator(text_prompt: str, reference_image_url: Optional[str]) -> str:
    print(f"Mock: Generating 3D model with prompt='{text_prompt}', ref_image='{reference_image_url}'")
    # In a real app, this would use NeRF-based models or other 3D generative AI
    return f"http://mock-cdn.com/generated-3d-model-{hash(text_prompt + str(reference_image_url))}.glb"

def mock_segmentation_model(image_base64: str, segmentation_prompt: str) -> str:
    print(f"Mock: Segmenting image (first 50 chars of base64): {image_base64[:50]}... for '{segmentation_prompt}'")
    # In a real app, this would use SAM/Detectron2
    return f"http://mock-cdn.com/segmentation-mask-{hash(image_base64 + segmentation_prompt)}.png"

# --- FastAPI Endpoints ---

@app.post("/generate-image")
async def generate_image(request: ImageGenerationRequest):
    try:
        generated_image_url = mock_image_generator(
            request.prompt, request.negative_prompt, request.modifiers
        )
        return {"image_url": generated_image_url, "explanation": "Image generated using diffusion model with prompts."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/paired-image-prompt")
async def paired_image_prompt(request: PairedImagePromptRequest):
    try:
        result_image_url = mock_paired_image_processor(
            str(request.base_image_url), str(request.reference_image_url)
        )
        return {"image_url": result_image_url, "explanation": "Style or texture transferred based on reference image using CLIP embeddings."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image-as-text-prompt")
async def image_as_text_prompt(request: ImageAsTextPromptRequest):
    try:
        # Step 1: Convert image to text description
        image_description = mock_image_captioner(request.image_base64)

        # Step 2: Use the description (and optional additional prompt) to generate a new image
        full_prompt = f"{image_description}. {request.additional_text_prompt or ''}".strip()
        generated_image_url = mock_image_generator(full_prompt, None, None)

        return {"image_url": generated_image_url, "image_description": image_description, "explanation": "Image converted to text, then used for new image generation."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-3d")
async def generate_3d_model(request: ThreeDGenerationRequest):
    try:
        generated_3d_url = mock_3d_generator(request.text_prompt, str(request.reference_image_url) if request.reference_image_url else None)
        return {"model_url": generated_3d_url, "explanation": "3D model generated or manipulated based on prompt."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment-product")
async def segment_product(request: SegmentationPromptRequest):
    try:
        segmentation_mask_url = mock_segmentation_model(request.image_base64, request.segmentation_prompt)
        return {"mask_url": segmentation_mask_url, "explanation": "Product region segmented based on prompt."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "AI Product Design Backend is running"}

# To run this application:
# 1. Save the code as main.py
# 2. Install dependencies: pip install fastapi uvicorn pydantic
# 3. Run: uvicorn main:app --reload
# 4. Access at http://127.0.0.1:8000/docs for interactive API documentation.