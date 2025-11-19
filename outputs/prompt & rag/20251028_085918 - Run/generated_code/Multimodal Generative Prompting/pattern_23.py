from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import uuid
import time

app = FastAPI(title="E-commerce AI Platform Backend Services")

# --- Pydantic Models for Request/Response Bodies ---

class PromptProcessRequest(BaseModel):
    text_prompt: str
    negative_prompt: str = ""
    modifiers: list[str] = []

class PromptProcessResponse(BaseModel):
    processed_instructions: dict

class ImageToTextRequest(BaseModel):
    image_url: str

class ImageToTextResponse(BaseModel):
    text_description: str

class GenerateImageRequest(BaseModel):
    text_prompt: str
    negative_prompt: str = ""
    modifiers: list[str] = []
    paired_image_url: str | None = None
    product_id: str | None = None # For associating generated images with products

class GenerateImageResponse(BaseModel):
    generated_image_url: str
    explanation: str

class Generate3DModelRequest(BaseModel):
    text_prompt: str
    model_type: str # e.g., "lamp", "ring", "chair"
    manipulation_prompt: str | None = None # e.g., "enlarge this diamond", "change material to wood"
    product_id: str | None = None # For associating generated 3D models with products

class Generate3DModelResponse(BaseModel):
    generated_3d_model_url: str
    explanation: str

class SegmentImageRequest(BaseModel):
    image_url: str
    target_object: str # e.g., "person", "clothing", "room_furniture"

class SegmentImageResponse(BaseModel):
    segmented_image_url: str
    mask_url: str
    explanation: str

# --- Backend Services (FastAPI Endpoints) ---

@app.post("/prompt_process", response_model=PromptProcessResponse)
async def prompt_processing_service(request: PromptProcessRequest):
    """Handles text-based prompts, including negative prompts and modifiers."""
    print(f"Received prompt processing request: {request.text_prompt}")
    # Simulate NLP processing and instruction generation
    instructions = {
        "core_idea": request.text_prompt,
        "exclude": request.negative_prompt.split(',') if request.negative_prompt else [],
        "include_styles": request.modifiers,
        "action": "generate_image" # Default action, could be more complex
    }
    return PromptProcessResponse(processed_instructions=instructions)

@app.post("/image_to_text", response_model=ImageToTextResponse)
async def image_to_text_service(request: ImageToTextRequest):
    """Converts an uploaded image into a descriptive text."""
    print(f"Received image-to-text request for image: {request.image_url}")
    # Simulate image captioning using a placeholder model
    time.sleep(1) # Simulate processing time
    dummy_descriptions = {
        "https://example.com/shoe.jpg": "A pair of red sneakers with white laces.",
        "https://example.com/room.jpg": "A modern living room with a grey sofa and a wooden coffee table.",
        "https://example.com/dress.jpg": "A blue floral summer dress with short sleeves."
    }
    description = dummy_descriptions.get(request.image_url, "A generic product image.")
    return ImageToTextResponse(text_description=description)

@app.post("/generate_image", response_model=GenerateImageResponse)
async def image_generation_service(request: GenerateImageRequest):
    """Generates customized product images or performs virtual try-on/placement."""
    print(f"Received image generation request for prompt: {request.text_prompt}")
    # Simulate image generation using diffusers/Stable Diffusion
    # For paired image prompting, this would involve using the paired_image_url
    # to guide the generation or perform image-to-image translation.
    time.sleep(2) # Simulate generation time
    unique_id = uuid.uuid4()
    generated_url = f"https://example.com/generated_images/{unique_id}.png"
    explanation = f"Generated image based on '{request.text_prompt}' with negative prompts '{request.negative_prompt}' and modifiers '{', '.join(request.modifiers)}'."
    if request.paired_image_url:
        explanation += f" Used paired image '{request.paired_image_url}' for context."
    return GenerateImageResponse(generated_image_url=generated_url, explanation=explanation)

@app.post("/generate_3d_model", response_model=Generate3DModelResponse)
async def generate_3d_model_service(request: Generate3DModelRequest):
    """Generates or manipulates 3D product models based on text prompts."""
    print(f"Received 3D model generation request for: {request.model_type} with prompt: {request.text_prompt}")
    # Simulate 3D model generation using specialized libraries (e.g., latent-diffusion-3D)
    time.sleep(3) # Simulate longer generation time for 3D
    unique_id = uuid.uuid4()
    generated_url = f"https://example.com/3d_models/{request.model_type}_{unique_id}.gltf"
    explanation = f"Generated 3D model of a '{request.model_type}' based on '{request.text_prompt}'."
    if request.manipulation_prompt:
        explanation += f" Applied manipulation: '{request.manipulation_prompt}'."
    return Generate3DModelResponse(generated_3d_model_url=generated_url, explanation=explanation)

@app.post("/segment_image", response_model=SegmentImageResponse)
async def segmentation_service(request: SegmentImageRequest):
    """Performs precise object delineation in images for virtual try-on or product placement."""
    print(f"Received segmentation request for image: {request.image_url} targeting: {request.target_object}")
    # Simulate image segmentation using a placeholder model (e.g., Mask R-CNN)
    time.sleep(1.5) # Simulate processing time
    unique_id = uuid.uuid4()
    segmented_url = f"https://example.com/segmented_images/{unique_id}_segmented.png"
    mask_url = f"https://example.com/segmented_images/{unique_id}_mask.png"
    explanation = f"Segmented '{request.target_object}' from image '{request.image_url}'."
    return SegmentImageResponse(segmented_image_url=segmented_url, mask_url=mask_url, explanation=explanation)

@app.get("/")
async def root():
    return {"message": "E-commerce AI Platform Backend is running! Access /docs for API documentation."}

if __name__ == "__main__":
    # To run this, save it as main.py and execute: uvicorn main:app --reload --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
