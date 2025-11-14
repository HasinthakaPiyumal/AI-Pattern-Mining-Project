import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

app = FastAPI(
    title="Intelligent Product Customization and Virtual Try-On Platform",
    description="API for multimodal product customization and virtual try-on.",
)

# --- Pydantic Models ---

class MultimodalInput(BaseModel):
    text_prompt: str = Field(..., description="Text description for the desired garment customization.")
    positive_image_examples: list[str] = Field(default_factory=list, description="List of base64 encoded positive image examples.")
    negative_text_prompt: str = Field(default="", description="Text prompt for elements to avoid in the generation.")
    
class DesignGenerationOutput(BaseModel):
    generated_design_image_b64: str = Field(..., description="Base64 encoded generated 2D garment design image.")
    explanation: str = Field(..., description="Explanation of the generated design.")

class VirtualTryOnInput(BaseModel):
    garment_design_image_b64: str = Field(..., description="Base64 encoded garment design image to try on.")
    avatar_image_b64: str = Field(..., description="Base64 encoded user avatar image for try-on.")

class VirtualTryOnOutput(BaseModel):
    try_on_image_b64: str = Field(..., description="Base64 encoded image of the garment on the avatar.")
    explanation: str = Field(..., description="Explanation of the virtual try-on result.")

# --- Utility Functions ---

def decode_base64_image(base64_string: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG") # Use PNG for lossless quality
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Placeholder AI Modules ---

def generate_2d_garment_design(
    text_prompt: str,
    positive_image_examples: list[Image.Image],
    negative_text_prompt: str
) -> Image.Image:
    """
    Placeholder for 2D garment design generation using a diffusion model (e.g., Hugging Face diffusers).
    In a real application, this would involve loading and running a sophisticated model.
    For demonstration, it creates a simple placeholder image based on the prompt.
    """
    # In a real scenario, integrate with a model like:
    # from diffusers import StableDiffusionPipeline
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # image = pipe(text_prompt, negative_prompt=negative_text_prompt).images[0]
    
    # Placeholder: Create a simple image with text
    width, height = 512, 512
    img = Image.new("RGB", (width, height), color = 'lightblue')
    # You would typically draw or generate content here
    # For simplicity, let's just add some text indicating the prompt
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 30) # Path to a font file
    except IOError:
        font = ImageFont.load_default()
    
    d.text((50,50), f"Design for: {text_prompt}", fill=(0,0,0), font=font)
    if negative_text_prompt: 
        d.text((50,100), f"Avoiding: {negative_text_prompt}", fill=(255,0,0), font=font)
    if positive_image_examples:
        d.text((50,150), f"Using {len(positive_image_examples)} image examples", fill=(0,0,255), font=font)

    return img

def perform_virtual_try_on(
    garment_design_image: Image.Image,
    avatar_image: Image.Image
) -> Image.Image:
    """
    Placeholder for virtual try-on functionality.
    In a real application, this would involve advanced image manipulation or a dedicated try-on model (e.g., using OpenCV for segmentation/blending, or a specialized GAN).
    For demonstration, it simply overlays the garment onto the avatar with basic resizing.
    """
    # Resize garment to fit avatar (this is a very simplistic approach)
    # In a real app, you'd perform segmentation, pose estimation, and sophisticated blending.
    garment_resized = garment_design_image.resize((avatar_image.width // 2, avatar_image.height // 2))

    # Create a copy of the avatar image to draw on
    try_on_image = avatar_image.copy()

    # Calculate position to paste (e.g., center-ish)
    paste_x = (try_on_image.width - garment_resized.width) // 2
    paste_y = (try_on_image.height - garment_resized.height) // 2

    # Paste the garment onto the avatar
    try_on_image.paste(garment_resized, (paste_x, paste_y), garment_resized if garment_resized.mode == "RGBA" else None)

    return try_on_image

# --- FastAPI Endpoints ---

@app.post("/generate_design", response_model=DesignGenerationOutput)
async def generate_design(input: MultimodalInput):
    """
    Generates a 2D garment design based on multimodal prompts.
    """
    positive_images = []
    for img_b64 in input.positive_image_examples:
        positive_images.append(decode_base64_image(img_b64))

    generated_img = generate_2d_garment_design(
        text_prompt=input.text_prompt,
        positive_image_examples=positive_images,
        negative_text_prompt=input.negative_text_prompt
    )

    return DesignGenerationOutput(
        generated_design_image_b64=encode_image_to_base64(generated_img),
        explanation=f"Generated a design based on your prompt: '{input.text_prompt}'. "
                    f"Image examples used: {len(positive_images)}. "
                    f"Negative prompt used: '{input.negative_text_prompt or 'None'}'. "
                    "This is a conceptual generation; a real system would use a diffusion model."
    )

@app.post("/try_on", response_model=VirtualTryOnOutput)
async def try_on_garment(input: VirtualTryOnInput):
    """
    Performs a virtual try-on of a generated garment design onto a user avatar.
    """
    garment_img = decode_base64_image(input.garment_design_image_b64)
    avatar_img = decode_base64_image(input.avatar_image_b64)

    try_on_result_img = perform_virtual_try_on(
        garment_design_image=garment_img,
        avatar_image=avatar_img
    )

    return VirtualTryOnOutput(
        try_on_image_b64=encode_image_to_base64(try_on_result_img),
        explanation="Successfully performed a conceptual virtual try-on. "
                    "A real system would employ advanced computer vision techniques for realistic fitting."
    )

@app.get("/")
async def root():
    return {"message": "Welcome to the Intelligent Product Customization and Virtual Try-On Platform API. Refer to /docs for API documentation."}


# To run this application:
# 1. Save the code as main.py
# 2. Install necessary libraries: pip install fastapi uvicorn Pillow
# 3. Run from your terminal: uvicorn main:app --reload
# 4. Access the API documentation at http://127.0.0.1:8000/docs