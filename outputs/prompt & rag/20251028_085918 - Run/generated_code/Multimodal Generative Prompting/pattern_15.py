from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
import io
import base64

app = FastAPI(
    title="Advanced Multimodal Fashion Design & Virtual Try-on",
    description="An e-commerce application leveraging advanced multimodal prompting for personalized fashion design and virtual try-on."
)

# --- Pydantic Models for API Request/Response ---
class DesignRequest(BaseModel):
    text_prompt: str = Field(..., example="an elegant evening gown, flowing")
    negative_prompt: str = Field(None, example="no stripes, less vibrant color")
    # image_reference_base64: Optional[str] = None # Handled as UploadFile
    # avatar_3d_scan_base64: Optional[str] = None # Handled as UploadFile

class DesignResponse(BaseModel):
    design_image_base64: str = Field(..., description="Base64 encoded image of the generated fashion design.")
    try_on_image_base64: str = Field(..., description="Base64 encoded image of the virtual try-on on an avatar.")
    explanation: str = Field(..., description="Explanation of the generated design and try-on.")

# --- Helper Functions ---
def decode_base64_image(base64_string: str) -> Image.Image:
    try:
        img_bytes = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image string: {e}")

def encode_image_to_base64(image_pil: Image.Image) -> str:
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG") # Use PNG for better quality and transparency if needed
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def mock_generate_dummy_image(text_input: str, width: int = 512, height: int = 512, background_color=(200, 200, 200), text_color=(0, 0, 0)) -> Image.Image:
    """Generates a dummy image with the given text input."""
    img = Image.new('RGB', (width, height), color = background_color)
    d = ImageDraw.Draw(img)
    try:
        # Try to use a default font if available, otherwise fall back
        font = ImageFont.truetype("arial.ttf", 20) # This might fail if 'arial.ttf' is not found
    except IOError:
        font = ImageFont.load_default() # Fallback font

    text = f"Design: {text_input}"
    # Calculate text size to center it
    bbox = d.textbbox((0,0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) / 2
    y = (height - text_height) / 2

    d.text((x, y), text, fill=text_color, font=font)
    return img


# --- Mock AI Modules (Replace with actual ML model calls in a real application) ---

def mock_multimodal_input_processor(
    text_prompt: str,
    image_ref_pil: Image.Image = None,
    avatar_3d_scan_data: bytes = None
) -> str:
    """
    Mocks the processing of multimodal inputs. In a real scenario, this would
    use models like CLIP/BLIP-2 to generate embeddings or descriptive text from images.
    """
    processed_prompt = f"Text description: {text_prompt}"
    if image_ref_pil:
        # In a real system: image_to_text_model.predict(image_ref_pil)
        # Or: image_to_embedding_model.encode(image_ref_pil)
        processed_prompt += "\nImage reference provided: (simulated description of style/pattern from image)"
    if avatar_3d_scan_data:
        # In a real system: 3d_scan_processor.process(avatar_3d_scan_data)
        processed_prompt += "\n3D avatar scan provided for fitting."
    return processed_prompt

def mock_generative_fashion_ai_model(processed_prompt: str, negative_prompt: str = None) -> Image.Image:
    """
    Mocks the generative AI model for fashion design. In a real system, this would
    be a Diffusion model (e.g., Stable Diffusion) taking the processed prompt.
    """
    full_prompt_for_gen = processed_prompt
    if negative_prompt:
        full_prompt_for_gen += f"\n(Negative aspects: {negative_prompt})"

    print(f"[Mock AI] Generating design with prompt: {full_prompt_for_gen}")
    # Simulate design generation based on prompt
    design_image = mock_generate_dummy_image(f"Generated Fashion Design based on: {full_prompt_for_gen}")
    return design_image

def mock_virtual_try_on_module(garment_image_pil: Image.Image, avatar_3d_scan_data: bytes = None) -> Image.Image:
    """
    Mocks the virtual try-on process. In a real system, this would involve 3D
    draping and rendering.
    """
    print(f"[Mock AI] Performing virtual try-on...")
    # Simulate try-on. For simplicity, we'll just overlay a text indicating try-on.
    # In a real application, this would use the avatar_3d_scan_data to render the garment on a 3D avatar.
    try_on_image = Image.new('RGB', (garment_image_pil.width, garment_image_pil.height), color = (150, 200, 255)) # Blueish background for avatar
    d = ImageDraw.Draw(try_on_image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    text = "Virtual Try-on: Your Avatar Here!"
    bbox = d.textbbox((0,0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (try_on_image.width - text_width) / 2
    y = (try_on_image.height - text_height) / 2

    d.text((x, y), text, fill=(255, 255, 255), font=font)

    # Optionally, superimpose a small version of the garment image
    garment_resized = garment_image_pil.resize((garment_image_pil.width // 3, garment_image_pil.height // 3))
    try_on_image.paste(garment_resized, (try_on_image.width - garment_resized.width - 20, 20))

    return try_on_image


# --- FastAPI Endpoints ---

@app.get("/", tags=["Health Check"])
async def read_root():
    return {"message": "Welcome to the Multimodal Fashion Design API!"}

@app.post("/design_and_try_on", response_model=DesignResponse, tags=["Fashion Design"])
async def design_and_try_on(
    text_prompt: str = Form(..., description="Textual description of the desired clothing style."),
    negative_prompt: str = Form(None, description="Negative prompt to avoid undesired elements."),
    image_reference: UploadFile = File(None, description="Optional image reference for style, pattern, or fabric."),
    avatar_3d_scan: UploadFile = File(None, description="Optional 3D scan of the user for virtual try-on.")
):
    """
    Generates a personalized fashion design and performs a virtual try-on based on multimodal input.

    - **text_prompt**: A textual description of the desired garment (e.g., "an elegant evening gown").
    - **negative_prompt**: Text describing elements to avoid (e.g., "no stripes, less vibrant color").
    - **image_reference**: An optional image file (PNG, JPG) providing visual inspiration.
    - **avatar_3d_scan**: An optional 3D file (e.g., .obj, .glb - mocked here) of the user's body for try-on.
    """
    image_ref_pil = None
    if image_reference:
        try:
            image_contents = await image_reference.read()
            image_ref_pil = Image.open(io.BytesIO(image_contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not process image_reference: {e}")

    avatar_scan_data = None
    if avatar_3d_scan:
        try:
            avatar_scan_data = await avatar_3d_scan.read()
            # In a real app, validate and process 3D data
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not process avatar_3d_scan: {e}")

    # 1. Multimodal Input Processor
    processed_prompt = mock_multimodal_input_processor(text_prompt, image_ref_pil, avatar_scan_data)

    # 2. Generative Fashion AI Model
    generated_design_image = mock_generative_fashion_ai_model(processed_prompt, negative_prompt)

    # 3. Virtual Try-on Module
    try_on_image = mock_virtual_try_on_module(generated_design_image, avatar_scan_data)

    # Encode images to base64 for response
    design_b64 = encode_image_to_base64(generated_design_image)
    try_on_b64 = encode_image_to_base64(try_on_image)

    explanation = (
        f"Fashion design generated based on your request: '{text_prompt}'. "
        f"Negative prompts '{negative_prompt or 'None'}' were considered. "
        f"{'An image reference was used. ' if image_ref_pil else ''}"
        f"{'A 3D avatar scan was used for try-on.' if avatar_scan_data else 'A default avatar was used for try-on.'}"
    )

    return DesignResponse(
        design_image_base64=design_b64,
        try_on_image_base64=try_on_b64,
        explanation=explanation
    )

# To run this application:
# 1. Save the code as `main.py`.
# 2. Install necessary libraries: `pip install fastapi uvicorn pillow`
# 3. Run from your terminal: `uvicorn main:app --reload`
# 4. Access the API documentation at `http://127.0.0.1:8000/docs` to test the endpoint.
