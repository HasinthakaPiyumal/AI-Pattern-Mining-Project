from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image
import io
from diffusers import StableDiffusionImg2ImgPipeline
import torch

app = FastAPI()

# Load the pre-trained diffusion model
# You might need to adjust the model name based on your specific requirements
# Ensure you have logged into Hugging Face if the model requires authentication
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipeline = pipeline.to(device)

@app.post("/transform-image/")
async def transform_image(file: UploadFile = File(...), prompt: str = ""):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read the uploaded image
        image_bytes = await file.read()
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Optional: Resize image if needed by the model or for consistency
        # init_image = init_image.resize((768, 512))

        # Perform the image transformation
        transformed_image = pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]

        # Save the transformed image to bytes
        output_bytes = io.BytesIO()
        transformed_image.save(output_bytes, format="PNG")
        output_bytes.seek(0)

        return Response(content=output_bytes.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image transformation failed: {str(e)}")
