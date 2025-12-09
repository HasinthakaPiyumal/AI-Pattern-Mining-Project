##### requirements.txt #####
streamlit
fastapi
uvicorn
torch
diffusers
transformers
accelerate
Pillow

##### generator.py #####
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe = pipe.to(device)

def generate_product_image(positive_prompt: str, negative_prompt: str) -> Image.Image:
    image = pipe(positive_prompt, negative_prompt=negative_prompt).images[0]
    return image

##### api.py #####
from fastapi import FastAPI
from pydantic import BaseModel
import io
import base64
from PIL import Image
import generator

app = FastAPI()

class ImageRequest(BaseModel):
    positive_prompt: str
    negative_prompt: str

@app.post("/generate_image")
async def generate_image_api(request: ImageRequest):
    generated_image = generator.generate_product_image(
        positive_prompt=request.positive_prompt,
        negative_prompt=request.negative_prompt
    )

    buffered = io.BytesIO()
    generated_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return {"image": img_str}

##### app.py #####
import streamlit as st
import requests
import base64
import io
from PIL import Image

st.set_page_config(layout="wide", page_title="AI Product Image Generator")

st.title("AI Product Image Generator with Negative Prompting")
st.markdown("Generate stunning product images and exclude undesirable elements.")

col1, col2 = st.columns(2)

with col1:
    st.header("Prompts")
    positive_prompt = st.text_area(
        "Positive Prompt (What you want to see)",
        "A sleek smartphone on a minimalist white background, studio lighting, high resolution, professional product photography",
        height=150
    )
    negative_prompt = st.text_area(
        "Negative Prompt (What you DON'T want to see)",
        "watermark, blurry, bad reflections, busy background, distorted, extra limbs, ugly, disfigured, poor quality, low resolution, bad hands, text",
        height=150
    )

    if st.button("Generate Image"):
        if not positive_prompt:
            st.warning("Please enter a positive prompt.")
        else:
            with st.spinner("Generating image... This may take a moment."):
                try:
                    response = requests.post(
                        "http://localhost:8000/generate_image",
                        json={
                            "positive_prompt": positive_prompt,
                            "negative_prompt": negative_prompt
                        },
                        timeout=300
                    )
                    response.raise_for_status()
                    result = response.json()
                    image_base64 = result["image"]

                    image_bytes = base64.b64decode(image_base64)
                    generated_image = Image.open(io.BytesIO(image_bytes))

                    with col2:
                        st.header("Generated Image")
                        st.image(generated_image, caption="Generated Product Image", use_column_width=True)

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend API. Please ensure the FastAPI server is running at http://localhost:8000.")
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

with col2:
    st.header("Generated Image Preview")
    st.write("Your generated image will appear here.")