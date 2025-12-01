import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# --- Configuration ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "generated_images"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.to("cuda")
    return pipe

pipe = load_model()

st.title("🛒 AI-powered E-commerce Product Image Generator")
st.markdown("Generate high-quality product images with negative prompting to avoid undesired elements.")

# --- User Inputs ---
product_description = st.text_area(
    "Product Description (e.g., 'a red leather handbag')",
    "a sleek silver smartphone"
)

scene_style = st.text_area(
    "Desired Scene/Style (e.g., 'studio lighting, minimalist background')",
    "on a pristine white pedestal, soft ambient lighting, bokeh background"
)

negative_prompt = st.text_area(
    "Negative Prompts (elements to avoid, comma-separated e.g., 'blurry, poor lighting, watermark')",
    "blurry, low resolution, watermark, poor lighting, distorted, bad anatomy, ugly, tiling, poorly drawn hands"
)

if st.button("Generate Product Image"):
    if product_description and scene_style:
        st.info("Generating image... This may take a moment.")
        full_prompt = f"{product_description}, {scene_style}"

        try:
            with st.spinner("Creating your masterpiece..."):
                image = pipe(prompt=full_prompt, negative_prompt=negative_prompt, num_inference_steps=25).images[0]
            
            st.success("Image generated successfully!")
            st.image(image, caption="Generated Product Image", use_column_width=True)

            # Save image locally
            filename = f"product_image_{len(os.listdir(OUTPUT_DIR)) + 1}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            image.save(filepath)
            st.success(f"Image saved as {filepath}")

            with open(filepath, "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name=filename,
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"An error occurred during image generation: {e}")
    else:
        st.warning("Please provide both a product description and a desired scene/style.")
