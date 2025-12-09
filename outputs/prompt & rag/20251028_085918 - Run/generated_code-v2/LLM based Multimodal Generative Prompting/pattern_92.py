import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

@st.cache_resource
def load_model():
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipeline.to("cuda")
    return pipeline

st.set_page_config(layout="wide")
st.title("E-commerce Product Image Generator with Negative Prompting")
st.markdown("Generate high-quality product images by specifying what you *want* and what you *don't want*.")

pipeline = load_model()

with st.sidebar:
    st.header("Image Generation Controls")
    guidance_scale = st.slider("Guidance Scale (CFG)", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
    num_inference_steps = st.slider("Number of Inference Steps", min_value=10, max_value=100, value=50, step=5)
    seed = st.number_input("Seed (for reproducibility)", value=None, placeholder="Leave empty for random", min_value=0)

positive_prompt = st.text_area(
    "Enter your **positive prompt** (what you want to see):",
    "A sleek, silver smartwatch on a minimalist wooden desk with natural lighting, high resolution, studio quality photo"
)

negative_prompt = st.text_area(
    "Enter your **negative prompt** (what you *don't* want to see):",
    "blurry, pixelated, distorted logo, bad reflections, busy background, unprofessional, text, watermark, low quality, cartoon, ugly, deformed, noisy"
)

if st.button("Generate Product Image", type="primary"):
    if positive_prompt:
        st.spinner("Generating your image... This may take a moment.")
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
        
        with st.empty():
            st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXJpMDY4aDNkbjR5b2U5MXo2eGFoZnJvOHp4cjVpYmF0cHVtdDlmMSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3y0oet7lstgKj0cfzE/giphy.gif", caption="Generating image...", use_column_width=True)
            
            try:
                image = pipeline(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                ).images[0]
                st.image(image, caption="Generated Product Image", use_column_width=True)
            except Exception as e:
                st.error(f"An error occurred during image generation: {e}")
    else:
        st.warning("Please enter a positive prompt to generate an image.")
