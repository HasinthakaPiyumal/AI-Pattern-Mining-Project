import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

st.set_page_config(layout="wide", page_title="E-commerce Product Image Generator")

st.title("🛒 AI-Powered E-commerce Product Image Generator")
st.markdown("Generate high-quality product images using AI, avoiding undesirable elements with negative prompts.")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipeline = pipeline.to(device)
    return pipeline

pipeline = load_model()

with st.sidebar:
    st.header("Image Generation Settings")
    positive_prompt = st.text_area(
        "Positive Prompt",
        "A beautiful product shot of a modern smartphone, clean white background, professional studio lighting, high resolution, sharp focus, vibrant colors, minimalist design"
    )
    negative_prompt = st.text_area(
        "Negative Prompt",
        "blurry, out of focus, distorted, bad shadows, inconsistent lighting, incorrect product colors, multiple hands, bad anatomy, deformed, ugly, disfigured, poor quality, low resolution"
    )
    height = st.slider("Height", 512, 1024, 768, step=64)
    width = st.slider("Width", 512, 1024, 768, step=64)
    num_inference_steps = st.slider("Inference Steps", 10, 100, 50, step=5)
    guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, step=0.5)
    seed = st.number_input("Seed (for reproducibility)", value=None, min_value=0, max_value=999999999)

    generate_button = st.button("Generate Image", type="primary")

st.subheader("Generated Product Image")

if generate_button:
    if not positive_prompt:
        st.warning("Please enter a positive prompt to generate an image.")
    else:
        with st.spinner("Generating your product image... This might take a moment."):
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed) if seed is not None else None
            
            image = pipeline(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            st.image(image, caption="Generated Product Image", use_column_width=True)
            
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="generated_product_image.png",
                mime="image/png"
            )
else:
    st.info("Enter your desired product features and any undesired elements in the sidebar, then click 'Generate Image'.")
