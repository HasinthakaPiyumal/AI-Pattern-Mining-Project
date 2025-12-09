import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.set_page_config(layout="wide", page_title="E-commerce Product Image Generator")

st.title("E-commerce Product Image Generator with Negative Prompting")

@st.cache_resource
def load_model():
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    if torch.cuda.is_available():
        pipeline.to("cuda")
    return pipeline

pipeline = load_model()

with st.sidebar:
    st.header("Generation Parameters")
    num_inference_steps = st.slider("Number of Inference Steps", 10, 100, 50)
    guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)

positive_prompt = st.text_input(
    "Positive Prompt",
    "A sleek silver laptop on a modern wooden desk with natural lighting, professional product photo, high resolution, detailed, studio quality"
)

negative_prompt = st.text_area(
    "Negative Prompt",
    "blurry, dark, shadows, reflections, distorted, messy background, other objects, low resolution, bad quality, low details, ugly, watermarks, text"
)

if st.button("Generate Image"):
    if positive_prompt:
        with st.spinner("Generating your image... This may take a moment."):
            image = pipeline(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            st.image(image, caption="Generated Product Image", use_column_width=True)
    else:
        st.warning("Please enter a positive prompt to generate an image.")

st.markdown("""
<style>
.stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)
