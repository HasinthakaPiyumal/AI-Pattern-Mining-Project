import streamlit as st
from PIL import Image
from io import BytesIO
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import os

@st.cache_resource
def load_model():
    # For a real application, uncomment the following lines and ensure a GPU is available.
    # It also requires significant download.
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    # pipe.to("cuda")
    # return pipe
    
    class MockDiffusionPipeline:
        def __call__(self, prompt, image, strength=0.75, guidance_scale=7.5):
            img_width, img_height = image.size
            overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(overlay)
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()
            draw.text((img_width * 0.1, img_height * 0.4), f"Transformed by: {prompt[:20]}...", font=font, fill=(0, 0, 0, 150))
            transformed_image = Image.alpha_composite(image.convert("RGBA"), overlay)
            return [transformed_image.convert("RGB")]
    
    return MockDiffusionPipeline()

st.set_page_config(layout="wide", page_title="E-commerce Virtual Try-On (PairedImage Prompting Demo)")

st.title("👗 E-commerce Virtual Try-On Demo")
st.markdown("""
This application demonstrates the 'PairedImage Prompting' concept for virtual try-on or style transfer.
Upload an image and provide a description of the desired transformation.
""")

col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader("Upload an image of yourself or an item:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    col1.subheader("Original Image")
    col1.image(original_image, use_column_width=True)

    st.sidebar.header("Transformation Settings")
    prompt = st.sidebar.text_area(
        "Describe the desired transformation (e.g., 'a person wearing a blue denim jacket', 'a red floral pattern shirt', 'a product on a minimalist white background'):",
        "a red floral pattern shirt"
    )
    
    guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
    strength = st.sidebar.slider("Transformation Strength", 0.0, 1.0, 0.75, 0.05)
    
    if st.sidebar.button("Apply Transformation"):
        with st.spinner("Applying transformation... This might take a moment."):
            try:
                # Mock model is used for quick demonstration without large downloads/GPU.
                # For actual diffusion model behavior, replace with the commented-out StableDiffusionImg2ImgPipeline.
                mock_pipe = load_model()
                transformed_image = mock_pipe(prompt=prompt, image=original_image, strength=strength, guidance_scale=guidance_scale)[0]

                col2.subheader("Transformed Image")
                col2.image(transformed_image, use_column_width=True)
                
                buf = BytesIO()
                transformed_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                col2.download_button(
                    label="Download Transformed Image",
                    data=byte_im,
                    file_name="transformed_image.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"An error occurred during transformation: {e}")
                st.info("Please ensure your environment is set up correctly for `diffusers` and PyTorch, and consider running on a GPU for performance.")
else:
    col2.subheader("Transformed Image")
    col2.info("Upload an image to see the transformation here.")
    st.markdown("""
--- 
### How 'PairedImage Prompting' works here:
In generative models like Stable Diffusion for image-to-image tasks, the model is pre-trained on vast image-text pairs. This training implicitly teaches it how to transform images based on textual descriptions (prompts). When you provide an original image and a prompt, the model uses its learned understanding (from numerous paired images in its training data) to generate a new image incorporating the described transformation onto your input image. This demo uses a mock pipeline for execution; conceptually, a StableDiffusionImg2ImgPipeline would be used.
""")