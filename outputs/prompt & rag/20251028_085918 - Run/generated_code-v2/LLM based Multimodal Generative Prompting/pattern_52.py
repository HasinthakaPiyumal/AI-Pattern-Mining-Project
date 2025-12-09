import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import io

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using a smaller model for quicker demonstration, can be changed to 'stabilityai/stable-diffusion-xl-base-1.0'
    pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipeline.to(device)
    return pipeline

st.title("🛒 E-commerce Product Image Generator")
st.write("Generate high-quality product images for your online store using AI with negative prompting.")

pipeline = load_model()

default_negative_prompts = (
    "blurry, poor lighting, shadows on product, distracting elements, incorrect product color, "
    "watermarks, low resolution, grainy, disfigured, deformed, ugly, bad anatomy, out of frame, "
    "mutated, extra limbs, missing limbs, text, signature, watermark, logo, duplicate"
)

# UI for positive prompt
positive_prompt = st.text_area(
    "Enter your product description (Positive Prompt):",
    "A stylish silver watch on a minimalist white background, high-resolution, professional product photo, studio light"
)

# UI for negative prompts
st.subheader("Negative Prompts (What to AVOID in the image)")
st.write("These are default negative prompts to ensure high-quality product images.")
st.text_area("Default Negative Prompts (unmodifiable):", default_negative_prompts, height=100, disabled=True)

additional_negative_prompts = st.text_area(
    "Add any additional negative prompts (comma-separated):",
    ""
)

# Generate button
if st.button("Generate Product Image"):
    if positive_prompt:
        combined_negative_prompt = default_negative_prompts
        if additional_negative_prompts:
            combined_negative_prompt += ", " + additional_negative_prompts

        with st.spinner("Generating your product image... This might take a moment."):
            # Generate image
            # Note: num_inference_steps can be adjusted for quality vs speed
            image = pipeline(prompt=positive_prompt, negative_prompt=combined_negative_prompt, num_inference_steps=30).images[0]

            st.success("Image Generated!")
            st.image(image, caption="Generated Product Image", use_column_width=True)

            # Add a download button
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
        st.warning("Please enter a positive prompt to generate an image.")