import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import os
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize BLIP model and processor
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, blip_model = load_blip_model()

# Streamlit App Title
st.title("Smart Product Description Generator for E-commerce")
st.markdown("Upload product images and provide additional details to generate compelling descriptions.")

# 1. User Inputs

st.header("1. Product Images")
uploaded_files = st.file_uploader("Choose multiple product images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

st.header("2. Product Details")
product_name = st.text_input("Product Name", "")
product_category = st.text_input("Product Category", "")
key_features = st.text_area("Key Features (one per line)", "")
target_audience = st.text_input("Target Audience", "Fashion-conscious individuals, young professionals")

# 2. Generate Product Description Button
if st.button("Generate Product Description"):
    if not uploaded_files and not product_name:
        st.warning("Please upload at least one image or provide a product name.")
    elif not openai.api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    else:
        st.spinner("Generating description...")

        # Image Captioning
        image_captions = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                inputs = processor(image, return_tensors="pt")
                out = blip_model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                image_captions.append(f"Image description: {caption}")

        # Prompt Construction
        prompt_parts = []
        if product_name: prompt_parts.append(f"Product Name: {product_name}")
        if product_category: prompt_parts.append(f"Category: {product_category}")
        if key_features: prompt_parts.append(f"Key Features:\n{key_features}")
        if target_audience: prompt_parts.append(f"Target Audience: {target_audience}")

        if image_captions: prompt_parts.extend(image_captions)

        base_prompt = "Generate a detailed, engaging, and SEO-friendly product description for an e-commerce website based on the following information. Highlight benefits and unique selling points. The tone should be persuasive and informative.\n\n"
        full_prompt = base_prompt + "\n".join(prompt_parts)

        # LLM Integration (OpenAI GPT-3.5/4)
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4"
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that writes compelling e-commerce product descriptions."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            generated_description = response.choices[0].message.content.strip()

            st.subheader("Generated Product Description")
            st.write(generated_description)

        except Exception as e:
            st.error(f"An error occurred with the LLM: {e}")
            st.error("Please ensure your OpenAI API key is valid and you have sufficient credits.")

        st.success("Description generated!")
