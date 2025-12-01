import streamlit as st
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

# --- Simulated Product Catalog ---
products_data = [
    {"ID": "P001", "Name": "Blue Denim Jeans", "Description": "Classic straight-fit blue denim jeans for everyday wear.", "Category": "Apparel", "Price": 49.99, "Image_URL": ""},
    {"ID": "P002", "Name": "White Cotton T-Shirt", "Description": "Soft and breathable white cotton t-shirt, perfect basic.", "Category": "Apparel", "Price": 19.99, "Image_URL": ""},
    {"ID": "P003", "Name": "Leather Biker Jacket", "Description": "Stylish black genuine leather biker jacket with zippers.", "Category": "Apparel", "Price": 199.99, "Image_URL": ""},
    {"ID": "P004", "Name": "Running Shoes", "Description": "Lightweight athletic running shoes with shock absorption.", "Category": "Footwear", "Price": 89.99, "Image_URL": ""},
    {"ID": "P005", "Name": "Silver Hoop Earrings", "Description": "Elegant sterling silver hoop earrings, small size.", "Category": "Jewelry", "Price": 29.99, "Image_URL": ""},
    {"ID": "P006", "Name": "Waterproof Backpack", "Description": "Durable and spacious waterproof backpack for outdoor activities.", "Category": "Accessories", "Price": 75.00, "Image_URL": ""},
    {"ID": "P007", "Name": "Smartwatch", "Description": "Fitness tracker and notification smartwatch with long battery life.", "Category": "Electronics", "Price": 120.00, "Image_URL": ""},
    {"ID": "P008", "Name": "Casual Linen Shirt", "Description": "Light blue linen shirt, perfect for a relaxed summer look.", "Category": "Apparel", "Price": 35.00, "Image_URL": ""},
    {"ID": "P009", "Name": "Yoga Mat", "Description": "Non-slip eco-friendly yoga mat for home or studio practice.", "Category": "Fitness", "Price": 40.00, "Image_URL": ""},
    {"ID": "P010", "Name": "Coffee Maker", "Description": "Programmable drip coffee maker with a glass carafe.", "Category": "Home Goods", "Price": 60.00, "Image_URL": ""}
]
products_df = pd.DataFrame(products_data)

# --- Model Loading (Cached) ---
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # Add a pad token if it doesn't exist, as some models (like GPT-2) don't have one by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

blip_processor, blip_model = load_blip_model()
llm_tokenizer, llm_model = load_llm_model()

# --- Functions for Image Captioning and LLM Inference ---
def get_image_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def get_recommendations_from_llm(image_caption, user_query, product_catalog_df):
    catalog_string = "\n".join([f"- {row['Name']}: {row['Description']} (Category: {row['Category']}, Price: ${row['Price']})" for index, row in product_catalog_df.iterrows()])

    prompt = f"""You are an E-commerce product recommendation assistant. Your goal is to suggest relevant products from the provided catalog based on an item's description and a user's specific request. Provide the recommended product names and a brief explanation of why they are relevant. If no relevant products are found, state that. Respond concisely. Make sure to only recommend products from the provided catalog. DO NOT hallucinate products. If the product description and user query doesn't match the catalog, suggest a similar category product from the catalog.

--- Product Catalog ---
{catalog_string}

--- User Input ---
Item Description: {image_caption}
User Query: {user_query}
---

Recommendations:
"""

    inputs = llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    
    # Generate text, ensuring pad_token_id is set for models that don't have it by default
    output_sequences = llm_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200, # Limit the length of the LLM's response
        num_return_sequences=1,
        pad_token_id=llm_tokenizer.pad_token_id
    )
    
    recommendations = llm_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    # The response will include the prompt itself, so we need to cut it off
    recommendations = recommendations.replace(prompt, "").strip()
    
    return recommendations

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🛍️ E-commerce Product Recommender")
st.markdown("Upload an image of a product and tell us what you're looking for! We'll use AI to understand the image and suggest items from our catalog.")

with st.sidebar:
    st.header("How it works:")
    st.info("1. Upload an image of a product you're interested in.")
    st.info("2. Enter a text query (e.g., 'find similar items', 'suggest accessories').")
    st.info("3. Our AI will describe the image and combine it with your query to find recommendations from our catalog.")
    st.write("--- Products in Catalog ---")
    st.dataframe(products_df[['Name', 'Category', 'Price']])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
user_text_query = st.text_input("Enter your query (e.g., 'find similar items', 'suggest accessories', 'cheaper alternatives'):")

if st.button("Get Recommendations"):
    if uploaded_file is not None:
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            st.subheader("Uploaded Image:")
            st.image(image, caption='Uploaded Image', use_column_width=True)

            with st.spinner("Generating image caption..."):
                caption = get_image_caption(image)
            st.subheader("Image Description:")
            st.info(caption)

            with st.spinner("Finding product recommendations..."):
                recommendations = get_recommendations_from_llm(caption, user_text_query, products_df)
            st.subheader("Our Recommendations:")
            st.write(recommendations)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload an image to get recommendations.")
