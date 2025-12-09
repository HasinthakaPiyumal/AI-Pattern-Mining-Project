import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load Models
# Image Captioning Model (BLIP)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_caption = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Sentence Transformer for embeddings
model_embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Mock Product Data
mock_products = [
    {"id": "P001", "name": "Blue Denim Jeans", "description": "Classic blue denim jeans for men, regular fit, comfortable cotton material.", "category": "Apparel", "image_url": "mock_images/jeans.jpg"},
    {"id": "P002", "name": "Red T-Shirt", "description": "Soft cotton red t-shirt, unisex, round neck, perfect for casual wear.", "category": "Apparel", "image_url": "mock_images/tshirt.jpg"},
    {"id": "P003", "name": "Leather Wallet", "description": "Premium black leather wallet with multiple card slots and coin pocket.", "category": "Accessories", "image_url": "mock_images/wallet.jpg"},
    {"id": "P004", "name": "Smartwatch", "description": "Fitness tracker smartwatch with heart rate monitor, GPS, and waterproof design.", "category": "Electronics", "image_url": "mock_images/smartwatch.jpg"},
    {"id": "P005", "name": "Running Shoes", "description": "Lightweight running shoes with breathable mesh upper and cushioned sole for optimal comfort.", "category": "Footwear", "image_url": "mock_images/shoes.jpg"},
    {"id": "P006", "name": "Cotton Dress", "description": "Summer floral cotton dress for women, knee-length, loose fit.", "category": "Apparel", "image_url": "mock_images/dress.jpg"},
]

# Pre-compute embeddings for mock product descriptions
product_descriptions = [p["description"] for p in mock_products]
product_embeddings = model_embedder.encode(product_descriptions, convert_to_tensor=True)

# 3. Backend Core Logic Functions
def generate_image_caption(image: Image.Image) -> str:
    """Generates a textual caption for a given image using BLIP."""
    if image is None:
        return ""
    inputs = processor(images=image, return_tensors="pt")
    out = model_caption.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def search_products(image: Image.Image, text_query: str) -> tuple[str, str, list]:
    """Combines image caption and text query for product search and recommendation."""
    image_caption = generate_image_caption(image)

    if image_caption and text_query:
        combined_prompt = f"A {image_caption}. User is looking for: {text_query}"
    elif image_caption:
        combined_prompt = f"A {image_caption}"
    elif text_query:
        combined_prompt = text_query
    else:
        return "", "Please provide an image or a text query.", []

    # Generate embedding for the combined prompt
    query_embedding = model_embedder.encode(combined_prompt, convert_to_tensor=True)

    # Calculate cosine similarity
    similarities = torch.nn.functional.cosine_similarity(query_embedding, product_embeddings)

    # Get top 3 most similar products
    top_product_indices = torch.topk(similarities, k=3).indices.tolist()
    recommended_products = [mock_products[i] for i in top_product_indices]

    return image_caption, combined_prompt, recommended_products

# 4. Gradio Interface
def gradio_interface(image_input, text_query_input):
    caption, full_prompt, recommendations = search_products(image_input, text_query_input)
    
    output_html = ""
    if recommendations:
        output_html += "<h3>Recommended Products:</h3>"
        for prod in recommendations:
            output_html += f"<p><b>{prod['name']}</b> ({prod['category']})<br>{prod['description']}</p>"
    else:
        output_html += "<p>No recommendations found or an error occurred.</p>"

    return caption, full_prompt, output_html

# Create a dummy folder for mock images
import os
if not os.path.exists("mock_images"):
    os.makedirs("mock_images")
    # Create dummy image files (e.g., empty files or placeholders)
    for prod in mock_products:
        with open(os.path.join("mock_images", os.path.basename(prod["image_url"])), "w") as f:
            f.write("Dummy image content") # In a real app, these would be actual images


iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="pil", label="Upload Product Image (Optional)"),
        gr.Textbox(label="Enter Text Search Query (Optional)", placeholder="e.g., 'comfortable apparel for summer'")
    ],
    outputs=[
        gr.Textbox(label="Generated Image Caption", interactive=False),
        gr.Textbox(label="Combined Search Prompt", interactive=False),
        gr.HTML(label="Product Recommendations")
    ],
    title="E-commerce Product Search & Recommendation (Image-as-Text)",
    description="Upload an image and/or enter a text query to find product recommendations."
)

if __name__ == "__main__":
    iface.launch()