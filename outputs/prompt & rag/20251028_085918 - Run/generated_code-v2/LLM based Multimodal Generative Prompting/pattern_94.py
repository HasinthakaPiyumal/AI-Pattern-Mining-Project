from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# 1. Image Captioning Model (BLIP)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. Sentence Transformer Model for embeddings
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# 3. In-memory Product Catalog
PRODUCT_CATALOG = [
    {"id": "P001", "name": "Blue Denim Jeans", "description": "Classic blue denim jeans, straight fit, comfortable and durable for everyday wear. Made from 100% cotton."},
    {"id": "P002", "name": "White Cotton T-Shirt", "description": "Soft white cotton t-shirt, breathable fabric, perfect for casual outings or layering. Available in various sizes."},
    {"id": "P003", "name": "Leather Ankle Boots", "description": "Stylish black leather ankle boots with a low heel, perfect for autumn and winter. Features a side zipper closure."},
    {"id": "P004", "name": "Striped Summer Dress", "description": "Lightweight striped summer dress, knee-length, with a comfortable elastic waistband. Ideal for beach days or casual events."},
    {"id": "P005", "name": "Sporty Running Shoes", "description": "High-performance running shoes with superior cushioning and grip. Designed for long-distance running and intense workouts."},
    {"id": "P006", "name": "Elegant Silk Scarf", "description": "Luxurious pure silk scarf with a floral pattern, ideal for adding a touch of elegance to any outfit."},
    {"id": "P007", "name": "Stainless Steel Watch", "description": "Classic men's watch with a stainless steel strap and a minimalist dial. Water-resistant and durable."},
    {"id": "P008", "name": "Waterproof Backpack", "description": "Durable and waterproof backpack with multiple compartments, suitable for hiking, travel, and daily commute."},
]

# Pre-compute embeddings for product descriptions
product_descriptions = [p["description"] for p in PRODUCT_CATALOG]
product_embeddings = model_sbert.encode(product_descriptions, convert_to_tensor=True)


async def generate_image_caption(image_bytes: bytes) -> str:
    raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model_blip.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def search_products(query_embedding: torch.Tensor, top_n: int = 5):
    query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
    product_embeddings_np = product_embeddings.cpu().numpy()
    
    similarities = cosine_similarity(query_embedding_np, product_embeddings_np)[0]
    
    # Get indices of top_n most similar products
    top_n_indices = np.argsort(similarities)[::-1][:top_n]
    
    recommended_products = []
    for idx in top_n_indices:
        product = PRODUCT_CATALOG[idx]
        product_score = float(similarities[idx]) # Convert to float for JSON serialization
        recommended_products.append({"product_id": product["id"], "name": product["name"], "description": product["description"], "similarity_score": product_score})
    
    return recommended_products


@app.post("/recommend")
async def recommend_products_multimodal(
    image: UploadFile = File(...),
    text_query: str = Form(None)
):
    image_bytes = await image.read()
    image_caption = await generate_image_caption(image_bytes)
    
    # Combine image caption and text query to form a multimodal prompt
    if text_query:
        multimodal_prompt = f"{text_query} {image_caption}"
    else:
        multimodal_prompt = image_caption
        
    # Generate embedding for the multimodal prompt
    prompt_embedding = model_sbert.encode(multimodal_prompt, convert_to_tensor=True)
    
    # Search for products
    recommendations = search_products(prompt_embedding)
    
    return {"multimodal_prompt_used": multimodal_prompt, "recommendations": recommendations}
