import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

product_catalog = [
    {"id": 1, "name": "Blue Denim Jacket", "description": "A classic blue denim jacket with button-front closure and two chest pockets. Made from 100% cotton, perfect for casual wear."},
    {"id": 2, "name": "Red Cotton T-Shirt", "description": "Soft and breathable red cotton t-shirt. Crew neck, short sleeves. Ideal for everyday comfort."},
    {"id": 3, "name": "Leather Messenger Bag", "description": "High-quality brown leather messenger bag with multiple compartments. Adjustable shoulder strap. Suitable for laptops and documents."},
    {"id": 4, "name": "White Sneakers", "description": "Comfortable white leather sneakers with a rubber sole. Lace-up design, suitable for sports and casual outfits."},
    {"id": 5, "name": "Striped Summer Dress", "description": "Lightweight blue and white striped summer dress. Midi length with adjustable straps. Perfect for warm weather."},
    {"id": 6, "name": "Black Formal Trousers", "description": "Tailored black formal trousers for men. Slim fit, made from a blend of polyester and rayon. Ideal for office wear."},
    {"id": 7, "name": "Silver Hoop Earrings", "description": "Elegant silver hoop earrings. Hypoallergenic material, lightweight design. A versatile accessory."},
    {"id": 8, "name": "Wooden Coffee Table", "description": "Modern wooden coffee table made from solid oak. Rectangular design with sturdy legs. Perfect for living rooms."},
    {"id": 9, "name": "Green Hiking Backpack", "description": "Durable green hiking backpack with a large main compartment and multiple exterior pockets. Padded shoulder straps, ideal for outdoor adventures."},
    {"id": 10, "name": "Yoga Mat with Strap", "description": "Non-slip yoga mat in vibrant purple. Comes with a carrying strap. Suitable for yoga, pilates, and floor exercises."},
]

catalog_descriptions = [product["description"] for product in product_catalog]
catalog_embeddings = embedding_model.encode(catalog_descriptions, convert_to_tensor=True)

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def recommend_products(image_caption, top_n=3):
    query_embedding = embedding_model.encode(image_caption, convert_to_tensor=True)
    
    similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), catalog_embeddings)
    
    top_indices = torch.topk(similarities, top_n).indices.tolist()
    recommendations = [product_catalog[i] for i in top_indices]
    return recommendations

def multimodal_recommendation(image):
    if image is None:
        return "Please upload an image.", []
    
    caption = generate_caption(image)
    
    recommended_products = recommend_products(caption)
    
    recommendation_output = "Recommended Products:\n"
    for prod in recommended_products:
        recommendation_output += f"- {prod["name"]} (ID: {prod["id"]})\n  Description: {prod["description"]}\n\n"
        
    return caption, recommendation_output

iface = gr.Interface(
    fn=multimodal_recommendation,
    inputs=gr.Image(type="pil", label="Upload Product Image"),
    outputs=[
        gr.Textbox(label="Generated Image Caption"),
        gr.Textbox(label="Product Recommendations")
    ],
    title="E-commerce Product Recommendation (Image-as-Text Prompting)",
    description="Upload an image of a product to get a textual description and recommendations for similar products from our catalog."
)

iface.launch()