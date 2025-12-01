from fastapi import FastAPI, HTTPException
from typing import List, Dict

# recommendation_engine.py content

PRODUCTS_DATA = [
    {
        "id": 1,
        "name": "Laptop Pro X",
        "category": "Electronics",
        "price": 1200.00,
        "description": "High-performance laptop for professionals."
    },
    {
        "id": 2,
        "name": "Mechanical Keyboard",
        "category": "Electronics",
        "price": 95.00,
        "description": "Tactile mechanical keyboard for typing enthusiasts."
    },
    {
        "id": 3,
        "name": "Wireless Mouse G300",
        "category": "Electronics",
        "price": 50.00,
        "description": "Ergonomic wireless mouse with long battery life."
    },
    {
        "id": 4,
        "name": "Fiction Novel: The Last Star",
        "category": "Books",
        "price": 15.99,
        "description": "A gripping science fiction novel."
    },
    {
        "id": 5,
        "name": "Cookbook: Italian Classics",
        "category": "Books",
        "price": 25.00,
        "description": "Traditional Italian recipes for home cooks."
    },
    {
        "id": 6,
        "name": "Running Shoes Alpha",
        "category": "Apparel",
        "price": 80.00,
        "description": "Lightweight running shoes for daily training."
    },
    {
        "id": 7,
        "name": "T-Shirt: Tech Enthusiast",
        "category": "Apparel",
        "price": 20.00,
        "description": "Comfortable cotton t-shirt with a tech-themed print."
    }
]

def load_product_data() -> List[Dict]:
    return PRODUCTS_DATA

def get_recommendations(product_id: int) -> List[Dict]:
    products = load_product_data()
    target_product = next((p for p in products if p["id"] == product_id), None)

    if not target_product:
        return []

    recommended_products = [
        p for p in products 
        if p["category"] == target_product["category"] and p["id"] != product_id
    ]
    return recommended_products

# app.py content

app = FastAPI()

@app.get("/recommend/{product_id}", response_model=List[Dict])
async def recommend_products(product_id: int):
    recommendations = get_recommendations(product_id)
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found for this product, or product does not exist.")
    return recommendations
