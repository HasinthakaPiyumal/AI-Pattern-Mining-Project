import pandas as pd
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI
import random
import requests
import streamlit as st

class Product(BaseModel):
    id: int
    name: str
    category: str
    price: float
    description: str
    image_url: str

class RecommendedProduct(Product):
    explanation: str

MOCK_PRODUCTS: List[Product] = [
    Product(id=1, name="Laptop Pro", category="Electronics", price=1200.00, description="Powerful laptop for professionals.", image_url="https://example.com/laptop.jpg"),
    Product(id=2, name="Wireless Earbuds", category="Electronics", price=150.00, description="Noise-cancelling earbuds with long battery life.", image_url="https://example.com/earbuds.jpg"),
    Product(id=3, name="Mechanical Keyboard", category="Electronics", price=100.00, description="Clicky keys for satisfying typing.", image_url="https://example.com/keyboard.jpg"),
    Product(id=4, name="Ergonomic Office Chair", category="Furniture", price=300.00, description="Comfortable chair for long working hours.", image_url="https://example.com/chair.jpg"),
    Product(id=5, name="Smartwatch X", category="Wearables", price=250.00, description="Track your fitness and receive notifications.", image_url="https://example.com/smartwatch.jpg"),
    Product(id=6, name="Coffee Maker Deluxe", category="Home Appliances", price=80.00, description="Brew perfect coffee every morning.", image_url="https://example.com/coffee.jpg"),
    Product(id=7, name="Fiction Bestseller", category="Books", price=15.00, description="A gripping novel you won't put down.", image_url="https://example.com/book.jpg"),
    Product(id=8, name="Running Shoes", category="Apparel", price=90.00, description="Lightweight and comfortable for your runs.", image_url="https://example.com/shoes.jpg"),
    Product(id=9, name="External SSD 1TB", category="Electronics", price=120.00, description="Fast and portable storage solution.", image_url="https://example.com/ssd.jpg"),
    Product(id=10, name="Yoga Mat", category="Fitness", price=30.00, description="Non-slip mat for your yoga sessions.", image_url="https://example.com/yogamat.jpg")
]

MOCK_USER_INTERACTIONS: Dict[int, List[int]] = {
    1: [1, 2, 5],
    2: [4, 6],
    3: [7],
    4: [8, 10]
}

class MockRecommendationEngine:
    def get_recommendations(self, user_id: int, num_recommendations: int = 5) -> List[int]:
        user_history = MOCK_USER_INTERACTIONS.get(user_id, [])
        
        all_product_ids = [p.id for p in MOCK_PRODUCTS]
        
        available_product_ids = [pid for pid in all_product_ids if pid not in user_history]
        
        if len(available_product_ids) <= num_recommendations:
            return available_product_ids
            
        return random.sample(available_product_ids, k=num_recommendations)

class MockLLMExplanationGenerator:
    def generate_explanation(self, user_id: int, product: Product, recommendation_reason: str) -> str:
        
        base_explanation = f"Based on your browsing history, we think you'll love the {product.name}. {recommendation_reason}"
        
        if "Electronics" in product.category:
            return f"{base_explanation} Its advanced features are perfect for tech enthusiasts like you."
        elif "Furniture" in product.category:
            return f"{base_explanation} This item is known for its durability and comfort, enhancing your living space."
        elif "Books" in product.category:
            return f"{base_explanation} Many users who enjoyed similar genres also loved this critically acclaimed title."
        else:
            return f"{base_explanation} This product aligns with your likely preferences."

app = FastAPI()
recommender = MockRecommendationEngine()
explainer = MockLLMExplanationGenerator()

@app.get("/recommendations/{user_id}", response_model=List[RecommendedProduct])
async def get_explainable_recommendations(user_id: int):
    recommended_product_ids = recommender.get_recommendations(user_id)
    
    recommendations_with_explanations: List[RecommendedProduct] = []
    
    for prod_id in recommended_product_ids:
        product = next((p for p in MOCK_PRODUCTS if p.id == prod_id), None)
        if product:
            mock_reason = "It complements your recent purchases." if prod_id in MOCK_USER_INTERACTIONS.get(user_id, []) else "It's a popular choice among users with similar interests."
            explanation = explainer.generate_explanation(user_id, product, mock_reason)
            recommendations_with_explanations.append(
                RecommendedProduct(
                    id=product.id,
                    name=product.name,
                    category=product.category,
                    price=product.price,
                    description=product.description,
                    image_url=product.image_url,
                    explanation=explanation
                )
            )
    return recommendations_with_explanations

if st.button("Run Streamlit Frontend"):
    st.title("E-commerce Product Recommender")

    user_id_input = st.number_input("Enter User ID", min_value=1, value=1)

    if st.button("Get Recommendations"):
        st.subheader(f"Recommendations for User {user_id_input}:")
        try:
            response = requests.get(f"http://127.0.0.1:8000/recommendations/{user_id_input}")
            response.raise_for_status()
            recommendations = [RecommendedProduct(**rec) for rec in response.json()]

            if recommendations:
                for rec in recommendations:
                    st.image(rec.image_url, width=100)
                    st.write(f"**{rec.name}** ({rec.category}) - ${rec.price:.2f}")
                    st.write(f"Explanation: {rec.explanation}")
                    st.markdown("---")
            else:
                st.write("No recommendations found for this user.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Please ensure it is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")