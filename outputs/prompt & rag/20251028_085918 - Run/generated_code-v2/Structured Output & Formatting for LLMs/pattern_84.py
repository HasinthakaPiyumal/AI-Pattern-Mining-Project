# recommendation_model.py
from typing import List, Dict

class ProductRecommender:
    def __init__(self):
        self.products = {
            1: {"id": 1, "name": "Laptop Pro", "price": 1200.00, "description": "Powerful laptop for professionals."},
            2: {"id": 2, "name": "Wireless Mouse X1", "price": 25.50, "description": "Ergonomic wireless mouse."},
            3: {"id": 3, "name": "Mechanical Keyboard K5", "price": 90.00, "description": "High-performance mechanical keyboard."},
            4: {"id": 4, "name": "USB-C Hub", "price": 45.00, "description": "Multi-port USB-C adapter."},
            5: {"id": 5, "name": "Monitor 27-inch 4K", "price": 350.00, "description": "Stunning 4K display for work and play."},
        }

    def get_recommendations(self, user_id: int) -> List[Dict]:
        return [
            self.products[1],
            self.products[5],
            self.products[3],
        ]

# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Assuming recommendation_model.py is in the same directory or importable
# from .recommendation_model import ProductRecommender # if it were a package
# For this combined output, we'll assume the class is available

class Product(BaseModel):
    id: int
    name: str
    price: float
    description: str

app = FastAPI()

# Instantiate the recommender globally or within the endpoint for simplicity in this combined file
recommender_instance = ProductRecommender()

@app.get("/recommendations/{user_id}", response_model=List[Product])
async def get_product_recommendations(user_id: int):
    recommendations = recommender_instance.get_recommendations(user_id)
    return recommendations

# requirements.txt
# fastapi
# uvicorn
# pydantic
