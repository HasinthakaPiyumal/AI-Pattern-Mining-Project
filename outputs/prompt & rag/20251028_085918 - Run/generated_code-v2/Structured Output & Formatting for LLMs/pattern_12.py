from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import random

class Product(BaseModel):
    id: str
    name: str
    category: str
    price: float

class Recommendation(BaseModel):
    user_id: str
    recommended_products: List[Product]

app = FastAPI()

MOCK_PRODUCTS = [
    {"id": "P001", "name": "Laptop Pro", "category": "Electronics", "price": 1200.00},
    {"id": "P002", "name": "Wireless Mouse", "category": "Electronics", "price": 25.00},
    {"id": "P003", "name": "Mechanical Keyboard", "category": "Electronics", "price": 75.00},
    {"id": "P004", "name": "Desk Chair Ergonomic", "category": "Furniture", "price": 300.00},
    {"id": "P005", "name": "Smartwatch X", "category": "Wearables", "price": 199.99},
    {"id": "P006", "name": "Noise Cancelling Headphones", "category": "Audio", "price": 250.00},
    {"id": "P007", "name": "Gaming Monitor 27 inch", "category": "Electronics", "price": 450.00},
]

def get_mock_recommendations(user_id: str, count: int = 3) -> List[Product]:
    recommended = random.sample(MOCK_PRODUCTS, min(count, len(MOCK_PRODUCTS)))
    return [Product(**p) for p in recommended]

@app.get("/recommendations/{user_id}", response_model=Recommendation)
async def get_recommendations(user_id: str):
    recommended_products = get_mock_recommendations(user_id, count=3)
    return Recommendation(user_id=user_id, recommended_products=recommended_products)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)