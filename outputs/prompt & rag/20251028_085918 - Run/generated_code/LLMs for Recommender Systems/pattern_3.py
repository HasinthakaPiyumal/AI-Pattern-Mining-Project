import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

class ProductEmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Initialize a sentence transformer model for generating embeddings
        self.model = SentenceTransformer(model_name)
        self.products = []
        self.product_embeddings = None
        self.faiss_index = None

    def load_products_data(self):
        # Simulate loading product data from a CSV or database
        # In a real application, this would fetch data from your e-commerce backend
        self.products = [
            {"id": "P001", "name": "High-Performance Gaming Laptop", "description": "Powerful laptop with latest CPU/GPU, 16GB RAM, 1TB SSD. Ideal for gaming and professional use.", "category": "Electronics", "price": 1500},
            {"id": "P002", "name": "Lightweight Travel Camera", "description": "Compact mirrorless camera with 4K video, 20MP sensor. Perfect for vloggers and travel photographers.", "category": "Electronics", "price": 800},
            {"id": "P003", "name": "Noise-Cancelling Headphones", "description": "Over-ear headphones with superior sound quality and active noise cancellation. Great for commutes and focus.", "category": "Electronics", "price": 250},
            {"id": "P004", "name": "Ergonomic Office Chair", "description": "Adjustable chair with lumbar support, breathable mesh. Designed for long hours of comfortable work.", "category": "Home & Office", "price": 350},
            {"id": "P005", "name": "Waterproof Hiking Jacket", "description": "Durable, waterproof, and breathable jacket with multiple pockets. Ideal for all weather hiking.", "category": "Apparel", "price": 180},
            {"id": "P006", "name": "Professional Graphic Design Software", "description": "Industry-standard software for vector graphics and photo editing. Subscription based.", "category": "Software", "price": 50},
            {"id": "P007", "name": "Beginner Photography Course", "description": "Online course covering basics of photography, camera settings, and composition. Self-paced learning.", "category": "Education", "price": 100},
            {"id": "P008", "name": "Smart Home Hub", "description": "Central control for all smart devices, compatible with various protocols. Voice assistant built-in.", "category": "Smart Home", "price": 120},
            {"id": "P009", "name": "Organic Coffee Beans (Dark Roast)", "description": "Ethically sourced, rich dark roast coffee beans. Perfect for espresso and French press.", "category": "Food & Beverage", "price": 25},
            {"id": "P010", "name": "Yoga Mat and Accessories Set", "description": "Premium non-slip yoga mat with carrying strap, blocks, and towel. For all yoga levels.", "category": "Fitness", "price": 70},
        ]
        print(f"Loaded {len(self.products)} products.")

    def generate_embeddings(self):
        if not self.products:
            self.load_products_data()

        # Combine name and description for richer context
        texts_to_embed = [f"{p['name']}. {p['description']}" for p in self.products]
        self.product_embeddings = self.model.encode(texts_to_embed, convert_to_tensor=False)
        print(f"Generated embeddings for {len(self.product_embeddings)} products. Embedding dimension: {self.product_embeddings.shape[1]}")
        return self.product_embeddings

    def create_faiss_index(self):
        if self.product_embeddings is None:
            self.generate_embeddings()

        dimension = self.product_embeddings.shape[1]
        # Use IndexFlatL2 for simple Euclidean distance search
        self.faiss_index = faiss.IndexFlatL2(dimension)
        # Add the product embeddings to the index
        self.faiss_index.add(np.array(self.product_embeddings).astype('float32'))
        print(f"FAISS index created with {self.faiss_index.ntotal} items.")
        return self.faiss_index

    def get_product_by_id(self, product_id):
        for p in self.products:
            if p["id"] == product_id:
                return p
        return None

    def get_all_products(self):
        return self.products
