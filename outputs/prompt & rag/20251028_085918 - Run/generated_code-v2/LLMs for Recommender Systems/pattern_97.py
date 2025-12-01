import pandas as pd
import networkx as nx
from fastapi import FastAPI
import uvicorn
import random

app = FastAPI()

class ProductKnowledgeGraph:
    def __init__(self):
        self.kg = nx.Graph()
        self.products = {}

    def add_product(self, product_id, name, category, attributes):
        if product_id not in self.products:
            self.products[product_id] = {"name": name, "category": category, "attributes": attributes}
            self.kg.add_node(f"product_{product_id}", type="product", name=name, category=category)
            self.kg.add_edge(f"product_{product_id}", f"category_{category}", relation="is_in_category")
            for attr, value in attributes.items():
                self.kg.add_node(f"attribute_{attr}_{value}", type="attribute", name=attr, value=value)
                self.kg.add_edge(f"product_{product_id}", f"attribute_{attr}_{value}", relation="has_attribute")

    def add_relation(self, source_node, target_node, relation_type):
        self.kg.add_edge(source_node, target_node, relation=relation_type)

    def mock_llm_enrichment(self, reviews_df):
        for _, row in reviews_df.iterrows():
            product_id = row['product_id']
            review_text = row['review_text']

            # Simulate LLM extracting new attributes or related entities from review text
            # For simplicity, we'll just add some predefined relations based on keywords
            if "gaming" in review_text.lower() and f"product_{product_id}" in self.kg:
                self.kg.add_node("use_case_gaming", type="use_case", name="gaming")
                self.add_relation(f"product_{product_id}", "use_case_gaming", "suited_for")
            if "budget" in review_text.lower() and f"product_{product_id}" in self.kg:
                self.kg.add_node("price_point_budget", type="price_point", name="budget")
                self.add_relation(f"product_{product_id}", "price_point_budget", "has_price_characteristic")
            if "work" in review_text.lower() and f"product_{product_id}" in self.kg:
                self.kg.add_node("use_case_work", type="use_case", name="work")
                self.add_relation(f"product_{product_id}", "use_case_work", "suited_for")

    def get_related_products(self, product_id, k=5):
        if f"product_{product_id}" not in self.kg:
            return []
        
        related = set()
        # Find products in the same category
        if f"category_{self.products[product_id]['category']}" in self.kg:
            for neighbor in self.kg.neighbors(f"category_{self.products[product_id]['category']}"):
                if neighbor.startswith("product_") and neighbor != f"product_{product_id}":
                    related.add(neighbor.replace("product_", ""))
        
        # Find products with similar attributes
        for attr, value in self.products[product_id]['attributes'].items():
            node_id = f"attribute_{attr}_{value}"
            if node_id in self.kg:
                for neighbor in self.kg.neighbors(node_id):
                    if neighbor.startswith("product_") and neighbor != f"product_{product_id}":
                        related.add(neighbor.replace("product_", ""))

        return list(related)[:k]

kg_manager = ProductKnowledgeGraph()

def load_dummy_data():
    products_data = {
        'product_id': [1, 2, 3, 4, 5, 6],
        'name': ['Laptop Pro', 'Gaming PC Elite', 'Smartphone X', 'Smart TV 4K', 'Wireless Headphones', 'Ergonomic Mouse'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories', 'Accessories'],
        'attributes': [
            {'brand': 'BrandA', 'screen_size': '15inch', 'processor': 'i7'},
            {'brand': 'BrandB', 'gpu': 'RTX3080', 'ram': '32GB'},
            {'brand': 'BrandC', 'os': 'Android', 'camera': '48MP'},
            {'brand': 'BrandD', 'resolution': '4K', 'size': '55inch'},
            {'brand': 'BrandE', 'type': 'over-ear', 'connection': 'bluetooth'},
            {'brand': 'BrandF', 'connection': 'wireless', 'dpi': '1200'}
        ]
    }
    products_df = pd.DataFrame(products_data)

    reviews_data = {
        'product_id': [1, 1, 2, 3, 4, 5, 6],
        'review_text': [
            'Great laptop for work and everyday tasks.',
            'The screen is vibrant, good for productivity.',
            'Amazing performance for gaming, worth every penny!',
            'Best smartphone, camera is fantastic.',
            'Watching movies on this TV is a delight. Excellent picture quality.',
            'Comfortable headphones, good for travel.',
            'Smooth and precise, perfect for work.'
        ]
    }
    reviews_df = pd.DataFrame(reviews_data)

    user_history_data = {
        'user_id': [101, 101, 102, 103, 104],
        'product_id': [1, 5, 2, 4, 6]
    }
    user_history_df = pd.DataFrame(user_history_data)

    return products_df, reviews_df, user_history_df

@app.on_event("startup")
async def startup_event():
    products_df, reviews_df, user_history_df = load_dummy_data()

    for _, row in products_df.iterrows():
        kg_manager.add_product(row['product_id'], row['name'], row['category'], row['attributes'])

    for _, row in user_history_df.iterrows():
        kg_manager.add_relation(f"user_{row['user_id']}", f"product_{row['product_id']}", "purchased")

    kg_manager.mock_llm_enrichment(reviews_df)
    print("KG initialized and enriched with LLM (mock) data.")

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: int):
    user_purchases = []
    for u, p, data in kg_manager.kg.edges(data=True):
        if data.get('relation') == 'purchased' and u == f"user_{user_id}":
            user_purchases.append(int(p.replace("product_", "")))

    if not user_purchases:
        # If no purchase history, recommend random popular items
        all_product_ids = [pid for pid in kg_manager.products.keys()]
        return random.sample(all_product_ids, min(5, len(all_product_ids)))

    recommended_products = set()
    for product_id in user_purchases:
        recommended_products.update(kg_manager.get_related_products(product_id, k=3))

    # Ensure not to recommend already purchased items
    final_recommendations = [pid for pid in list(recommended_products) if int(pid) not in user_purchases]

    return final_recommendations[:5]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
