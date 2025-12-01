import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

def get_mock_data():
    products_data = [
        {"product_id": "P1", "name": "Smartwatch X", "description": "Advanced smartwatch with health tracking and GPS. Compatible with Android and iOS.", "category": "Electronics", "brand": "BrandA"},
        {"product_id": "P2", "name": "Fitness Tracker Y", "description": "Basic fitness tracker for daily steps and sleep monitoring. Long battery life.", "category": "Electronics", "brand": "BrandB"},
        {"product_id": "P3", "name": "Premium Headphones Z", "description": "Noise-cancelling over-ear headphones. Great for music lovers. Accessory for phones and laptops.", "category": "Electronics", "brand": "BrandC"},
        {"product_id": "P4", "name": "Water Bottle A", "description": "Durable stainless steel water bottle. Perfect for gym and outdoor activities.", "category": "Sports & Outdoors", "brand": "BrandD"},
        {"product_id": "P5", "name": "Yoga Mat B", "description": "Eco-friendly yoga mat with non-slip surface. Ideal for all yoga styles.", "category": "Sports & Outdoors", "brand": "BrandE"}
    ]
    user_reviews_data = [
        {"product_id": "P1", "review": "Amazing smartwatch, love the health features!", "user_id": "U1"},
        {"product_id": "P3", "review": "Sound quality is superb, but a bit heavy.", "user_id": "U2"},
        {"product_id": "P1", "review": "Integrates perfectly with my Android phone.", "user_id": "U2"}
    ]
    return pd.DataFrame(products_data), pd.DataFrame(user_reviews_data)

class MockLLM:
    def extract_entities_relations(self, text):
        entities = []
        relations = []
        if "smartwatch" in text.lower():
            entities.append("Smartwatch")
        if "health tracking" in text.lower():
            entities.append("HealthTracking")
            relations.append(("Smartwatch", "HAS_FEATURE", "HealthTracking"))
        if "gps" in text.lower():
            entities.append("GPS")
            relations.append(("Smartwatch", "HAS_FEATURE", "GPS"))
        if "compatible with android" in text.lower():
            entities.append("Android")
            relations.append(("Smartwatch", "COMPATIBLE_WITH", "Android"))
        if "compatible with ios" in text.lower():
            entities.append("iOS")
            relations.append(("Smartwatch", "COMPATIBLE_WITH", "iOS"))
        if "fitness tracker" in text.lower():
            entities.append("FitnessTracker")
        if "headphones" in text.lower():
            entities.append("Headphones")
        if "accessory for phones" in text.lower() or "accessory for laptops" in text.lower():
            entities.append("Phone")
            entities.append("Laptop")
            relations.append(("Headphones", "ACCESSORY_FOR", "Phone"))
            relations.append(("Headphones", "ACCESSORY_FOR", "Laptop"))
        if "water bottle" in text.lower():
            entities.append("WaterBottle")
        if "yoga mat" in text.lower():
            entities.append("YogaMat")
        return list(set(entities)), list(set(relations))

    def complete_kg_facts(self, existing_triples, entity_type="Product", relation_type="HAS_FEATURE"):
        new_triples = []
        if any(t == ("Smartwatch", "HAS_FEATURE", "HealthTracking") for t in existing_triples) and any(t == ("Smartwatch", "COMPATIBLE_WITH", "Android") for t in existing_triples):
            new_triples.append(("Smartwatch", "TARGET_AUDIENCE", "TechEnthusiast"))
        if any(t == ("Headphones", "ACCESSORY_FOR", "Phone") for t in existing_triples) and any(t == ("Headphones", "ACCESSORY_FOR", "Laptop") for t in existing_triples):
            new_triples.append(("Headphones", "BEST_FOR", "Multimedia"))
        return new_triples

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_product_node(self, product_id, name, category, brand):
        self.graph.add_node(product_id, type="Product", name=name, category=category, brand=brand)

    def add_entity_node(self, entity_name, entity_type="Feature"):
        if not self.graph.has_node(entity_name):
            self.graph.add_node(entity_name, type=entity_type)

    def add_relation(self, source, target, relation_type):
        self.graph.add_edge(source, target, type=relation_type)

    def build_from_data(self, products_df, reviews_df, llm):
        for _, row in products_df.iterrows():
            self.add_product_node(row["product_id"], row["name"], row["category"], row["brand"])
            text = row["description"]
            entities, relations = llm.extract_entities_relations(text)
            for entity in entities:
                self.add_entity_node(entity)
            for s, r, t in relations:
                if not self.graph.has_node(s):
                    self.graph.add_node(s)
                if not self.graph.has_node(t):
                    self.graph.add_node(t)
                self.add_relation(row["product_id"], t, r)

        for _, row in reviews_df.iterrows():
            text = row["review"]
            entities, relations = llm.extract_entities_relations(text)
            for entity in entities:
                self.add_entity_node(entity)
            for s, r, t in relations:
                if not self.graph.has_node(s):
                    self.graph.add_node(s)
                if not self.graph.has_node(t):
                    self.graph.add_node(t)
                self.add_relation(row["product_id"], t, r)

        existing_triples = [(u, d["type"], v) for u, v, d in self.graph.edges(data=True)]
        completed_triples = llm.complete_kg_facts(existing_triples)
        for s, r, t in completed_triples:
            if not self.graph.has_node(s):
                self.graph.add_node(s)
            if not self.graph.has_node(t):
                self.graph.add_node(t)
            self.add_relation(s, t, r)

class RecommendationEngine:
    def __init__(self, products_df, kg):
        self.products_df = products_df
        self.kg = kg
        self.product_features = self._generate_product_features()
        self.vectorizer = TfidfVectorizer()
        self.product_vectors = self.vectorizer.fit_transform(self.product_features["combined_text"])

    def _generate_product_features(self):
        features = []
        for _, row in self.products_df.iterrows():
            product_id = row["product_id"]
            product_description = row["description"]
            product_category = row["category"]
            product_brand = row["brand"]

            kg_info = []
            if self.kg.graph.has_node(product_id):
                for neighbor in self.kg.graph.neighbors(product_id):
                    edge_data = self.kg.graph.get_edge_data(product_id, neighbor)
                    if edge_data and "type" in edge_data:
                        kg_info.append(f"{edge_data['type'].lower()}_{neighbor.lower()}")

            combined_text = f"{product_description} {product_category} {product_brand} {' '.join(kg_info)}"
            features.append({"product_id": product_id, "combined_text": combined_text})
        return pd.DataFrame(features)

    def get_recommendations(self, product_id, num_recommendations=5):
        if product_id not in self.product_features["product_id"].values:
            return []

        idx = self.product_features[self.product_features["product_id"] == product_id].index[0]
        similarities = cosine_similarity(self.product_vectors[idx:idx+1], self.product_vectors).flatten()
        similar_indices = similarities.argsort()[-num_recommendations-1:-1][::-1]
        recommended_product_ids = [self.product_features.iloc[i]["product_id"] for i in similar_indices]

        return recommended_product_ids

app = FastAPI()

class RecommendRequest(BaseModel):
    product_id: str
    num_recommendations: int = 5

products_df, reviews_df = get_mock_data()
mock_llm = MockLLM()
kg = KnowledgeGraph()
kg.build_from_data(products_df, reviews_df, mock_llm)
recommendation_engine = RecommendationEngine(products_df, kg)

@app.post("/recommend")
async def get_recommendations_endpoint(request: RecommendRequest):
    recommendations = recommendation_engine.get_recommendations(request.product_id, request.num_recommendations)
    return {"product_id": request.product_id, "recommendations": recommendations}

@app.get("/health")
async def health_check():
    return {"status": "ok"}