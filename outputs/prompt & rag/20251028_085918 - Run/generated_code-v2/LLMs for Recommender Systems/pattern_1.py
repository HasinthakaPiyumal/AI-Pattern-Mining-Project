import pandas as pd
import networkx as nx
from fastapi import FastAPI
from typing import List, Dict, Any
from sklearn.neighbors import NearestNeighbors
import random

# 1. Data Ingestion & Storage Layer
class ProductData:
    def __init__(self):
        self.products_df = pd.DataFrame({
            "product_id": ["P1", "P2", "P3", "P4", "P5"],
            "name": ["Laptop", "Mouse", "Keyboard", "Monitor", "Webcam"],
            "category": ["Electronics", "Electronics", "Electronics", "Electronics", "Electronics"],
            "brand": ["BrandA", "BrandB", "BrandC", "BrandA", "BrandD"],
            "description": [
                "High performance laptop with 16GB RAM and SSD.",
                "Ergonomic wireless mouse with customizable buttons.",
                "Mechanical keyboard with RGB lighting and tactile switches.",
                "27-inch 4K monitor with HDR support.",
                "Full HD webcam with built-in microphone for clear calls."
            ]
        })
        self.user_interactions_df = pd.DataFrame({
            "user_id": ["U1", "U1", "U2", "U3", "U3", "U4"],
            "product_id": ["P1", "P2", "P1", "P3", "P4", "P5"],
            "interaction_type": ["view", "purchase", "view", "purchase", "view", "purchase"]
        })

    def get_products(self) -> pd.DataFrame:
        return self.products_df

    def get_user_interactions(self) -> pd.DataFrame:
        return self.user_interactions_df

# 2. Knowledge Graph (KG) Management Layer
class KnowledgeGraphManager:
    def __init__(self):
        self.kg = nx.Graph()

    def add_entity(self, entity_id: str, entity_type: str, attributes: Dict[str, Any]):
        self.kg.add_node(entity_id, type=entity_type, **attributes)

    def add_relation(self, source_id: str, target_id: str, relation_type: str, attributes: Dict[str, Any] = None):
        if attributes is None:
            attributes = {}
        self.kg.add_edge(source_id, target_id, type=relation_type, **attributes)

    def build_initial_kg(self, products_df: pd.DataFrame):
        for _, row in products_df.iterrows():
            product_id = row["product_id"]
            self.add_entity(product_id, "Product", {"name": row["name"], "category": row["category"], "brand": row["brand"], "description": row["description"]})
            self.add_entity(row["brand"], "Brand", {"name": row["brand"]})
            self.add_entity(row["category"], "Category", {"name": row["category"]})
            self.add_relation(product_id, row["brand"], "HAS_BRAND")
            self.add_relation(product_id, row["category"], "HAS_CATEGORY")

    def get_kg(self) -> nx.Graph:
        return self.kg

# 3. LLM-powered KG Enrichment Layer (Placeholders)
class LLMKGEnricher:
    def __init__(self, llm_api_key: str = "dummy_key"):
        self.llm_api_key = llm_api_key

    def _preprocess_text(self, text: str) -> str:
        # Placeholder for spacy/nltk preprocessing
        return text.lower().strip()

    def discover_entities_llm(self, text: str) -> List[Dict[str, str]]:
        # Simulate LLM entity discovery - in a real scenario, this would call an LLM API
        processed_text = self._preprocess_text(text)
        if "laptop" in processed_text:
            return [{"entity_id": "feature_ssd", "entity_type": "Feature", "name": "SSD"}]
        return []

    def extract_relations_llm(self, text: str, existing_entities: List[str]) -> List[Dict[str, str]]:
        # Simulate LLM relation extraction
        processed_text = self._preprocess_text(text)
        if "laptop" in processed_text and "ssd" in processed_text:
            if "P1" in existing_entities:
                return [{"source": "P1", "target": "feature_ssd", "type": "HAS_FEATURE"}]
        return []

    def complete_kg_llm(self, graph: nx.Graph) -> nx.Graph:
        # Simulate LLM KG completion
        # For demonstration, let's add a dummy 'popularity' attribute to products
        enriched_graph = graph.copy()
        for node_id in enriched_graph.nodes:
            if enriched_graph.nodes[node_id].get("type") == "Product":
                if "P1" == node_id:
                    enriched_graph.nodes[node_id]["popularity"] = 0.85
                elif "P2" == node_id:
                    enriched_graph.nodes[node_id]["popularity"] = 0.70
                else:
                    enriched_graph.nodes[node_id]["popularity"] = random.uniform(0.5, 0.9)
        return enriched_graph

# 4. Recommendation Engine Layer
class RecommenderEngine:
    def __init__(self, kg: nx.Graph):
        self.kg = kg
        self.model = None
        self.product_features = {}

    def _generate_features(self):
        # Simple feature engineering: using product attributes directly
        product_data = []
        product_ids = []
        for node_id, attributes in self.kg.nodes(data=True):
            if attributes.get("type") == "Product":
                product_ids.append(node_id)
                # Example: one-hot encode category and brand, use popularity if available
                features = [
                    1 if attributes.get("category") == "Electronics" else 0,
                    1 if attributes.get("brand") == "BrandA" else 0,
                    1 if attributes.get("brand") == "BrandB" else 0,
                    1 if attributes.get("brand") == "BrandC" else 0,
                    1 if attributes.get("brand") == "BrandD" else 0,
                    attributes.get("popularity", 0.0) # Use 0 if not completed by LLM
                ]
                product_data.append(features)
        self.product_features = dict(zip(product_ids, product_data))
        return pd.DataFrame(product_data, index=product_ids)

    def train_model(self):
        feature_df = self._generate_features()
        if not feature_df.empty:
            self.model = NearestNeighbors(n_neighbors=3, metric='cosine')
            self.model.fit(feature_df)

    def get_recommendations(self, product_id: str, num_recommendations: int = 5) -> List[str]:
        if self.model is None or product_id not in self.product_features:
            return []

        input_features = [self.product_features[product_id]]
        distances, indices = self.model.kneighbors(input_features, n_neighbors=num_recommendations + 1)

        recommended_product_indices = indices.flatten()[1:] # Exclude itself
        recommended_product_ids = [self._generate_features().index[i] for i in recommended_product_indices]
        return recommended_product_ids

# 5. API & User Interface Layer
app = FastAPI()

# Initialize components
product_data_store = ProductData()
kg_manager = KnowledgeGraphManager()
llm_enricher = LLMKGEnricher() # In a real app, API key would be passed

# Build initial KG
products_df = product_data_store.get_products()
kg_manager.build_initial_kg(products_df)

# Enrich KG with LLM
enriched_kg = kg_manager.get_kg()
for _, row in products_df.iterrows():
    product_id = row["product_id"]
    description = row["description"]
    # Simulate LLM operations
    discovered_entities = llm_enricher.discover_entities_llm(description)
    for entity in discovered_entities:
        if not enriched_kg.has_node(entity["entity_id"]):
            kg_manager.add_entity(entity["entity_id"], entity["entity_type"], {"name": entity["name"]})
    existing_entities = list(enriched_kg.nodes)
    extracted_relations = llm_enricher.extract_relations_llm(description, existing_entities)
    for rel in extracted_relations:
        if enriched_kg.has_node(rel["source"]) and enriched_kg.has_node(rel["target"]):
            kg_manager.add_relation(rel["source"], rel["target"], rel["type"])
enriched_kg = llm_enricher.complete_kg_llm(enriched_kg)

# Initialize and train recommender engine with the enriched KG
recommender = RecommenderEngine(enriched_kg)
recommender.train_model()

@app.get("/recommendations/{product_id}", response_model=List[str])
async def get_product_recommendations(product_id: str, num_recommendations: int = 5):
    recommendations = recommender.get_recommendations(product_id, num_recommendations)
    return recommendations

@app.get("/kg")
async def get_knowledge_graph() -> Dict:
    nodes_data = []
    for node_id, attributes in enriched_kg.nodes(data=True):
        nodes_data.append({"id": node_id, **attributes})
    edges_data = []
    for u, v, attributes in enriched_kg.edges(data=True):
        edges_data.append({"source": u, "target": v, **attributes})
    return {"nodes": nodes_data, "edges": edges_data}

# To run this FastAPI app, save it as main.py and run: uvicorn main:app --reload