import pandas as pd
import nltk
from nltk.corpus import stopwords
import networkx as nx
from fastapi import FastAPI
import uvicorn
from typing import List, Dict, Any
import json

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

app = FastAPI()

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        return " ".join(filtered_tokens)

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_entity(self, entity_id: str, entity_type: str, attributes: Dict[str, Any] = None):
        if not self.graph.has_node(entity_id):
            self.graph.add_node(entity_id, type=entity_type, **(attributes if attributes else {}))
            return True
        return False

    def add_relation(self, entity1_id: str, relation_type: str, entity2_id: str, attributes: Dict[str, Any] = None):
        if self.graph.has_node(entity1_id) and self.graph.has_node(entity2_id):
            self.graph.add_edge(entity1_id, entity2_id, type=relation_type, **(attributes if attributes else {}))
            return True
        return False

    def get_neighbors(self, entity_id: str, relation_type: str = None) -> List[Dict[str, Any]]:
        neighbors_data = []
        if self.graph.has_node(entity_id):
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if relation_type is None or edge_data.get("type") == relation_type:
                    neighbors_data.append({
                        "node_id": neighbor,
                        "node_type": self.graph.nodes[neighbor].get("type"),
                        "relation_type": edge_data.get("type"),
                        "node_attributes": {k: v for k, v in self.graph.nodes[neighbor].items() if k != "type"},
                        "relation_attributes": {k: v for k, v in edge_data.items() if k != "type"}
                    })
        return neighbors_data

    def get_entity_attributes(self, entity_id: str) -> Dict[str, Any]:
        if self.graph.has_node(entity_id):
            return {k: v for k, v in self.graph.nodes[entity_id].items() if k != "type"}
        return {}

class LLMEnhancer:
    def __init__(self):
        pass

    def _mock_llm_response(self, prompt: str) -> str:
        if "extract entities and relations" in prompt:
            if "Apple iPhone 15" in prompt:
                return json.dumps({
                    "entities": [
                        {"id": "product_iphone15", "type": "product", "name": "Apple iPhone 15"},
                        {"id": "brand_apple", "type": "brand", "name": "Apple"},
                        {"id": "feature_camerasystem", "type": "feature", "name": "advanced camera system"},
                        {"id": "feature_durability", "type": "feature", "name": "improved durability"}
                    ],
                    "relations": [
                        {"head": "product_iphone15", "type": "has_brand", "tail": "brand_apple"},
                        {"head": "product_iphone15", "type": "has_feature", "tail": "feature_camerasystem"},
                        {"head": "product_iphone15", "type": "has_feature", "tail": "feature_durability"}
                    ]
                })
            elif "Bluetooth headphones" in prompt:
                return json.dumps({
                    "entities": [
                        {"id": "product_headphones", "type": "product", "name": "Bluetooth headphones"},
                        {"id": "tech_bluetooth", "type": "technology", "name": "Bluetooth"},
                        {"id": "attribute_wireless", "type": "attribute", "name": "wireless"}
                    ],
                    "relations": [
                        {"head": "product_headphones", "type": "uses_technology", "tail": "tech_bluetooth"},
                        {"head": "product_headphones", "type": "is_type_of", "tail": "attribute_wireless"}
                    ]
                })
            elif "summer dress" in prompt:
                 return json.dumps({
                    "entities": [
                        {"id": "product_summerdress", "type": "product", "name": "elegant summer dress"},
                        {"id": "attribute_material_cotton", "type": "material", "name": "cotton"},
                        {"id": "attribute_style_flowy", "type": "style", "name": "flowy"},
                        {"id": "occasion_summer", "type": "occasion", "name": "summer"}
                    ],
                    "relations": [
                        {"head": "product_summerdress", "type": "made_of", "tail": "attribute_material_cotton"},
                        {"head": "product_summerdress", "type": "has_style", "tail": "attribute_style_flowy"},
                        {"head": "product_summerdress", "type": "suitable_for", "tail": "occasion_summer"}
                    ]
                })
        elif "complete missing attributes" in prompt:
            if "P004" in prompt: # Men's Casual T-Shirt
                return json.dumps({"material": "cotton", "neckline": "crew neck", "sleeve_length": "short sleeve"})
            elif "P005" in prompt: # Gaming Laptop
                return json.dumps({"processor": "Intel i7", "ram_gb": 16, "storage_gb": 512, "os": "Windows 11"})
        return json.dumps({"entities": [], "relations": []})

    def extract_entities_relations(self, text: str) -> Dict[str, Any]:
        prompt = f"Extract entities and relations from: {text}"
        mock_response = self._mock_llm_response(prompt)
        return json.loads(mock_response)

    def complete_attributes(self, entity_id: str, existing_attributes: Dict[str, Any], context: str) -> Dict[str, Any]:
        prompt = f"complete missing attributes for entity '{entity_id}' with existing attributes {existing_attributes} and context '{context}'"
        mock_response = self._mock_llm_response(prompt)
        return json.loads(mock_response)

class ProductRecommender:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.user_profiles: Dict[str, Dict[str, Any]] = {}

    def get_recommendations_for_user(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        if user_id not in self.user_profiles:
            all_products = [node for node, data in self.kg.graph.nodes(data=True) if data.get("type") == "product"]
            import random
            random.shuffle(all_products)
            return [{"product_id": p, **self.kg.get_entity_attributes(p)} for p in all_products[:limit]]

        user_prefs = self.user_profiles[user_id].get("preferences", {})
        liked_products = self.user_profiles[user_id].get("liked_products", [])

        candidate_products = set()

        for product_id in liked_products:
            for neighbor_data in self.kg.get_neighbors(product_id):
                if neighbor_data["node_type"] == "product" and neighbor_data["node_id"] not in liked_products:
                    candidate_products.add(neighbor_data["node_id"])

            product_attributes = self.kg.get_entity_attributes(product_id)
            for attr_key, attr_value in product_attributes.items():
                for node, data in self.kg.graph.nodes(data=True):
                    if data.get("type") == "product" and node != product_id and node not in liked_products and \
                       data.get(attr_key) == attr_value:
                        candidate_products.add(node)

        if user_prefs:
            filtered_candidates = []
            for prod_id in candidate_products:
                prod_attrs = self.kg.get_entity_attributes(prod_id)
                match = True
                for pref_key, pref_value in user_prefs.items():
                    if prod_attrs.get(pref_key) != pref_value:
                        match = False
                        break
                if match:
                    filtered_candidates.append(prod_id)
            candidate_products = filtered_candidates

        recommendations = []
        for prod_id in list(candidate_products)[:limit]:
            recommendations.append({"product_id": prod_id, **self.kg.get_entity_attributes(prod_id)})

        return recommendations

    def update_user_profile(self, user_id: str, new_data: Dict[str, Any]):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {"liked_products": [], "preferences": {}}
        self.user_profiles[user_id].update(new_data)


text_preprocessor = TextPreprocessor()
kg_store = KnowledgeGraph()
llm_enhancer = LLMEnhancer()
recommender = ProductRecommender(kg_store)

def ingest_and_enhance_data():
    raw_products_data = [
        {"id": "P001", "name": "Apple iPhone 15", "description": "The latest iPhone with an advanced camera system and improved durability.", "category": "Electronics"},
        {"id": "P002", "name": "Wireless Bluetooth Headphones", "description": "Comfortable over-ear Bluetooth headphones with noise cancellation.", "category": "Electronics"},
        {"id": "P003", "name": "Elegant Summer Dress", "description": "A flowy cotton dress perfect for summer outings.", "category": "Apparel", "material": "cotton"},
        {"id": "P004", "name": "Men's Casual T-Shirt", "description": "Soft cotton t-shirt, great for everyday wear.", "category": "Apparel"},
        {"id": "P005", "name": "Gaming Laptop", "description": "High-performance laptop for gaming and creative tasks.", "category": "Electronics"}
    ]

    print("--- Ingesting and Enhancing Data ---")
    for product in raw_products_data:
        product_id = product["id"]
        product_name = product["name"]
        product_desc = product["description"]
        category = product["category"]

        processed_desc = text_preprocessor.preprocess_text(product_desc)
        print(f"Processing product: {product_name}")

        kg_store.add_entity(product_id, "product", {"name": product_name, "category": category, "description_raw": product_desc, "description_processed": processed_desc})

        extracted_data = llm_enhancer.extract_entities_relations(product_desc)
        for entity in extracted_data.get("entities", []):
            kg_store.add_entity(entity["id"], entity["type"], {k: v for k, v in entity.items() if k not in ["id", "type"]})
        for relation in extracted_data.get("relations", []):
            kg_store.add_relation(relation["head"], relation["type"], relation["tail"])

        current_attributes = kg_store.get_entity_attributes(product_id)
        if not current_attributes.get("material") and category == "Apparel" or not current_attributes.get("processor") and category == "Electronics":
            print(f"Attempting to complete attributes for {product_name}")
            completed_attrs = llm_enhancer.complete_attributes(product_id, current_attributes, product_desc)
            if completed_attrs:
                for k, v in completed_attrs.items():
                    kg_store.graph.nodes[product_id][k] = v
                print(f"Completed attributes for {product_id}: {completed_attrs}")


    recommender.update_user_profile("user123", {
        "liked_products": ["P001", "P002"],
        "preferences": {"category": "Electronics"}
    })
    recommender.update_user_profile("user456", {
        "liked_products": ["P003"],
        "preferences": {"material": "cotton", "category": "Apparel"}
    })

    print("\n--- Knowledge Graph Content (Sample) ---")
    print(f"Number of nodes: {kg_store.graph.number_of_nodes()}")
    print(f"Number of edges: {kg_store.graph.number_of_edges()}")
    print("\nP001 (Apple iPhone 15) details:")
    print(kg_store.get_entity_attributes("P001"))
    print("Neighbors:")
    for neighbor in kg_store.get_neighbors("P001"):
        print(f"- {neighbor['node_id']} ({neighbor['node_type']}) via {neighbor['relation_type']}")

ingest_and_enhance_data()

@app.get("/")
async def read_root():
    return {"message": "Smart Product Recommender API"}

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str, limit: int = 5):
    recommendations = recommender.get_recommendations_for_user(user_id, limit)
    if not recommendations:
        return {"message": f"No recommendations found for user {user_id}. Try updating user profile or check KG."}
    return {"user_id": user_id, "recommendations": recommendations}

@app.post("/user_profile/{user_id}")
async def update_user_profile_api(user_id: str, profile_data: Dict[str, Any]):
    recommender.update_user_profile(user_id, profile_data)
    return {"message": f"User {user_id} profile updated."}

@app.get("/product/{product_id}")
async def get_product_details(product_id: str):
    attributes = kg_store.get_entity_attributes(product_id)
    if not attributes:
        return {"message": f"Product {product_id} not found."}
    relations = kg_store.get_neighbors(product_id)
    return {"product_id": product_id, "attributes": attributes, "relations": relations}
