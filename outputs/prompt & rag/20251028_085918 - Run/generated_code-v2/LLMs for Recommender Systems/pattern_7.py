import pandas as pd
import re
import uuid
from fastapi import FastAPI
from typing import List, Dict, Any

# --- 1. Data Ingestion & Preprocessing Layer ---
class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def ingest_product_data(self, product_data: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(product_data)
        df['processed_description'] = df['description'].apply(self.preprocess_text)
        return df

# --- 2. LLM-Enhanced Knowledge Graph Module ---
class LLMKnowledgeGraphModule:
    def __init__(self, llm_api_key: str = "dummy_key"):
        self.llm_api_key = llm_api_key
        self.knowledge_graph_triples = [] # Stores (subject, predicate, object)
        self.entities = {}

    def _call_llm(self, prompt: str) -> str:
        # Placeholder for LLM API call
        # In a real application, this would interact with OpenAI, Gemini, Hugging Face, etc.
        if "extract entities" in prompt:
            if "smartphone" in prompt:
                return "smartphone, brand:samsung, model:galaxy s23, feature:camera, feature:screen"
            return "entity:product, entity:brand"
        elif "extract relations" in prompt:
            if "samsung galaxy s23" in prompt and "camera" in prompt:
                return "samsung galaxy s23 HAS_FEATURE camera"
            return "product IS_A electronic"
        elif "predict missing fact" in prompt:
            if "weight of iphone 15" in prompt:
                return "weight:171g"
            return "attribute:value"
        return "LLM_RESPONSE_PLACEHOLDER"

    def discover_entities(self, text: str, product_id: str) -> List[Dict[str, str]]:
        prompt = f"Extract entities (product, brand, feature, material, use case) from the following text: {text}"
        llm_response = self._call_llm(prompt)
        extracted_entities = []
        for item in llm_response.split(', '):
            if ':' in item:
                key, value = item.split(':', 1)
                extracted_entities.append({"type": key.strip(), "value": value.strip(), "source_text": text, "product_id": product_id})
            else:
                extracted_entities.append({"type": "generic_entity", "value": item.strip(), "source_text": text, "product_id": product_id})
        return extracted_entities

    def extract_relations(self, subject_entity: Dict[str, str], object_entity: Dict[str, str], context_text: str) -> List[Dict[str, str]]:
        prompt = f"Given '{subject_entity['value']}' and '{object_entity['value']}' in the context: '{context_text}', extract the relationship between them."
        llm_response = self._call_llm(prompt)
        relations = []
        for rel_str in llm_response.split('; '):
            parts = rel_str.split(' ')
            if len(parts) >= 3:
                relations.append({"subject": subject_entity['value'], "predicate": parts[1], "object": object_entity['value']})
        return relations

    def complete_kg_facts(self, entity_name: str, missing_attribute: str) -> Dict[str, str]:
        prompt = f"Predict the missing fact for '{missing_attribute}' of '{entity_name}'."
        llm_response = self._call_llm(prompt)
        if ':' in llm_response:
            key, value = llm_response.split(':', 1)
            return {key.strip(): value.strip()}
        return {}

    def build_knowledge_graph(self, preprocessed_data: pd.DataFrame):
        for idx, row in preprocessed_data.iterrows():
            product_id = str(row['product_id'])
            description = row['processed_description']

            # Entity Discovery
            entities = self.discover_entities(description, product_id)
            product_entity = {"type": "product", "value": row['name'], "product_id": product_id}
            if product_entity not in entities:
                entities.append(product_entity)
            self.entities[product_id] = entities

            for ent1 in entities:
                self.knowledge_graph_triples.append((product_id, "HAS_ENTITY", ent1['value']))

            # Simplified Relation Extraction (between product and its discovered entities)
            for ent in entities:
                if ent['type'] != "product" and ent['value'] != row['name']:
                    relations = self.extract_relations(product_entity, ent, description)
                    for rel in relations:
                        self.knowledge_graph_triples.append((rel['subject'], rel['predicate'], rel['object']))

            # Knowledge Graph Completion (example: if price is missing, predict it)
            if 'price' not in row or pd.isna(row['price']):
                missing_fact = self.complete_kg_facts(row['name'], "price")
                if missing_fact: # Placeholder, would ideally update the original product data or KG
                    self.knowledge_graph_triples.append((row['name'], "HAS_PRICE", missing_fact.get('price')))

# --- 3. Knowledge Graph Storage (Simulated) ---
class GraphDatabase:
    def __init__(self):
        self._graph = [] # List of (subject, predicate, object) tuples

    def add_triple(self, s: str, p: str, o: str):
        self._graph.append((s, p, o))

    def query(self, subject: str = None, predicate: str = None, object: str = None) -> List[tuple]:
        results = []
        for s, p, o in self._graph:
            match_s = (subject is None or s == subject)
            match_p = (predicate is None or p == predicate)
            match_o = (object is None or o == object)
            if match_s and match_p and match_o:
                results.append((s, p, o))
        return results

    def bulk_add_triples(self, triples: List[tuple]):
        self._graph.extend(triples)

# --- 4. Recommendation Engine ---
class RecommendationEngine:
    def __init__(self, graph_db: GraphDatabase):
        self.graph_db = graph_db

    def get_similar_products(self, product_id: str, limit: int = 5) -> List[str]:
        # In a real system, this would involve KG embeddings, GNNs, or complex graph queries.
        # For this simulation, we'll find products sharing common entities/features.
        
        product_entities = [o for s, p, o in self.graph_db.query(subject=product_id, predicate="HAS_ENTITY")]
        
        similar_products = {}
        for entity in product_entities:
            # Find other products that have this entity
            related_triples = self.graph_db.query(object=entity, predicate="HAS_ENTITY")
            for s, p, o in related_triples:
                if s != product_id:
                    similar_products[s] = similar_products.get(s, 0) + 1
        
        # Sort by number of shared entities and return top N
        sorted_similar = sorted(similar_products.items(), key=lambda item: item[1], reverse=True)
        return [prod_id for prod_id, _ in sorted_similar[:limit]]

# --- 5. API & Frontend Layer ---
app = FastAPI()

# Initialize components (simulated)
data_preprocessor = DataPreprocessor()
llm_kg_module = LLMKnowledgeGraphModule()
graph_db = GraphDatabase()
recommendation_engine = RecommendationEngine(graph_db)

# Dummy product data
dummy_products_data = [
    {"product_id": "P1", "name": "Samsung Galaxy S23", "description": "Powerful Android smartphone with an amazing camera and long-lasting battery.", "category": "Electronics", "price": 799.99},
    {"product_id": "P2", "name": "iPhone 15 Pro", "description": "Latest Apple flagship phone with A17 Bionic chip and ProMotion display.", "category": "Electronics", "price": 999.99},
    {"product_id": "P3", "name": "Noise Cancelling Headphones", "description": "Over-ear headphones with superior sound quality and active noise cancellation. Great for travel.", "category": "Audio", "price": 249.00},
    {"product_id": "P4", "name": "Smartwatch Series 8", "description": "Fitness tracker and health monitor with GPS and heart rate sensor.", "category": "Wearables", "price": 349.50},
    {"product_id": "P5", "name": "Gaming Laptop RGB", "description": "High-performance laptop for gaming and creative tasks with RGB keyboard.", "category": "Computers", "price": 1499.00},
    {"product_id": "P6", "name": "Portable Bluetooth Speaker", "description": "Compact speaker with loud sound, waterproof design, and 10-hour battery life.", "category": "Audio", "price": 79.99},
    {"product_id": "P7", "name": "Android Tablet X", "description": "Large display tablet perfect for media consumption and light productivity.", "category": "Electronics", "price": None} # Missing price to test KG completion
]

@app.on_event("startup")
async def startup_event():
    print("Initializing data and building knowledge graph...")
    global processed_data
    processed_data = data_preprocessor.ingest_product_data(dummy_products_data)
    
    # Simulate product entities and their attributes for the graph
    for idx, row in processed_data.iterrows():
        product_id = str(row['product_id'])
        product_name = row['name']
        graph_db.add_triple(product_id, "IS_NAMED", product_name)
        graph_db.add_triple(product_id, "HAS_CATEGORY", row['category'])
        if row['price'] is not None:
            graph_db.add_triple(product_id, "HAS_PRICE", str(row['price']))
        else:
            # Simulate KG completion for missing price
            completed_fact = llm_kg_module.complete_kg_facts(product_name, "price")
            if completed_fact and "price" in completed_fact:
                graph_db.add_triple(product_id, "HAS_PRICE", completed_fact["price"])
                print(f"KG Completion: Predicted price for {product_name}: {completed_fact['price']}")

    # LLM-Enhanced KG construction
    llm_kg_module.build_knowledge_graph(processed_data)
    graph_db.bulk_add_triples(llm_kg_module.knowledge_graph_triples)
    print("Knowledge graph built successfully with LLM enhancements.")
    # print("Knowledge Graph Triples:", graph_db._graph) # For debugging

@app.get("/recommend/{product_id}", response_model=List[str])
async def get_recommendations(product_id: str):
    if product_id not in [p['product_id'] for p in dummy_products_data]:
        return []
    recommendations = recommendation_engine.get_similar_products(product_id)
    return recommendations

@app.get("/products", response_model=List[Dict[str, Any]])
async def get_all_products():
    return dummy_products_data

@app.get("/kg_triples", response_model=List[List[str]])
async def get_kg_triples():
    return [[str(s), str(p), str(o)] for s, p, o in graph_db._graph]

# To run the FastAPI app:
# Save the code as main.py
# Install uvicorn: pip install uvicorn fastapi pandas
# Run: uvicorn main:app --reload
# Access in browser: http://127.0.0.1:8000/docs

