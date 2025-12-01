import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from py2neo import Graph, Node, Relationship
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# NLTK downloads (run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Configuration ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 1. Data Ingestion & Preprocessing Layer ---
class DataIngestor:
    def __init__(self, data_path: str = None):
        self.data_path = data_path

    def load_product_data(self) -> pd.DataFrame:
        # Simulate loading data from a CSV or similar source
        if self.data_path and os.path.exists(self.data_path):
            try:
                df = pd.read_csv(self.data_path)
                return df
            except Exception as e:
                print(f"Error loading data from {self.data_path}: {e}")
                # Fallback to dummy data if loading fails
        
        print("Loading dummy product data for demonstration.")
        data = {
            "product_id": ["P001", "P002", "P003", "P004"],
            "product_name": ["Laptop X1", "Mechanical Keyboard Z", "Wireless Mouse M", "Monitor Pro-HDR"],
            "description": [
                "Powerful laptop with 16GB RAM, 512GB SSD, and 14-inch display, ideal for professionals and students.",
                "High-performance mechanical keyboard with RGB backlighting and tactile switches, perfect for gaming and typing.",
                "Ergonomic wireless mouse with long battery life and adjustable DPI, suitable for office and casual use.",
                "4K HDR monitor with 27-inch IPS panel and USB-C connectivity, excellent for creative work and entertainment."
            ],
            "category": ["Electronics", "Electronics", "Electronics", "Electronics"],
            "brand": ["TechPro", "GamerGear", "OfficeMate", "VisualTech"],
            "price": [1200.00, 85.00, 30.00, 450.00]
        }
        return pd.DataFrame(data)

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(filtered_tokens)

# --- 2. LLM-Enhanced Knowledge Graph (KG) Module ---
class LLMKGBuilder:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in your .env file.")
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o") # Using gpt-4o for better performance
        
    def _call_llm(self, prompt_template: str, input_variables: Dict[str, Any]) -> str:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = chain.invoke(input_variables)
        return response.content

    def extract_entities_relations(self, text: str) -> Dict[str, Any]:
        prompt_template = """
        Extract entities and relationships from the following product text. 
        Entities should be 'Product', 'Brand', 'Category', 'Feature'.
        Relationships should describe connections between these entities.
        Respond in a JSON format with 'entities' (list of dicts with 'type' and 'name') and 'relationships' (list of dicts with 'source', 'target', 'type').

        Example Input: "The 'Laptop X1' by 'TechPro' is a powerful 'laptop' with '16GB RAM'."
        Example Output: 
        {{
            "entities": [
                {{"type": "Product", "name": "Laptop X1"}},
                {{"type": "Brand", "name": "TechPro"}},
                {{"type": "Category", "name": "laptop"}},
                {{"type": "Feature", "name": "16GB RAM"}}
            ],
            "relationships": [
                {{"source": "Laptop X1", "target": "TechPro", "type": "HAS_BRAND"}},
                {{"source": "Laptop X1", "target": "laptop", "type": "IS_A"}},
                {{"source": "Laptop X1", "target": "16GB RAM", "type": "HAS_FEATURE"}}
            ]
        }}

        Product Text: {text}
        """
        try:
            response_content = self._call_llm(prompt_template, {"text": text})
            # LLM might add extra text or formatting, try to parse JSON robustly
            import json
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response_content[start_idx : end_idx + 1]
                return json.loads(json_str)
            else:
                print(f"Warning: Could not find valid JSON in LLM response: {response_content}")
                return {"entities": [], "relationships": []}
        except Exception as e:
            print(f"Error extracting entities/relations: {e}")
            return {"entities": [], "relationships": []}

    def complete_kg_facts(self, product_description: str, existing_facts: Dict[str, Any]) -> Dict[str, Any]:
        prompt_template = """
        Given the following product description and existing known facts, infer and provide 2-3 additional facts (features or relationships) that are highly likely but not explicitly stated. 
        Focus on common usage, complementary products, or ideal user scenarios.
        Respond in a JSON format with 'inferred_facts' (list of dicts, each with 'type', 'subject', 'predicate', 'object').

        Product Description: {product_description}
        Existing Facts: {existing_facts}

        Example Output:
        {{
            "inferred_facts": [
                {{"type": "Relationship", "subject": "Laptop X1", "predicate": "SUITABLE_FOR", "object": "Software Development"}},
                {{"type": "Relationship", "subject": "Laptop X1", "predicate": "COMPLEMENTARY_WITH", "object": "Monitor Pro-HDR"}}
            ]
        }}
        """
        try:
            response_content = self._call_llm(prompt_template, {"product_description": product_description, "existing_facts": existing_facts})
            import json
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response_content[start_idx : end_idx + 1]
                return json.loads(json_str)
            else:
                print(f"Warning: Could not find valid JSON in LLM response for KG completion: {response_content}")
                return {"inferred_facts": []}
        except Exception as e:
            print(f"Error completing KG facts: {e}")
            return {"inferred_facts": []}

    def distill_commonsense_knowledge(self, entity_name: str, entity_type: str) -> List[str]:
        prompt_template = """
        Provide 2-3 common-sense facts or typical uses related to '{entity_name}' (a {entity_type}). 
        Each fact should be a short, distinct statement.
        Example Output: ["Often used for professional design work.", "Requires a high-end graphics card."]
        """
        try:
            response_content = self._call_llm(prompt_template, {"entity_name": entity_name, "entity_type": entity_type})
            import json
            start_idx = response_content.find('[')
            end_idx = response_content.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = response_content[start_idx : end_idx + 1]
                return json.loads(json_str)
            else:
                print(f"Warning: Could not find valid JSON list in LLM response for commonsense: {response_content}")
                return []
        except Exception as e:
            print(f"Error distilling commonsense: {e}")
            return []

# --- 3. Knowledge Graph (KG) Storage Layer ---
class KnowledgeGraphManager:
    def __init__(self, uri, user, password):
        try:
            self.graph = Graph(uri, auth=(user, password))
            self.graph.verify_connectivity()
            print("Connected to Neo4j successfully.")
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")
            self.graph = None

    def add_product(self, product_id: str, name: str, description: str, category: str, brand: str, price: float):
        if not self.graph: return
        product_node = Node("Product", id=product_id, name=name, description=description, category=category, brand=brand, price=price)
        self.graph.merge(product_node, "Product", "id")
        return product_node

    def add_entity(self, entity_type: str, name: str):
        if not self.graph: return
        node = Node(entity_type, name=name)
        self.graph.merge(node, entity_type, "name")
        return node

    def add_relationship(self, source_name: str, source_type: str, target_name: str, target_type: str, rel_type: str):
        if not self.graph: return
        source_node = self.graph.nodes.match(source_type, name=source_name).first()
        target_node = self.graph.nodes.match(target_type, name=target_name).first()
        if source_node and target_node:
            rel = Relationship(source_node, rel_type, target_node)
            self.graph.merge(rel)
            return rel
        print(f"Warning: Could not create relationship between {source_name} ({source_type}) and {target_name} ({target_type}). Nodes not found.")
        return None

    def get_product_kg_data(self, product_id: str) -> List[Dict[str, Any]]:
        if not self.graph: return []
        query = f"""
        MATCH (p:Product {{id: '{product_id}'}})-[r]-(o)
        RETURN p.name AS product_name, TYPE(r) AS relationship_type, o.name AS related_entity_name, labels(o) AS related_entity_labels
        """
        records = self.graph.query(query).data()
        return records

    def get_related_products(self, product_id: str, limit: int = 5) -> List[str]:
        if not self.graph: return []
        # Simple example: find products sharing a category or brand
        query = f"""
        MATCH (p:Product {{id: '{product_id}'}})-[]->(b:Brand)<-[]-(rec:Product)
        WHERE p <> rec
        RETURN DISTINCT rec.id AS recommended_product_id
        LIMIT {limit}
        """
        records = self.graph.query(query).data()
        return [r["recommended_product_id"] for r in records]

# --- 4. Recommendation Engine Layer ---
class RecommendationEngine:
    def __init__(self, kg_manager: KnowledgeGraphManager):
        self.kg_manager = kg_manager

    def get_recommendations(self, user_id: str, product_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        # In a real system, user_id would influence recommendations (e.g., via user-item graph, past purchases)
        # For this demo, we'll focus on product-product recommendations via KG.
        
        # Get directly related products from KG (e.g., same brand, similar features)
        recommended_product_ids = self.kg_manager.get_related_products(product_id, limit)
        
        recommendations = []
        for rec_id in recommended_product_ids:
            # Fetch details for recommended products (simplified for demo)
            product_node = self.kg_manager.graph.nodes.match("Product", id=rec_id).first()
            if product_node:
                recommendations.append({"product_id": product_node["id"], "name": product_node["name"], "reason": "Related product"})
            
        # Fallback/enrichment: if not enough direct recommendations, use LLM for complementary suggestions
        if len(recommendations) < limit:
            print(f"Generating LLM-based complementary suggestions for {product_id}...")
            current_product_node = self.kg_manager.graph.nodes.match("Product", id=product_id).first()
            if current_product_node:
                product_desc = current_product_node['description']
                # This LLM call would be more sophisticated in a real app, perhaps considering user persona
                llm_kg_builder = LLMKGBuilder(OPENAI_API_KEY)
                complementary_suggestions = llm_kg_builder.distill_commonsense_knowledge(current_product_node['name'], "Product")
                
                for suggestion in complementary_suggestions:
                    # In a real system, we'd try to map these to actual products in the KG
                    if len(recommendations) < limit:
                        recommendations.append({"product_id": None, "name": suggestion, "reason": "LLM inferred complementary use"})

        return recommendations

# --- 5. API & User Interface Layer (FastAPI) ---
app = FastAPI(
    title="LLM-Enhanced E-commerce Recommender",
    description="API for personalized product recommendations leveraging an LLM-enriched Knowledge Graph."
)

# Initialize modules
# DataIngestor is used to simulate data source
data_ingestor = DataIngestor()

# Initialize LLM KG Builder
llm_kg_builder = None
if OPENAI_API_KEY:
    try:
        llm_kg_builder = LLMKGBuilder(OPENAI_API_KEY)
    except ValueError as e:
        print(e)
        print("LLM functionality will be limited.")
else:
    print("OPENAI_API_KEY not found. LLM functionalities will be disabled.")

# Initialize Knowledge Graph Manager
kg_manager = KnowledgeGraphManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Initialize Recommendation Engine
recommendation_engine = RecommendationEngine(kg_manager)


@app.on_event("startup")
async def startup_event():
    print("Application startup: Initializing KG and populating with data...")
    # --- Populate KG on startup (for demonstration) ---
    df = data_ingestor.load_product_data()
    if df.empty: return

    # Clear existing data for a clean start if needed (caution in production)
    if kg_manager.graph:
        kg_manager.graph.delete_all()
        print("Cleared existing Neo4j data.")

    for index, row in df.iterrows():
        product_id = row["product_id"]
        product_name = row["product_name"]
        description = row["description"]
        category = row["category"]
        brand = row["brand"]
        price = row["price"]
        
        # Add product node
        product_node = kg_manager.add_product(product_id, product_name, description, category, brand, price)
        if not product_node: continue

        # Add Brand node and relationship
        brand_node = kg_manager.add_entity("Brand", brand)
        if brand_node:
            kg_manager.add_relationship(product_name, "Product", brand, "Brand", "HAS_BRAND")
        
        # Add Category node and relationship
        category_node = kg_manager.add_entity("Category", category)
        if category_node:
            kg_manager.add_relationship(product_name, "Product", category, "Category", "BELONGS_TO_CATEGORY")

        if llm_kg_builder:
            # Use LLM for Entity & Relation Extraction
            print(f"Extracting entities/relations for {product_name}...")
            extracted_data = llm_kg_builder.extract_entities_relations(description)
            for entity in extracted_data.get("entities", []):
                if entity["type"] not in ["Product", "Brand", "Category"]: # Already added
                    kg_manager.add_entity(entity["type"], entity["name"])
            for rel in extracted_data.get("relationships", []):
                # Ensure source/target nodes exist before adding relation from LLM output
                source_node_type = next((e["type"] for e in extracted_data.get("entities", []) if e["name"] == rel["source"]), "Unknown")
                target_node_type = next((e["type"] for e in extracted_data.get("entities", []) if e["name"] == rel["target"]), "Unknown")
                if source_node_type != "Unknown" and target_node_type != "Unknown":
                    kg_manager.add_entity(source_node_type, rel["source"]) # Ensure nodes exist before linking
                    kg_manager.add_entity(target_node_type, rel["target"])
                    kg_manager.add_relationship(rel["source"], source_node_type, rel["target"], target_node_type, rel["type"])

            # Use LLM for KG Completion (inferring new facts)
            print(f"Completing KG facts for {product_name}...")
            existing_facts = kg_manager.get_product_kg_data(product_id) # Get current facts for context
            inferred_facts_data = llm_kg_builder.complete_kg_facts(description, existing_facts)
            for fact in inferred_facts_data.get("inferred_facts", []):
                if fact["type"] == "Relationship":
                    # This is simplified. In a real scenario, you'd map these to existing nodes or create new generic ones.
                    subject_type = "Product" if fact["subject"] == product_name else "Feature"
                    object_type = "Product" if kg_manager.graph.nodes.match("Product", name=fact["object"]).first() else "Concept"
                    kg_manager.add_entity(subject_type, fact["subject"])
                    kg_manager.add_entity(object_type, fact["object"])
                    kg_manager.add_relationship(fact["subject"], subject_type, fact["object"], object_type, fact["predicate"])

            # Use LLM for Commonsense Knowledge Distillation
            print(f"Distilling commonsense for {product_name}...")
            commonsense_facts = llm_kg_builder.distill_commonsense_knowledge(product_name, "Product")
            for fact_text in commonsense_facts:
                # Add commonsense as a direct relationship to the product or a general Concept node
                fact_node = kg_manager.add_entity("CommonsenseFact", fact_text)
                if fact_node:
                    kg_manager.add_relationship(product_name, "Product", fact_text, "CommonsenseFact", "HAS_COMMONSENSE")
    
    print("KG population complete.")


class RecommendRequest(BaseModel):
    user_id: str
    product_id: str
    limit: int = 5

class Recommendation(BaseModel):
    product_id: str = None # Can be None if it's an LLM-inferred concept
    name: str
    reason: str

@app.post("/recommend", response_model=List[Recommendation])
async def get_product_recommendations(request: RecommendRequest):
    if not kg_manager.graph:
        raise HTTPException(status_code=500, detail="Knowledge Graph not initialized or connected.")
    
    # Check if the requested product_id exists
    product_exists = kg_manager.graph.nodes.match("Product", id=request.product_id).first()
    if not product_exists:
        raise HTTPException(status_code=404, detail=f"Product with ID {request.product_id} not found in KG.")

    try:
        recommendations = recommendation_engine.get_recommendations(request.user_id, request.product_id, request.limit)
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found for this product.")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.get("/product_kg_details/{product_id}")
async def get_product_kg_details(product_id: str) -> List[Dict[str, Any]]:
    if not kg_manager.graph:
        raise HTTPException(status_code=500, detail="Knowledge Graph not initialized or connected.")
    
    product_exists = kg_manager.graph.nodes.match("Product", id=product_id).first()
    if not product_exists:
        raise HTTPException(status_code=404, detail=f"Product with ID {product_id} not found in KG.")

    try:
        kg_data = kg_manager.get_product_kg_data(product_id)
        if not kg_data:
            raise HTTPException(status_code=404, detail="No KG details found for this product.")
        return kg_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving KG details: {str(e)}")

# --- Instructions to run this application: ---
# 1.  Install dependencies: 
#     pip install pandas nltk py2neo "fastapi[all]" python-dotenv langchain-openai
# 2.  Set up a Neo4j instance (Docker recommended):
#     docker run --name neo4j-llm-kg -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
# 3.  Create a '.env' file in the same directory as this script with your API keys and Neo4j credentials:
#     OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
#     NEO4J_URI="bolt://localhost:7687"
#     NEO4J_USER="neo4j"
#     NEO4J_PASSWORD="password"
# 4.  Run the FastAPI application:
#     uvicorn main:app --reload --port 8000
# 5.  Access the API documentation at: http://localhost:8000/docs
#
# To test recommendations:
# POST to http://localhost:8000/recommend with body: {"user_id": "user123", "product_id": "P001", "limit": 3}
# To view KG details:
# GET http://localhost:8000/product_kg_details/P001
