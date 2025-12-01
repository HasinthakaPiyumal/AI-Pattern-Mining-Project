import os
import json
from neo4j import GraphDatabase
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Config:
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    LLM_API_KEY = os.getenv("LLM_API_KEY", "YOUR_LLM_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    PROMPT_TEMPLATE_ENTITY_EXTRACTION = "Extract key entities (Product, Brand, Category, Feature) and their attributes (e.g., color, material, use_case) from the following product description. Provide output in JSON format: {description}"
    PROMPT_TEMPLATE_RELATION_EXTRACTION = "Given the following product description and a list of known products, identify any relationships (e.g., 'is_complementary_to', 'is_alternative_to', 'belongs_to_category') between the described product and known products. Also, extract relationships within the description. Provide output in JSON format. Product description: {description}. Known products (JSON array of {'id': '...', 'name': '...'}): {known_products}"
    PROMPT_TEMPLATE_ATTRIBUTE_INFERENCE = "Infer additional relevant attributes for the product described by: {description}. Focus on attributes useful for recommendations, such as 'use_case', 'target_audience', 'material', 'style'. Provide output in JSON format: {description}"

class ProductInput(BaseModel):
    id: str
    name: str
    description: str
    price: float
    category: Optional[str] = "General"
    brand: Optional[str] = "Unknown"

class KnowledgeGraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        self.llm_client = None # Placeholder. In a real app, initialize OpenAI() or similar

    def close(self):
        if self.driver:
            self.driver.close()

    def _run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def _interact_with_llm(self, prompt: str, model: str = Config.LLM_MODEL) -> Dict[str, Any]:
        print(f"[LLM Call] Prompt: {prompt[:200]}...")

        # --- DUMMY LLM RESPONSE FOR DEMONSTRATION ---
        if "Extract key entities" in prompt:
            # Simulate extraction for a generic product
            if "Smartwatch Pro X" in prompt:
                return {"entities": {"Product": "Smartwatch Pro X", "Brand": "TechGear", "Category": "Wearables", "Feature": "GPS"}, "attributes": {"color": "black", "waterproof": True}}
            elif "Wireless Earbuds Elite" in prompt:
                return {"entities": {"Product": "Wireless Earbuds Elite", "Brand": "AudioWave", "Category": "Audio", "Feature": "Noise Cancellation"}, "attributes": {"connectivity": "Bluetooth 5.0", "battery_life_hours": 8}}
            return {"entities": {"Product": "Generic Product", "Brand": "Generic Brand", "Category": "Misc", "Feature": "Standard"}, "attributes": {}}
        elif "identify any relationships" in prompt:
            # Simulate relationship inference
            if "Smartwatch Pro X" in prompt and "Wireless Earbuds Elite" in prompt:
                return {"relationships": [{"product1": "prod123", "type": "is_complementary_to", "product2": "prod124"}]}
            return {"relationships": []}
        elif "Infer additional relevant attributes" in prompt:
            # Simulate attribute inference
            if "Smartwatch Pro X" in prompt:
                return {"inferred_attributes": {"use_case": "fitness tracking, daily communication", "target_audience": "active individuals"}}
            elif "Wireless Earbuds Elite" in prompt:
                return {"inferred_attributes": {"use_case": "music listening, calls, workouts", "sound_profile": "balanced"}}
            return {"inferred_attributes": {}}
        return {}
        # --- END DUMMY LLM RESPONSE ---

    def _create_product_node(self, tx, product_id, name, brand, category, attributes):
        query = (
            f"MERGE (p:Product {{id: $product_id}})"
            f" ON CREATE SET p.name = $name, p.brand = $brand, p.category = $category, p.created_at = timestamp()"
            f" ON MATCH SET p.name = $name, p.brand = $brand, p.category = $category, p.updated_at = timestamp()"
            f" SET p += $attributes"
            f" RETURN p"
        )
        tx.run(query, product_id=product_id, name=name, brand=brand, category=category, attributes=attributes)

    def _create_relationship(self, tx, product_id1, relation_type, product_id2):
        query = (
            f"MATCH (p1:Product {{id: $product_id1}})"
            f"MATCH (p2:Product {{id: $product_id2}})"
            f"MERGE (p1)-[r:{relation_type}]->(p2)"
            f" RETURN r"
        )
        tx.run(query, product_id1=product_id1, product_id2=product_id2)

    def enrich_product_data(self, product_data: ProductInput):
        product_id = product_data.id
        description = product_data.description
        name = product_data.name
        initial_category = product_data.category
        initial_brand = product_data.brand

        # Step 1: Extract entities and initial attributes using LLM
        entity_extraction_prompt = Config.PROMPT_TEMPLATE_ENTITY_EXTRACTION.format(description=description)
        extracted_info = self._interact_with_llm(entity_extraction_prompt)
        
        entities = extracted_info.get("entities", {})
        extracted_attributes = extracted_info.get("attributes", {})
        
        # Combine initial product data with LLM extracted attributes
        attributes = {"description": description, "price": product_data.price}
        attributes.update(extracted_attributes)

        # Use LLM extracted entity if more specific, otherwise use initial data
        final_name = entities.get("Product", name)
        final_brand = entities.get("Brand", initial_brand)
        final_category = entities.get("Category", initial_category)

        # Step 2: Infer additional attributes using LLM
        attribute_inference_prompt = Config.PROMPT_TEMPLATE_ATTRIBUTE_INFERENCE.format(description=description)
        inferred_info = self._interact_with_llm(attribute_inference_prompt)
        attributes.update(inferred_info.get("inferred_attributes", {}))

        # Step 3: Update/Create Product Node in KG
        with self.driver.session() as session:
            session.write_transaction(self._create_product_node,
                                      product_id,
                                      final_name,
                                      final_brand,
                                      final_category,
                                      attributes)
            print(f"Product {product_id} node created/updated in KG.")
            
            # Step 4: Extract and create relationships using LLM
            known_products_query = "MATCH (p:Product) RETURN p.id, p.name LIMIT 100"
            known_products_data = self._run_query(known_products_query)
            known_products_str = json.dumps([{"id": r["p.id"], "name": r["p.name"]} for r in known_products_data])

            relation_extraction_prompt = Config.PROMPT_TEMPLATE_RELATION_EXTRACTION.format(
                description=description,
                known_products=known_products_str
            )
            relations_info = self._interact_with_llm(relation_extraction_prompt)

            for rel in relations_info.get("relationships", []):
                product1_id = rel.get("product1")
                relation_type = rel.get("type")
                product2_id = rel.get("product2")
                
                # Ensure product IDs are valid and not the same
                if product1_id and relation_type and product2_id and product1_id != product2_id:
                    # Prioritize relationships involving the current product
                    if product1_id == product_id:
                        session.write_transaction(self._create_relationship, product_id, relation_type, product2_id)
                        print(f"Relationship {product_id} -[{relation_type}]-> {product2_id} created.")
                    elif product2_id == product_id:
                        # Handle reverse relationship if LLM identifies it
                        session.write_transaction(self._create_relationship, product1_id, relation_type, product_id)
                        print(f"Relationship {product1_id} -[{relation_type}]-> {product_id} created.")

class RecommendationEngine:
    def __init__(self, kg_builder: KnowledgeGraphBuilder):
        self.kg_builder = kg_builder

    def get_complementary_products(self, product_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        query = (
            f"MATCH (p:Product {{id: $product_id}})-[:is_complementary_to]->(rec:Product)"
            f"RETURN rec.id AS id, rec.name AS name, rec.description AS description, rec.price AS price"
            f" LIMIT $limit"
        )
        return self.kg_builder._run_query(query, {"product_id": product_id, "limit": limit})

    def get_alternative_products(self, product_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        query = (
            f"MATCH (p:Product {{id: $product_id}})-[:is_alternative_to]->(rec:Product)"
            f"RETURN rec.id AS id, rec.name AS name, rec.description AS description, rec.price AS price"
            f" LIMIT $limit"
        )
        return self.kg_builder._run_query(query, {"product_id": product_id, "limit": limit})

    def get_similar_products_by_category(self, product_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        query = (
            f"MATCH (p:Product {{id: $product_id}})"
            f"MATCH (p)-[:BELONGS_TO_CATEGORY]->(c:Category)<-[:BELONGS_TO_CATEGORY]-(rec:Product)"
            f"WHERE p <> rec"
            f"RETURN rec.id AS id, rec.name AS name, rec.description AS description, rec.price AS price"
            f" LIMIT $limit"
        )
        return self.kg_builder._run_query(query, {"product_id": product_id, "limit": limit})

    def get_recommendations_for_product(self, product_id: str, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        complementary = self.get_complementary_products(product_id, limit)
        alternatives = self.get_alternative_products(product_id, limit)
        similar_by_category = self.get_similar_products_by_category(product_id, limit)

        return {
            "complementary_products": complementary,
            "alternative_products": alternatives,
            "similar_products_by_category": similar_by_category
        }

app = FastAPI()
kg_builder = KnowledgeGraphBuilder()
recommender = RecommendationEngine(kg_builder)

@app.on_event("startup")
async def startup_event():
    # Ensure Neo4j constraints/indexes are in place
    with kg_builder.driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")

@app.on_event("shutdown")
async def shutdown_event():
    kg_builder.close()

@app.post("/ingest_product", summary="Ingest and enrich new product data into the Knowledge Graph")
async def ingest_product_data(product: ProductInput):
    try:
        kg_builder.enrich_product_data(product)
        return {"message": f"Product {product.id} ingested and KG enriched successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest product: {e}")

@app.get("/recommendations/{product_id}", summary="Get recommendations for a given product")
async def get_product_recommendations(product_id: str):
    try:
        recommendations = recommender.get_recommendations_for_product(product_id)
        if not any(recommendations.values()): # Check if any recommendation list is non-empty
            raise HTTPException(status_code=404, detail=f"No recommendations found for product {product_id}. It might not exist or have no relationships.")
        return recommendations
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recommendations: {e}")

# To run this application:
# 1. Ensure Neo4j is running (e.g., via Docker: docker run --name neo4j-llm-kg -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j)
# 2. Install dependencies: pip install fastapi uvicorn neo4j pydantic
# 3. Save this code as main.py
# 4. Run from your terminal: uvicorn main:app --reload
# 5. Access the API documentation at http://127.0.0.1:8000/docs

# Example usage in FastAPI Swagger UI (http://127.0.0.1:8000/docs):
# POST /ingest_product with body:
# {
#   "id": "prod123",
#   "name": "Smartwatch Pro X",
#   "description": "A cutting-edge smartwatch with health tracking, GPS, and a long-lasting battery. Compatible with iOS and Android. Features a high-resolution AMOLED display and waterproof design.",
#   "price": 299.99
# }
# {
#   "id": "prod124",
#   "name": "Wireless Earbuds Elite",
#   "description": "Premium wireless earbuds with noise cancellation and crystal-clear audio. Perfect companion for your smartwatch during workouts. Comes with a portable charging case.",
#   "price": 149.99
# }

# GET /recommendations/prod123
