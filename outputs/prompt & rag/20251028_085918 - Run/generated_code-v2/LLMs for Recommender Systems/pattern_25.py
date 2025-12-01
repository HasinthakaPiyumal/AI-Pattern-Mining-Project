import uvicorn
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class MockLLM:
    def __init__(self):
        pass

    def extract_entities(self, text: str) -> List[str]:
        if "dress" in text.lower():
            return ["dress", "clothing", "fashion"]
        if "shirt" in text.lower():
            return ["shirt", "clothing", "fashion"]
        if "jeans" in text.lower():
            return ["jeans", "pants", "denim"]
        if "bohemian" in text.lower():
            return ["bohemian", "style"]
        if "silk" in text.lower():
            return ["silk", "material"]
        if "cotton" in text.lower():
            return ["cotton", "material"]
        if "vintage" in text.lower():
            return ["vintage", "style"]
        if "retro" in text.lower():
            return ["retro", "style"]
        if "casual" in text.lower():
            return ["casual", "occasion"]
        if "summer" in text.lower():
            return ["summer", "season"]
        if "sandals" in text.lower():
            return ["sandals", "footwear"]
        return []

    def extract_relations(self, text: str, entities: List[str]) -> List[Dict[str, str]]:
        relations = []
        if "dress" in entities and "bohemian" in entities:
            relations.append({"source": "dress", "relation": "has_style", "target": "bohemian"})
        if "shirt" in entities and "cotton" in text.lower():
            relations.append({"source": "shirt", "relation": "made_of", "target": "cotton"})
        if "jeans" in entities and "denim" in entities:
            relations.append({"source": "jeans", "relation": "made_of", "target": "denim"})
        if "dress" in entities and "summer" in entities:
            relations.append({"source": "dress", "relation": "suitable_for_season", "target": "summer"})
        if "sandals" in entities and "bohemian" in entities:
            relations.append({"source": "sandals", "relation": "has_style", "target": "bohemian"})
        return relations

    def infer_missing_facts(self, kg: nx.Graph, product_nodes: List[str]) -> List[Dict[str, str]]:
        inferred_facts = []
        for i in range(len(product_nodes)):
            for j in range(i + 1, len(product_nodes)):
                prod1 = product_nodes[i]
                prod2 = product_nodes[j]

                # Simple common attribute check for similar_style or similar_occasion
                prod1_attrs = set([n for n in kg.neighbors(prod1) if kg.nodes[n].get("type") == "attribute"])
                prod2_attrs = set([n for n in kg.neighbors(prod2) if kg.nodes[n].get("type") == "attribute"])
                common_attrs = prod1_attrs.intersection(prod2_attrs)

                if "bohemian" in common_attrs and not kg.has_edge(prod1, prod2, "similar_style") and not kg.has_edge(prod2, prod1, "similar_style"):
                    inferred_facts.append({"source": prod1, "relation": "similar_style", "target": prod2})
                if "casual" in common_attrs and not kg.has_edge(prod1, prod2, "similar_occasion") and not kg.has_edge(prod2, prod1, "similar_occasion"):
                    inferred_facts.append({"source": prod1, "relation": "similar_occasion", "target": prod2})
                if "vintage" in common_attrs and "retro" in common_attrs and not kg.has_edge(prod1, prod2, "similar_style") and not kg.has_edge(prod2, prod1, "similar_style"):
                    inferred_facts.append({"source": prod1, "relation": "similar_style", "target": prod2})
        return inferred_facts

class KnowledgeGraphManager:
    def __init__(self):
        self.kg = nx.Graph()
        self.llm = MockLLM()

    def add_product_to_kg(self, product_id: str, description: str, attributes: List[str]):
        if not self.kg.has_node(product_id):
            self.kg.add_node(product_id, type="product", description=description, attributes=attributes)
        else:
            self.kg.nodes[product_id]["description"] = description
            self.kg.nodes[product_id]["attributes"] = list(set(self.kg.nodes[product_id]["attributes"] + attributes))

        for attr in attributes:
            if not self.kg.has_node(attr):
                self.kg.add_node(attr, type="attribute")
            self.kg.add_edge(product_id, attr, relation="has_attribute")

        extracted_entities = self.llm.extract_entities(description)
        for entity in extracted_entities:
            if not self.kg.has_node(entity):
                self.kg.add_node(entity, type="attribute")
            if not self.kg.has_edge(product_id, entity, relation="has_attribute"):
                self.kg.add_edge(product_id, entity, relation="has_attribute")

        extracted_relations = self.llm.extract_relations(description, extracted_entities + attributes)
        for rel in extracted_relations:
            source = rel["source"]
            target = rel["target"]
            relation_type = rel["relation"]
            if not self.kg.has_node(source):
                self.kg.add_node(source, type="extracted_entity")
            if not self.kg.has_node(target):
                self.kg.add_node(target, type="extracted_entity")
            if not self.kg.has_edge(source, target, relation_type):
                self.kg.add_edge(source, target, relation=relation_type)


    def complete_kg(self):
        product_nodes = [node for node, data in self.kg.nodes(data=True) if data.get("type") == "product"]
        inferred_facts = self.llm.infer_missing_facts(self.kg, product_nodes)
        for fact in inferred_facts:
            source = fact["source"]
            target = fact["target"]
            relation_type = fact["relation"]
            if self.kg.has_node(source) and self.kg.has_node(target) and not self.kg.has_edge(source, target, relation=relation_type):
                self.kg.add_edge(source, target, relation=relation_type)

    def get_related_products(self, product_id: str, depth: int = 2, max_recommendations: int = 10) -> List[str]:
        if not self.kg.has_node(product_id):
            return []

        related_nodes = set()
        try:
            for neighbor in nx.bfs_tree(self.kg, source=product_id, depth_limit=depth):
                if neighbor != product_id and self.kg.nodes[neighbor].get("type") == "product":
                    related_nodes.add(neighbor)
                elif neighbor != product_id and self.kg.nodes[neighbor].get("type") == "attribute":
                    for connected_node in self.kg.neighbors(neighbor):
                        if connected_node != product_id and self.kg.nodes[connected_node].get("type") == "product":
                            related_nodes.add(connected_node)
        except nx.NetworkXNoPath:
            pass

        return list(related_nodes)[:max_recommendations]

app = FastAPI(title="Fashion Recommendation System with LLM-enhanced KG")

kg_manager = KnowledgeGraphManager()

dummy_products = [
    {"product_id": "P001", "description": "Elegant bohemian floral maxi dress, perfect for summer.", "attributes": ["dress", "bohemian", "floral", "summer"]},
    {"product_id": "P002", "description": "Comfortable cotton t-shirt, suitable for casual wear.", "attributes": ["shirt", "cotton", "casual"]},
    {"product_id": "P003", "description": "Vintage high-waisted denim jeans, a classic look.", "attributes": ["jeans", "denim", "vintage"]},
    {"product_id": "P004", "description": "Flowy silk blouse, ideal for a chic evening.", "attributes": ["blouse", "silk", "evening", "chic"]},
    {"product_id": "P005", "description": "Retro-style midi skirt with a playful pattern.", "attributes": ["skirt", "retro", "midi", "patterned"]},
    {"product_id": "P006", "description": "Casual linen shorts for warm weather.", "attributes": ["shorts", "linen", "casual", "warm weather"]},
    {"product_id": "P007", "description": "Bohemian inspired sandals, completes the summer look.", "attributes": ["sandals", "bohemian", "summer"]}
]

for product in dummy_products:
    kg_manager.add_product_to_kg(product["product_id"], product["description"], product["attributes"])
kg_manager.complete_kg()

class ProductIngestRequest(BaseModel):
    product_id: str
    description: str
    attributes: List[str]

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[str]

class ProductRelatedResponse(BaseModel):
    product_id: str
    related_products: List[str]

@app.post("/products/ingest", response_model=ProductRelatedResponse)
async def ingest_product(product_data: ProductIngestRequest):
    kg_manager.add_product_to_kg(product_data.product_id, product_data.description, product_data.attributes)
    kg_manager.complete_kg()
    related = kg_manager.get_related_products(product_data.product_id)
    return {"product_id": product_data.product_id, "related_products": related}

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations(user_id: str):
    import random
    seed_product_id = None
    if user_id in kg_manager.kg.nodes and kg_manager.kg.nodes[user_id].get("type") == "product":
        seed_product_id = user_id
    else:
        if dummy_products:
            seed_product_id = random.choice([p["product_id"] for p in dummy_products])
        else:
            raise HTTPException(status_code=404, detail="No products available for recommendations.")

    recommendations = kg_manager.get_related_products(seed_product_id, depth=3, max_recommendations=5)
    return {"user_id": user_id, "recommendations": recommendations}

@app.get("/product/{product_id}/related", response_model=ProductRelatedResponse)
async def get_related_for_product(product_id: str):
    if not kg_manager.kg.has_node(product_id):
        raise HTTPException(status_code=404, detail="Product not found in knowledge graph.")
    related_products = kg_manager.get_related_products(product_id, depth=3)
    return {"product_id": product_id, "related_products": related_products}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)