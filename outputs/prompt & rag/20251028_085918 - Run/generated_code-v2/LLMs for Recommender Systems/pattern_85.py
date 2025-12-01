import networkx as nx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List

# --- Knowledge Graph Builder Class ---
class KnowledgeGraphBuilder:
    """
    Manages the construction and completion of the knowledge graph using simulated LLM interactions.
    In a real application, this would integrate with an actual LLM API (e.g., OpenAI, Google Gemini).
    """
    def __init__(self, llm_client=None):
        self.kg = nx.DiGraph() # Using NetworkX for in-memory graph representation
        self.llm_client = llm_client # Placeholder for actual LLM client instance

    def _simulate_llm_response(self, prompt: str, task_type: str) -> str:
        """
        Simulates LLM responses for knowledge graph completion and construction tasks.
        This function would typically make an API call to a large language model.
        """
        if task_type == "kg_completion":
            # Simulate predicting missing attributes/relationships
            if "material for Nike Running Shoes" in prompt:
                return "material: mesh, rubber"
            elif "style for Elegant Evening Dress" in prompt:
                return "style: elegant, evening"
            elif "capacity for Automatic Coffee Machine" in prompt:
                return "capacity: 10 cups"
            elif "compatibility for Nike Running Shoes" in prompt:
                return "compatible_with: running apps, smartwatches"
            return ""
        elif task_type == "kg_construction":
            # Simulate extracting entities and relations from text (e.g., reviews)
            if "review for product_1" in prompt: # Nike Running Shoes review
                return "entities: product_1, soft, comfortable, durable\nrelations: product_1 HAS_ATTRIBUTE soft, product_1 HAS_ATTRIBUTE comfortable, product_1 HAS_ATTRIBUTE durable"
            elif "review for product_2" in prompt: # Elegant Evening Dress review
                return "entities: product_2, beautiful, fits well, elegant\nrelations: product_2 HAS_ATTRIBUTE beautiful, product_2 HAS_ATTRIBUTE fits_well, product_2 HAS_ATTRIBUTE elegant"
            elif "review for product_4" in prompt: # Adidas Training Shorts review
                return "entities: product_4, breathable, flexible\nrelations: product_4 HAS_ATTRIBUTE breathable, product_4 HAS_ATTRIBUTE flexible"
            return ""
        return ""

    def enrich_kg_from_product_data(self, product_id: str, product_description: str, product_attributes: Dict[str, str]):
        """
        Adds product data to the KG and uses LLM to predict missing attributes.
        """
        # Add product node with its initial attributes
        self.kg.add_node(product_id, type="product", description=product_description, attributes=product_attributes.copy())

        # Use LLM for Knowledge Graph Completion: Predict missing attributes
        missing_attrs_prompt = (
            f"Given the product '{product_description}' with attributes {product_attributes}, "
            f"predict a highly relevant missing attribute or relationship for an e-commerce recommender system. "
            f"For example, 'material: cotton' or 'compatible_with: USB-C Hubs'. "
            f"Provide only the 'key: value' pair. Product ID: {product_id}."
        )
        llm_response = self._simulate_llm_response(missing_attrs_prompt, "kg_completion")

        if llm_response:
            try:
                key, value = llm_response.split(": ", 1)
                # Add predicted attribute if it's new
                if key not in self.kg.nodes[product_id]['attributes']:
                    self.kg.nodes[product_id]['attributes'][key] = value
                    # Add an edge from the product to the attribute value for graph traversal
                    if not self.kg.has_node(value):
                        self.kg.add_node(value, type="attribute_value")
                    self.kg.add_edge(product_id, value, relation=key)
            except ValueError:
                pass # LLM response was not in the expected 'key: value' format

    def enrich_kg_from_reviews(self, product_id: str, review_text: str):
        """
        Uses LLM to extract entities and relations from customer reviews and adds them to the KG.
        """
        # Use LLM for Knowledge Graph Construction: Entity Discovery, Relation Extraction
        extract_prompt = (
            f"From the following customer review for product '{product_id}': \"{review_text}\", "
            f"extract key entities and their relationships relevant for product understanding. "
            f"Format as 'entities: entity1, entity2\\nrelations: entity1 RELATION entity2'. "
            f"Example: 'entities: phone, screen\\nrelations: phone HAS_PART screen'."
        )
        llm_response = self._simulate_llm_response(extract_prompt, "kg_construction")

        if llm_response:
            if "entities:" in llm_response and "relations:" in llm_response:
                entities_str = llm_response.split("entities:")[1].split("relations:")[0].strip()
                relations_str = llm_response.split("relations:")[1].strip()

                # Add entities to the KG
                entities = [e.strip() for e in entities_str.split(',') if e.strip()]
                for entity in entities:
                    if not self.kg.has_node(entity):
                        self.kg.add_node(entity, type="concept_from_review")
                    self.kg.add_edge(product_id, entity, relation="mentioned_in_review") # Link product to extracted concept

                # Add relations to the KG
                for relation_triple in relations_str.split(', '):
                    parts = relation_triple.split(' ', 2)
                    if len(parts) == 3:
                        source, rel, target = parts
                        # Ensure source and target nodes exist before adding edge
                        if not self.kg.has_node(source):
                            self.kg.add_node(source, type="concept")
                        if not self.kg.has_node(target):
                            self.kg.add_node(target, type="concept")
                        self.kg.add_edge(source, target, relation=rel)

    def get_knowledge_graph(self) -> nx.DiGraph:
        """Returns the current state of the knowledge graph."""
        return self.kg

# --- Product Recommender Class ---
class ProductRecommender:
    """
    Generates product recommendations based on the enriched knowledge graph.
    """
    def __init__(self, knowledge_graph: nx.DiGraph):
        self.kg = knowledge_graph

    def get_similar_products(self, product_id: str, num_recommendations: int = 3) -> List[str]:
        """
        Recommends similar products by comparing shared attributes and explicit relationships
        in the knowledge graph.
        """
        if product_id not in self.kg or self.kg.nodes[product_id].get('type') != 'product':
            return []

        target_product_attributes = self.kg.nodes[product_id].get('attributes', {})
        similarities = {}

        for other_product_id in self.kg.nodes:
            if other_product_id == product_id or self.kg.nodes[other_product_id].get('type') != 'product':
                continue

            other_product_attributes = self.kg.nodes[other_product_id].get('attributes', {})
            
            # Count shared attributes (both initial and LLM-predicted)
            shared_attrs_count = 0
            for attr, val in target_product_attributes.items():
                if other_product_attributes.get(attr) == val:
                    shared_attrs_count += 1
            
            # Consider explicit "goes_well_with" or "alternative_to" relations
            explicit_relations_score = 0
            # Check for edges between the target product and other products
            if self.kg.has_edge(product_id, other_product_id):
                edge_data = self.kg.get_edge_data(product_id, other_product_id)
                if edge_data and edge_data.get('relation') in ['goes_well_with', 'alternative_to']:
                    explicit_relations_score += 5 # Boost for strong explicit relations

            # Simple similarity score
            similarity_score = shared_attrs_count + explicit_relations_score
            if similarity_score > 0:
                similarities[other_product_id] = similarity_score

        # Sort by similarity and return top recommendations
        sorted_products = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return [prod for prod, score in sorted_products[:num_recommendations]]

    def get_cross_domain_recommendations(self, product_id: str, num_recommendations: int = 3) -> List[str]:
        """
        Provides cross-domain recommendations, leveraging LLM's broader understanding
        of conceptual relationships. This is a simplified simulation.
        """
        if product_id not in self.kg or self.kg.nodes[product_id].get('type') != 'product':
            return []

        # Simulate cross-domain links based on product description or category
        product_description = self.kg.nodes[product_id].get('description', '').lower()
        product_category = self.kg.nodes[product_id].get('category', '').lower()

        if "running shoes" in product_description or "footwear" in product_category:
            return ["Smartwatch (Fitness)", "Wireless Earbuds (Audio)", "Sports Water Bottle (Hydration)"]
        elif "evening dress" in product_description or "apparel" in product_category:
            return ["Matching Handbag (Fashion)", "Elegant Necklace (Jewelry)", "High Heels (Footwear)"]
        elif "coffee machine" in product_description or "home appliances" in product_category:
            return ["Gourmet Coffee Beans (Food)", "Espresso Cups (Kitchenware)", "Milk Frother (Kitchenware)"]
        elif "training shorts" in product_description or "apparel" in product_category:
            return ["Athletic T-shirt (Apparel)", "Fitness Tracker (Electronics)"]
        elif "leather handbag" in product_description or "accessories" in product_category:
            return ["Leather Wallet (Accessories)", "Fashion Scarf (Apparel)"]
        else:
            return ["Generic Cross-Domain Item A", "Generic Cross-Domain Item B"]

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Intelligent Product Recommender",
    description="An e-commerce recommender system leveraging LLM-enhanced knowledge graphs for improved personalization and cross-domain recommendations."
)

# --- Sample E-commerce Data ---
sample_products_data = [
    {
        "product_id": "product_1",
        "name": "Nike Running Shoes",
        "description": "Comfortable and lightweight running shoes for daily training. Ideal for athletes.",
        "category": "Footwear",
        "brand": "Nike",
        "attributes": {"color": "black", "size": "US 10", "gender": "unisex"}
    },
    {
        "product_id": "product_2",
        "name": "Elegant Evening Dress",
        "description": "A stunning black gown perfect for formal events. Made from luxurious fabric.",
        "category": "Apparel",
        "brand": "Versace",
        "attributes": {"color": "black", "size": "M", "occasion": "formal"}
    },
    {
        "product_id": "product_3",
        "name": "Automatic Coffee Machine",
        "description": "Brew your favorite coffee at home with this easy-to-use machine. Features programmable settings.",
        "category": "Home Appliances",
        "brand": "DeLonghi",
        "attributes": {"color": "silver", "power": "1200W", "type": "espresso"}
    },
    {
        "product_id": "product_4",
        "name": "Adidas Training Shorts",
        "description": "Lightweight and breathable shorts for workouts. Designed for maximum flexibility.",
        "category": "Apparel",
        "brand": "Adidas",
        "attributes": {"color": "grey", "size": "L", "gender": "male"}
    },
    {
        "product_id": "product_5",
        "name": "Stylish Leather Handbag",
        "description": "A chic accessory for any outfit, made from genuine leather. Perfect for everyday use.",
        "category": "Accessories",
        "brand": "Coach",
        "attributes": {"color": "brown", "material": "leather", "style": "casual-chic"}
    }
]

sample_customer_reviews = [
    {"product_id": "product_1", "review_text": "These running shoes are incredibly soft and comfortable, perfect for long runs. Very durable too!"},
    {"product_id": "product_2", "review_text": "The dress is absolutely beautiful and fits perfectly. I felt so elegant wearing it. Highly recommend."},
    {"product_id": "product_4", "review_text": "Great shorts for the gym, very breathable and flexible. Exactly what I needed for workouts."},
    {"product_id": "product_5", "review_text": "Lovely handbag, the leather feels premium and it's quite spacious for my daily essentials."}
]

# --- Initialize and Build Knowledge Graph ---
kg_builder = KnowledgeGraphBuilder()

# Enrich KG from product data (including LLM-based completion)
print("Building Knowledge Graph from product data...")
for product in sample_products_data:
    kg_builder.enrich_kg_from_product_data(
        product_id=product["product_id"],
        product_description=product["description"],
        product_attributes=product["attributes"]
    )
print("Enriching Knowledge Graph from customer reviews...")
# Enrich KG from customer reviews (including LLM-based construction)
for review in sample_customer_reviews:
    kg_builder.enrich_kg_from_reviews(
        product_id=review["product_id"],
        review_text=review["review_text"]
    )

# --- Initialize Recommender with the Enriched KG ---
product_recommender = ProductRecommender(kg_builder.get_knowledge_graph())

# --- API Models ---
class RecommendRequest(BaseModel):
    product_id: str
    num_recommendations: int = 3

class RecommendationResponse(BaseModel):
    product_id: str
    recommendations: List[str]
    cross_domain_recommendations: List[str] = []

class KGNodeDetails(BaseModel):
    id: str
    type: str
    description: str | None = None
    attributes: Dict[str, Any] = {}
    neighbors: List[Dict[str, Any]] = []

# --- API Endpoints ---
@app.get("/", summary="Root Endpoint", description="Check if the API is running.")
async def read_root():
    return {"message": "Intelligent Product Recommender API is running!"}

@app.post("/recommend", response_model=RecommendationResponse, summary="Get Product Recommendations",
          description="Provides similar product recommendations and cross-domain recommendations based on the LLM-enhanced knowledge graph.")
async def recommend_products_api(request: RecommendRequest):
    similar_products = product_recommender.get_similar_products(request.product_id, request.num_recommendations)
    cross_domain_recs = product_recommender.get_cross_domain_recommendations(request.product_id, request.num_recommendations)
    return RecommendationResponse(
        product_id=request.product_id,
        recommendations=similar_products,
        cross_domain_recommendations=cross_domain_recs
    )

@app.get("/kg/node/{node_id}", response_model=KGNodeDetails, summary="Get Knowledge Graph Node Details",
         description="Retrieves details and neighbors for a specific node in the knowledge graph, showing LLM-enhanced data.")
async def get_kg_node_details(node_id: str):
    kg = kg_builder.get_knowledge_graph()
    if node_id not in kg:
        return KGNodeDetails(id=node_id, type="unknown", description=f"Node '{node_id}' not found in KG.")

    node_data = kg.nodes[node_id]
    neighbors_list = []
    for neighbor in kg.neighbors(node_id):
        edge_data = kg.get_edge_data(node_id, neighbor)
        neighbors_list.append({"node_id": neighbor, "relation": edge_data.get('relation', 'unknown') if edge_data else 'unknown'})

    return KGNodeDetails(
        id=node_id,
        type=node_data.get('type', 'unknown'),
        description=node_data.get('description'),
        attributes=node_data.get('attributes', {}),
        neighbors=neighbors_list
    )

