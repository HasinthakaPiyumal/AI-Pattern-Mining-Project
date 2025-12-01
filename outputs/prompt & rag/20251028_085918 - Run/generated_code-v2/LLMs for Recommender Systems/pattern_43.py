import pandas as pd
import networkx as nx
from fastapi import FastAPI
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import random

# --- 1. Data Ingestion & Preprocessing Layer (Mock Data) ---
class DataIngestor:
    def __init__(self):
        self.products_data = {
            "product_id": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
            "name": [
                "Red Summer Dress", "Blue Jeans", "Leather Handbag", 
                "Running Shoes", "Smartwatch", "Gourmet Coffee Beans", 
                "Espresso Machine", "Denim Jacket"
            ],
            "description": [
                "A beautiful red summer dress, perfect for beach holidays. Lightweight and breathable fabric.",
                "Classic blue denim jeans, comfortable fit for everyday wear. Durable material.",
                "Elegant leather handbag, spacious and stylish. Ideal for evening events.",
                "Lightweight running shoes with excellent cushioning. Great for long runs.",
                "Advanced smartwatch with fitness tracking and heart rate monitor. Syncs with phone.",
                "Premium single-origin gourmet coffee beans. Rich aroma and smooth taste.",
                "High-pressure espresso machine for perfect coffee at home. Easy to use and clean.",
                "Stylish denim jacket, a timeless piece for all seasons. Made from organic cotton."
            ],
            "category": [
                "Apparel", "Apparel", "Accessories", 
                "Footwear", "Electronics", "Food & Beverage", 
                "Home Appliances", "Apparel"
            ],
            "material": [
                "Cotton", "Denim", "Leather", 
                "Synthetic", "Metal/Plastic", "Coffee", 
                "Metal/Plastic", "Denim"
            ],
            "color": [
                "Red", "Blue", "Black", 
                "Black", "Black", "Brown", 
                "Silver", "Blue"
            ],
            "attributes": [
                "summer, beach, breathable", "casual, durable", "elegant, evening, spacious",
                "sporty, comfortable, lightweight", "fitness, smart, wearable", "premium, aromatic",
                "home, easy-use", "classic, organic"
            ]
        }
        self.reviews_data = [
            {"product_id": "P1", "review": "Lovely dress, very comfortable for summer!"},
            {"product_id": "P2", "review": "My favorite jeans, fit perfectly."},
            {"product_id": "P3", "review": "Great quality bag, looks very high-end."},
            {"product_id": "P4", "review": "These shoes are amazing for my morning jogs."},
            {"product_id": "P5", "review": "Love the fitness tracking feature on this watch."},
            {"product_id": "P6", "review": "The best coffee I've ever tasted, truly gourmet."},
            {"product_id": "P7", "review": "Makes excellent espresso, a bit noisy though."},
            {"product_id": "P8", "review": "Goes with everything, essential for my wardrobe."},
            {"product_id": "P1", "review": "Matches well with a straw hat and sandals."},
            {"product_id": "P3", "review": "Perfect with the red summer dress."},
            {"product_id": "P6", "review": "Bought with the espresso machine for a perfect combo."},
        ]

    def get_products_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.products_data)

    def get_reviews_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.reviews_data)


# --- 2. Knowledge Graph (KG) Management Layer ---
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_product(self, product_id: str, name: str, category: str):
        self.graph.add_node(product_id, type="product", name=name, category=category)

    def add_attribute(self, product_id: str, attr_type: str, attr_value: str):
        attr_node_id = f"{attr_type}:{attr_value}"
        if not self.graph.has_node(attr_node_id):
            self.graph.add_node(attr_node_id, type=attr_type, value=attr_value)
        self.graph.add_edge(product_id, attr_node_id, relation=f"has_{attr_type}")

    def add_relation(self, entity1: str, relation_type: str, entity2: str):
        if self.graph.has_node(entity1) and self.graph.has_node(entity2):
            self.graph.add_edge(entity1, entity2, relation=relation_type)
        else:
            print(f"Warning: Could not add relation {relation_type} between {entity1} and {entity2}. One or both entities not in graph.")

    def get_related_entities(self, entity_id: str, relation_type: str = None) -> List[str]:
        related = []
        for neighbor in self.graph.successors(entity_id):
            if relation_type is None or self.graph[entity_id][neighbor]["relation"] == relation_type:
                related.append(neighbor)
        return related
    
    def get_all_product_attributes(self, product_id: str) -> List[str]:
        attributes = []
        for neighbor in self.graph.successors(product_id):
            edge_data = self.graph.get_edge_data(product_id, neighbor)
            if edge_data and edge_data.get("relation", "").startswith("has_"):
                attributes.append(self.graph.nodes[neighbor].get("value", neighbor))
        return attributes


# --- 3. LLM-based Knowledge Augmentation Layer (Mocked LLM interactions) ---
class LLMKGAugmenter:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        # Mocking LLM capabilities
        # In a real scenario, this would involve API calls to LLM models (e.g., via transformers, langchain)

    def kg_completion(self, product_id: str, description: str) -> Dict[str, str]:
        # Mock: Infer a style attribute for dresses or a use case for electronics
        inferred_attributes = {}
        if "dress" in description.lower() and "beach" in description.lower():
            inferred_attributes["style"] = "beachwear"
        elif "smartwatch" in description.lower() and "fitness" in description.lower():
            inferred_attributes["use_case"] = "fitness_tracking"
        return inferred_attributes

    def entity_discovery(self, text: str) -> List[str]:
        # Mock: Simple keyword extraction for entities
        entities = []
        if "red summer dress" in text.lower(): entities.append("Red Summer Dress")
        if "blue jeans" in text.lower(): entities.append("Blue Jeans")
        if "leather handbag" in text.lower(): entities.append("Leather Handbag")
        if "running shoes" in text.lower(): entities.append("Running Shoes")
        if "smartwatch" in text.lower(): entities.append("Smartwatch")
        if "coffee beans" in text.lower(): entities.append("Gourmet Coffee Beans")
        if "espresso machine" in text.lower(): entities.append("Espresso Machine")
        if "denim jacket" in text.lower(): entities.append("Denim Jacket")
        if "summer" in text.lower(): entities.append("Summer")
        if "beach" in text.lower(): entities.append("Beach")
        if "cotton" in text.lower(): entities.append("Cotton")
        if "leather" in text.lower(): entities.append("Leather")
        return list(set(entities)) # Remove duplicates

    def relation_extraction(self, text: str, entities: List[str]) -> List[Dict[str, str]]:
        # Mock: Simple rule-based relation extraction
        relations = []
        if "matches well with" in text.lower() or "perfect with" in text.lower():
            if "red summer dress" in entities and "leather handbag" in entities:
                relations.append({"source": "P1", "target": "P3", "type": "complements"})
            if "red summer dress" in entities and "sandals" in text.lower(): # Assuming sandals is an inferred entity
                relations.append({"source": "P1", "target": "sandals_concept", "type": "complements"})
            if "coffee beans" in entities and "espresso machine" in entities:
                relations.append({"source": "P6", "target": "P7", "type": "often_bought_with"})
        return relations

    def coreference_resolution(self, text: str) -> Dict[str, str]:
        # Mock: Simple resolution for common phrases
        resolved = {}
        if "apple's latest flagship phone" in text.lower():
            resolved["apple's latest flagship phone"] = "iPhone 15 Pro Max"
        return resolved

    def distill_commonsense(self, query: str) -> List[str]:
        # Mock: Return pre-defined commonsense rules/facts
        if "accessories typically go with a smartphone" in query.lower():
            return ["headphones", "phone case", "charger", "screen protector"]
        elif "what to wear for a beach holiday" in query.lower():
            return ["swimsuit", "sunscreen", "hat", "sandals", "light dress"]
        return []

    def augment_kg_with_llm(self, products_df: pd.DataFrame, reviews_df: pd.DataFrame):
        # 1. Initial KG population
        for _, row in products_df.iterrows():
            self.kg.add_product(row["product_id"], row["name"], row["category"])
            for attr_type in ["category", "material", "color"]:
                self.kg.add_attribute(row["product_id"], attr_type, row[attr_type])
            # Add existing 'attributes' from data
            for attr in str(row["attributes"]).split(', '):
                if attr: self.kg.add_attribute(row["product_id"], "generic_attr", attr)

        # 2. KG Completion
        print("\n--- Performing KG Completion ---")
        for _, row in products_df.iterrows():
            inferred = self.kg_completion(row["product_id"], row["description"])
            for attr_type, attr_value in inferred.items():
                self.kg.add_attribute(row["product_id"], attr_type, attr_value)
                print(f"Inferred: {row['name']} has_{attr_type} {attr_value}")
        
        # 3. KG Construction from Reviews
        print("\n--- Performing KG Construction from Reviews ---")
        for _, row in reviews_df.iterrows():
            resolved_text = row["review"]
            # Mock Coreference Resolution (simplified for reviews)
            resolved_entities = self.coreference_resolution(resolved_text)
            for k, v in resolved_entities.items():
                resolved_text = resolved_text.replace(k, v)
            
            entities_in_review = self.entity_discovery(resolved_text)
            # Ensure product_id is considered as an entity in its own review context
            if self.kg.graph.has_node(row["product_id"]) and row["product_id"] not in entities_in_review:
                entities_in_review.append(row["product_id"])

            # Map discovered entities to existing product IDs or create new concept nodes if applicable
            mapped_entities = []
            for entity in entities_in_review:
                # Check if it's a product_id we know
                if self.kg.graph.has_node(entity) and self.kg.graph.nodes[entity].get("type") == "product":
                    mapped_entities.append(entity)
                else:
                    # Try to find product by name
                    found_pid = None
                    for node_id, data in self.kg.graph.nodes(data=True):
                        if data.get("type") == "product" and data.get("name", "").lower() == entity.lower():
                            found_pid = node_id
                            break
                    if found_pid:
                        mapped_entities.append(found_pid)
                    else:
                        # Add as a generic concept node if not found as product or attribute
                        concept_node_id = f"concept:{entity.replace(' ', '_').lower()}"
                        if not self.kg.graph.has_node(concept_node_id):
                            self.kg.graph.add_node(concept_node_id, type="concept", value=entity)
                        mapped_entities.append(concept_node_id)

            # Extract relations
            extracted_relations = self.relation_extraction(resolved_text, entities_in_review)
            for rel in extracted_relations:
                # Ensure source and target are valid KG nodes (product_id or concept_node_id)
                source_node = rel["source"]
                target_node = rel["target"]

                # Attempt to map target_node if it's a concept that might correspond to a product
                if target_node.endswith("_concept"): # For cases like 'sandals_concept'
                    # Heuristically check if this concept is related to an existing product category
                    if "sandals" in target_node:
                        # Add a general relation, or link to a hypothetical 'sandals' product/category
                        if not self.kg.graph.has_node("concept:sandals"):
                            self.kg.graph.add_node("concept:sandals", type="concept", value="sandals")
                        target_node = "concept:sandals"

                if self.kg.graph.has_node(source_node) and self.kg.graph.has_node(target_node):
                    self.kg.add_relation(source_node, rel["type"], target_node)
                    print(f"Extracted Relation: {source_node} -[{rel['type']}]-> {target_node}")
                else:
                    print(f"Skipping relation {rel['type']} due to missing nodes: {source_node}, {target_node}")


        # 4. Commonsense Distillation (Example usage, not directly modifying KG here)
        print("\n--- Performing Commonsense Distillation ---")
        smartphone_accessories = self.distill_commonsense("what accessories typically go with a smartphone")
        print(f"Distilled commonsense for smartphone accessories: {smartphone_accessories}")


# --- 4. Recommendation Engine Layer ---
class RecommendationEngine:
    def __init__(self, kg: KnowledgeGraph, products_df: pd.DataFrame):
        self.kg = kg
        self.products_df = products_df.set_index("product_id")
        self.vectorizer = TfidfVectorizer()
        self.product_vectors = None
        self.product_ids = []

    def _prepare_for_similarity(self):
        # Combine product attributes from KG for content-based similarity
        product_descriptions = []
        self.product_ids = []
        for pid in self.products_df.index:
            attributes = self.kg.get_all_product_attributes(pid)
            # Include original description as well
            original_desc = self.products_df.loc[pid, "description"]
            product_descriptions.append(original_desc + " " + " ".join(attributes))
            self.product_ids.append(pid)
        
        if product_descriptions:
            self.product_vectors = self.vectorizer.fit_transform(product_descriptions)
        else:
            self.product_vectors = None

    def get_content_based_recommendations(self, product_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        if self.product_vectors is None:
            self._prepare_for_similarity()
        
        if product_id not in self.product_ids:
            return []

        idx = self.product_ids.index(product_id)
        similarities = cosine_similarity(self.product_vectors[idx:idx+1], self.product_vectors).flatten()
        
        # Get top_n similar products, excluding itself
        similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
        
        recommendations = []
        for i in similar_indices:
            rec_pid = self.product_ids[i]
            if rec_pid != product_id:
                recommendations.append({
                    "product_id": rec_pid,
                    "name": self.products_df.loc[rec_pid, "name"],
                    "similarity": round(similarities[i], 4)
                })
        return recommendations

    def get_kg_based_recommendations(self, product_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        recommendations = []
        # Directly use KG relations like 'complements' or 'often_bought_with'
        complementary_products = self.kg.get_related_entities(product_id, "complements")
        often_bought_with_products = self.kg.get_related_entities(product_id, "often_bought_with")

        related_pids = list(set(complementary_products + often_bought_with_products))
        
        # Filter to actual product IDs and retrieve names
        for pid_node in related_pids:
            if self.kg.graph.has_node(pid_node) and self.kg.graph.nodes[pid_node].get("type") == "product":
                recommendations.append({
                    "product_id": pid_node,
                    "name": self.products_df.loc[pid_node, "name"],
                    "reason": "KG-based related product"
                })
            elif self.kg.graph.has_node(pid_node) and self.kg.graph.nodes[pid_node].get("type") == "concept":
                 # For concept recommendations, we might need to find products matching the concept
                 concept_value = self.kg.graph.nodes[pid_node].get("value", "")
                 if concept_value:
                     # Simple search for products related to the concept
                     for p_id, p_row in self.products_df.iterrows():
                         if concept_value.lower() in p_row["description"].lower() or \
                            concept_value.lower() in p_row["attributes"].lower() or \
                            concept_value.lower() in p_row["name"].lower():
                             if p_id != product_id:
                                 recommendations.append({
                                     "product_id": p_id,
                                     "name": self.products_df.loc[p_id, "name"],
                                     "reason": f"KG-based concept: {concept_value}"
                                 })
                                 if len(recommendations) >= top_n: break

        return recommendations[:top_n]

    def recommend(self, product_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        content_recs = self.get_content_based_recommendations(product_id, top_n=top_n)
        kg_recs = self.get_kg_based_recommendations(product_id, top_n=top_n)
        
        # Combine and deduplicate recommendations
        combined_recs = {rec["product_id"]: rec for rec in content_recs}
        for rec in kg_recs:
            if rec["product_id"] not in combined_recs:
                combined_recs[rec["product_id"]] = rec
            # If it's a KG rec and also content rec, prioritize KG reason if available
            elif "reason" in rec and "reason" not in combined_recs[rec["product_id"]]:
                combined_recs[rec["product_id"]]["reason"] = rec["reason"]

        final_recs = list(combined_recs.values())
        random.shuffle(final_recs) # Mix for variety
        return final_recs[:top_n]


# --- 5. API & Deployment Layer ---
app = FastAPI(
    title="Smart Product Recommender API",
    description="API for personalized product recommendations leveraging LLM-augmented Knowledge Graphs."
)

# Initialize components (global scope for simplicity, in production use dependency injection)
data_ingestor = DataIngestor()
products_df = data_ingestor.get_products_df()
reviews_df = data_ingestor.get_reviews_df()

product_kg = KnowledgeGraph()
kg_augmenter = LLMKGAugmenter(product_kg)
kg_augmenter.augment_kg_with_llm(products_df, reviews_df)

recommender_engine = RecommendationEngine(product_kg, products_df)


@app.get("/recommend/{product_id}", response_model=List[Dict[str, Any]])
async def get_recommendations(product_id: str, top_n: int = 5):
    """Get product recommendations for a given product ID."""
    if product_id not in products_df["product_id"].values:
        return []
    return recommender_engine.recommend(product_id, top_n)

@app.get("/products", response_model=List[Dict[str, Any]])
async def list_products():
    """List all available products."""
    return products_df.to_dict(orient="records")


if __name__ == "__main__":
    print("\n--- Starting FastAPI Application ---")
    print("Access API at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)

