from typing import List, Dict, Any
import networkx as nx
from kg_builder import KnowledgeGraphBuilder, MockLLMClient
from prompts import PERSONALIZATION_FACT_PROMPT
import random

class ProductRecommender:
    def __init__(self, kg_builder: KnowledgeGraphBuilder, llm_client: MockLLMClient):
        self.kg_builder = kg_builder
        self.llm_client = llm_client
        print("Initialized ProductRecommender")

    def _get_related_products_from_kg(self, product_id: str, relation_types: List[str] = None, depth: int = 1) -> List[str]:
        """Traverses the KG to find related products based on specified relation types and depth."""
        related_products = set()
        if not self.kg_builder.kg.has_node(product_id):
            return []

        # Use NetworkX for graph traversal
        for u, v, data in nx.dfs_edges(self.kg_builder.kg, source=product_id, depth_limit=depth):
            if relation_types is None or data.get('type') in relation_types:
                # Consider both outgoing and incoming edges if relevant
                if self.kg_builder.kg.nodes[v].get('type') == 'product': # Assuming we tag product nodes
                    related_products.add(v)
        
        # Also consider neighbors directly connected by desired relation types
        for neighbor in self.kg_builder.kg.neighbors(product_id):
            for _, _, data in self.kg_builder.kg.get_edge_data(product_id, neighbor).values():
                if relation_types is None or data.get('type') in relation_types:
                    if self.kg_builder.kg.nodes[neighbor].get('type') == 'product':
                        related_products.add(neighbor)

        return list(related_products)

    def generate_user_personalization_facts(self, user_history_summary: str) -> List[str]:
        """Uses LLM to generate personalized facts based on user history."""
        prompt = PERSONALIZATION_FACT_PROMPT.format(user_history=user_history_summary)
        llm_response = self.llm_client.generate_text(prompt)
        facts = [f.strip() for f in llm_response.split(';') if f.strip()]
        print(f"Generated personalization facts: {facts}")
        return facts

    def recommend_products(self, user_id: str, user_history_summary: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Generates product recommendations based on user history and KG."""
        print(f"Generating recommendations for user: {user_id}")
        personalization_facts = self.generate_user_personalization_facts(user_history_summary)

        candidate_products = set()
        # 1. Recommendations based on direct relations (e.g., recently viewed/purchased items)
        # In a real system, you'd fetch actual user interaction data
        mock_last_viewed_product = "MacBook Pro"
        if mock_last_viewed_product:
            candidate_products.update(self._get_related_products_from_kg(mock_last_viewed_product, 
                                                                         relation_types=["compatible_with", "similar_to", "best_for"]))

        # 2. Incorporate personalization facts into candidate generation (e.g., query KG with facts)
        # This is a simplified approach; a more advanced system would embed facts and query a vector store
        for fact in personalization_facts:
            # Mock: If a fact mentions a category or attribute, find products related to it
            if "smart home devices" in fact.lower():
                # For demonstration, manually add a product that fits this description
                self.kg_builder.add_entity("Smart Speaker X", {'type': 'product', 'category': 'smart home'})
                candidate_products.add("Smart Speaker X")
            if "sustainable products" in fact.lower():
                self.kg_builder.add_entity("Eco-Friendly Water Bottle", {'type': 'product', 'attribute': 'sustainable'})
                candidate_products.add("Eco-Friendly Water Bottle")

        # Filter out the mock_last_viewed_product itself if it's in candidates
        if mock_last_viewed_product in candidate_products:
            candidate_products.remove(mock_last_viewed_product)

        # Fallback for cold-start or if not enough candidates found
        if len(candidate_products) < num_recommendations and self.kg_builder.kg.number_of_nodes() > 0:
            # Add some popular or random products from the KG
            all_products_in_kg = [node for node, data in self.kg_builder.kg.nodes(data=True) if data.get('type') == 'product']
            if len(all_products_in_kg) > 0:
                random.shuffle(all_products_in_kg)
                candidate_products.update(all_products_in_kg[:num_recommendations * 2]) # Get more than needed
        
        final_recommendations = []
        for product_id in list(candidate_products)[:num_recommendations]:
            final_recommendations.append({
                "product_id": product_id,
                "description": self.kg_builder.kg.nodes[product_id].get('description', 'No description available'),
                # In a real system, you'd fetch more product details from a database
            })
        
        print(f"Final recommendations for user {user_id}: {final_recommendations}")
        return final_recommendations

# Example Usage
if __name__ == "__main__":
    from config import LLM_API_KEY

    llm_client = MockLLMClient(api_key=LLM_API_KEY)
    kg_builder = KnowledgeGraphBuilder(llm_client=llm_client)

    # Populate KG with some example data first
    product_desc_macbook = "The new MacBook Pro with M2 chip delivers incredible performance for creative professionals. Features a stunning Liquid Retina XDR display and advanced thermal architecture."
    reviews_macbook = ["Amazing laptop, super fast for video editing!", "Battery life is fantastic."]
    kg_builder.build_kg_from_product_data("MacBook Pro", product_desc_macbook, reviews_macbook)

    product_desc_boots = "Durable waterproof hiking boots for challenging trails. Excellent grip and ankle support."
    reviews_boots = ["Great for long hikes!", "Kept my feet dry in the rain."]
    kg_builder.build_kg_from_product_data("Hiking Boots", product_desc_boots, reviews_boots)

    product_desc_coffee = "Automatic drip coffee maker with programmable timer and brew strength control."
    reviews_coffee = ["Makes great coffee every morning.", "Easy to clean."]
    kg_builder.build_kg_from_product_data("Coffee Maker", product_desc_coffee, reviews_coffee)

    recommender = ProductRecommender(kg_builder=kg_builder, llm_client=llm_client)

    user_history_summary_1 = "User recently viewed high-end electronics, especially Apple products. Has purchased smart home devices in the past."
    recommendations_1 = recommender.recommend_products(user_id="user_A", user_history_summary=user_history_summary_1)
    print("\n--- Recommendations for User A ---")
    for rec in recommendations_1:
        print(f"- {rec['product_id']}: {rec['description']}")

    user_history_summary_2 = "User frequently buys outdoor gear and is planning a hiking trip. Prefers sustainable products."
    recommendations_2 = recommender.recommend_products(user_id="user_B", user_history_summary=user_history_summary_2)
    print("\n--- Recommendations for User B ---")
    for rec in recommendations_2:
        print(f"- {rec['product_id']}: {rec['description']}")