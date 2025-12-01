import random

class BatchPromptBuilder:
    def build_prompt(self, candidates: list[dict], user_query: str, num_to_recommend: int) -> str:
        prompt_parts = [
            "Rank the following products based on the user's query. Output only the IDs of the top {} products, separated by commas.".format(num_to_recommend),
            "User Query: {}\n".format(user_query),
            "Available Products:"
        ]
        for product in candidates:
            prompt_parts.append(
                "Product ID: {}\nName: {}\nDescription: {}\n".format(
                    product['id'], product['name'], product['description']
                )
            )
        return "\n".join(prompt_parts)

class LLMServiceSimulator:
    def call_llm(self, prompt: str) -> str:
        # Simulate LLM processing: extract product IDs and return a pseudo-ranked list.
        # In a real scenario, an actual LLM API call would be made here.
        candidate_ids = []
        for line in prompt.split('\n'):
            if "Product ID: " in line:
                candidate_ids.append(line.split(": ")[1].strip())
        
        # Simulate ranking by shuffling and taking a subset
        random.shuffle(candidate_ids)
        # The LLM is instructed to output 'num_to_recommend', but for simplicity here,
        # we'll just return a subset of the candidates it was given.
        # A real LLM would produce a more intelligent ranking.
        simulated_ranked_ids = candidate_ids[:min(len(candidate_ids), 3)] # Just taking top 3 for simulation
        return ",".join(simulated_ranked_ids)

class ProductRecommendationEngine:
    def __init__(self, product_catalog: list[dict]):
        self.product_catalog = {p['id']: p for p in product_catalog}
        self.prompt_builder = BatchPromptBuilder()
        self.llm_simulator = LLMServiceSimulator()

    def get_batch_recommendations(self, user_query: str, num_candidates: int, num_to_return: int) -> list[dict]:
        # 1. Candidate Selection (for simplicity, random selection)
        all_product_ids = list(self.product_catalog.keys())
        if len(all_product_ids) < num_candidates:
            selected_candidate_ids = all_product_ids
        else:
            selected_candidate_ids = random.sample(all_product_ids, num_candidates)
        
        candidates = [self.product_catalog[pid] for pid in selected_candidate_ids]

        # 2. Prompt Batching
        batch_prompt = self.prompt_builder.build_prompt(candidates, user_query, num_to_return)

        # 3. LLM Call
        llm_response = self.llm_simulator.call_llm(batch_prompt)

        # 4. Result Parsing
        ranked_ids = [pid.strip() for pid in llm_response.split(',') if pid.strip()]
        
        recommended_products = []
        for pid in ranked_ids:
            if pid in self.product_catalog:
                recommended_products.append(self.product_catalog[pid])
            if len(recommended_products) >= num_to_return:
                break
        
        return recommended_products

if __name__ == "__main__":
    # Sample Product Catalog
    sample_product_catalog = [
        {'id': 'P001', 'name': 'Running Shoes', 'description': 'Lightweight and comfortable running shoes for daily use.'},
        {'id': 'P002', 'name': 'Hiking Boots', 'description': 'Durable and waterproof boots for tough trails.'},
        {'id': 'P003', 'name': 'Casual Sneakers', 'description': 'Stylish sneakers for everyday wear.'},
        {'id': 'P004', 'name': 'Dress Shoes', 'description': 'Elegant leather shoes for formal occasions.'},
        {'id': 'P005', 'name': 'Sandals', 'description': 'Comfortable sandals for summer outings.'},
        {'id': 'P006', 'name': 'Workout Gloves', 'description': 'Provides grip and protection during gym workouts.'},
        {'id': 'P007', 'name': 'Yoga Mat', 'description': 'Non-slip mat for yoga and pilates.'},
        {'id': 'P008', 'name': 'Water Bottle', 'description': 'Insulated stainless steel water bottle.'},
        {'id': 'P009', 'name': 'Smartwatch', 'description': 'Tracks fitness, heart rate, and notifications.'},
        {'id': 'P010', 'name': 'Headphones', 'description': 'Noise-cancelling over-ear headphones.'},
    ]

    # Initialize the Recommendation Engine
    engine = ProductRecommendationEngine(sample_product_catalog)

    # Simulate a user query
    user_query = "comfortable footwear for sports"
    num_candidates_to_consider = 5  # Number of products to send to LLM in one batch
    num_recommendations_to_return = 3 # Desired number of final recommendations

    print(f"User Query: '{user_query}'")
    print(f"Considering {num_candidates_to_consider} candidates, returning top {num_recommendations_to_return} recommendations.\n")

    recommended_products = engine.get_batch_recommendations(user_query, num_candidates_to_consider, num_recommendations_to_return)

    print("Recommended Products:")
    if recommended_products:
        for product in recommended_products:
            print(f"  - {product['name']} (ID: {product['id']})")
    else:
        print("  No recommendations found.")

    print("\n--- Demonstrating another query ---")
    user_query_2 = "accessories for fitness"
    num_candidates_to_consider_2 = 4
    num_recommendations_to_return_2 = 2

    print(f"User Query: '{user_query_2}'")
    print(f"Considering {num_candidates_to_consider_2} candidates, returning top {num_recommendations_to_return_2} recommendations.\n")

    recommended_products_2 = engine.get_batch_recommendations(user_query_2, num_candidates_to_consider_2, num_recommendations_to_return_2)

    print("Recommended Products:")
    if recommended_products_2:
        for product in recommended_products_2:
            print(f"  - {product['name']} (ID: {product['id']})")
    else:
        print("  No recommendations found.")