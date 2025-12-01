import os
from kg_builder import KnowledgeGraphBuilder
from recommender import ProductWiseRecommender
import networkx as nx

# Set your OpenAI API key as an environment variable or replace 'YOUR_OPENAI_API_KEY' directly
# For better security, always use environment variables.
# Example: export OPENAI_API_KEY="sk-your-key-here"
if "OPENAI_API_KEY" not in os.environ:
    # This is for demonstration purposes. In a real application, ensure this is set securely.
    print("WARNING: OPENAI_API_KEY environment variable not set. Please set it for the LLM calls to work.")
    print("Attempting to proceed, but LLM calls may fail without a valid API key.")
    # os.environ["OPENAI_API_KEY"] = "sk-your-fallback-key-if-any" # Uncomment and replace if you want a fallback

def main():
    print("\n--- Starting ProductWise AI Recommender System ---")

    # 1. Initialize Knowledge Graph Builder
    # You can specify a different LLM model if needed, e.g., "gpt-4"
    kg_builder = KnowledgeGraphBuilder(llm_model_name="gpt-3.5-turbo")

    # 2. Process example product data to build the initial Knowledge Graph
    product_descriptions = [
        "The iPhone 15 Pro features an A17 Bionic chip, a titanium design, and a Pro camera system with a 5x Telephoto lens. It runs iOS and has a stunning Super Retina XDR display.",
        "Sony WH-1000XM5 headphones offer industry-leading noise cancellation, exceptional sound quality, and comfortable over-ear design. They support Bluetooth 5.2 and have a long battery life.",
        "Dell XPS 15 laptop comes with a 13th Gen Intel Core i9 processor, NVIDIA GeForce RTX 4070 GPU, and a vibrant OLED touch display. It's ideal for creative professionals and gamers.",
        "High-performance running shoes with breathable mesh upper and responsive foam cushioning. Perfect for long-distance running and daily workouts.",
        "Smart Home Hub with voice assistant integration. Control all your smart devices, from lights to thermostats, with simple voice commands or through its intuitive app."
    ]

    for desc in product_descriptions:
        kg_builder.process_text_for_kg(desc)

    # 3. Demonstrate Knowledge Graph Completion (Fact Prediction)
    # Let's assume 'Dell XPS 15' might have some missing info that LLM can infer
    kg_builder.predict_and_add_fact("Dell XPS 15")

    # 4. Demonstrate Knowledge Graph Completion (Commonsense Injection)
    kg_builder.inject_commonsense_knowledge("running shoes")
    kg_builder.inject_commonsense_knowledge("Smart Home Hub")

    print("\n--- Current Knowledge Graph Edges (Sample) ---")
    for u, v, data in list(kg_builder.get_graph().edges(data=True))[:10]: # Print first 10 edges
        print(f"({u}) -[{data['relation']}]-> ({v})")
    if len(kg_builder.get_graph().edges) > 10:
        print("... and more edges.")

    # 5. Initialize Recommender System
    recommender = ProductWiseRecommender(kg_builder)

    # 6. Get Recommendations
    print("\n--- Generating Recommendations ---")
    seed_product_1 = "iPhone 15 Pro"
    iphone_recommendations = recommender.get_recommendations(seed_product_1)
    print(f"Recommendations for '{seed_product_1}': {iphone_recommendations}")

    seed_product_2 = "OLED display"
    oled_recommendations = recommender.get_recommendations(seed_product_2)
    print(f"Recommendations for '{seed_product_2}': {oled_recommendations}")

    seed_product_3 = "running shoes"
    shoes_recommendations = recommender.get_recommendations(seed_product_3)
    print(f"Recommendations for '{seed_product_3}': {shoes_recommendations}")

    # Example for a product not initially in the text, to see if LLM inferences helped
    seed_product_4 = "Gaming Laptop"
    gaming_laptop_recs = recommender.get_recommendations(seed_product_4)
    print(f"Recommendations for '{seed_product_4}': {gaming_laptop_recs}")

    print("\n--- ProductWise AI Recommender System Finished ---")

if __name__ == "__main__":
    main()