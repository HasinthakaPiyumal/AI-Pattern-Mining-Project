
import random

def generate_synthetic_queries():
    """Generates a set of synthetic queries for demonstration."""
    faq_queries = [
        "What are your shipping options?",
        "How do I track my order?",
        "What is your return policy?",
        "Do you offer international shipping?",
        "How can I reset my password?"
    ]
    product_queries_simple = [
        "Tell me about the 'Evergreen' t-shirt.",
        "What colors does the 'Voyager' backpack come in?",
        "Is the 'Zenith' smartwatch waterproof?",
        "What material is the 'Comfort' hoodie made from?",
        "What's the price of the 'Starlight' necklace?"
    ]
    product_queries_complex = [
        "Compare the 'Evergreen' t-shirt with the 'Forest' t-shirt in terms of material and fit.",
        "I ordered the 'Voyager' backpack last week, but I also want to buy the 'Explorer' version. Can I get a discount?",
        "My 'Zenith' smartwatch isn't syncing with my phone after the latest update. What should I do?",
        "I'm looking for a comfortable and durable hoodie for hiking, suitable for cold weather. Do you have any recommendations between the 'Comfort' and 'Adventure' hoodies?",
        "I want to return the 'Starlight' necklace but I've lost the original packaging. What are my options?"
    ]
    general_queries = [
        "Hello",
        "Thank you",
        "I need help",
        "Can you assist me?",
        "What can you do?"
    ]

    return {
        "faq": faq_queries,
        "product_simple": product_queries_simple,
        "product_complex": product_queries_complex,
        "general": general_queries
    }

def simulate_rag_success(query, strategy_level):
    """Simulates if a given RAG strategy can 'answer' a query.
    Simplistic simulation based on perceived complexity and strategy level.
    strategy_level: 0 (No RAG), 1 (Single-step RAG), 2 (Multi-step RAG)
    """
    # A query is 'answered' if its inherent complexity is less than or equal to the strategy's capability
    if "compare" in query.lower() or "discount" in query.lower() or "syncing" in query.lower() or "recommendations" in query.lower() or "return" in query.lower() and "lost" in query.lower():
        query_complexity = 2 # Complex
    elif "what are" in query.lower() or "how do i" in query.lower() or "what is your" in query.lower() or "do you offer" in query.lower() or "tell me about" in query.lower() or "what colors" in query.lower() or "is the" in query.lower() or "what material" in query.lower() or "price of" in query.lower():
        query_complexity = 1 # Moderate/Simple RAG
    else:
        query_complexity = 0 # Very Simple/No RAG

    # Introduce some randomness for more realistic simulation
    if random.random() < 0.2: # 20% chance of failure even if complexity matches
        return False

    return strategy_level >= query_complexity

def generate_data_from_model_outcomes(queries_dict):
    """Labels queries based on which simulated RAG strategy succeeds first.
    Prioritizes simpler models.
    """
    labeled_data = []
    all_queries = []
    for category, query_list in queries_dict.items():
        all_queries.extend(query_list)

    random.shuffle(all_queries) # Process queries in a semi-random order

    for query in all_queries:
        label = "unlabeled"
        if simulate_rag_success(query, 0): # Try No RAG first
            label = "simple"
        elif simulate_rag_success(query, 1): # Then Single-step RAG
            label = "moderate"
        elif simulate_rag_success(query, 2): # Finally Multi-step RAG
            label = "complex"

        # If still unlabeled, assign based on a more general fallback or assume it's complex
        if label == "unlabeled":
             # For the purpose of this simulation, if all fail, assume it's complex
            label = "complex"

        labeled_data.append((query, label))

    return labeled_data

def generate_data_from_dataset_biases(queries_dict):
    """Assigns labels based on known inherent dataset biases.
    """
    labeled_data = []
    # FAQ-like queries -> simple/moderate
    for query in queries_dict["faq"]:
        labeled_data.append((query, "simple"))
    for query in queries_dict["general"]:
        labeled_data.append((query, "simple"))
    # Product queries -> moderate
    for query in queries_dict["product_simple"]:
        labeled_data.append((query, "moderate"))
    # Complex product queries -> complex
    for query in queries_dict["product_complex"]:
        labeled_data.append((query, "complex"))

    return labeled_data

def combine_and_deduplicate_datasets(model_outcome_data, bias_data):
    """Combines and deduplicates labeled datasets, prioritizing model outcomes where available.
    If a query has conflicting labels, the one from model outcome is preferred if it's more specific.
    For simplicity, we'll assume model outcome data is more precise or takes precedence if a conflict occurs.
    """
    combined_data = {}

    for query, label in model_outcome_data:
        combined_data[query] = label

    # Add bias data, only if query not already present from model outcomes
    for query, label in bias_data:
        if query not in combined_data:
            combined_data[query] = label

    return list(combined_data.items())

if __name__ == "__main__":
    print("Generating synthetic training data...")
    synthetic_queries = generate_synthetic_queries()

    # Strategy 1: Model Prediction Outcomes
    model_outcome_labels = generate_data_from_model_outcomes(synthetic_queries)
    print("\nLabels from Model Prediction Outcomes:")
    for q, l in model_outcome_labels:
        print(f"  Query: '{q}' -> Label: {l}")

    # Strategy 2: Inherent Dataset Biases
    bias_labels = generate_data_from_dataset_biases(synthetic_queries)
    print("\nLabels from Inherent Dataset Biases:")
    for q, l in bias_labels:
        print(f"  Query: '{q}' -> Label: {l}")

    # Combine and Deduplicate
    final_training_data = combine_and_deduplicate_datasets(model_outcome_labels, bias_labels)
    print("\nFinal Combined Training Data:")
    for q, l in final_training_data:
        print(f"  Query: '{q}' -> Label: {l}")

    print(f"\nGenerated {len(final_training_data)} training data points.")

    # You would typically save this data to a file (e.g., CSV, JSON) here
    # For demonstration, we'll just return it for use in other modules.

