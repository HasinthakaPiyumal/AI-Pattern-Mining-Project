
import random
from typing import List, Tuple, Dict

def generate_synthetic_queries(num_queries: int) -> List[Dict]:
    """
    Generates a list of synthetic e-commerce customer queries with simulated source types.
    """
    query_templates_simple = [
        "What is the price of product X?",
        "Is product Y in stock?",
        "How do I track my order?",
        "What are your shipping options?"
    ]
    query_templates_moderate = [
        "Compare product A and product B features.",
        "What is the return policy for electronics?",
        "How do I reset my password if I forgot my old one and my email isn't working?",
        "Can I get a discount if I buy multiple items from different categories?"
    ]
    query_templates_complex = [
        "Explain the difference between material Z and material W for durability and sustainability, considering our ethical sourcing guidelines and current supplier certifications.",
        "My order #12345 was marked delivered but I didn't receive it. I also tried contacting the carrier but got no response. What should I do next?",
        "How does the loyalty program integrate with third-party payment gateways, and what are the implications for international customers regarding tax and currency conversion?"
    ]

    synthetic_queries = []
    for i in range(num_queries):
        query_type = random.choices(['simple', 'moderate', 'complex'], weights=[0.4, 0.35, 0.25], k=1)[0]
        if query_type == 'simple':
            query = random.choice(query_templates_simple).replace('X', f'item{i}').replace('Y', f'widget{i}')
            source_type = 'single-hop-faq' if random.random() < 0.8 else 'direct-query'
        elif query_type == 'moderate':
            query = random.choice(query_templates_moderate).replace('A', f'prodA{i}').replace('B', f'prodB{i}')
            source_type = 'multi-hop-product-info' if random.random() < 0.7 else 'forum-post'
        else:
            query = random.choice(query_templates_complex)
            source_type = 'multi-hop-complex-scenario' if random.random() < 0.9 else 'escalated-ticket'

        synthetic_queries.append({"query": query, "source_type": source_type, "original_complexity": query_type})
    return synthetic_queries

def _simulate_llm_response_and_evaluate(query: str, strategy: str) -> Tuple[bool, str]:
    """
    Simulates an LLM attempting to answer a query using a given strategy and evaluates success.
    Returns (success_flag, simulated_response).
    """
    # This is a highly simplified simulation. In a real system, this would involve
    # actual LLM calls and evaluation metrics (e.g., answer correctness, relevance).
    
    # Simple queries are likely to be answered by simpler strategies
    if "price" in query.lower() or "stock" in query.lower() or "track order" in query.lower():
        if strategy == 'no_rag':
            return True, f"Simulated success with {strategy} for simple query."
        elif strategy == 'single_step_rag':
            return True, f"Simulated success with {strategy} for simple query (overkill, but works)."
        elif strategy == 'multi_step_rag':
            return True, f"Simulated success with {strategy} for simple query (very overkill, but works)."

    # Moderate queries might need single-step RAG
    if "compare" in query.lower() or "policy" in query.lower() or "reset password" in query.lower():
        if strategy == 'no_rag':
            return random.random() < 0.3, f"Simulated partial success/failure with {strategy} for moderate query."
        elif strategy == 'single_step_rag':
            return True, f"Simulated success with {strategy} for moderate query."
        elif strategy == 'multi_step_rag':
            return True, f"Simulated success with {strategy} for moderate query (overkill, but works)."

    # Complex queries likely need multi-step RAG
    if "difference between material" in query.lower() or "order #" in query.lower() and "not received" in query.lower() or "loyalty program integrate" in query.lower():
        if strategy == 'no_rag':
            return random.random() < 0.1, f"Simulated failure with {strategy} for complex query."
        elif strategy == 'single_step_rag':
            return random.random() < 0.5, f"Simulated partial success with {strategy} for complex query."
        elif strategy == 'multi_step_rag':
            return True, f"Simulated success with {strategy} for complex query."
            
    # Default for unhandled specific cases (random chance)
    if strategy == 'no_rag':
        return random.random() < 0.4, f"Simulated response with {strategy}."
    elif strategy == 'single_step_rag':
        return random.random() < 0.7, f"Simulated response with {strategy}."
    elif strategy == 'multi_step_rag':
        return random.random() < 0.9, f"Simulated response with {strategy}."


def generate_training_data(num_queries: int = 100) -> List[Dict]:
    """
    Automatically generates training data for a query complexity classifier.
    Combines Model Prediction Outcomes and Inherent Dataset Biases strategies.
    """
    raw_queries = generate_synthetic_queries(num_queries)
    labeled_data = []

    # Strategy 1: Model Prediction Outcomes
    # Try strategies from simplest to most complex and label based on first success.
    for item in raw_queries:
        query = item['query']
        assigned_label = None

        strategies = ['no_rag', 'single_step_rag', 'multi_step_rag']
        complexity_map = {'no_rag': 'simple', 'single_step_rag': 'moderate', 'multi_step_rag': 'complex'}

        for strategy in strategies:
            success, _ = _simulate_llm_response_and_evaluate(query, strategy)
            if success:
                assigned_label = complexity_map[strategy]
                break # Prioritize simpler models if multiple succeed
        
        item['complexity_label'] = assigned_label
        labeled_data.append(item)

    # Strategy 2: Inherent Dataset Biases (for queries still unlabeled or to refine)
    for item in labeled_data:
        if item['complexity_label'] is None:
            source_type = item['source_type']
            if 'single-hop-faq' in source_type or 'direct-query' in source_type:
                item['complexity_label'] = 'simple'
            elif 'multi-hop-product-info' in source_type or 'forum-post' in source_type:
                item['complexity_label'] = 'moderate'
            elif 'multi-hop-complex-scenario' in source_type or 'escalated-ticket' in source_type:
                item['complexity_label'] = 'complex'
            else:
                # Fallback for truly unclassifiable by bias, perhaps assign 'moderate' or 'unknown'
                item['complexity_label'] = random.choice(['simple', 'moderate', 'complex']) # Assign randomly if no strong bias
        
        # In a more sophisticated system, Strategy 2 might also be used to 'boost' labels
        # or serve as a prior even when Strategy 1 gives a label.
        
    # Ensure all queries have a label for training
    for item in labeled_data:
        if item['complexity_label'] is None:
            item['complexity_label'] = 'moderate' # Default to moderate if all else fails

    return labeled_data

if __name__ == '__main__':
    print("Generating sample training data...")
    training_data = generate_training_data(num_queries=50)
    for i, data in enumerate(training_data[:10]): # Print first 10 for inspection
        print(f"Query {i+1}: '{data['query']}' -> Label: {data['complexity_label']} (Original: {data['original_complexity']}, Source: {data['source_type']})")

    # Check distribution of labels
    from collections import Counter
    labels = [d['complexity_label'] for d in training_data]
    print("\nLabel Distribution:", Counter(labels))
