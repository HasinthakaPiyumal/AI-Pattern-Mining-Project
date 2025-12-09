"""
This module simulates the preparation and blending of various data sources
for instruction tuning an LLM, following the 'Multi-Source Data Blending'
pattern for an E-commerce Customer Support Assistant.

It defines functions to generate synthetic examples for each data type
and a function to blend them into a unified format suitable for conceptual LLM tuning.
"""

def _get_sft_data():
    """Simulates loading general Supervised Fine-Tuning (SFT) data."""
    return [
        {"instruction": "Summarize the following text: 'The quick brown fox jumps over the lazy dog.'", "output": "A fox jumps over a dog."},
        {"instruction": "What is the capital of France?", "output": "Paris."},
        {"instruction": "Explain what a 'cookie' is in web development.", "output": "A cookie is a small piece of data stored on the user's computer by the web browser while browsing a website."},
    ]

def _get_context_rich_qa_data():
    """Simulates loading context-rich Question-Answering (QA) data."""
    return [
        {"instruction": "Given the context, answer the question.\nContext: The return policy states items can be returned within 30 days if unused.\nQuestion: How many days do I have to return an item?", "output": "You have 30 days to return an item."},
        {"instruction": "Based on the information, what is the warranty period?\nContext: All electronics come with a 1-year manufacturer's warranty.\nQuestion: What is the warranty on electronics?", "output": "The warranty on electronics is 1 year."},
        {"instruction": "From the text below, what are the shipping options?\nContext: Standard shipping takes 5-7 days. Expedited shipping is also available.\nQuestion: What shipping choices are there?", "output": "There is standard shipping (5-7 days) and expedited shipping."},
    ]

def _get_retrieval_augmented_qa_data():
    """Simulates loading Retrieval-Augmented Question-Answering (RA-QA) data,
    including potentially irrelevant contexts."""
    return [
        {"instruction": "Using the provided information, answer the query. Prioritize relevant facts.\nContexts:\n1. Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging.\n2. Check out our latest spring collection! New styles arriving daily with discounts up to 50% off.\nQuery: What's your return policy?", "output": "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging."},
        {"instruction": "Answer the query based on these contexts.\nContexts:\n1. To track your order, please visit the 'My Orders' section on our website and enter your order number.\n2. This product is made with sustainable materials.\nQuery: How do I track my delivery?", "output": "You can track your delivery by visiting the 'My Orders' section on our website and entering your order number."},
    ]

def _get_context_ranking_data():
    """Simulates loading context ranking data (query, passage, relevance score)."""
    return [
        {"instruction": "Rate the relevance of the following passage to the query (1=irrelevant, 5=highly relevant).\nQuery: Shipping costs\nPassage: Standard shipping usually takes 5-7 business days. Expedited options are available at checkout.\nRelevance:", "output": "5"},
        {"instruction": "Rate the relevance of the following passage to the query.\nQuery: Return timeframe\nPassage: Our contact support is available 24/7 via chat.\nRelevance:", "output": "1"},
        {"instruction": "Rate the relevance of the following passage to the query.\nQuery: Product features\nPassage: This product features a 1080p display and a 5000 mAh battery.\nRelevance:", "output": "4"},
    ]

def _get_retrieval_augmented_ranking_data():
    """Simulates loading retrieval-augmented ranking data (query, multiple contexts, ranked list)."""
    return [
        {"instruction": "Rank the following passages by relevance to the query, from most to least relevant (provide passage numbers).\nQuery: About product returns\nPassages:\n1. Our return policy allows returns within 30 days of purchase.\n2. Standard shipping takes 5-7 days.\n3. Contact our support for technical issues.\nRanking:", "output": "1, 3, 2"}, # Assuming 'contact support' might be somewhat relevant for return issues
        {"instruction": "Rank the passages for the query.\nQuery: Order tracking\nPassages:\n1. To track your order, visit 'My Orders'.\n2. Our new autumn collection is here!\n3. Warranty covers manufacturing defects for 1 year.\nRanking:", "output": "1, 3, 2"}, # 3 might be less relevant than 2 depending on query nuance, but 1 is clearly best
    ]

def blend_datasets(ratios=None):
    """
    Blends all simulated datasets into a unified format for instruction tuning.
    The 'ratios' parameter could control the proportion of each data type.
    For this simulation, it simply concatenates and standardizes the format.
    """
    if ratios is None:
        # Default equal ratio for demonstration
        ratios = {
            "sft": 1,
            "context_qa": 1,
            "ra_qa": 1,
            "ranking": 1,
            "ra_ranking": 1,
        }

    all_data = []

    # SFT data
    for _ in range(ratios["sft"]):
        for item in _get_sft_data():
            all_data.append({"text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"})

    # Context-rich QA data
    for _ in range(ratios["context_qa"]):
        for item in _get_context_rich_qa_data():
            all_data.append({"text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"})

    # Retrieval-Augmented QA data
    for _ in range(ratios["ra_qa"]):
        for item in _get_retrieval_augmented_qa_data():
            all_data.append({"text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"})

    # Context Ranking data
    for _ in range(ratios["ranking"]):
        for item in _get_context_ranking_data():
            all_data.append({"text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"})

    # Retrieval-Augmented Ranking data
    for _ in range(ratios["ra_ranking"]):
        for item in _get_retrieval_augmented_ranking_data():
            all_data.append({"text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"})

    # In a real scenario, you'd shuffle and potentially apply more sophisticated sampling
    # For this simulation, simple concatenation is sufficient.
    print(f"Blended dataset created with {len(all_data)} total examples.")
    return all_data

if __name__ == "__main__":
    # Example usage:
    blended_data = blend_datasets()
    # You would then pass this 'blended_data' to your LLM fine-tuning pipeline.
    # For demonstration, print a few examples:
    print("\nFirst 3 blended examples:")
    for i, example in enumerate(blended_data[:3]):
        print(f"\n--- Example {i+1} ---")
        print(example["text"])
