import pandas as pd

def get_initial_training_data():
    """
    Provides initial training data for the query classifier.
    In a real scenario, this would come from historical labeled data.
    """
    data = {
        "query": [
            "Where is my order?",
            "How do I reset my password?",
            "What are your return policies?",
            "My laptop is not turning on, can you help?",
            "I need to change my shipping address for order #12345.",
            "What is the price of product X?",
            "The website is not loading correctly.",
            "How can I track my recent purchase?",
            "Tell me about your privacy policy.",
            "I received a damaged item, what should I do?"
        ],
        "label": [
            "order_issue",
            "technical_support",
            "simple_info",
            "technical_support",
            "order_issue",
            "simple_info",
            "technical_support",
            "order_issue",
            "simple_info",
            "order_issue"
        ]
    }
    return pd.DataFrame(data)

def generate_new_training_data(new_queries_with_labels: list[dict], existing_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Simulates automatic generation of new training data.
    In a real system, this would come from new, potentially LLM-labeled
    and human-validated queries.

    Args:
        new_queries_with_labels (list[dict]): A list of dictionaries, each with 'query' and 'label'.
        existing_df (pd.DataFrame, optional): Existing training data to append to.

    Returns:
        pd.DataFrame: Updated training data.
    """
    new_df = pd.DataFrame(new_queries_with_labels)
    if existing_df is not None:
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    return updated_df

if __name__ == "__main__":
    # Example usage
    initial_data = get_initial_training_data()
    print("Initial Training Data:")
    print(initial_data)

    print("\n--- Simulating new data generation ---\n")
    new_data_points = [
        {"query": "Can I pay with PayPal?", "label": "simple_info"},
        {"query": "My payment failed for order #67890.", "label": "order_issue"},
        {"query": "The app keeps crashing on my phone.", "label": "technical_support"}
    ]
    updated_data = generate_new_training_data(new_data_points, initial_data)
    print("Updated Training Data:")
    print(updated_data)

    # In a real scenario, this updated_data would then be used to retrain the classifier.
