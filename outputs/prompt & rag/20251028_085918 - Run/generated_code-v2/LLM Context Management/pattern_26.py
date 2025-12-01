"""Module for automatically generating training data for the Query Complexity Classifier."""

import random
import pandas as pd

def generate_data_from_model_outcomes(historical_queries):
    """Labels queries based on which LLM strategy successfully answered them.

    Args:
        historical_queries (list of dict): Each dict should contain 'query' and
                                          boolean flags for LLM strategy success
                                          (e.g., 'simple_llm_success', 'single_rag_success',
                                          'multi_rag_success').

    Returns:
        list of dict: Labeled queries with 'query' and 'complexity_label'.
    """
    labeled_data = []
    for item in historical_queries:
        query = item["query"]
        label = None

        if item.get("simple_llm_success", False):
            label = "simple"
        elif item.get("single_rag_success", False):
            label = "moderate"
        elif item.get("multi_rag_success", False):
            label = "complex"
        
        # If a query was handled by multiple, prioritize the simpler one
        if item.get("simple_llm_success", False):
            label = "simple"
        elif item.get("single_rag_success", False) and label != "simple":
            label = "moderate"
        elif item.get("multi_rag_success", False) and label not in ["simple", "moderate"]:
            label = "complex"
            
        if label:
            labeled_data.append({"query": query, "complexity_label": label})
    return labeled_data

def generate_data_from_dataset_biases(unlabeled_queries, dataset_bias_rules):
    """Labels queries based on inherent dataset biases (e.g., keywords, source).

    Args:
        unlabeled_queries (list of str): List of queries to be labeled.
        dataset_bias_rules (dict): A dictionary mapping keywords/patterns to complexity labels.
                                  Example: {'faq': 'simple', 'troubleshoot': 'complex'}

    Returns:
        list of dict: Labeled queries with 'query' and 'complexity_label'.
    """
    labeled_data = []
    for query in unlabeled_queries:
        label = "unknown" # Default label if no rule matches
        for keyword, complexity in dataset_bias_rules.items():
            if keyword.lower() in query.lower():
                label = complexity
                break # Apply the first matching rule
        labeled_data.append({"query": query, "complexity_label": label})
    return labeled_data

def combine_and_prepare_data(model_outcome_data, bias_data):
    """Combines and de-duplicates the generated datasets, removes 'unknown' labels.

    Args:
        model_outcome_data (list of dict): Data labeled by model outcomes.
        bias_data (list of dict): Data labeled by dataset biases.

    Returns:
        pd.DataFrame: A DataFrame with combined and cleaned training data.
    """
    combined_df = pd.DataFrame(model_outcome_data + bias_data)
    
    # Remove duplicates, keeping the first label encountered (which implicitly prioritizes
    # model outcome labels if they appear first due to concatenation order).
    combined_df.drop_duplicates(subset=["query"], keep="first", inplace=True)
    
    # Filter out queries that are still labeled as 'unknown'
    combined_df = combined_df[combined_df["complexity_label"] != "unknown"]
    
    return combined_df
