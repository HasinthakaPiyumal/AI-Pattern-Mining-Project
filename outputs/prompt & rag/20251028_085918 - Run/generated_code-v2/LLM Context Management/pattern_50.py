import random
from typing import List, Tuple, Dict

class DataGenerator:
    def __init__(self):
        # Placeholder for different retrieval strategies.
        # In a real system, these would call actual LLMs and retrieval systems.
        pass

    def _simulate_no_retrieval(self, query: str) -> bool:
        """Simulates a simple LLM response without retrieval."""
        return "faq" in query.lower() or "what is" in query.lower()

    def _simulate_single_step_retrieval(self, query: str) -> bool:
        """Simulates single-step retrieval (e.g., knowledge base lookup)."""
        return "how to" in query.lower() or "error" in query.lower()

    def _simulate_multi_step_retrieval(self, query: str) -> bool:
        """Simulates multi-step retrieval (e.g., complex troubleshooting)."""
        return "troubleshoot" in query.lower() or "integration" in query.lower() or "performance issue" in query.lower()

    def generate_labels_from_model_outcomes(self, queries: List[str]) -> List[Tuple[str, str]]:
        """
        Labels queries based on the success of different retrieval strategies,
        prioritizing simpler models.
        """
        labeled_data = []
        for query in queries:
            label = None
            if self._simulate_no_retrieval(query):
                label = "simple"
            elif self._simulate_single_step_retrieval(query):
                label = "moderate"
            elif self._simulate_multi_step_retrieval(query):
                label = "complex"

            # Fallback for demonstration if no strategy matches directly
            if label is None:
                if len(query.split()) < 5:
                    label = "simple"
                elif len(query.split()) < 15:
                    label = "moderate"
                else:
                    label = "complex"

            labeled_data.append((query, label))
        return labeled_data

    def generate_labels_from_dataset_biases(self, queries_with_source: List[Tuple[str, str]], dataset_biases: Dict[str, str]) -> List[Tuple[str, str]]:
        """
        Labels queries based on inherent dataset biases if source information is available.
        Queries that don't have a specific bias mapping will be returned with a None label.
        """
        labeled_data = []
        for query, source in queries_with_source:
            label = dataset_biases.get(source, None)
            labeled_data.append((query, label))
        return labeled_data

    def generate_training_data(self,
                               raw_queries: List[str],
                               queries_with_source: List[Tuple[str, str]],
                               dataset_biases: Dict[str, str]) -> List[Tuple[str, str]]:
        """
        Combines model prediction outcomes and inherent dataset biases to generate
        a comprehensive training dataset.
        """
        # Step 1: Label based on model prediction outcomes
        data_from_models = self.generate_labels_from_model_outcomes(raw_queries)
        
        # Create a temporary dictionary to easily update labels based on sources
        temp_data_dict = {query: label for query, label in data_from_models}
        
        # Step 2: Apply dataset biases, potentially overwriting or adding new labels
        for query_with_src, source in queries_with_source:
            if source in dataset_biases:
                # Prioritize dataset bias if known and specific
                temp_data_dict[query_with_src] = dataset_biases[source]
            elif query_with_src not in temp_data_dict:
                # Add queries only present in queries_with_source if no bias is found yet
                temp_data_dict[query_with_src] = None # Will be given a default label later

        # Final pass to ensure all queries have a label
        combined_labeled_data: List[Tuple[str, str]] = []
        for query, current_label in temp_data_dict.items():
            if current_label is None:
                # Apply a default label if still none after both strategies
                if len(query.split()) < 5:
                    final_label = "simple"
                elif len(query.split()) < 15:
                    final_label = "moderate"
                else:
                    final_label = "complex"
                combined_labeled_data.append((query, final_label))
            else:
                combined_labeled_data.append((query, current_label))

        return combined_labeled_data