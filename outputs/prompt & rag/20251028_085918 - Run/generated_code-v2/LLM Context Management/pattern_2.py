
import pandas as pd
import random

class DataGenerator:
    def __init__(self):
        # Simulate different LLM strategies' success rates for different complexities
        # This is a simplified model for data generation purposes
        self.strategy_success_matrix = {
            "simple": {"direct": 0.9, "single_step_rag": 0.95, "multi_step_rag": 0.98},
            "moderate": {"direct": 0.2, "single_step_rag": 0.8, "multi_step_rag": 0.9},
            "complex": {"direct": 0.05, "single_step_rag": 0.3, "multi_step_rag": 0.75}
        }
        self.complexity_map = {
            "direct": "simple",
            "single_step_rag": "moderate",
            "multi_step_rag": "complex"
        }

    def _simulate_llm_strategies(self, query: str) -> str or None:
        """Simulates running a query through different LLM strategies and returns the simplest successful label."""
        strategies = ["direct", "single_step_rag", "multi_step_rag"]
        random.shuffle(strategies) # Prioritize simpler strategies if multiple succeed

        for strategy in strategies:
            # In a real scenario, you'd run the actual LLM strategy and evaluate its output
            # Here, we simulate success based on a random chance
            # For data generation, we assume some underlying 'true' complexity for the simulation
            # Let's assume a query's inherent complexity dictates which strategy would 'truly' succeed.
            # We'll assign a 'simulated_true_complexity' for each query for this simulation.
            # This is an internal detail for generating data.

            # For the purpose of data generation based on model outcomes, we'll iterate
            # through strategies from simplest to complex and assign the label
            # based on the *first* one that 'succeeds' if we assume a query
            # can be answered by multiple but we prefer the simplest.

            # For the sake of data generation, let's simplify: A strategy 'succeeds'
            # if a random number is below a certain threshold. We'll iterate
            # in order of complexity (direct -> single -> multi) to ensure
            # the 'simplest successful' logic for labeling.
            if strategy == "direct" and random.random() < 0.7: # High chance for simple queries
                return "simple"
            elif strategy == "single_step_rag" and random.random() < 0.6: # Medium chance
                return "moderate"
            elif strategy == "multi_step_rag" and random.random() < 0.5: # Lower chance
                return "complex"
        return None # No strategy succeeded

    def generate_data_from_llm_outcomes(self, raw_queries: list[str]) -> pd.DataFrame:
        """Generates training data by simulating LLM strategy outcomes."""
        labeled_data = []
        for query in raw_queries:
            label = self._simulate_llm_strategies(query)
            if label:
                labeled_data.append({"query": query, "complexity": label})
            else:
                labeled_data.append({"query": query, "complexity": None}) # Mark as unlabeled for next step
        return pd.DataFrame(labeled_data)

    def apply_dataset_biases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies labels based on inherent dataset biases for unlabeled queries.
        In a real scenario, this would involve integrating with actual benchmark datasets.
        Here, we simulate based on keywords or patterns.
        """
        # Simulating dataset biases by looking for keywords
        # In a real-world scenario, you would have actual source datasets
        # (e.g., 'billing_faq_dataset', 'troubleshooting_guide_dataset')
        # and assign labels based on the source of the query.
        
        def _assign_bias_label(row):
            if pd.isna(row["complexity"]):
                query = row["query"].lower()
                if any(keyword in query for keyword in ["bill", "payment", "invoice", "charge"]):
                    return "moderate" # Billing queries often require specific info, not super complex
                elif any(keyword in query for keyword in ["setup", "configure", "troubleshoot", "fix", "internet not working"]):
                    return "complex" # Troubleshooting can be multi-step
                elif any(keyword in query for keyword in ["plan", "data", "minutes", "contract", "upgrade"]):
                    return "moderate"
                elif any(keyword in query for keyword in ["what is", "how to", "definition"]):
                    return "simple" # Simple informational queries
            return row["complexity"]
        
        df["complexity"] = df.apply(_assign_bias_label, axis=1)
        return df

    def generate_training_data(self, initial_raw_queries: list[str]) -> pd.DataFrame:
        """Orchestrates the data generation process."""
        print("\n--- Generating data from LLM outcomes ---")
        df_llm_outcomes = self.generate_data_from_llm_outcomes(initial_raw_queries)
        print(f"Initial data generated. Labeled: {df_llm_outcomes['complexity'].count()}/{len(df_llm_outcomes)}")
        
        print("\n--- Applying dataset biases to unlabeled data ---")
        df_final = self.apply_dataset_biases(df_llm_outcomes)
        print(f"Final labeled data: {df_final['complexity'].count()}/{len(df_final)}")
        
        # Filter out any remaining unlabeled data if necessary, or keep for further processing
        df_final_labeled = df_final.dropna(subset=["complexity"])
        print(f"Data points for training: {len(df_final_labeled)}")
        return df_final_labeled

# Example Usage (for testing the module independently):
if __name__ == "__main__":
    generator = DataGenerator()
    sample_queries = [
        "What is my current data plan?",
        "My internet is not working, how do I fix it?",
        "How do I pay my bill?",
        "Explain the difference between 4G and 5G.",
        "I want to upgrade my phone, what are my options and how does it affect my contract?",
        "What are the roaming charges in France?",
        "Can I change my billing cycle date?",
        "Why is my call dropping frequently?",
        "What is the best plan for a family of four?",
        "Where can I find the terms and conditions?"
    ]

    training_data = generator.generate_training_data(sample_queries)
    print("\n--- Generated Training Data ---")
    print(training_data)

    # Verify distribution
    print("\n--- Complexity Distribution ---")
    print(training_data["complexity"].value_counts())
