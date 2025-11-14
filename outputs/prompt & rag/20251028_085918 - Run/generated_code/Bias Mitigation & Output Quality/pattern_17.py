import random
from collections import Counter

class ExemplarManager:
    """Manages a pool of diverse few-shot exemplars for various query categories."""
    def __init__(self):
        # Example exemplars. In a real system, these would be loaded from a dataset.
        self.exemplars = {
            "shipping_inquiry": [
                {"query": "Where is my order #12345?", "classification": "shipping_inquiry"},
                {"query": "Has my package shipped yet?", "classification": "shipping_inquiry"},
                {"query": "When will my delivery arrive?", "classification": "shipping_inquiry"},
                {"query": "Tracking for order 67890 please.", "classification": "shipping_inquiry"},
                {"query": "My order hasn't moved for days.", "classification": "shipping_inquiry"},
                {"query": "Is order 54321 out for delivery?", "classification": "shipping_inquiry"},
            ],
            "product_inquiry": [
                {"query": "What are the dimensions of the X-model TV?", "classification": "product_inquiry"},
                {"query": "Does the Z-phone come in blue?", "classification": "product_inquiry"},
                {"query": "Tell me about the features of the new laptop.", "classification": "product_inquiry"},
                {"query": "Is this product compatible with iOS?", "classification": "product_inquiry"},
                {"query": "What materials is the jacket made of?", "classification": "product_inquiry"},
                {"query": "Does the blender have multiple speed settings?", "classification": "product_inquiry"},
            ],
            "billing_issue": [
                {"query": "I was double charged for my purchase.", "classification": "billing_issue"},
                {"query": "My credit card was declined, but I have funds.", "classification": "billing_issue"},
                {"query": "Can you explain this charge on my statement?", "classification": "billing_issue"},
                {"query": "I need a refund for an incorrect amount.", "classification": "billing_issue"},
                {"query": "Why was my subscription renewed early?", "classification": "billing_issue"},
                {"query": "There's an unrecognized charge from your store.", "classification": "billing_issue"},
            ],
            "return_request": [
                {"query": "I want to return item #AB123.", "classification": "return_request"},
                {"query": "How do I initiate a return for a faulty product?", "classification": "return_request"},
                {"query": "What is your return policy?", "classification": "return_request"},
                {"query": "Can I exchange this shirt for a different size?", "classification": "return_request"},
                {"query": "The item I received is damaged, I need to return it.", "classification": "return_request"},
                {"query": "I bought the wrong item, how can I return it?", "classification": "return_request"},
            ]
        }
        self.categories = list(self.exemplars.keys())

    def get_distinct_exemplar_subsets(self, num_subsets: int, examples_per_subset: int = 3):
        """
        Provides multiple distinct subsets of exemplars.
        It tries to pick exemplars from different categories to make subsets diverse.
        For simplicity, it just shuffles and takes unique ones.
        """
        all_available_exemplars = []
        for category in self.categories:
            all_available_exemplars.extend(self.exemplars[category])

        if len(all_available_exemplars) < num_subsets * examples_per_subset:
            print(f"Warning: Not enough unique exemplars to create {num_subsets} distinct subsets of {examples_per_subset} examples each. Some subsets might be identical or smaller.")

        subsets = []
        for _ in range(num_subsets):
            random.shuffle(all_available_exemplars)
            subset = random.sample(all_available_exemplars, min(examples_per_subset, len(all_available_exemplars)))
            subsets.append(subset)
        return subsets


class PromptGenerator:
    """Constructs few-shot prompts for the LLM."""
    def generate_prompt(self, customer_query: str, exemplars_subset: list):
        """
        Generates a few-shot prompt string.
        Each exemplar is formatted as "Q: <query>\nA: <classification>".
        """
        prompt_parts = []
        for ex in exemplars_subset:
            prompt_parts.append(f"Q: {ex['query']}\nA: {ex['classification']}")
        
        prompt_parts.append(f"Q: {customer_query}\nA:") # The LLM should complete this.

        return "\n\n".join(prompt_parts)


class SimulatedLLM:
    """Simulates the interaction with a Large Language Model."""
    def __init__(self):
        # A simple mapping for "expected" responses, but with built-in variance.
        self.expected_responses = {
            "shipping": "shipping_inquiry",
            "delivery": "shipping_inquiry",
            "order": "shipping_inquiry",
            "track": "shipping_inquiry",
            "package": "shipping_inquiry",

            "product": "product_inquiry",
            "dimensions": "product_inquiry",
            "features": "product_inquiry",
            "compatible": "product_inquiry",
            "materials": "product_inquiry",

            "charge": "billing_issue",
            "billed": "billing_issue",
            "refund": "billing_issue",
            "statement": "billing_issue",
            "subscription": "billing_issue",

            "return": "return_request",
            "exchange": "return_request",
            "damaged": "return_request",
            "faulty": "return_request",
            "wrong item": "return_request",
        }
        self.all_possible_categories = list(set(self.expected_responses.values()))

    def simulate_llm_response(self, prompt: str) -> str:
        """
        Simulates an LLM response by trying to extract a classification.
        Introduces some "noise" or "variance" to simulate different LLM outputs.
        """
        # Extract the customer query from the last part of the prompt
        query_start = prompt.rfind("Q: ") + 3
        query_end = prompt.rfind("\nA:")
        customer_query = prompt[query_start:query_end].lower()

        predicted_category = None
        for keyword, category in self.expected_responses.items():
            if keyword in customer_query:
                predicted_category = category
                break
        
        # Introduce some "noise" or "variance" to simulate different LLM outputs
        if random.random() < 0.2:  # 20% chance of a "wrong" or "varied" classification
            if predicted_category and len(self.all_possible_categories) > 1:
                # Pick a random category that is NOT the predicted one
                other_categories = [cat for cat in self.all_possible_categories if cat != predicted_category]
                if other_categories:
                    return random.choice(other_categories)
            else: # If no strong prediction or only one category, just pick randomly
                return random.choice(self.all_possible_categories)
        
        return predicted_category if predicted_category else random.choice(self.all_possible_categories)


class OutputAggregator:
    """Aggregates classification outputs from multiple LLM interactions."""
    def aggregate_outputs(self, llm_outputs: list) -> str:
        """
        Applies a majority voting strategy to determine the final classification.
        """
        if not llm_outputs:
            return "unknown"
        
        # Count occurrences of each classification
        counts = Counter(llm_outputs)
        
        # Find the most common classification(s)
        most_common = counts.most_common(1)
        if most_common:
            return most_common[0][0]
        return "unknown"


class CustomerQueryClassifier:
    """
    Main Orchestrator for the Enhanced Customer Query Classifier.
    Coordinates the entire DENSE process.
    """
    def __init__(self):
        self.exemplar_manager = ExemplarManager()
        self.prompt_generator = PromptGenerator()
        self.simulated_llm = SimulatedLLM()
        self.output_aggregator = OutputAggregator()

    def classify_query(self, customer_query: str, num_ensembles: int = 5, examples_per_subset: int = 3) -> dict:
        """
        Classifies a customer query using the Demonstration Ensembling (DENSE) pattern.
        
        Args:
            customer_query (str): The new customer query to classify.
            num_ensembles (int): The number of distinct prompts/LLM interactions to run.
            examples_per_subset (int): The number of few-shot exemplars to include in each prompt.
            
        Returns:
            dict: A dictionary containing the final classification, individual LLM outputs,
                  and the prompts used.
        """
        individual_llm_outputs = []
        generated_prompts = []

        # 1. Request multiple distinct exemplar subsets
        exemplar_subsets = self.exemplar_manager.get_distinct_exemplar_subsets(
            num_ensembles, examples_per_subset
        )

        for i, subset in enumerate(exemplar_subsets):
            # 2. For each subset, generate a prompt
            prompt = self.prompt_generator.generate_prompt(customer_query, subset)
            generated_prompts.append(f"--- Prompt {i+1} ---\n{prompt}")

            # 3. Send the prompt to the simulated LLM Interaction Module
            llm_classification = self.simulated_llm.simulate_llm_response(prompt)
            individual_llm_outputs.append(llm_classification)

        # 4. Pass these classifications to the Output Aggregation Module to get the final result
        final_classification = self.output_aggregator.aggregate_outputs(individual_llm_outputs)

        return {
            "customer_query": customer_query,
            "individual_llm_outputs": individual_llm_outputs,
            "final_classification": final_classification,
            "generated_prompts_preview": generated_prompts # For demonstration
        }

# Example Usage:
if __name__ == "__main__":
    classifier = CustomerQueryClassifier()

    queries = [
        "Where is my order 98765?",
        "Does the new smartwatch support NFC?",
        "I was charged twice for item ABC.",
        "How do I return a broken blender?",
        "What are the specs of the XYZ drone?",
        "My tracking link for order 11223 is not working.",
        "I need a refund for my subscription.",
        "Can I exchange this size M shirt for a size L?",
        "I want to know if the headphones come with a case.",
        "I see an unknown charge of $50 from your company.",
    ]

    print("--- Demonstrating DENSE for Customer Query Classification ---")
    print("-" * 50)

    for query in queries:
        print(f"\nClassifying query: \"{query}\"")
        result = classifier.classify_query(query, num_ensembles=5) # Using 5 ensembles
        
        print(f"  Individual LLM Outputs: {result['individual_llm_outputs']}")
        print(f"  Final Aggregated Classification (DENSE): {result['final_classification']}")
        # for prompt_text in result['generated_prompts_preview']:
        #     print(f"\n{prompt_text}")
        print("-" * 50)

    print("\n--- End of Demonstration ---")