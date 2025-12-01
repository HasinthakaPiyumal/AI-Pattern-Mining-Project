# few_shot_chatbot.py

class FewShotClassifier:
    def __init__(self, base_instruction: str, initial_examples: dict = None):
        self.base_instruction = base_instruction
        self.examples = initial_examples if initial_examples is not None else {}

    def add_exemplar(self, category: str, query: str, label: str = None):
        """Adds a new exemplar (example query-label pair) for a given category."""
        if category not in self.examples:
            self.examples[category] = []
        # If label is not provided, assume the category name is the label for the exemplar
        self.examples[category].append({"query": query, "label": label if label else category})
        print(f"Added exemplar for category '{category}': Query='{query}', Label='{label if label else category}'")

    def _construct_prompt(self, new_query: str) -> str:
        """Constructs the few-shot prompt including base instruction and exemplars."""
        prompt_parts = [self.base_instruction]

        for category, exemplars in self.examples.items():
            for example in exemplars:
                prompt_parts.append(f"Customer query: \"{example['query']}\"")
                prompt_parts.append(f"Category: {example['label']}")
            prompt_parts.append("") # Add a newline for separation between categories or after exemplars

        prompt_parts.append(f"Customer query: \"{new_query}\"")
        prompt_parts.append(f"Category:") # The LLM will complete this

        return "\n".join(prompt_parts)

    def _call_simulated_llm(self, prompt: str) -> str:
        """
        Simulates an LLM call to classify the query.
        In a real application, this would interact with an actual LLM API (e.g., OpenAI, Cohere).
        For this simulation, it tries to find the most similar category from existing examples.
        """
        # Simple heuristic: find if the new_query contains keywords from existing categories
        # A real LLM would be much more sophisticated.
        # The prompt itself contains the `new_query` at the very end.
        # We need to extract it to use in the simulation logic.
        query_line_prefix = "Customer query: "
        new_query_from_prompt = ""
        for line in prompt.split('\n'):
            if line.startswith(query_line_prefix) and line != self.base_instruction and not any(ex["query"] in line for ex_list in self.examples.values() for ex in ex_list):
                new_query_from_prompt = line[len(query_line_prefix):].strip().strip('"')
                break

        if not new_query_from_prompt:
             # Fallback if parsing fails, assume the last query added to prompt_parts
             # This is a bit fragile, better to pass the original query directly.
             pass # The main classify method passes the original query, so this is fine.

        new_query_lower = new_query_from_prompt.lower() if new_query_from_prompt else prompt.split('\n')[-2].replace('Customer query: "', '').replace('"', '').lower() # Fallback for simulation
        
        predicted_category = "Uncategorized" # Default
        best_match_score = -1

        for category, exemplars in self.examples.items():
            current_category_score = 0
            for example in exemplars:
                # A very basic keyword matching simulation
                if any(word in new_query_lower for word in example['query'].lower().split()):
                    current_category_score += 1
            if current_category_score > best_match_score:
                best_match_score = current_category_score
                predicted_category = category
        
        # If no strong match, or to simulate an LLM guessing
        if best_match_score == 0 and len(self.examples) > 0:
            # Fallback: if no keywords match, just pick the first category from examples
            # This is a very crude simulation. A real LLM would use semantic understanding.
            predicted_category = list(self.examples.keys())[0] if self.examples else "General Inquiry"
        elif not self.examples:
            predicted_category = "General Inquiry"


        # In a real scenario, the LLM would return text like "Category: Billing Inquiry"
        # We need to parse that. For simulation, we directly return the "predicted_category".
        print(f"(Simulated LLM Processed Prompt and predicted: {predicted_category})")
        return predicted_category


    def classify_customer_query(self, query: str) -> str:
        """Classifies a new customer query using few-shot prompting."""
        prompt = self._construct_prompt(query)
        print("\n--- Generated Few-Shot Prompt ---")
        print(prompt)
        print("-------------------------------\n")

        # In a real system:
        # response = llm_api.generate(prompt, max_tokens=20, stop=['\n'])
        # predicted_category = parse_llm_response(response)

        predicted_category = self._call_simulated_llm(prompt) # Pass the full prompt to the simulated LLM
        return predicted_category

def main():
    print("Welcome to the Few-Shot Customer Support Chatbot!")

    base_instruction = "Classify the following customer queries into one of the provided categories. Provide only the category name."

    # Initial examples for common categories
    initial_examples = {
        "Billing Inquiry": [
            {"query": "My last bill seems incorrect.", "label": "Billing Inquiry"},
            {"query": "How can I update my payment method?", "label": "Billing Inquiry"}
        ],
        "Technical Support": [
            {"query": "My internet is not working.", "label": "Technical Support"},
            {"query": "I can't log into my account.", "label": "Technical Support"}
        ],
        "Product Feature Request": [
            {"query": "Can you add a dark mode to your app?", "label": "Product Feature Request"},
            {"query": "I'd like to suggest a new search filter.", "label": "Product Feature Request"}
        ]
    }

    classifier = FewShotClassifier(base_instruction, initial_examples)

    print("\n--- Initial Chatbot Capabilities ---")
    print("Known categories:", list(classifier.examples.keys()))

    while True:
        user_input = input("\nEnter a customer query (or 'add example' to teach, 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'add example':
            category = input("Enter the NEW or existing category name for the example: ").strip()
            query = input(f"Enter an example query for '{category}': ").strip()
            # For simplicity, we assume the label is the category name for new examples
            classifier.add_exemplar(category, query, category)
        else:
            print("\n--- Classifying Query ---")
            predicted_category = classifier.classify_customer_query(user_input)
            print(f"Chatbot classified the query as: {predicted_category}")

if __name__ == "__main__":
    main()