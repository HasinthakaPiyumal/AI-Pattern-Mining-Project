class AutomatedPromptOptimizer:
    def __init__(self, exemplars):
        self.exemplars = exemplars

    def _initial_prompt_creation(self):
        return [
            "Respond to customer queries about product availability and shipping.",
            "Provide information on order status and return policies.",
            "Help customers with common product-related questions."
        ]

    def _prompt_variation_paraphrasing(self, prompt):
        variations = [
            prompt, # Keep original
            prompt.replace("Respond to", "Answer").replace("queries about", "questions regarding"),
            prompt.replace("Provide information on", "Give details about"),
            prompt.replace("Help customers with", "Assist shoppers on").replace("common product-related questions.", "frequently asked product questions.")
        ]
        return list(set(variations)) # Remove duplicates

    def _simulate_chatbot_response(self, prompt, query):
        response_keywords = []
        if "product availability" in prompt.lower() and "stock" in query.lower():
            response_keywords.append("We check stock levels for you.")
        if "shipping" in prompt.lower() and ("delivery" in query.lower() or "ship" in query.lower()):
            response_keywords.append("Information about shipping is available.")
        if "order status" in prompt.lower() and "order" in query.lower() and "status" in query.lower():
            response_keywords.append("We can provide your order status.")
        if "return policies" in prompt.lower() and ("return" in query.lower() or "refund" in query.lower()):
            response_keywords.append("Details on our return policy.")
        if "product-related questions" in prompt.lower() and ("how to use" in query.lower() or "feature" in query.lower()):
            response_keywords.append("We assist with product details.")

        if not response_keywords:
            return "I'm sorry, I don't have information on that currently."
        return " ".join(response_keywords) + " Based on your query: '" + query + "'."

    def _score_prompt(self, simulated_response, ideal_response):
        score = 0
        ideal_keywords = [word.lower() for word in ideal_response.split() if len(word) > 2]
        simulated_keywords = [word.lower() for word in simulated_response.split() if len(word) > 2]

        for keyword in ideal_keywords:
            if keyword in simulated_keywords:
                score += 1
        return score

    def optimize_prompts(self, max_iterations=5, performance_threshold=0.8):
        current_prompts = self._initial_prompt_creation()
        best_prompt = ""
        highest_avg_score = -1

        for iteration in range(max_iterations):
            all_prompts_to_score = set()
            for p in current_prompts:
                all_prompts_to_score.update(self._prompt_variation_paraphrasing(p))

            iteration_scores = {}

            for prompt in all_prompts_to_score:
                total_score_for_prompt = 0
                for exemplar in self.exemplars:
                    simulated_resp = self._simulate_chatbot_response(prompt, exemplar["query"])
                    score = self._score_prompt(simulated_resp, exemplar["ideal_response"])
                    total_score_for_prompt += score
                
                avg_score = total_score_for_prompt / len(self.exemplars)
                iteration_scores[prompt] = avg_score

            if not iteration_scores:
                break

            sorted_prompts = sorted(iteration_scores.items(), key=lambda item: item[1], reverse=True)
            current_best_prompt, current_best_score = sorted_prompts[0]

            if current_best_score > highest_avg_score:
                highest_avg_score = current_best_score
                best_prompt = current_best_prompt
            
            if highest_avg_score >= performance_threshold:
                break

            current_prompts = [p for p, score in sorted_prompts[:min(3, len(sorted_prompts))]] # Take top 3 for next iteration

        return best_prompt, highest_avg_score


if __name__ == "__main__":
    # Exemplar Data Management
    customer_exemplars = [
        {"query": "Where is my order?", "ideal_response": "We can track your order status for you. Please provide your order ID."},
        {"query": "How do I return a product?", "ideal_response": "Details regarding our return policy can be found on our website, or we can guide you through the process."},
        {"query": "Do you have this shirt in stock?", "ideal_response": "Let me check the product availability for that shirt. What size and color are you looking for?"},
        {"query": "When will my package arrive?", "ideal_response": "Information about your shipping delivery date can be found in your order details."},
        {"query": "Tell me about the features of this laptop.", "ideal_response": "I can assist you with the features of the laptop. What aspects are you interested in?"}
    ]

    optimizer = AutomatedPromptOptimizer(customer_exemplars)
    optimized_prompt, final_score = optimizer.optimize_prompts(max_iterations=10, performance_threshold=3)

    print(f"Optimized Prompt: {optimized_prompt}")
    print(f"Final Average Score: {final_score:.2f}")

    # Test with a new query using the optimized prompt
    test_query = "I need to know if the blue dress is available."
    simulated_response = optimizer._simulate_chatbot_response(optimized_prompt, test_query)
    print(f"\nTest Query: '{test_query}'")
    print(f"Chatbot Response with Optimized Prompt: '{simulated_response}'")

    test_query_2 = "What's the process for returning a defective item?"
    simulated_response_2 = optimizer._simulate_chatbot_response(optimized_prompt, test_query_2)
    print(f"\nTest Query: '{test_query_2}'")
    print(f"Chatbot Response with Optimized Prompt: '{simulated_response_2}'")