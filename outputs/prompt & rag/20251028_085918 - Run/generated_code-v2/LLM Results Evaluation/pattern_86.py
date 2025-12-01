import random

class MockLLM:
    """A mock Language Model for demonstration purposes."""
    def generate(self, prompt_text):
        """Generates a mock response based on the prompt."""
        # Simulate different quality responses for different prompts
        if "detailed solution" in prompt_text.lower():
            return "Here is a detailed solution to your problem. Please follow these steps carefully..."
        elif "quick answer" in prompt_text.lower():
            return "Here's a quick answer: Check the FAQ or your order status."
        elif "general inquiry" in prompt_text.lower():
            return "Thank you for your inquiry. How can I assist you further?"
        else:
            return "I am a helpful assistant. How can I help with your e-commerce query?"

    def get_log_likelihood_score(self, prompt_text, generated_response):
        """
        Simulates obtaining a log-likelihood-like score for a prompt-response pair.
        In a real scenario, this would involve passing both to the LLM and
        getting the log probabilities of the generated tokens given the prompt.
        For demonstration, we'll assign a score based on some heuristics.
        A higher score implies better alignment/information.
        """
        score = 0
        if "detailed solution" in prompt_text.lower() and "detailed solution" in generated_response.lower():
            score += 0.8
        elif "quick answer" in prompt_text.lower() and "quick answer" in generated_response.lower():
            score += 0.6
        elif "general inquiry" in prompt_text.lower() and "assist you further" in generated_response.lower():
            score += 0.5
        
        # Add some randomness to simulate variation
        score += random.uniform(-0.1, 0.1)
        return score

class MaxMutualInformationPromptSelector:
    """
    Selects the optimal prompt template using a proxy for Max Mutual Information.
    The proxy here is based on a simulated log-likelihood score from the LLM,
    where a higher score indicates a better alignment between prompt and output
    for a given query.
    """
    def __init__(self, llm_model, prompt_templates):
        self.llm = llm_model
        self.prompt_templates = prompt_templates

    def _calculate_mutual_information_proxy(self, formatted_prompt, generated_response):
        """
        Calculates a proxy for mutual information between the formatted prompt
        and the LLM's generated response.

        In a real application, this would involve more sophisticated techniques
        like:
        - Obtaining log-probabilities of the generated tokens given the prompt.
        - Using a separate model to evaluate relevance/coherence.
        - Information-theoretic measures if the LLM provides sufficient data.
        
        For this demonstration, we use the MockLLM's internal scoring.
        """
        return self.llm.get_log_likelihood_score(formatted_prompt, generated_response)

    def select_optimal_prompt(self, customer_query):
        """
        Evaluates each prompt template for a given customer query and
        selects the one that maximizes the mutual information proxy.
        """
        best_prompt_template = None
        max_mi_score = -float('inf')
        optimal_formatted_prompt = ""

        print(f"\nEvaluating prompts for query: '{customer_query}'")

        for template_name, template_string in self.prompt_templates.items():
            formatted_prompt = template_string.format(query=customer_query)
            
            # Generate a "test" response with the current prompt
            generated_response = self.llm.generate(formatted_prompt)
            
            # Calculate mutual information proxy
            current_mi_score = self._calculate_mutual_information_proxy(formatted_prompt, generated_response)
            
            print(f"  - Template '{template_name}': Score = {current_mi_score:.4f}")

            if current_mi_score > max_mi_score:
                max_mi_score = current_mi_score
                best_prompt_template = template_name
                optimal_formatted_prompt = formatted_prompt

        print(f"Optimal prompt selected: '{best_prompt_template}' with score {max_mi_score:.4f}")
        return optimal_formatted_prompt, max_mi_score # Returning the score for completeness

class CustomerSupportChatbot:
    """
    An AI-powered customer support chatbot for an e-commerce platform.
    It uses MaxMutualInformationPromptSelector to choose the best prompt.
    """
    def __init__(self, llm_model, prompt_templates):
        self.llm = llm_model
        self.prompt_selector = MaxMutualInformationPromptSelector(llm_model, prompt_templates)

    def get_chatbot_response(self, customer_query):
        """
        Generates a response to a customer query by first selecting
        the optimal prompt template and then using it with the LLM.
        """
        optimal_formatted_prompt, _ = self.prompt_selector.select_optimal_prompt(customer_query)
        
        # Generate the final response using the selected optimal prompt
        final_response = self.llm.generate(optimal_formatted_prompt)

        print(f"\nChatbot's Final Response (using optimal prompt): {final_response}")
        return final_response

# --- Main execution ---
if __name__ == "__main__":
    # 1. Initialize the Mock LLM
    mock_llm = MockLLM()

    # 2. Define multiple prompt templates for different types of queries
    # These templates have placeholders for the actual customer query
    ecommerce_prompt_templates = {
        "detailed_solution_template": "You are a helpful e-commerce support agent. Provide a detailed solution to the customer's problem related to '{query}'. Focus on actionable steps and clear explanations.",
        "quick_answer_template": "As an e-commerce support bot, give a concise and quick answer to '{query}'. Prioritize brevity and directness.",
        "general_inquiry_template": "You are a friendly e-commerce assistant. Respond to the customer's general inquiry about '{query}' in a polite and informative manner.",
        "problem_solving_template": "An e-commerce customer has a problem: '{query}'. Analyze the issue and suggest a resolution path.",
    }

    # 3. Initialize the Chatbot system
    chatbot = CustomerSupportChatbot(mock_llm, ecommerce_prompt_templates)

    # 4. Simulate customer queries
    queries = [
        "My order #12345 hasn't shipped yet. What should I do?",
        "How do I return a product?",
        "Do you offer international shipping?",
        "I received a damaged item, what's the process for replacement?",
        "Can I track my guest order?",
    ]

    for query in queries:
        print("\n" + "="*80)
        print(f"Customer Query: {query}")
        chatbot_response = chatbot.get_chatbot_response(query)
        print("="*80)
