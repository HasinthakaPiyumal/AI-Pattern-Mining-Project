import random

class MockLLM:
    def __init__(self, responses=None):
        self.responses = responses if responses is not None else {
            "product details": "This product features a 1080p display and a 5000mAh battery.",
            "return policy": "Our return policy allows returns within 30 days of purchase with the original receipt.",
            "technical issue": "Please try restarting your device and checking the cable connections.",
            "shipping status": "Your order is currently in transit and expected to arrive by Friday."
        }

    def generate_response(self, prompt):
        for keyword, response in self.responses.items():
            if keyword in prompt.lower():
                return f"[Mock LLM Response based on '{keyword}']: {response}"
        return "[Mock LLM Response]: I am sorry, I couldn't find a direct answer for that. Could you please rephrase?"

class ExemplarOrderingPrompt:
    def __init__(self, base_instruction):
        self.base_instruction = base_instruction
        self.exemplars = []

    def add_exemplar(self, query, response):
        self.exemplars.append({"query": query, "response": response})

    def _format_exemplar(self, exemplar):
        return f"User: {exemplar['query']}\nBot: {exemplar['response']}"

    def _get_ordered_exemplars(self, ordering_strategy):
        if ordering_strategy == 'original':
            return self.exemplars
        elif ordering_strategy == 'random':
            shuffled_exemplars = self.exemplars[:]
            random.shuffle(shuffled_exemplars)
            return shuffled_exemplars
        else:
            return self.exemplars # Default to original if strategy not recognized

    def generate_prompt(self, user_query, ordering_strategy='original'):
        ordered_exemplars = self._get_ordered_exemplars(ordering_strategy)
        
        prompt_parts = [self.base_instruction]
        for exemplar in ordered_exemplars:
            prompt_parts.append(self._format_exemplar(exemplar))
        
        prompt_parts.append(f"User: {user_query}\nBot:")
        
        return "\n\n".join(prompt_parts)

# Chatbot Interaction Simulation
if __name__ == "__main__":
    # 1. Initialize MockLLM
    mock_llm = MockLLM()

    # 2. Initialize ExemplarOrderingPrompt manager
    base_instruction = "You are a helpful customer support chatbot. Provide concise and accurate answers based on the examples provided."
    prompt_manager = ExemplarOrderingPrompt(base_instruction)

    # 3. Populate with example customer support interactions
    prompt_manager.add_exemplar(
        "What are the specifications of the new XYZ smartphone?",
        "The XYZ smartphone features a 6.5-inch OLED display, a triple-lens camera system, and 8GB RAM."
    )
    prompt_manager.add_exemplar(
        "How can I return a faulty product?",
        "To return a faulty product, please visit our returns portal online and follow the instructions to generate a shipping label."
    )
    prompt_manager.add_exemplar(
        "My internet connection is not working.",
        "First, restart your router and modem. If the issue persists, check all cable connections and then contact technical support."
    )
    prompt_manager.add_exemplar(
        "What is your warranty policy for electronics?",
        "All electronics come with a 1-year manufacturer's warranty. Extended warranties are available for purchase."
    )

    # 4. Define a sample user query
    sample_user_query = "I need to know about the warranty for my recently purchased laptop."

    print("--- Demonstrating Exemplar Ordering Effect ---")
    print("\n--- Original Ordering Strategy ---")
    original_prompt = prompt_manager.generate_prompt(sample_user_query, ordering_strategy='original')
    print(f"Generated Prompt (Original Order):\n{original_prompt}")
    original_response = mock_llm.generate_response(original_prompt)
    print(f"Mock LLM Response: {original_response}")

    print("\n--- Random Ordering Strategy ---")
    random_prompt = prompt_manager.generate_prompt(sample_user_query, ordering_strategy='random')
    print(f"Generated Prompt (Random Order):\n{random_prompt}")
    random_response = mock_llm.generate_response(random_prompt)
    print(f"Mock LLM Response: {random_response}")

    print("\n--- Explanation of Potential Impact ---")
    print("Different exemplar orderings can lead the LLM to focus on different aspects of the examples, potentially \nresulting in varied responses or a shift in the perceived 'most relevant' example. For instance, \nif an example about 'returns' is placed immediately before a query about 'warranty', the LLM might \nerroneously associate the two, even if they are distinct policies. Careful ordering, or even random \nordering to assess robustness, is crucial for optimizing few-shot prompt performance and mitigating \nprompt sensitivity, which can significantly impact accuracy.")