from collections import defaultdict

class ExternalMemory:
    def __init__(self):
        self.memory = defaultdict(list)

    def store_interaction(self, customer_id, interaction_data):
        self.memory[customer_id].append(interaction_data)

    def retrieve_interactions(self, customer_id, num_interactions=5):
        return self.memory[customer_id][-num_interactions:]

    def store_preference(self, customer_id, preference_data):
        # For simplicity, overwrites existing preferences for a given key
        self.memory[customer_id + "_preferences"].append(preference_data)

    def retrieve_preferences(self, customer_id):
        return self.memory.get(customer_id + "_preferences", [])

class LLMChatbot:
    def __init__(self, external_memory):
        self.external_memory = external_memory
        # In a real application, this would be an actual LLM API call
        self.llm_model = self._mock_llm_response

    def _mock_llm_response(self, prompt):
        if "past interactions" in prompt and "bought a shoe" in prompt:
            return "Welcome back! I see you recently bought a shoe. Are you looking for accessories or perhaps another pair?"
        elif "preferences" in prompt and "dresses" in prompt:
            return "Given your preference for dresses, I can recommend some new arrivals. Would you like to see them?"
        elif "product inquiry" in prompt:
            return "I can help you with product information. What specific product are you interested in?"
        else:
            return "Hello! How can I assist you today?"

    def chat(self, customer_id, message):
        # Retrieve past interactions and preferences
        past_interactions = self.external_memory.retrieve_interactions(customer_id)
        customer_preferences = self.external_memory.retrieve_preferences(customer_id)

        # Augment the LLM prompt with retrieved memory
        augmented_prompt = f"Customer ID: {customer_id}\n"
        if past_interactions:
            augmented_prompt += f"Past interactions: {past_interactions}\n"
        if customer_preferences:
            augmented_prompt += f"Customer preferences: {customer_preferences}\n"
        augmented_prompt += f"Customer message: {message}\n"
        augmented_prompt += "Based on the above, please provide a personalized response."

        # Get LLM response
        llm_response = self.llm_model(augmented_prompt)

        # Store the current interaction for future use
        self.external_memory.store_interaction(customer_id, {"message": message, "response": llm_response})

        return llm_response

# Example Usage:
if __name__ == "__main__":
    memory = ExternalMemory()
    chatbot = LLMChatbot(memory)

    customer_1_id = "user_123"

    # Simulate initial interactions and preferences
    memory.store_interaction(customer_1_id, {"query": "Hi, I'm looking for a new pair of running shoes.", "response": "Sure, we have a great selection. Any particular brand or size?"})
    memory.store_interaction(customer_1_id, {"query": "I bought a shoe last month.", "response": "Great! How are they working out for you?"})
    memory.store_preference(customer_1_id, {"category": "dresses", "style": "bohemian"})

    print(f"Customer 1 Chat 1: {chatbot.chat(customer_1_id, 'Hello, I need some help.')}")
    print(f"Customer 1 Chat 2: {chatbot.chat(customer_1_id, 'What are the latest offers on dresses?')}")

    customer_2_id = "user_456"
    print(f"Customer 2 Chat 1: {chatbot.chat(customer_2_id, 'I need information about my recent order.')}")
