class LLMClient:
    """A mock LLM client to simulate interactions with a Large Language Model."""

    def __init__(self, model_name="mock-llm-model"):
        self.model_name = model_name
        print(f"Initialized LLMClient with model: {self.model_name}")

    def generate_response(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Simulates generating a response from an LLM.
        In a real application, this would involve calling an actual LLM API (e.g., OpenAI, Hugging Face, Google Gemini).
        """
        print(f"\n--- LLM Prompt ---\n{prompt}\n--- End Prompt ---")
        # Mock responses for demonstration purposes
        if "entities" in prompt.lower() and "extract" in prompt.lower():
            if "iPhone 15" in prompt:
                return "[\"iPhone 15\", \"Smartphone\", \"Apple\", \"iOS\", \"Camera\"]"
            elif "Gaming Laptop" in prompt:
                return "[\"Gaming Laptop\", \"Laptop\", \"Processor\", \"GPU\", \"RAM\", \"Storage\", \"Screen\"]"
            else:
                return "[\"product\", \"feature\", \"brand\"]"
        elif "relations" in prompt.lower() and "extract" in prompt.lower():
            if "iPhone 15" in prompt:
                return "[[\"iPhone 15\", \"is_a_type_of\", \"Smartphone\"], [\"Smartphone\", \"manufactured_by\", \"Apple\"]]"
            elif "Gaming Laptop" in prompt:
                return "[[\"Gaming Laptop\", \"has_component\", \"GPU\"], [\"Gaming Laptop\", \"has_component\", \"Processor\"]]"
            else:
                return "[[\"entity1\", \"has_relation\", \"entity2\"]]"
        elif "coreference" in prompt.lower():
            if "Apple's latest phone" in prompt:
                return "{\"Apple's latest phone\": \"iPhone 15\"]}"
            else:
                return "{}"
        elif "missing facts" in prompt.lower() or "complete" in prompt.lower():
            if "Gaming Laptop" in prompt and "GPU" in prompt:
                return "[[\"Gaming Laptop\", \"has_GPU\", \"NVIDIA GeForce RTX 4080\"]]"
            else:
                return "[[\"product\", \"has_attribute\", \"value\"]]"
        elif "commonsense" in prompt.lower():
            if "Winter Jacket" in prompt:
                return "[[\"Winter Jacket\", \"is_suitable_for\", \"Cold Weather\"], [\"Winter Jacket\", \"is_used_for\", \"Warmth\"]]"
            else:
                return "[[\"entity\", \"has_commonsense_property\", \"value\"]]"
        else:
            return f"Mock LLM response for prompt: {prompt[:50]}..."
