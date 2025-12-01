class LLMExplanationService:
    def __init__(self, llm_model_placeholder="LLM_API_ENDPOINT"):
        """
        Initializes the LLM Explanation Service.
        In a real application, llm_model_placeholder would be an actual LLM client or API endpoint.
        """
        self.llm_model = llm_model_placeholder
        print(f"LLMExplanationService initialized with model: {self.llm_model}")

    def _call_llm(self, prompt: str) -> str:
        """
        Simulates an LLM API call.
        In a real application, this would interact with an actual LLM (e.g., OpenAI GPT, Google Gemini, Hugging Face model).
        """
        print(f"\n--- Simulating LLM Call with Prompt ---\n{prompt}\n--------------------------------------")
        # Placeholder for actual LLM interaction
        if "why is this recommended" in prompt.lower():
            if "running shoes" in prompt.lower() and "active lifestyle" in prompt.lower():
                return "Based on your active lifestyle and recent interest in fitness, these running shoes are recommended because they offer excellent cushioning and support for long-distance running, which aligns with your stated preference for comfortable and durable athletic gear."
            elif "coffee machine" in prompt.lower() and "morning routine" in prompt.lower():
                return "Given your browsing history indicating an interest in morning routines and kitchen gadgets, this coffee machine is recommended due to its quick brewing time and programmable features, perfect for a busy start to your day. It also has great reviews for ease of use."
            elif "tell me more about" in prompt.lower() or "elaborate" in prompt.lower():
                return "The previous explanation focused on user fit. To elaborate, this product also features a sleek design, energy-saving mode, and comes with a 2-year warranty, adding to its overall value."
            else:
                return "This recommendation is based on a complex interplay of factors including your past interactions and similar user preferences. Specifically, it aligns with your interest in [specific category] and [specific attribute]."
        elif "rephrase" in prompt.lower() or "simpler terms" in prompt.lower():
            return "Simply put, we think you'll like this because it matches what you've looked at before and what people similar to you have enjoyed."
        elif "too technical" in prompt.lower():
            return "Let me simplify that: This item is a good match because we've noticed you like similar things, and other users with tastes like yours also bought it."
        else:
            return "I need more context to generate a specific explanation. Could you please provide details about the recommendation and user context?"

    def generate_explanation(self, recommendation: dict, user_profile: dict, product_details: dict) -> str:
        """
        Generates a natural language explanation for a product recommendation.

        Args:
            recommendation (dict): Details about the recommended product and the core reason (e.g., item_id, recommendation_score, primary_reason).
            user_profile (dict): User's preferences, history, and inferred traits (e.g., user_id, interests, recent_searches).
            product_details (dict): Full details of the product (e.g., product_id, name, description, features).

        Returns:
            str: A natural language explanation.
        """
        product_name = product_details.get("name", "an unknown product")
        user_interests = ", ".join(user_profile.get("interests", []))
        primary_reason = recommendation.get("primary_reason", "your general browsing history")
        product_features = ", ".join(product_details.get("features", []))

        prompt = (
            f"Generate a personalized, natural language explanation for recommending '{product_name}'.\n"
            f"User Profile: Interests include {user_interests}. Recently viewed: {user_profile.get('recent_views', 'nothing specific')}.\n"
            f"Product Details: Features include {product_features}. Description: {product_details.get('description', '')}.\n"
            f"Core Recommendation Reason: {primary_reason}.\n"
            "Explain why this product is a good fit for the user, focusing on personalization and clarity."
        )
        return self._call_llm(prompt)

    def refine_explanation(self, current_explanation: str, user_feedback: str) -> str:
        """
        Refines an existing explanation based on user feedback.

        Args:
            current_explanation (str): The explanation previously generated.
            user_feedback (str): User's feedback (e.g., "Too technical", "Tell me more", "Why this feature?").

        Returns:
            str: A refined explanation.
        """
        prompt = (
            f"Refine the following product recommendation explanation based on the user's feedback.\n"
            f"Current Explanation: '{current_explanation}'\n"
            f"User Feedback: '{user_feedback}'\n"
            "Please provide a revised explanation that addresses the feedback, making it more helpful or clearer."
        )
        return self._call_llm(prompt)
