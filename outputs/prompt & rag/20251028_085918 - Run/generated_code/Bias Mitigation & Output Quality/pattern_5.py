import random
import json

# --- Simulated LLM Interaction --- 
# In a real application, this would be an API call to a large language model.
# For demonstration, we'll return a deterministic or slightly varied mock response.
def _simulate_llm_response(prompt: str) -> str:
    """Simulates an LLM API call and returns a mock response."""
    print(f"\n--- SIMULATING LLM CALL ---\nPrompt: {prompt[:200]}...\n") # Print first 200 chars of prompt
    if "return policy" in prompt.lower():
        return "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging. For electronics, a 15-day return window applies. Please visit our 'Returns & Refunds' page for more details." 
    elif "shipping costs" in prompt.lower():
        return "Standard shipping within the US is $5.99. Expedited shipping is available for $12.99. International shipping costs vary by destination and item weight. Free standard shipping is offered on orders over $75." 
    elif "product availability" in prompt.lower():
        return "Product availability can vary. Please check the specific product page for real-time stock information. If an item is out of stock, you can sign up for email notifications to be alerted when it's back." 
    elif "cultural context: Japan" in prompt:
        return "\n[Culturally Adapted Response for Japan]: Konnichiwa! Thank you for contacting us. We value your patience and understanding. Regarding your inquiry, our services are designed to be efficient and respectful of your time. We aim for high quality and meticulous attention to detail. Is there anything specific I can assist you with today?\n" 
    elif "cultural context: Germany" in prompt:
        return "\n[Culturally Adapted Response for Germany]: Guten Tag! We appreciate your directness and efficiency. Please let us know your precise concern, and we will provide a clear and factual response. Our focus is on reliable information and excellent technical support.\n" 
    elif "biased synthetic data" in prompt.lower() and "vary attributes" in prompt.lower():
        return "Generated diverse synthetic customer review: 'This product is amazing, but the red color is a bit too bright for my taste.' Another: 'The delivery was fast, but the packaging could be more eco-friendly.'\n"
    elif "for and against" in prompt.lower():
        if "argument for" in prompt.lower():
            return "\n[Evidence FOR]: Customers often report faster issue resolution and higher satisfaction when interacting with AI assistants for common queries, freeing human agents for complex problems. AI operates 24/7, providing constant support.\n"
        else: # argument against
            return "\n[Evidence AGAINST]: Some customers prefer human interaction for empathetic support, especially for sensitive or complex issues. AI may struggle with nuanced language, emotional cues, and unexpected deviations from pre-programmed scripts.\n"
    elif "sensitive topic" in prompt.lower() and "gender" in prompt.lower():
        return "When discussing product recommendations, our system ensures a wide range of options without defaulting to gendered assumptions, promoting inclusivity and respect for all customers.\n"
    elif "demonstration ensembling query" in prompt.lower():
        return f"Ensembled LLM output for: '{prompt.split('Query:')[-1].strip()[:50]}...'. This is an aggregate of several perspectives, leading to a more robust answer.\n"
    else:
        return "I am an AI assistant. How can I help you today? Please provide more details about your request."


# --- Prompt Engineering Module --- 
class PromptEngineer:
    def __init__(self, general_demonstrations: list, balanced_demonstrations: list):
        self.general_demonstrations = general_demonstrations
        self.balanced_demonstrations = balanced_demonstrations

    def _select_exemplars(self, demonstrations: list, subset_size: int) -> list:
        """Randomly selects a subset of exemplars for few-shot prompting."""
        if len(demonstrations) <= subset_size:
            return demonstrations
        return random.sample(demonstrations, subset_size)

    def _create_base_prompt(self, query: str, exemplars: list, cultural_context: str = "") -> str:
        """Constructs a base prompt with optional cultural context and exemplars."""
        prompt_parts = []
        if cultural_context:
            prompt_parts.append(f"Cultural context: {cultural_context}. Please tailor your response accordingly.")

        if exemplars:
            prompt_parts.append("Here are some examples of previous interactions:")
            for i, ex in enumerate(exemplars):
                prompt_parts.append(f"Example {i+1}:\nUser: {ex['user_query']}\nAssistant: {ex['assistant_response']}")
            prompt_parts.append("\nBased on the above examples, please answer the following query:")

        prompt_parts.append(f"Query: {query}")
        return "\n".join(prompt_parts)

    def demonstration_ensembling(self, query: str, num_ensembles: int = 3, subset_size: int = 2) -> list:
        """Applies Demonstration Ensembling (DENSE) to get robust LLM outputs."""
        ensemble_outputs = []
        for i in range(num_ensembles):
            exemplars = self._select_exemplars(self.general_demonstrations, subset_size)
            prompt = self._create_base_prompt(f"Demonstration Ensembling query: {query}", exemplars)
            response = _simulate_llm_response(prompt)
            ensemble_outputs.append(response)
        return ensemble_outputs # In a real scenario, these would be aggregated (e.g., majority vote)

    def selecting_balanced_demonstrations(self, query: str) -> str:
        """Constructs a prompt with balanced demonstrations to reduce bias."""
        # Assumes self.balanced_demonstrations are already balanced
        prompt = self._create_base_prompt(f"Consider this a sensitive topic. Use balanced information. Query: {query}", self.balanced_demonstrations)
        return _simulate_llm_response(prompt)

    def cultural_awareness_prompt(self, query: str, cultural_context: str) -> str:
        """Injects cultural sensitivity into the prompt for relevant outputs."""
        # Using general demonstrations here, but could also use culturally specific ones if available
        prompt = self._create_base_prompt(query, self.general_demonstrations, cultural_context=cultural_context)
        return _simulate_llm_response(prompt)

    def attr_prompt_for_diverse_generation(self, base_query: str, attributes_to_vary: dict) -> str:
        """Uses AttrPrompt to encourage LLM to vary specific attributes in generative tasks."""
        attr_prompt_parts = [f"Generate data based on: {base_query}"]
        attr_prompt_parts.append("Ensure the output varies the following attributes:")
        for attr, values in attributes_to_vary.items():
            attr_prompt_parts.append(f" - {attr}: {', '.join(values)}")
        prompt = "\n".join(attr_prompt_parts)
        return _simulate_llm_response(prompt)


# --- Bias and Evidence Aggregation Module --- 
class BiasAndEvidenceAggregator:
    def bias_aware_mitigation(self, llm_output: str) -> str:
        """Conceptual component for monitoring and mitigating biases in LLM outputs.
        In a real system, this would involve bias detection models or rules.
        """
        # For demonstration, we'll just acknowledge the mitigation attempt.
        if "gender" in llm_output.lower() and "product recommendations" in llm_output.lower():
            return f"[Bias Mitigated]: Reviewed for gender bias in recommendations. Output: {llm_output}"
        return f"[Bias Check Performed]: {llm_output}"

    def debate_style_evidence_aggregation(self, claim: str, query: str) -> str:
        """Trains models to present evidence both for and against claims.
        Simulates getting 'for' and 'against' arguments from the LLM.
        """
        # Simulate getting arguments for and against from the LLM based on the query
        prompt_for = f"Provide an argument FOR the claim: '{claim}' based on query: '{query}'"
        response_for = _simulate_llm_response(prompt_for)

        prompt_against = f"Provide an argument AGAINST the claim: '{claim}' based on query: '{query}'"
        response_against = _simulate_llm_response(prompt_against)

        return (
            f"Claim: {claim}\n\n"
            f"Arguments FOR:\n{response_for}\n\n"
            f"Arguments AGAINST:\n{response_against}\n\n"
            f"Conclusion: Consider both sides for a balanced perspective."
        )


# --- Main Customer Support Assistant --- 
class CustomerSupportAssistant:
    def __init__(self):
        # Define some general demonstrations for the LLM
        self.general_demonstrations = [
            {"user_query": "What is your return policy?", "assistant_response": "You can return items within 30 days."}, 
            {"user_query": "How much is shipping?", "assistant_response": "Shipping cost depends on your location and chosen speed."}, 
            {"user_query": "Can I track my order?", "assistant_response": "Yes, you'll receive a tracking number via email once your order ships."} 
        ]
        # Define some balanced demonstrations for sensitive topics
        self.balanced_demonstrations = [
            {"user_query": "Recommend a gift for a child.", "assistant_response": "Consider age-appropriate toys, books, or craft kits without gender stereotypes."}, 
            {"user_query": "What are the best products for wellness?", "assistant_response": "Wellness products vary by individual needs; consider items for physical activity, mental well-being, or healthy eating."} 
        ]
        self.prompt_engineer = PromptEngineer(self.general_demonstrations, self.balanced_demonstrations)
        self.bias_aggregator = BiasAndEvidenceAggregator()

    def get_answer(self, query: str, customer_locale: str = "en_US") -> str:
        print(f"\n--- Processing query: '{query}' for locale: {customer_locale} ---")
        final_response_parts = []

        # 1. Cultural Awareness (always applied based on locale)
        cultural_context = {
            "en_US": "United States, casual and direct",
            "ja_JP": "Japan, formal and polite, emphasizes respect and quality",
            "de_DE": "Germany, direct, factual, and efficient",
            "fr_FR": "France, polite, emphasizes elegance and quality",
        }.get(customer_locale, "general, polite")
        
        cultural_response = self.prompt_engineer.cultural_awareness_prompt(query, cultural_context)
        final_response_parts.append(f"Cultural Adaptation: {cultural_response}")

        # 2. Demonstration Ensembling (DENSE) for general accuracy
        ensemble_results = self.prompt_engineer.demonstration_ensembling(query)
        final_response_parts.append(f"DENSE Ensembled Output (aggregated for robustness): {ensemble_results[0]}") # Taking first for simplicity

        # 3. Selecting Balanced Demonstrations (if query is potentially sensitive)
        if any(keyword in query.lower() for keyword in ["gender", "age", "religion", "ethnicity", "sensitive topic"]):
            balanced_response = self.prompt_engineer.selecting_balanced_demonstrations(query)
            final_response_parts.append(f"Balanced Demonstrations (for sensitive query): {balanced_response}")
        
        # 4. AttrPrompt for Generative AI (if query implies generation with varied attributes)
        if "generate a review" in query.lower() or "create synthetic data" in query.lower():
            attributes = {"sentiment": ["positive", "negative", "neutral"], "product_aspect": ["delivery", "quality", "price"]}
            attr_gen_response = self.prompt_engineer.attr_prompt_for_diverse_generation(query, attributes)
            final_response_parts.append(f"AttrPrompt Generated Output: {attr_gen_response}")

        # 5. Bias-Aware Design & Mitigation (post-processing)
        # We'll apply this to a combined response for simplicity.
        combined_llm_output = " ".join(final_response_parts)
        mitigated_output = self.bias_aggregator.bias_aware_mitigation(combined_llm_output)
        final_response_parts.append(f"Bias Mitigation Applied: {mitigated_output}")

        # 6. Debate-Style Evidence Aggregation (if query involves a claim or complex decision)
        if "should" in query.lower() or "pros and cons" in query.lower() or "claim:" in query.lower():
            # For demonstration, we'll extract a simple claim or use a generic one
            claim = query.replace("Should", "The idea that").replace("?", ".") if "should" in query.lower() else "AI assistants improve customer satisfaction"
            debate_response = self.bias_aggregator.debate_style_evidence_aggregation(claim, query)
            final_response_parts.append(f"Debate-Style Evidence: {debate_response}")

        return "\n\n".join(final_response_parts)


# --- Example Usage --- 
if __name__ == "__main__":
    assistant = CustomerSupportAssistant()

    print("\n--- Scenario 1: General Query with Cultural Awareness (Japan) ---")
    response1 = assistant.get_answer("I have a question about my recent order.", customer_locale="ja_JP")
    print(f"Final Assistant Response:\n{response1}")

    print("\n--- Scenario 2: Query about Return Policy with DENSE and Bias Check ---")
    response2 = assistant.get_answer("What's your return policy for electronics?")
    print(f"Final Assistant Response:\n{response2}")

    print("\n--- Scenario 3: Sensitive Product Recommendation Query (Balanced Demonstrations) ---")
    response3 = assistant.get_answer("Can you recommend a toy for a 5-year-old?", customer_locale="en_US")
    print(f"Final Assistant Response:\n{response3}")

    print("\n--- Scenario 4: Generative AI (AttrPrompt) ---")
    response4 = assistant.get_answer("Generate a short customer review for a new smartphone.")
    print(f"Final Assistant Response:\n{response4}")

    print("\n--- Scenario 5: Debate-Style Evidence Aggregation ---")
    response5 = assistant.get_answer("Should AI completely replace human customer service agents?")
    print(f"Final Assistant Response:\n{response5}")

    print("\n--- Scenario 6: Query with Cultural Awareness (Germany) ---")
    response6 = assistant.get_answer("I need technical specifications for product XYZ.", customer_locale="de_DE")
    print(f"Final Assistant Response:\n{response6}")