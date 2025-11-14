"""Python code for a Smart Customer Support Assistant demonstrating various LLM prompting patterns.

This code conceptually implements the following AI design patterns:
- Demonstration Ensembling (DENSE)
- Selecting Balanced Demonstrations
- Cultural Awareness
- AttrPrompt (for synthetic data generation)
- Bias-Aware Design & Mitigation (integrated into prompt design)
- Debate-Style Evidence Aggregation

It focuses on prompt construction and response aggregation logic, with placeholders for actual LLM API calls.
"""

import random

class SmartCustomerSupportAssistant:
    def __init__(self, llm_model_name="mock_llm"):
        self.llm_model_name = llm_model_name

    def _mock_llm_call(self, prompt, temperature=0.7):
        """Simulates an LLM API call. In a real application, this would use a library like langchain or openai.
        It returns a placeholder string indicating the LLM's hypothetical response.
        """
        print(f"\n--- Mock LLM Call with Temperature: {temperature} ---")
        print(f"Prompt: {prompt}")
        # Simulate a simple response based on keywords
        if "cultural context" in prompt.lower() or "culturally relevant" in prompt.lower():
            return f"[LLM Response with Cultural Awareness for '{prompt.split('Customer Query:')[-1].strip()}']"
        elif "pro argument" in prompt.lower() and "con argument" in prompt.lower():
            return f"[LLM Debate Response for '{prompt.split('Claim:')[-1].strip()}']"
        elif "vary attribute" in prompt.lower():
            return f"[LLM Synthetic Data Sample for '{prompt.split('Generate data with:')[-1].strip()}']"
        elif "exemplars:" in prompt.lower():
            return f"[LLM Few-Shot Response for '{prompt.split('Customer Query:')[-1].strip()}']"
        else:
            return f"[LLM Generic Response for '{prompt.split('Customer Query:')[-1].strip()}']"

    def generate_response_dense(self, query: str, exemplar_subsets: list[list[str]]) -> str:
        """Demonstration Ensembling (DENSE) pattern.
        Aggregates outputs from multiple prompts with distinct exemplar subsets to reduce variance.
        """
        responses = []
        for i, subset in enumerate(exemplar_subsets):
            exemplars_str = "\n".join([f"Example {j+1}: {ex}" for j, ex in enumerate(subset)])
            prompt = f"""
            You are a helpful customer support assistant. Analyze the following examples and then answer the customer query.
            
            Exemplars:
            {exemplars_str}

            Customer Query: {query}
            Assistant:
            """
            response = self._mock_llm_call(prompt, temperature=0.7) # Using a moderate temperature
            responses.append(response)
        
        # Simple aggregation strategy (e.g., majority vote, or concatenation here for demonstration)
        aggregated_response = f"Aggregated DENSE Response:\n" + "\n---\n".join(responses)
        return aggregated_response

    def generate_response_balanced_demonstrations(self, query: str, balanced_exemplars: list[str]) -> str:
        """Selecting Balanced Demonstrations pattern.
        Uses a set of exemplars balanced across potential biases to mitigate bias in few-shot LLM outputs.
        """
        exemplars_str = "\n".join([f"Example {i+1}: {ex}" for i, ex in enumerate(balanced_exemplars)])
        prompt = f"""
        You are an impartial customer support assistant. Carefully consider the following balanced examples to provide a fair and unbiased answer to the customer query.
        
        Exemplars:
        {exemplars_str}

        Customer Query: {query}
        Assistant:
        """
        response = self._mock_llm_call(prompt, temperature=0.4) # Lower temperature for less variance
        return f"Balanced Demonstration Response: {response}"

    def generate_response_cultural_awareness(self, query: str, cultural_context: str, language: str = "English") -> str:
        """Cultural Awareness pattern.
        Injects cultural sensitivity into prompts to ensure culturally relevant and appropriate outputs.
        """
        prompt = f"""
        You are a culturally aware customer support assistant. The customer is from a region with the following cultural context: {cultural_context}. Ensure your response is sensitive to this background and provided in {language}.
        
        Customer Query: {query}
        Assistant:
        """
        response = self._mock_llm_call(prompt, temperature=0.6)
        return f"Culturally Aware Response: {response}"

    def generate_synthetic_data_attrprompt(self, base_prompt: str, attributes_to_vary: dict, num_samples: int = 1) -> list[str]:
        """AttrPrompt pattern.
        Prompts LLMs to vary specific attributes for generating diverse, bias-mitigated synthetic data.
        """
        generated_samples = []
        for _ in range(num_samples):
            varying_attrs_str = ", ".join([f"{k}: {random.choice(v)}" for k, v in attributes_to_vary.items()])
            prompt = f"""
            Generate a diverse synthetic customer inquiry based on the following context. Vary the attributes as specified to ensure broad coverage and mitigate biases in synthetic data generation.
            
            Context: {base_prompt}
            Generate data with: {varying_attrs_str}
            Synthetic Inquiry:
            """
            response = self._mock_llm_call(prompt, temperature=0.9) # Higher temperature for diversity
            generated_samples.append(response)
        return generated_samples

    def generate_debate_style_response(self, query: str) -> str:
        """Debate-Style Evidence Aggregation pattern.
        Trains models to present evidence both for and against claims, leading to more balanced and robust answers.
        """
        prompt_for_pros = f"""
        Present a strong argument in favor of the following claim, citing potential benefits or supporting evidence.
        Claim: {query}
        Pro argument:
        """
        pro_argument = self._mock_llm_call(prompt_for_pros, temperature=0.5)

        prompt_for_cons = f"""
        Present a strong argument against the following claim, citing potential drawbacks or opposing evidence.
        Claim: {query}
        Con argument:
        """
        con_argument = self._mock_llm_call(prompt_for_cons, temperature=0.5)

        debate_response = f"""
        Debate-Style Analysis for: "{query}"
        
        Argument For:
        {pro_argument}
        
        Argument Against:
        {con_argument}
        
        Consider both perspectives for a comprehensive understanding.
        """
        return debate_response

# --- Example Usage --- #
if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()

    print("\n--- Demonstrating DENSE Pattern ---")
    query_dense = "My internet is slow, what can I do?"
    exemplar_subsets_dense = [
        ["Problem: Slow internet. Solution: Restart router.", "Problem: Connection drops. Solution: Check cable connection."],
        ["Issue: Intermittent speed. Solution: Contact ISP for line test.", "Issue: Wi-Fi dead zones. Solution: Use a repeater."]
    ]
    dense_output = assistant.generate_response_dense(query_dense, exemplar_subsets_dense)
    print(dense_output)

    print("\n--- Demonstrating Balanced Demonstrations Pattern ---")
    query_balanced = "How do I reset my password?"
    balanced_exemplars = [
        "User (tech-savvy): 'I need to reset my credentials. Guide me through the API.'",
        "User (novice): 'I forgot my password. How do I get a new one?'",
        "User (mobile): 'Can't log in on my phone, help with password reset.'",
        "User (desktop): 'Having trouble on my computer, password reset needed.'"
    ]
    balanced_output = assistant.generate_response_balanced_demonstrations(query_balanced, balanced_exemplars)
    print(balanced_output)

    print("\n--- Demonstrating Cultural Awareness Pattern ---")
    query_culture = "What is your refund policy for electronics?"
    cultural_context_jp = "In Japan, customer service highly values politeness, indirect communication, and preserving harmony (wa). Direct 'no' might be avoided; apologies and detailed explanations are common."
    cultural_context_us = "In the United States, customer service is often direct, solution-oriented, and values efficiency. Clear terms and conditions are expected."
    culture_jp_output = assistant.generate_response_cultural_awareness(query_culture, cultural_context_jp, language="Japanese (simulated)") # Language parameter to show intent
    print(culture_jp_output)
    culture_us_output = assistant.generate_response_cultural_awareness(query_culture, cultural_context_us)
    print(culture_us_output)

    print("\n--- Demonstrating AttrPrompt for Synthetic Data Generation ---")
    base_prompt_attr = "Generate a customer service ticket about a product issue."
    attributes_to_vary_attr = {
        "product_type": ["smartphone", "laptop", "smartwatch", "headphones"],
        "issue_severity": ["minor", "moderate", "critical"],
        "customer_demographic": ["elderly", "teenager", "business professional", "student"]
    }
    synthetic_tickets = assistant.generate_synthetic_data_attrprompt(base_prompt_attr, attributes_to_vary_attr, num_samples=3)
    print("Generated Synthetic Tickets:")
    for i, ticket in enumerate(synthetic_tickets):
        print(f"  Sample {i+1}: {ticket}")

    print("\n--- Demonstrating Debate-Style Evidence Aggregation Pattern ---")
    query_debate = "Is AI generally beneficial for society?"
    debate_output = assistant.generate_debate_style_response(query_debate)
    print(debate_output)
