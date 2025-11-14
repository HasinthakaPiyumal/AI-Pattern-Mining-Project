import random
from collections import Counter

class IntelligentChatbot:
    """An Intelligent Customer Support Chatbot incorporating various AI design patterns.
    This is a conceptual implementation, simulating LLM interactions and data generation.
    """

    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base if knowledge_base is not None else {
            "product_info": {
                "feature_A": "This is a core feature with benefits X, Y, Z.",
                "feature_B": "Feature B enhances user experience through P, Q, R."
            },
            "policy_info": {
                "return_policy": "Items can be returned within 30 days with a valid receipt.",
                "warranty": "All products come with a 1-year warranty against manufacturing defects."
            },
            "complex_issue": {
                "climate_change": {
                    "for": ["Scientific consensus indicates human activity contributes to global warming.", "Rising sea levels and extreme weather events are observable."],
                    "against": ["Some argue natural cycles are the primary driver of climate change.", "Economic impacts of aggressive climate policies are a concern."]
                }
            }
        }
        self.demonstrations_pool = {
            "general_inquiry": [
                {"query": "What is feature A?", "response": "Feature A is our flagship feature offering X, Y, Z benefits.", "bias_attrs": {"tone": "neutral"}},
                {"query": "How does feature B work?", "response": "Feature B improves user experience by enabling P, Q, R.", "bias_attrs": {"tone": "neutral"}},
                {"query": "Tell me about your return policy.", "response": "Our return policy allows returns within 30 days with proof of purchase.", "bias_attrs": {"tone": "neutral"}},
                {"query": "What is the warranty period?", "response": "All products have a 1-year warranty for manufacturing defects.", "bias_attrs": {"tone": "neutral"}},
                {"query": "Is feature C available?", "response": "Feature C is currently under development and will be released next quarter.", "bias_attrs": {"tone": "neutral"}},
            ],
            "biased_demo": [
                {"query": "Why is your competitor better?", "response": "Our products are superior due to advanced technology and customer support.", "bias_attrs": {"tone": "defensive", "competitor_bias": True}},
                {"query": "Can you recommend a product for men?", "response": "Our \"Alpha"" series is very popular among male customers for its rugged design.", "bias_attrs": {"gender": "male", "stereotypical": True}}
            ],
            "cultural_demo_us": [
                {"query": "Happy Holidays!", "response": "Happy Holidays to you too!", "culture": "US"},
                {"query": "How are you doing today?", "response": "I'm doing great, thanks for asking!", "culture": "US"}
            ],
            "cultural_demo_jp": [
                {"query": "Kon'nichiwa!", "response": "Kon'nichiwa!", "culture": "JP"},
                {"query": "Dōmo arigatō gozaimasu.", "response": "Dōitashimashite.", "culture": "JP"}
            ]
        }

    def _simulate_llm_response(self, prompt: str) -> str:
        """Simulates an LLM response based on the prompt content.
        In a real scenario, this would call an actual LLM API.
        """
        print(f"[Simulating LLM for prompt]: {prompt[:100]}...")
        # Simple keyword-based response simulation
        if "feature A" in prompt.lower():
            return self.knowledge_base["product_info"]["feature_A"]
        elif "feature B" in prompt.lower():
            return self.knowledge_base["product_info"]["feature_B"]
        elif "return policy" in prompt.lower():
            return self.knowledge_base["policy_info"]["return_policy"]
        elif "warranty" in prompt.lower():
            return self.knowledge_base["policy_info"]["warranty"]
        elif "climate change" in prompt.lower() and "for and against" in prompt.lower():
            for_args = " ".join(self.knowledge_base["complex_issue"]["climate_change"]["for"])
            against_args = " ".join(self.knowledge_base["complex_issue"]["climate_change"]["against"])
            return f"Arguments for climate change: {for_args}. Arguments against: {against_args}."
        elif "cultural_us_greeting" in prompt:
            return "Hello! How can I assist you today?"
        elif "cultural_jp_greeting" in prompt:
            return "Konnichiwa! How may I help you?"
        elif "Alpha" in prompt and "men" in prompt:
            return "The Alpha series is indeed a popular choice known for its durability and design."
        return "I am an intelligent chatbot. How can I help you today?"

    def _apply_dense_ensembling(self, query: str, num_ensembles: int = 3, examples_per_ensemble: int = 2) -> str:
        """Demonstration Ensembling (DENSE): Aggregates outputs from multiple prompts.
        Each prompt uses a distinct subset of exemplars.
        """
        responses = []
        available_demos = self.demonstrations_pool["general_inquiry"]

        if not available_demos:
            return self._simulate_llm_response(query)

        for _ in range(num_ensembles):
            # Select distinct random subsets of exemplars
            selected_demos = random.sample(available_demos, min(examples_per_ensemble, len(available_demos)))
            
            prompt_parts = []
            for demo in selected_demos:
                prompt_parts.append(f"Q: {demo['query']}\nA: {demo['response']}")
            
            ensemble_prompt = "\n\n".join(prompt_parts) + f"\n\nQ: {query}\nA:"
            responses.append(self._simulate_llm_response(ensemble_prompt))
        
        # Simple aggregation: majority vote or return the most common response
        if responses:
            return Counter(responses).most_common(1)[0][0]
        return self._simulate_llm_response(query)

    def _select_balanced_demonstrations(self, query: str, sensitive_attributes: dict = None) -> list:
        """Selecting Balanced Demonstrations: Aims to reduce bias by using balanced exemplars.
        For demonstration, we'll try to balance based on a 'tone' attribute.
        """
        if sensitive_attributes is None:
            sensitive_attributes = {"tone": "neutral"}

        balanced_demos = []
        # In a real scenario, this would involve a more sophisticated selection
        # based on a dataset labeled with sensitive attributes.
        
        # For simplicity, we'll filter general inquiries that are deemed 'neutral' 
        # and avoid explicitly 'biased_demo' examples for the main query path.
        for demo in self.demonstrations_pool["general_inquiry"]:
            if demo["bias_attrs"].get("tone") == sensitive_attributes["tone"]:
                balanced_demos.append(demo)
        
        # If no balanced demos, fallback to general ones (or raise error)
        if not balanced_demos:
            print("[Warning]: No perfectly balanced demonstrations found. Using general demos.")
            balanced_demos = self.demonstrations_pool["general_inquiry"]

        # Return a subset for a few-shot prompt
        return random.sample(balanced_demos, min(2, len(balanced_demos)))

    def _inject_cultural_awareness(self, prompt: str, culture: str = "US") -> str:
        """Cultural Awareness: Injects cultural sensitivity into prompts.
        """
        if culture.upper() == "JP":
            return f"[CULTURAL_JP_GREETING] Konnichiwa. Please provide a response suitable for a Japanese audience: {prompt}"
        elif culture.upper() == "US":
            return f"[CULTURAL_US_GREETING] Hello. Please provide a response suitable for an American audience: {prompt}"
        return prompt # Default to original prompt

    def _generate_attr_prompt_data(self, base_prompt: str, attribute_variations: dict) -> list:
        """AttrPrompt: Generates synthetic data by varying specific attributes in prompts.
        Conceptual utility for generating training data.
        """
        generated_prompts = []
        for attr, values in attribute_variations.items():
            for value in values:
                # Simple replacement for demonstration
                varied_prompt = base_prompt.replace(f"{{{attr}}}", value)
                generated_prompts.append(varied_prompt)
        return generated_prompts

    def _check_for_bias(self, response: str) -> str:
        """Bias-Aware Response Checker: Post-processing to detect and potentially rephrase biased outputs.
        This is a conceptual check.
        """
        if "Alpha series is very popular among male customers" in response:
            print("[BIAS DETECTED]: Response shows gender stereotyping. Rephrasing...")
            return "The Alpha series is a highly-rated product known for its durability and design."
        if "Our products are superior" in response:
            print("[BIAS DETECTED]: Response shows competitive bias. Rephrasing...")
            return "We continuously strive to offer high-quality products and excellent customer support."
        return response

    def _aggregate_debate_evidence(self, topic: str) -> str:
        """Debate-Style Evidence Aggregation: Presents evidence for and against claims.
        """
        if topic in self.knowledge_base["complex_issue"]:
            data = self.knowledge_base["complex_issue"][topic]
            for_args = "\n- ".join(data["for"])
            against_args = "\n- ".join(data["against"])
            return (f"Regarding '{topic}', here are arguments for and against the claim:\n\n"
                    f"Arguments FOR:\n- {for_args}\n\n"
                    f"Arguments AGAINST:\n- {against_args}")
        return f"I don't have detailed debate-style evidence for '{topic}' at the moment."

    def ask_chatbot(self, query: str, use_dense: bool = False, use_balanced_demos: bool = False, 
                    culture: str = None, generate_attr_data: dict = None, 
                    debate_topic: str = None) -> str:
        """Main method to interact with the chatbot, applying selected design patterns.
        """
        final_response = ""
        initial_prompt = query

        # 1. Cultural Awareness
        if culture:
            initial_prompt = self._inject_cultural_awareness(initial_prompt, culture)

        # 2. AttrPrompt Data Generation (conceptual - usually for training, not direct interaction)
        if generate_attr_data:
            print("\n--- Generating AttrPrompt Data (Conceptual) ---")
            generated_prompts = self._generate_attr_prompt_data(generate_attr_data['base_prompt'], generate_attr_data['variations'])
            print(f"Generated prompts: {generated_prompts}")
            # This would typically be saved or used for model fine-tuning
            return "AttrPrompt data generation simulated. Check console for output."

        # 3. Debate-Style Evidence Aggregation
        if debate_topic:
            print(f"\n--- Aggregating Debate Evidence for '{debate_topic}' ---")
            final_response = self._aggregate_debate_evidence(debate_topic)
            return self._check_for_bias(final_response)

        # Prepare for LLM call with few-shot examples if applicable
        few_shot_examples_str = ""
        if use_balanced_demos:
            print("\n--- Applying Selecting Balanced Demonstrations ---")
            selected_demos = self._select_balanced_demonstrations(query, sensitive_attributes={"tone": "neutral"})
            if selected_demos:
                few_shot_examples_str = "\n\n" + "\n\n".join([f"Q: {d['query']}\nA: {d['response']}" for d in selected_demos])

        if use_dense:
            print("\n--- Applying Demonstration Ensembling (DENSE) ---")
            # DENSE handles its own internal LLM calls and aggregation
            response = self._apply_dense_ensembling(initial_prompt + few_shot_examples_str, num_ensembles=3)
            final_response = response
        else:
            # Standard LLM call with or without balanced demos
            full_prompt = initial_prompt + few_shot_examples_str
            final_response = self._simulate_llm_response(full_prompt)
        
        # 4. Bias-Aware Response Checker (post-processing)
        final_response = self._check_for_bias(final_response)

        return final_response

# --- Example Usage ---
if __name__ == "__main__":
    chatbot = IntelligentChatbot()

    print("\n--- Scenario 1: Standard Query ---")
    response = chatbot.ask_chatbot("What is feature A?")
    print(f"Chatbot: {response}")

    print("\n--- Scenario 2: Demonstration Ensembling (DENSE) ---")
    response = chatbot.ask_chatbot("Tell me about the return process.", use_dense=True)
    print(f"Chatbot: {response}")

    print("\n--- Scenario 3: Selecting Balanced Demonstrations ---")
    response = chatbot.ask_chatbot("How does feature B help customers?", use_balanced_demos=True)
    print(f"Chatbot: {response}")

    print("\n--- Scenario 4: Cultural Awareness (Japanese Context) ---")
    response = chatbot.ask_chatbot("What are your operating hours?", culture="JP")
    print(f"Chatbot: {response}")

    print("\n--- Scenario 5: Bias-Aware Design & Mitigation (Gender Stereotyping) ---")
    # This query would trigger a biased demo, but the checker rephrases
    response = chatbot.ask_chatbot("Can you recommend a product for men?") # Will be caught by _check_for_bias
    print(f"Chatbot: {response}")

    print("\n--- Scenario 6: Debate-Style Evidence Aggregation ---")
    response = chatbot.ask_chatbot("", debate_topic="climate_change")
    print(f"Chatbot: {response}")

    print("\n--- Scenario 7: AttrPrompt Data Generation (Conceptual) ---")
    attr_data_config = {
        'base_prompt': 'Describe the product with {design} and {color}.',
        'variations': {
            'design': ['sleek', 'rugged', 'minimalist'],
            'color': ['blue', 'red', 'black']
        }
    }
    response = chatbot.ask_chatbot("", generate_attr_data=attr_data_config)
    print(f"Chatbot: {response}")

    print("\n--- Scenario 8: Bias-Aware Design & Mitigation (Competitive Bias) ---")
    response = chatbot.ask_chatbot("Why is your competitor better?") # Will be caught by _check_for_bias
    print(f"Chatbot: {response}")