import os
import random
from collections import Counter

# --- 1. Configuration (simulated) ---
# In a real application, you would load this from a .env file using python-dotenv
# For this example, we'll use a placeholder.
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-YOUR_OPENAI_API_KEY")

# --- 2. LLM Utilities ---

class LLMClient:
    """Handles secure interaction with the chosen LLM (e.g., OpenAI API)."""
    def __init__(self, api_key: str):
        # In a real application, you would initialize the OpenAI client here:
        # from openai import OpenAI
        # self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        print("LLMClient initialized (using mock for actual API calls).")

    def text_completion(self, prompt: str) -> str:
        """Simulates an LLM text completion call."""
        # This is a mock implementation. In a real scenario, it would call the LLM API.
        # For example, using OpenAI:
        # response = self.client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.choices[0].message.content
        print(f"\n--- LLM CALL (MOCK) ---")
        print(f"Prompt: {prompt[:200]}...") # Print first 200 chars of prompt
        mock_responses = [
            "Thank you for your inquiry. Let me look into that for you.",
            "I understand your concern. Here's what I found.",
            "Based on the information, this is the most likely solution.",
            "Your question is important. I need a moment to gather all the details."
        ]
        return random.choice(mock_responses) + " [MOCK LLM Response]"


class PromptGenerator:
    """Generates various types of prompts for the LLM."""
    def __init__(self):
        print("PromptGenerator initialized.")

    def select_balanced_demonstrations(self, examples: list, query_attributes: dict, num_demos: int = 3) -> list:
        """ 
        Selects a diverse and balanced subset of exemplars to mitigate bias.
        This is a simplified mock. In a real scenario, this would involve:
        - Analyzing `query_attributes` (e.g., product category, user demographic, sentiment).
        - Using embedding similarity, clustering (e.g., k-means on example embeddings),
          or stratified sampling to pick `num_demos` that represent diverse facets
          or balance certain attributes from the `examples` pool.
        """
        if not examples:
            return []
        
        # Simple random sampling for demonstration purposes
        if len(examples) <= num_demos:
            return examples
        return random.sample(examples, num_demos)

    def inject_cultural_awareness(self, prompt_template: str, cultural_context: dict) -> str:
        """ 
        Modifies prompt templates to include cultural nuances relevant to the user.
        `cultural_context` might include language, locale, common idioms, or sensitivities.
        """
        if not cultural_context:
            return prompt_template
        
        # Example: Injecting a greeting based on language
        language = cultural_context.get("language", "en").lower()
        greeting = "Hello" # Default
        if language == "es":
            greeting = "Hola"
        elif language == "fr":
            greeting = "Bonjour"
        
        # More sophisticated cultural injection would involve checking for sensitive topics,
        # preferred communication styles, or specific cultural references.
        return f"{greeting}! {prompt_template} Please ensure your response is appropriate for a {language.upper()} speaker." 

    def generate_few_shot_prompt(self, query: str, demonstrations: list, cultural_context: dict = None) -> str:
        """Generates a few-shot prompt with selected demonstrations and cultural awareness."""
        demo_string = ""
        for i, (input_text, output_text) in enumerate(demonstrations):
            demo_string += f"Example {i+1} Input: {input_text}\nExample {i+1} Output: {output_text}\n"

        base_prompt = f"""
        You are an AI customer support agent. Please provide a helpful and accurate response.

        {demo_string}
        User Query: {query}
        Agent Response:
        """
        if cultural_context:
            return self.inject_cultural_awareness(base_prompt, cultural_context)
        return base_prompt


class DemonstrationEnsembler:
    """Aggregates outputs from multiple prompts with distinct exemplar subsets."""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_generator = PromptGenerator() # Use an internal prompt generator for consistency
        print("DemonstrationEnsembler initialized.")

    def ensemble_few_shot_prompts(self, query: str, all_demonstrations: list, 
                                  num_ensembles: int = 3, demos_per_ensemble: int = 3,
                                  cultural_context: dict = None) -> str:
        """
        Creates `num_ensembles` variations of few-shot prompts by sampling distinct
        subsets of `all_demonstrations`. Calls the LLM for each, and aggregates their outputs.
        """
        if not all_demonstrations or demos_per_ensemble == 0:
            # Fallback to single LLM call if no demonstrations or no demos needed per ensemble
            base_prompt = self.prompt_generator.generate_few_shot_prompt(query, [], cultural_context)
            return self.llm_client.text_completion(base_prompt)

        responses = []
        for i in range(num_ensembles):
            # Ensure distinct subsets if possible, or sample with replacement if demonstrations are few
            selected_demos = random.sample(all_demonstrations, min(demos_per_ensemble, len(all_demonstrations)))
            
            prompt = self.prompt_generator.generate_few_shot_prompt(query, selected_demos, cultural_context)
            response = self.llm_client.text_completion(prompt)
            responses.append(response)
            print(f"Ensemble {i+1} response: {response[:100]}...")

        # Simple aggregation: for generative tasks, could be concatenation, re-ranking, or another LLM call
        # For classification, a majority vote would be used.
        # Here, we'll just join them, indicating an ensembled output.
        return "Ensembled Response:\n" + "\n---\n".join(responses)


class DebateAggregator:
    """Helps overcome 'cherry-picking' of sources by presenting evidence for and against claims."""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        print("DebateAggregator initialized.")

    def aggregate_debate_evidence(self, claim: str, evidence_sources: list) -> str:
        """
        Crafts prompts to encourage the LLM to generate arguments for and against
        a given claim based on provided evidence. Synthesizes a balanced response.
        """
        evidence_string = "\n".join([f"- {e}" for e in evidence_sources])

        prompt = f"""
        You are an impartial analyst. A claim has been made: "{claim}".
        Here is a list of evidence:
        {evidence_string}

        Please provide a balanced perspective by presenting arguments supporting the claim,
        arguments refuting the claim, and then synthesize a balanced conclusion.
        
        Arguments FOR the claim:
        Arguments AGAINST the claim:
        Balanced Conclusion:
        """
        print("\n--- DEBATE AGGREGATOR PROMPT ---")
        print(prompt)
        return self.llm_client.text_completion(prompt)

# --- 3. Synthetic Data Generation ---

class AttrPromptGenerator:
    """Uses AttrPrompt to generate diverse synthetic data by varying attributes."""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        print("AttrPromptGenerator initialized.")

    def generate_diverse_data(self, base_prompt: str, attribute_variations: dict, num_samples_per_variation: int = 1) -> list:
        """
        Generates synthetic data instances by systematically varying attributes.
        `attribute_variations` is a dict like {'sentiment': ['positive', 'negative'], 'product_type': ['electronics', 'clothing']}
        """
        generated_data = []
        # This is a simplified iteration. Real AttrPrompt might use a more complex combinatorial approach
        # or more intelligent sampling of attribute combinations.
        
        # Get all attribute names and their possible values
        attr_names = list(attribute_variations.keys())
        attr_values_lists = list(attribute_variations.values())

        # Generate combinations of attribute values
        import itertools
        for combination in itertools.product(*attr_values_lists):
            current_attributes = dict(zip(attr_names, combination))
            
            attr_instruction = ", ".join([f"{k}: {v}" for k, v in current_attributes.items()])
            
            for _ in range(num_samples_per_variation):
                prompt = f"""
                {base_prompt}
                Ensure the generated data has the following attributes: {attr_instruction}.
                """
                data_instance = self.llm_client.text_completion(prompt)
                generated_data.append({"attributes": current_attributes, "data": data_instance})
                print(f"Generated data with attributes {current_attributes}: {data_instance[:50]}...")
                
        return generated_data

# --- 4. Main Agent Logic ---

class CustomerSupportAgent:
    """Main AI customer support agent utilizing various LLM patterns."""
    def __init__(self):
        self.llm_client = LLMClient(api_key=Config.OPENAI_API_KEY)
        self.prompt_generator = PromptGenerator()
        self.demonstration_ensembler = DemonstrationEnsembler(self.llm_client)
        self.debate_aggregator = DebateAggregator(self.llm_client)
        self.attr_prompt_generator = AttrPromptGenerator(self.llm_client) # For generating synthetic training data
        
        # Mock examples for few-shot prompting
        self.few_shot_examples = [
            ("My order #12345 hasn't shipped yet.", "Please allow 2-3 business days for processing. Your order #12345 is currently being prepared for shipment."),
            ("How do I return a product?", "You can initiate a return through your account page under 'Order History'. Select the item and follow the return instructions."),
            ("Is the item #XYZ available in blue?", "Let me check our inventory. The item #XYZ is available in blue in sizes S, M, L."),
            ("My account is locked.", "For security reasons, please contact our support hotline to unlock your account."),
            ("Can I change my shipping address for order #67890?", "Yes, if the order #67890 has not yet shipped, we can update the address. Please provide the new address."),
        ]
        print("CustomerSupportAgent initialized.")

    def handle_query(self, user_query: str, user_culture: str = "en-US", product_context: dict = None) -> str:
        """
        Handles a user query, applying appropriate AI design patterns.
        `product_context` could contain details about the product the user is asking about.
        """
        print(f"\nHandling query: '{user_query}' for culture '{user_culture}'")
        cultural_context = {"language": user_culture.split('-')[0], "locale": user_culture}

        # Determine if the query is complex/controversial (requiring debate aggregation)
        # This would typically involve NLP classification or keyword detection.
        is_complex_query = any(keyword in user_query.lower() for keyword in ["controversy", "dispute", "policy issue"])

        if is_complex_query:
            print("Query identified as complex/controversial, using Debate Aggregation.")
            # For a real scenario, evidence_sources would be retrieved from a knowledge base
            mock_evidence = [
                "Company policy states returns are accepted within 30 days.",
                "Customer received item 45 days ago.",
                "A special exception was made for a similar case last month."
            ]
            return self.debate_aggregator.aggregate_debate_evidence(user_query, mock_evidence)
        else:
            print("Query identified as standard, using Demonstration Ensembling with Balanced Demonstrations.")
            # Simulate query attributes for balanced demonstration selection
            query_attributes = {"intent": "shipping status", "urgency": "medium"}
            if "return" in user_query.lower():
                query_attributes["intent"] = "return policy"
            
            balanced_demos = self.prompt_generator.select_balanced_demonstrations(
                self.few_shot_examples, query_attributes, num_demos=3
            )
            return self.demonstration_ensembler.ensemble_few_shot_prompts(
                user_query, balanced_demos, cultural_context=cultural_context
            )

    def generate_synthetic_training_data(self, base_scenario_prompt: str, attribute_variations: dict, num_samples_per_variation: int = 1) -> list:
        """
        Generates synthetic data for training/fine-tuning the agent, varying specified attributes.
        """
        print(f"\nGenerating synthetic training data using AttrPrompt for: {attribute_variations}")
        return self.attr_prompt_generator.generate_diverse_data(
            base_scenario_prompt, attribute_variations, num_samples_per_variation
        )

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the agent
    agent = CustomerSupportAgent()

    # Example 1: Standard query with cultural awareness
    response1 = agent.handle_query(
        "Where is my package for order #XYZ789?", 
        user_culture="es-ES"
    )
    print(f"\nFinal Agent Response 1: {response1}")

    print("\n========================================\n")

    # Example 2: Complex query triggering debate aggregation
    response2 = agent.handle_query(
        "I need to return this item, but it's past the 30-day window. What's the policy dispute here?",
        user_culture="en-US"
    )
    print(f"\nFinal Agent Response 2: {response2}")

    print("\n========================================\n")

    # Example 3: Generating synthetic data for agent improvement
    synthetic_data = agent.generate_synthetic_training_data(
        base_scenario_prompt="Generate a customer support interaction for a product issue.",
        attribute_variations={
            "sentiment": ["positive", "negative", "neutral"],
            "product_type": ["electronics", "apparel"],
            "issue_type": ["defect", "delivery", "billing"]
        },
        num_samples_per_variation=1
    )
    print(f"\nGenerated {len(synthetic_data)} synthetic data instances.")
    # for item in synthetic_data:
    #     print(item)
