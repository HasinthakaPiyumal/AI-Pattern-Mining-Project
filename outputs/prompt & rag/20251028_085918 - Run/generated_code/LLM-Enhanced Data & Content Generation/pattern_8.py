"""
synthetic_data_generator.py: Generates synthetic product data using an LLM.
"""

import json
from llm_service import LLMService

class SyntheticDataGenerator:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def generate_synthetic_product(self, product_category: str, existing_data_context: str = "") -> dict:
        """
        Generates synthetic product data (name, features, price) for a given category.
        Optionally takes existing data as context to ensure consistency or variation.
        """
        prompt = (
            f"Generate realistic and varied synthetic product data for the \'{product_category}\' category. "
            f"The output should be a JSON object with keys: \'product_name\', \'description\', \'category\', \'price\', and \'features\'. "
            f"Features should be a list of strings. Make the description engaging. "
            f"If any existing data context is provided: {existing_data_context} ensure the new data is distinct yet plausible within the domain."
            "Ensure the JSON is perfectly formed and can be parsed directly."
        )

        raw_llm_output = self.llm_service.generate_text(prompt, max_tokens=300)

        try:
            # Attempt to parse the LLM output as JSON
            synthetic_data = json.loads(raw_llm_output)
            # Add a unique ID for simulation purposes
            synthetic_data["product_id"] = f"synth_{hash(raw_llm_output) % 1000000}"
            return synthetic_data
        except json.JSONDecodeError:
            print(f"Warning: LLM did not return valid JSON. Raw output: {raw_llm_output}")
            # Fallback to a predefined structure or re-prompting in a real scenario
            return {
                "product_id": f"synth_fallback_{hash(raw_llm_output) % 1000000}",
                "product_name": f"Synthetic {product_category} Item",
                "description": "A generic description for a synthetic product.",
                "category": product_category,
                "price": 0.0,
                "features": ["generated", "placeholder"]
            }
