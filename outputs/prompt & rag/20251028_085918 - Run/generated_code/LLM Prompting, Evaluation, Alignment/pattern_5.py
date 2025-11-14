from typing import Dict, Any, List

class PromptManager:
    def __init__(self, templates: Dict[str, str], few_shot_examples: Dict[str, List[Dict[str, str]]]):
        self.templates = templates
        self.few_shot_examples = few_shot_examples
        self.default_persona = "You are a polite, helpful, and professional customer support representative."

    def _generate_role_based_prompt(self, query: str, persona: str = None) -> str:
        actual_persona = persona if persona else self.default_persona
        return f"{actual_persona} Please respond to the following customer query:\n\nCustomer Query: {query}"

    def _generate_template_driven_prompt(self, query: str, template_key: str, context: Dict[str, str]) -> str:
        if template_key not in self.templates:
            # In a real application, you might raise an error or log a warning
            return self._generate_zero_shot_prompt(query) # Fallback to zero-shot
        template = self.templates[template_key]
        # Simple substitution for context variables in the template
        try:
            formatted_template = template.format(**context)
        except KeyError as e:
            print(f"Warning: Missing context key {e} for template {template_key}. Falling back to zero-shot.")
            return self._generate_zero_shot_prompt(query)
        
        return f"{self.default_persona} Using the following information, respond to the customer:\n\nInformation: {formatted_template}\nCustomer Query: {query}"

    def _generate_few_shot_prompt(self, query: str, example_key: str) -> str:
        if example_key not in self.few_shot_examples:
            return self._generate_zero_shot_prompt(query) # Fallback
        examples = "\n\n".join([
            f"Customer: {ex['input']}\nAgent: {ex['output']}" for ex in self.few_shot_examples[example_key]
        ])
        return f"{self.default_persona} Here are some examples of how to respond to similar queries:\n\n{examples}\n\nNow, please respond to the following customer query:\n\nCustomer Query: {query}"

    def _generate_zero_shot_prompt(self, query: str) -> str:
        return self._generate_role_based_prompt(query) # Zero-shot is essentially just role-based without specific examples/templates

    def select_and_generate_prompt(self, query: str, query_type: str, context: Dict[str, str] = None) -> str:
        """
        Dynamically selects and generates a prompt based on query type and context.
        query_type could be 'password_reset', 'shipping_info', 'new_issue', 'complex_issue'
        """
        if query_type in self.templates and context:
            print(f"DEBUG: Using template-driven prompt for type: {query_type}")
            return self._generate_template_driven_prompt(query, query_type, context)
        elif query_type in self.few_shot_examples:
            print(f"DEBUG: Using few-shot prompt for type: {query_type}")
            return self._generate_few_shot_prompt(query, query_type)
        else: # Default to zero-shot or role-based if no specific match
            print(f"DEBUG: Using zero-shot/role-based prompt for type: {query_type}")
            return self._generate_zero_shot_prompt(query)

# Example Usage (commented out for direct tool output)
# if __name__ == "__main__":
#     templates_data = {
#         "password_reset": "To reset your password, please go to {website_url} and click on the 'Forgot Password' link.",
#         "shipping_info": "Standard shipping takes {min_days}-{max_days} days. Track at {tracking_url}."
#     }
#     few_shot_data = {
#         "tech_support": [
#             {"input": "My printer is offline.", "output": "Please ensure your printer is powered on and connected to Wi-Fi. Try restarting both."},
#             {"input": "My software won't open.", "output": "Could you verify your operating system and software version? A reinstall might be needed."}
#         ]
#     }
#     pm = PromptManager(templates_data, few_shot_data)

#     # Test template-driven
#     prompt1 = pm.select_and_generate_prompt("I need to reset my password.", "password_reset", {"website_url": "https://example.com/login"})
#     print(f"\nTemplate-driven prompt:\n{prompt1}")

#     # Test few-shot
#     prompt2 = pm.select_and_generate_prompt("My software isn't launching.", "tech_support")
#     print(f"\nFew-shot prompt:\n{prompt2}")

#     # Test zero-shot
#     prompt3 = pm.select_and_generate_prompt("What are your business hours?", "general_question")
#     print(f"\nZero-shot prompt:\n{prompt3}")

#     # Test template with missing context
#     prompt4 = pm.select_and_generate_prompt("Shipping details please.", "shipping_info", {"min_days": "3"}) # Missing max_days, tracking_url
#     print(f"\nTemplate with missing context fallback:\n{prompt4}")