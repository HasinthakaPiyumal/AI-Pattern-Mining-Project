class PromptManager:
    def __init__(self):
        self.prompt_templates = {
            "password_reset": [
                "Generate a step-by-step guide for resetting a user's password.",
                "Provide instructions on password recovery for an account.",
                "Explain the process to change a forgotten password.",
                "How do I reset my password? Give me clear steps."
            ],
            "billing_inquiry": [
                "Explain how to understand a recent billing statement.",
                "Provide information on disputing a charge on my bill.",
                "What are the common reasons for unexpected charges?"
            ],
            "product_info": [
                "Describe the key features of your new product line.",
                "What are the technical specifications for product X?",
                "Compare product A and product B, highlighting their differences."
            ]
        }

    def get_templates_for_query_type(self, query_type: str) -> list[str]:
        """Retrieves prompt templates for a given query type."""
        return self.prompt_templates.get(query_type, [])

    def add_template(self, query_type: str, template: str):
        """Adds a new prompt template for a specific query type."""
        if query_type not in self.prompt_templates:
            self.prompt_templates[query_type] = []
        self.prompt_templates[query_type].append(template)
        print(f"Added template for {query_type}: {template}")

    def list_all_query_types(self) -> list[str]:
        """Lists all available query types."""
        return list(self.prompt_templates.keys())

# Example Usage:
# if __name__ == "__main__":
#     pm = PromptManager()
#     password_prompts = pm.get_templates_for_query_type("password_reset")
#     print("Password Reset Prompts:", password_prompts)
#     pm.add_template("shipping_status", "How can I check the status of my order?")
#     print("All Query Types:", pm.list_all_query_types())