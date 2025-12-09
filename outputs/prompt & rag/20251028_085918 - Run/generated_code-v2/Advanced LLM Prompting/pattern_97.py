""" A simple AI-powered E-commerce Product Description Generator.
    Uses template-based prompting to create product descriptions.
"""

class ProductDescriptionGenerator:
    def __init__(self):
        self.templates = {
            "standard": (
                "Discover the amazing {{product_name}}!\n\n"\
                "{{short_description}} With features like {{features}}, "\
                "it's designed to {{benefits}}.\n\n"\
                "Perfect for {{target_audience}}, this product also boasts "\
                "{{unique_selling_points}}. Elevate your experience today!\n\n"\
                "Keywords: {{keywords}}."
            ),
            "concise": (
                "Introducing {{product_name}}: {{short_description}} "\
                "Key features include {{features}}. Designed to {{benefits}}."\
                " Keywords: {{keywords}}."
            ),
            "feature_rich": (
                "Unlock the full potential with {{product_name}}. "\
                "This product stands out with its comprehensive features:\n"\
                "- {{feature_list_bulleted}}\n\n"\
                "Experience unparalleled {{benefits}} and make a smart choice. "\
                "Ideal for {{target_audience}}. Keywords: {{keywords}}."
            )
        }

    def _get_template(self, template_name: str) -> str:
        """Retrieves a product description template by its name."""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found. Available templates: {list(self.templates.keys())}")
        return template

    def _fill_template(self, template: str, product_data: dict) -> str:
        """Fills placeholders in a template with actual product data.
           Handles special formatting for lists (e.g., bullet points).
        """
        filled_template = template
        for key, value in product_data.items():
            placeholder = f"{{{{{key}}}}}"
            if isinstance(value, list):
                if key.endswith('_list_bulleted'):
                    # Format list as bullet points
                    formatted_value = "\n".join([f"  - {item}" for item in value])
                else:
                    # Default comma separated for other lists
                    formatted_value = ", ".join(value)
            else:
                formatted_value = str(value)
            filled_template = filled_template.replace(placeholder, formatted_value)
        return filled_template

    def _call_llm(self, prompt: str) -> str:
        """Simulates an LLM call to refine or generate text based on the prompt.
           In a real application, this would interact with an actual LLM API
           (e.g., OpenAI, Hugging Face, etc.).
        """
        print("\n--- Simulating LLM Call ---")
        print("Prompt provided to LLM:")
        print("------------------------")
        print(prompt)
        print("------------------------")
        print("\nLLM would now generate/refine the description. For this example,")
        print("we'll return the filled template directly, assuming LLM refines it.")
        return prompt # In a real scenario, LLM's output would be different

    def generate_description(self, product_data: dict, template_name: str = "standard") -> str:
        """Generates a product description using a specified template and product data."""
        try:
            template = self._get_template(template_name)
            filled_prompt = self._fill_template(template, product_data)
            final_description = self._call_llm(filled_prompt)
            return final_description
        except ValueError as e:
            return f"Error: {e}"


# --- Example Usage ---
if __name__ == "__main__":
    generator = ProductDescriptionGenerator()

    product_info_1 = {
        "product_name": "SmartFit Pro Fitness Tracker",
        "short_description": "An advanced fitness tracker with heart rate monitoring and GPS.",
        "features": "heart rate tracking, GPS, sleep analysis, calorie counter, waterproof design",
        "feature_list_bulleted": [
            "24/7 Heart Rate Monitoring",
            "Built-in GPS for route tracking",
            "Advanced Sleep Stage Analysis",
            "IP68 Waterproof up to 50 meters",
            "Long-lasting battery life (7 days)"
        ],
        "benefits": "help you achieve your fitness goals, monitor your health, and stay motivated",
        "target_audience": "fitness enthusiasts, health-conscious individuals, active professionals",
        "unique_selling_points": "customizable watch faces, seamless smartphone integration, personalized coaching",
        "keywords": "fitness tracker, smartwatch, health monitor, GPS watch, waterproof, SmartFit Pro"
    }

    product_info_2 = {
        "product_name": "AeroGlide Robotic Vacuum",
        "short_description": "Intelligent robotic vacuum with advanced navigation and powerful suction.",
        "features": "Lidar navigation, automatic dirt disposal, multi-floor mapping, quiet operation",
        "benefits": "effortlessly keep your home clean, save time, and enjoy a spotless environment",
        "target_audience": "busy homeowners, pet owners, tech enthusiasts",
        "keywords": "robotic vacuum, smart home, automatic cleaner, Lidar, pet hair vacuum, AeroGlide"
    }

    print("\n--- Generating Standard Description for SmartFit Pro ---")
    desc_1_standard = generator.generate_description(product_info_1, "standard")
    print("\nGenerated Description (Standard):")
    print(desc_1_standard)

    print("\n--- Generating Feature-Rich Description for SmartFit Pro ---")
    desc_1_feature_rich = generator.generate_description(product_info_1, "feature_rich")
    print("\nGenerated Description (Feature-Rich):")
    print(desc_1_feature_rich)

    print("\n--- Generating Concise Description for AeroGlide Robotic Vacuum ---")
    desc_2_concise = generator.generate_description(product_info_2, "concise")
    print("\nGenerated Description (Concise):")
    print(desc_2_concise)

    print("\n--- Attempting to use a non-existent template ---")
    error_desc = generator.generate_description(product_info_1, "non_existent_template")
    print(error_desc)
