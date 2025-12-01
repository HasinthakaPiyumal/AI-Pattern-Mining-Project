class ProductDescriptionRefiner:
    def __init__(self, max_iterations=3):
        # In a real application, 'self.llm' would be an actual LLM client (e.g., from OpenAI, Anthropic).
        # For this example, we use a mock function to simulate LLM responses.
        self.llm = self._mock_llm
        self.max_iterations = max_iterations

        self.initial_prompt_template = """
        You are an expert e-commerce copywriter. Generate a compelling product description for the following product.
        Product Name: {product_name}
        Key Features: {key_features}
        Target Audience: {target_audience}
        Tone: {tone}
        Description:
        """

        self.feedback_prompt_template = """
        You are an expert editor for e-commerce product descriptions. Review the following product description and provide constructive feedback to improve its clarity, conciseness, SEO-friendliness (mentioning relevant keywords if applicable), and factual accuracy.
        Product Name: {product_name}
        Current Description:
        {description}

        Provide feedback as a list of actionable points.
        Feedback:
        """

        self.refine_prompt_template = """
        You are an expert e-commerce copywriter. Refine the following product description based on the provided feedback.
        Product Name: {product_name}
        Original Description:
        {original_description}
        Feedback:
        {feedback}

        Revised Description:
        """

    def _mock_llm(self, prompt_template, **kwargs):
        # This function simulates an LLM call without external libraries.
        # In a real scenario, this would involve an API call to a large language model.
        full_prompt = prompt_template.format(**kwargs)

        if "Generate a compelling product description" in full_prompt:
            # Simulate initial generation
            product_name = kwargs.get("product_name", "a product")
            return f"Initial description for {product_name}: This amazing item offers fantastic features for all users. It's truly great!"
        elif "Review the following product description and provide constructive feedback" in full_prompt:
            # Simulate feedback generation
            return (
                "1. Make the description more specific about features. "
                "2. Add some strong SEO keywords like 'durable' and 'innovative'. "
                "3. Improve the call to action."
            )
        elif "Refine the following product description based on the provided feedback" in full_prompt:
            # Simulate refinement
            product_name = kwargs.get("product_name", "a product")
            original_description = kwargs.get("original_description", "")
            feedback = kwargs.get("feedback", "")
            return (
                f"Refined description for {product_name}: Experience the durable and innovative design of our {product_name}. "
                "Now with specific details and a clear call to action! (Based on feedback: {feedback})"
            )
        return "LLM Response Placeholder"

    def _generate_initial_description(self, product_details):
        return self.llm(self.initial_prompt_template, **product_details)

    def _generate_feedback(self, product_details, description):
        return self.llm(self.feedback_prompt_template, product_name=product_details["product_name"], description=description)

    def _refine_description(self, product_details, original_description, feedback):
        return self.llm(self.refine_prompt_template,
                       product_name=product_details["product_name"],
                       original_description=original_description,
                       feedback=feedback)

    def refine(self, product_details):
        print(f"\n--- Starting refinement for product: {product_details['product_name']} ---")
        current_description = self._generate_initial_description(product_details)
        print(f"\nInitial Description:\n{current_description}")

        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")
            feedback = self._generate_feedback(product_details, current_description)
            print(f"Feedback:\n{feedback}")

            new_description = self._refine_description(product_details, current_description, feedback)
            print(f"Refined Description:\n{new_description}")

            if new_description == current_description:
                print("\nDescription did not change significantly. Stopping early.")
                break
            current_description = new_description

        print("\n--- Final Refined Description ---")
        return current_description

# Example Usage (demonstrates how to use the refiner)
if __name__ == "__main__":
    product_details = {
        "product_name": "SuperSonic Wireless Earbuds",
        "key_features": "Noise cancellation, 30-hour battery, Ergonomic design, Bluetooth 5.2",
        "target_audience": "Music lovers, commuters, fitness enthusiasts",
        "tone": "Exciting, high-tech, user-friendly"
    }

    refiner = ProductDescriptionRefiner(max_iterations=3)
    final_description = refiner.refine(product_details)
    print(f"\nFINAL OUTPUT:\n{final_description}")