import json

class ProductDescriptionGenerator:
    def _call_llm(self, prompt: str) -> str:
        # Simulate LLM response in JSON format
        # In a real application, this would be an API call to an LLM service
        if "Laptop" in prompt:
            return json.dumps({
                "product_name": "High-Performance Ultrabook",
                "description": "A sleek and powerful ultrabook designed for professionals and students alike. Featuring a stunning display, long-lasting battery, and robust performance for all your computing needs.",
                "features": [
                    "13.3-inch Retina Display",
                    "Intel Core i7 Processor",
                    "16GB RAM, 512GB SSD",
                    "All-day battery life",
                    "Lightweight aluminum chassis"
                ],
                "price_range": "$1200-$1800"
            })
        elif "Coffee Maker" in prompt:
            return json.dumps({
                "product_name": "Smart Brew Coffee Maker",
                "description": "Start your day right with the Smart Brew Coffee Maker. Program your brew time, adjust strength, and enjoy perfectly brewed coffee every morning with smart connectivity.",
                "features": [
                    "Programmable timer",
                    "Adjustable brew strength",
                    "Wi-Fi connectivity",
                    "12-cup capacity",
                    "Keep warm function"
                ],
                "price_range": "$75-$120"
            })
        else:
            return json.dumps({
                "product_name": "Generic Product",
                "description": "A description for a generic product, illustrating the structured output.",
                "features": [
                    "Generic Feature 1",
                    "Generic Feature 2"
                ],
                "price_range": "$10-$50"
            })

    def generate_description(self, product_name: str, category: str, key_features: list, target_audience: str) -> dict:
        prompt = (
            f"Generate a product description for an e-commerce website.\n"
            f"Product Name: {product_name}\n"
            f"Category: {category}\n"
            f"Key Features: {', '.join(key_features)}\n"
            f"Target Audience: {target_audience}\n\n"
            f"Please provide the output in a JSON format with the following keys: "
            f"`product_name`, `description`, `features` (as a list of strings), and `price_range`."
        )

        llm_response_json_str = self._call_llm(prompt)
        
        try:
            parsed_response = json.loads(llm_response_json_str)
            # Basic validation to ensure the expected keys are present
            required_keys = ["product_name", "description", "features", "price_range"]
            if not all(key in parsed_response for key in required_keys):
                raise ValueError("LLM response missing required keys.")
            return parsed_response
        except json.JSONDecodeError:
            raise ValueError("LLM returned invalid JSON.")
        except Exception as e:
            raise ValueError(f"Error processing LLM response: {e}")

if __name__ == "__main__":
    generator = ProductDescriptionGenerator()

    # Example 1: Laptop
    product_details_laptop = {
        "product_name": "High-Performance Ultrabook",
        "category": "Electronics",
        "key_features": ["Lightweight", "Fast Processor", "Long Battery Life", "High Resolution Display"],
        "target_audience": "Professionals and Students"
    }
    try:
        description_laptop = generator.generate_description(**product_details_laptop)
        print("\n--- Generated Description (Laptop) ---")
        print(json.dumps(description_laptop, indent=2))
    except ValueError as e:
        print(f"Error generating description for laptop: {e}")

    # Example 2: Coffee Maker
    product_details_coffee_maker = {
        "product_name": "Smart Brew Coffee Maker",
        "category": "Home Appliances",
        "key_features": ["Programmable", "Smart Connectivity", "Large Capacity"],
        "target_audience": "Coffee Enthusiasts"
    }
    try:
        description_coffee_maker = generator.generate_description(**product_details_coffee_maker)
        print("\n--- Generated Description (Coffee Maker) ---")
        print(json.dumps(description_coffee_maker, indent=2))
    except ValueError as e:
        print(f"Error generating description for coffee maker: {e}")

    # Example 3: Generic Product (demonstrates default simulated response)
    product_details_generic = {
        "product_name": "Mystery Gadget",
        "category": "Gadgets",
        "key_features": ["Innovative", "Portable"],
        "target_audience": "Tech Savvy Individuals"
    }
    try:
        description_generic = generator.generate_description(**product_details_generic)
        print("\n--- Generated Description (Generic) ---")
        print(json.dumps(description_generic, indent=2))
    except ValueError as e:
        print(f"Error generating description for generic product: {e}")
