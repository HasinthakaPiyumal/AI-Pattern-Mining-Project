import json

def generate_product_description_structured(product_name: str, product_features: list):
    prompt = f"""Generate a detailed product description for '{product_name}'.\n\nInclude the following features: {', '.join(product_features)}.\n\nOutput the description in JSON format with the following keys: 'product_name', 'description', 'features' (as a list), and 'price_range'.\n\nExample:\n{{\n  "product_name": "Example Product",\n  "description": "A brief and engaging description of the example product.",\n  "features": ["Feature 1", "Feature 2"],\n  "price_range": "$XX - $YY"\n}}"""

    # Simulate an LLM call. In a real application, this would involve an API call to a GenAI model.
    # The LLM would be instructed to follow the JSON format specified in the prompt.
    # For this demonstration, we'll return a hardcoded structured response.
    if product_name == "Smartwatch X":
        generated_content = {
            "product_name": "Smartwatch X",
            "description": "Revolutionize your daily routine with the Smartwatch X, a blend of style and cutting-edge technology. Track your fitness, receive notifications, and manage your day with ease.",
            "features": [
                "Heart Rate Monitoring",
                "GPS Tracking",
                "Water Resistant",
                "Long Battery Life",
                "Customizable Watch Faces"
            ],
            "price_range": "$199 - $249"
        }
    else:
        generated_content = {
            "product_name": product_name,
            "description": f"A generic description for {product_name} highlighting its key attributes based on {', '.join(product_features)}.",
            "features": product_features,
            "price_range": "Price varies"
        }

    return json.dumps(generated_content, indent=2)

if __name__ == "__main__":
    product_name = "Smartwatch X"
    product_features = ["Health Tracking", "Notifications", "Fitness Modes"]
    structured_output = generate_product_description_structured(product_name, product_features)
    print(f"Generated structured product description for '{product_name}':\n{structured_output}")

    product_name_2 = "Ergonomic Office Chair"
    product_features_2 = ["Adjustable Lumbar Support", "Breathable Mesh", "Swivel Base"]
    structured_output_2 = generate_product_description_structured(product_name_2, product_features_2)
    print(f"\nGenerated structured product description for '{product_name_2}':\n{structured_output_2}")