import itertools

class LLM_Interface:
    def generate_response(self, prompt: str) -> str:
        # Mock LLM response for demonstration purposes
        return f"Generated description based on: {prompt}"

class AttributeIdentifier:
    def identify_attributes(self, product_name: str, base_description: str) -> dict:
        identified_attrs = {}
        product_name_lower = product_name.lower()

        if "shirt" in product_name_lower or "top" in product_name_lower:
            identified_attrs["style"] = ["casual", "formal", "athletic"]
            identified_attrs["material"] = ["cotton", "polyester", "linen"]
            identified_attrs["fit"] = ["slim-fit", "regular-fit", "oversized"]
        elif "dress" in product_name_lower:
            identified_attrs["style"] = ["evening", "daytime", "cocktail"]
            identified_attrs["material"] = ["silk", "chiffon", "rayon"]
            identified_attrs["length"] = ["mini", "midi", "maxi"]
        elif "shoe" in product_name_lower or "sneaker" in product_name_lower:
            identified_attrs["style"] = ["sporty", "classic", "modern"]
            identified_attrs["material"] = ["leather", "mesh", "synthetic"]
            identified_attrs["occasion"] = ["running", "daily wear", "formal"]
        else:
            # Default attributes if no specific product type is matched
            identified_attrs["color"] = ["red", "blue", "green"]
            identified_attrs["feature"] = ["durable", "comfortable", "stylish"]

        return identified_attrs

class DescriptionGenerator:
    def __init__(self, llm_interface: LLM_Interface):
        self.llm_interface = llm_interface

    def generate_descriptions(self, product_name: str, base_description: str, attributes: dict) -> list:
        generated_descriptions = []
        
        attribute_keys = list(attributes.keys())
        attribute_values = [attributes[key] for key in attribute_keys]

        for combo in itertools.product(*attribute_values):
            current_attributes_str = ", ".join(f"{key}: {value}" for key, value in zip(attribute_keys, combo))
            
            prompt = (
                f"Generate a product description for '{product_name}'. "
                f"Base description: '{base_description}'. "
                f"Ensure the description highlights the following attributes: {current_attributes_str}."
            )
            
            description = self.llm_interface.generate_response(prompt)
            generated_descriptions.append(description)

        return generated_descriptions

if __name__ == "__main__":
    # 1. Instantiate modules
    llm_mock = LLM_Interface()
    attr_identifier = AttributeIdentifier()
    desc_generator = DescriptionGenerator(llm_mock)

    # 2. Sample Product Information
    product_name_1 = "Classic Cotton T-Shirt"
    base_description_1 = "A comfortable and versatile t-shirt for everyday wear."

    product_name_2 = "Elegant Silk Evening Dress"
    base_description_2 = "Stunning dress perfect for special occasions."

    # 3. Generate descriptions for Product 1
    print(f"\n--- Generating descriptions for '{product_name_1}' ---")
    identified_attrs_1 = attr_identifier.identify_attributes(product_name_1, base_description_1)
    print(f"Identified Attributes for '{product_name_1}': {identified_attrs_1}")
    descriptions_1 = desc_generator.generate_descriptions(product_name_1, base_description_1, identified_attrs_1)
    for i, desc in enumerate(descriptions_1):
        print(f"Description {i+1}: {desc}")

    # 4. Generate descriptions for Product 2
    print(f"\n--- Generating descriptions for '{product_name_2}' ---")
    identified_attrs_2 = attr_identifier.identify_attributes(product_name_2, base_description_2)
    print(f"Identified Attributes for '{product_name_2}': {identified_attrs_2}")
    descriptions_2 = desc_generator.generate_descriptions(product_name_2, base_description_2, identified_attrs_2)
    for i, desc in enumerate(descriptions_2):
        print(f"Description {i+1}: {desc}")
