import json
import xml.etree.ElementTree as ET

def get_product_info():
    product_name = input("Enter product name: ")
    category = input("Enter product category: ")
    features_input = input("Enter key features (comma-separated): ")
    key_features = [f.strip() for f in features_input.split(",") if f.strip()]
    output_format = input("Enter desired output format (json or xml): ").lower()

    return {
        "product_name": product_name,
        "category": category,
        "key_features": key_features,
        "output_format": output_format
    }

def generate_description(product_details):
    product_name = product_details["product_name"]
    category = product_details["category"]
    key_features = product_details["key_features"]
    output_format = product_details["output_format"]

    base_description = (
        f"Discover the {product_name}, a versatile {category} designed to enhance your daily life. "
        "With its innovative features, it stands out in its class." # Placeholder for LLM generated part
    )

    # Simulate LLM adding more details based on features
    if key_features:
        features_str = ", ".join(key_features)
        base_description += f" Key features include: {features_str}."

    llm_generated_content = {
        "product_name": product_name,
        "category": category,
        "description": base_description,
        "features": key_features
    }

    if output_format == "json":
        return json.dumps(llm_generated_content, indent=4)
    elif output_format == "xml":
        root = ET.Element("product")
        name_elem = ET.SubElement(root, "name")
        name_elem.text = product_name
        category_elem = ET.SubElement(root, "category")
        category_elem.text = category
        description_elem = ET.SubElement(root, "description")
        description_elem.text = base_description

        features_elem = ET.SubElement(root, "features")
        for feature in key_features:
            feature_item_elem = ET.SubElement(features_elem, "feature")
            feature_item_elem.text = feature

        return ET.tostring(root, encoding="unicode", pretty_print=True)
    else:
        return base_description

if __name__ == "__main__":
    product_info = get_product_info()
    generated_description = generate_description(product_info)
    print("\n--- Generated Product Description ---")
    print(generated_description)
    print("-------------------------------------")
