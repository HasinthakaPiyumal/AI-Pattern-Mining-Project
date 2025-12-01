import json
import xml.etree.ElementTree as ET

def generate_product_description_ai_sim(product_name, category, features):
    base_description = f"Discover the {product_name}, a premium {category} designed for excellence. "
    features_text = ", ".join(features)
    return f"{base_description}It boasts a range of innovative features including: {features_text}. Perfect for your needs."

def format_output(description, product_name, category, features, output_format):
    if output_format.lower() == "json":
        output_data = {
            "product_name": product_name,
            "category": category,
            "features": features,
            "description": description
        }
        return json.dumps(output_data, indent=2)
    elif output_format.lower() == "xml":
        product_element = ET.Element("Product")
        name_element = ET.SubElement(product_element, "Name")
        name_element.text = product_name
        category_element = ET.SubElement(product_element, "Category")
        category_element.text = category
        features_element = ET.SubElement(product_element, "Features")
        for feature in features:
            feature_item = ET.SubElement(features_element, "Feature")
            feature_item.text = feature
        description_element = ET.SubElement(product_element, "Description")
        description_element.text = description
        return ET.tostring(product_element, encoding="unicode", pretty_print=True)
    elif output_format.lower() == "markdown":
        markdown_output = f"# {product_name}\n\n"
        markdown_output += f"**Category:** {category}\n\n"
        markdown_output += "**Features:**\n"
        for feature in features:
            markdown_output += f"- {feature}\n"
        markdown_output += f"\n## Description\n{description}\n"
        return markdown_output
    else:
        return f"Unsupported output format. Here is the raw description:\n\n{description}"

def main():
    print("E-commerce Product Description Generator")
    product_name = input("Enter product name: ")
    category = input("Enter product category: ")
    features_input = input("Enter key features (comma-separated): ")
    features = [f.strip() for f in features_input.split(",")]

    output_format = input("Enter desired output format (json, xml, markdown): ")

    generated_description = generate_product_description_ai_sim(product_name, category, features)
    formatted_output = format_output(generated_description, product_name, category, features, output_format)

    print("\n--- Generated Product Description ---")
    print(formatted_output)
    print("-------------------------------------")

if __name__ == "__main__":
    main()