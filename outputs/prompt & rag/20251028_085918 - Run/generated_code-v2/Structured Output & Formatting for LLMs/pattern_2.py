import json
import csv
import xml.etree.ElementTree as ET
import io

# Placeholder for LLM interaction (Information Extraction Module)
def extract_product_info_llm_placeholder(product_listing: str) -> dict:
    # In a real application, this would use langchain to call an LLM
    # The prompt to the LLM would explicitly ask for structured output like:
    # "Extract the product name, price, description, category, and SKU from the following text."
    # "Output the information as a JSON object with keys: name, price, description, category, sku."
    
    # Simulate LLM extracting information
    if "Laptop XYZ" in product_listing:
        return {"name": "Laptop XYZ", "price": "1200.00", "description": "Powerful laptop with 16GB RAM and 512GB SSD.", "category": "Electronics", "sku": "LTXYZ001"}
    elif "Coffee Maker Pro" in product_listing:
        return {"name": "Coffee Maker Pro", "price": "89.99", "description": "Automatic coffee maker with grinder.", "category": "Home Appliances", "sku": "CMPRO123"}
    else:
        return {"name": "Unknown Product", "price": "N/A", "description": "N/A", "category": "N/A", "sku": "N/A"}

# Output Formatting Module
def format_output(data: dict, output_format: str) -> str:
    if output_format.lower() == "json":
        return json.dumps(data, indent=4)
    elif output_format.lower() == "csv":
        output = io.StringIO()
        fieldnames = data.keys()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(data)
        return output.getvalue().strip()
    elif output_format.lower() == "xml":
        root = ET.Element("product")
        for key, value in data.items():
            child = ET.SubElement(root, key)
            child.text = str(value)
        return ET.tostring(root, encoding="unicode", pretty_print=True)
    else:
        raise ValueError("Unsupported output format. Choose from 'json', 'csv', 'xml'.")

# Application Interface
if __name__ == "__main__":
    # Sample product listings
    product_listing_1 = """Latest Laptop XYZ model, featuring a 15.6-inch display, Intel i7 processor, 16GB RAM, and 512GB SSD. 
                         Perfect for professionals and gamers. Price: $1200.00. Category: Electronics. SKU: LTXYZ001."""
    product_listing_2 = """Coffee Maker Pro - Brew delicious coffee at home with this automatic machine. Built-in grinder and programmable settings. 
                         Only $89.99! Home Appliances category. SKU is CMPRO123."""

    print("\n--- Extracting Product 1 ---")
    extracted_data_1 = extract_product_info_llm_placeholder(product_listing_1)
    print("Extracted Raw Data:", extracted_data_1)

    # Demonstrate JSON output
    json_output = format_output(extracted_data_1, "json")
    print("\nJSON Output:\n", json_output)

    # Demonstrate CSV output
    csv_output = format_output(extracted_data_1, "csv")
    print("\nCSV Output:\n", csv_output)
    
    # Demonstrate XML output
    xml_output = format_output(extracted_data_1, "xml")
    print("\nXML Output:\n", xml_output)

    print("\n--- Extracting Product 2 ---")
    extracted_data_2 = extract_product_info_llm_placeholder(product_listing_2)
    print("Extracted Raw Data:", extracted_data_2)

    # Demonstrate JSON output for product 2
    json_output_2 = format_output(extracted_data_2, "json")
    print("\nJSON Output:\n", json_output_2)

    # Demonstrate unsupported format
    try:
        format_output(extracted_data_1, "yaml")
    except ValueError as e:
        print(f"\nError: {e}")