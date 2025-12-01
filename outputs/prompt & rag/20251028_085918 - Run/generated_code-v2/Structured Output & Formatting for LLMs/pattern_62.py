import json

def standardize_product_description(unstructured_description: str) -> dict:
    prompt = f"""Extract the following product details: product_name, brand, category, key_features (as a list), price_range, and material from the following description. Respond ONLY in a JSON object with these keys.\n\nDescription:\n{unstructured_description}\n\nJSON Output:\n"""

    # In a real application, this would be an actual LLM call, e.g.:
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant that extracts product information into a strict JSON format."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     response_format={"type": "json_object"} # For models that support this
    # )
    # llm_output_json_string = response.choices[0].message.content

    # For demonstration, we simulate the LLM output based on the example given in the problem description.
    # This simulation directly provides the desired JSON structure.
    # In a real scenario, the LLM would generate this based on the `unstructured_description`.
    if "SuperSonic Wireless Headphones" in unstructured_description:
        llm_output_json_string = """
{
  "product_name": "SuperSonic Wireless Headphones",
  "brand": "AudioTech",
  "category": "Electronics > Audio > Headphones",
  "key_features": [
    "Active Noise Cancellation",
    "Bluetooth 5.2",
    "30-hour battery life",
    "Ergonomic design"
  ],
  "price_range": "$150 - $200",
  "material": "Premium ABS Plastic, Memory Foam"
}
"""
    elif "Sturdy Oak Dining Table" in unstructured_description:
        llm_output_json_string = """
{
  "product_name": "Sturdy Oak Dining Table",
  "brand": "HomeEssentials",
  "category": "Furniture > Dining Room > Tables",
  "key_features": [
    "Solid Oak Wood",
    "Seats 6-8 people",
    "Easy assembly",
    "Water-resistant finish"
  ],
  "price_range": "$500 - $800",
  "material": "Solid Oak"
}
"""
    else:
        llm_output_json_string = """
{
  "product_name": "Unknown Product",
  "brand": "N/A",
  "category": "N/A",
  "key_features": [],
  "price_range": "N/A",
  "material": "N/A"
}
"""


    try:
        standardized_data = json.loads(llm_output_json_string)
        return standardized_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM output: {e}")
        print(f"LLM Output:\n{llm_output_json_string}")
        return {"error": "Failed to parse LLM output", "raw_output": llm_output_json_string}

if __name__ == "__main__":
    description1 = "Introducing the new SuperSonic Wireless Headphones from AudioTech. Experience immersive sound with Active Noise Cancellation and Bluetooth 5.2. Enjoy up to 30 hours of battery life and an ergonomic design for maximum comfort. Made with premium ABS plastic and memory foam. Available for $150-$200."
    description2 = "A beautiful and sturdy dining table, perfect for your home. Made from solid oak, it comfortably seats 6 to 8 people. Features easy assembly and a water-resistant finish. Get yours from HomeEssentials today!"
    description3 = "A generic product description without specific keywords."

    print("--- Processing Description 1 ---")
    output1 = standardize_product_description(description1)
    print(json.dumps(output1, indent=2))
    print("\n")

    print("--- Processing Description 2 ---")
    output2 = standardize_product_description(description2)
    print(json.dumps(output2, indent=2))
    print("\n")

    print("--- Processing Description 3 ---")
    output3 = standardize_product_description(description3)
    print(json.dumps(output3, indent=2))
    print("\n")
