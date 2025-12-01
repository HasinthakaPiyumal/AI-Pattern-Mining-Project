import json

def simulate_llm_response(product_info):
    product_name = product_info.get("name", "Unknown Product")
    category = product_info.get("category", "General")
    key_features = ", ".join(product_info.get("key_features", []))
    selling_points = ", ".join(product_info.get("selling_points", []))
    specs = ", ".join([f"{k}: {v}" for k, v in product_info.get("specs", {}).items()])

    response_template = (
        f"Introducing the new {product_name}, a cutting-edge {category} product designed to revolutionize your experience. "
        f"It comes packed with essential features like {key_features}. "
        f"Enjoy the advantages of {selling_points}. "
        f"Technical details include: {specs}. "
        f"This device is perfect for those who seek efficiency and advanced technology. "
    )
    return response_template

def format_product_description(product_name, llm_raw_output, original_product_info):

    short_description_candidate = llm_raw_output.split(". ")[0] + "." if ". " in llm_raw_output else llm_raw_output[:150] + "..."

    features_list = original_product_info.get("key_features", [])
    benefits_list = original_product_info.get("selling_points", [])
    tech_specs_dict = original_product_info.get("specs", {})

    structured_output = {
        "product_name": product_name,
        "short_description": short_description_candidate.strip(),
        "features": features_list,
        "benefits": benefits_list,
        "technical_specifications": tech_specs_dict
    }

    return json.dumps(structured_output, indent=2)

def generate_product_description(product_info):
    product_name = product_info.get("name", "Unnamed Product")
    raw_llm_output = simulate_llm_response(product_info)
    formatted_json_output = format_product_description(product_name, raw_llm_output, product_info)
    return formatted_json_output
