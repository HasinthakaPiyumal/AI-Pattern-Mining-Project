import json

def generate_product_image_coi(product_description: str):
    """
    Simulates the ChainofImages (CoI) process for product customization.
    It takes a product description and conceptually generates intermediate
    image steps, returning a final imagined product description.
    """
    print(f"Processing request: '{product_description}' using ChainofImages...")
    steps = []
    final_image_description = ""

    # Simple keyword-based parsing for demonstration
    keywords = product_description.lower().split()

    base_product = "generic product"
    if "sofa" in keywords:
        base_product = "a sleek sofa base"
    elif "chair" in keywords:
        base_product = "a basic chair design"
    elif "table" in keywords:
        base_product = "a simple table structure"
    # Add more base product logic as needed

    steps.append({"step": 1, "description": f"Generating base image: {base_product}"})

    current_image_state = base_product

    # Apply modifications based on keywords
    if "modern" in keywords:
        current_image_state += ", with a modern aesthetic"
        steps.append({"step": 2, "description": f"Applying modern styling: {current_image_state}"})
    if "blue" in keywords and "velvet" in keywords:
        current_image_state += ", upholstered in rich blue velvet"
        steps.append({"step": len(steps) + 1, "description": f"Adding blue velvet upholstery: {current_image_state}"})
    elif "velvet" in keywords:
        current_image_state += ", upholstered in velvet"
        steps.append({"step": len(steps) + 1, "description": f"Adding velvet upholstery: {current_image_state}"})
    elif "blue" in keywords:
        current_image_state += ", in a calming blue color"
        steps.append({"step": len(steps) + 1, "description": f"Applying blue color: {current_image_state}"})
    
    if "golden" in keywords and "legs" in keywords:
        current_image_state += ", supported by elegant golden legs"
        steps.append({"step": len(steps) + 1, "description": f"Adding golden legs: {current_image_state}"})
    elif "legs" in keywords:
        current_image_state += ", with distinct legs"
        steps.append({"step": len(steps) + 1, "description": f"Adding legs: {current_image_state}"})

    final_image_description = f"Final customized product: {current_image_state}."
    steps.append({"step": len(steps) + 1, "description": final_image_description})

    print("\n--- Chain of Images Steps ---")
    for step in steps:
        print(f"Step {step['step']}: {step['description']}")
    print("-----------------------------")
    
    return {
        "intermediate_steps": steps,
        "final_product_description": final_image_description
    }

# Example Usage:
if __name__ == "__main__":
    description1 = "modern blue velvet sofa with golden legs"
    result1 = generate_product_image_coi(description1)
    print("\nGenerated Result for Description 1:", json.dumps(result1, indent=2))

    print("\n" + "="*50 + "\n")

    description2 = "vintage wooden table with intricate carvings"
    result2 = generate_product_image_coi(description2)
    print("\nGenerated Result for Description 2:", json.dumps(result2, indent=2))

    print("\n" + "="*50 + "\n")

    description3 = "simple red chair"
    result3 = generate_product_image_coi(description3)
    print("\nGenerated Result for Description 3:", json.dumps(result3, indent=2))
