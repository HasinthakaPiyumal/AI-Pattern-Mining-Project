import json

def generate_formatted_output(data):
    prompt = "Output the following information as a JSON array of objects, where each object has \'name\' and \'quantity\' fields."
    
    # In a real application, you would send the prompt and data to a GenAI model.
    # For this example, we'll simulate the GenAI's response by formatting the data directly.
    
    formatted_data = []
    for item in data:
        formatted_data.append({"name": item["name"], "quantity": item["quantity"]})
        
    return json.dumps(formatted_data, indent=2)

if __name__ == "__main__":
    example_data = [
        {"name": "Apples", "quantity": 10},
        {"name": "Bananas", "quantity": 5},
        {"name": "Oranges", "quantity": 8}
    ]
    
    output = generate_formatted_output(example_data)
    print(output)