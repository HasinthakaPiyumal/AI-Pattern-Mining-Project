import os
import openai
import json
import csv
from io import StringIO

# Set your OpenAI API key
# It's recommended to set this as an environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

# For demonstration purposes, you can replace with your actual key or ensure it's set in your environment
# Replace 'YOUR_OPENAI_API_KEY' with your actual key if not using environment variable
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY") 

def generate_product_description(
    product_name: str,
    features: list,
    benefits: list,
    price: float,
    output_format: str = "markdown",
) -> str:
    """
    Generates a product description using an LLM in a specified structured format.

    Args:
        product_name: The name of the product.
        features: A list of key features of the product.
        benefits: A list of benefits for the customer.
        price: The price of the product.
        output_format: The desired output format ('json', 'markdown', 'csv').

    Returns:
        The generated product description in the specified format.
    """

    features_str = ", ".join(features)
    benefits_str = ", ".join(benefits)

    base_prompt = f"""
    Generate a product description for the following product:
    Product Name: {product_name}
    Features: {features_str}
    Benefits: {benefits_str}
    Price: ${price:.2f}
    """

    if output_format == "json":
        format_instruction = """
        Please provide the product description as a JSON object with the following keys: "product_name", "description", "features", "benefits", "price". The "features" and "benefits" should be arrays of strings.
        """
    elif output_format == "csv":
        format_instruction = """
        Please provide the product description as a single CSV row, including a header row. The columns should be: "Product Name", "Description", "Features", "Benefits", "Price". Features and benefits should be comma-separated strings within their respective cells.
        """
    elif output_format == "markdown":
        format_instruction = """
        Please provide the product description in a well-structured Markdown format, including headings for Product Name, Description, Features, Benefits, and Price.
        """
    else:
        raise ValueError("Unsupported output format. Choose from 'json', 'markdown', 'csv'.")

    full_prompt = f"{base_prompt}\n\n{format_instruction}"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # You can choose other models like "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates product descriptions."}, 
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        generated_text = response.choices[0].message.content.strip()

        if output_format == "csv":
            # OpenAI might return some introductory text, try to extract the CSV part
            # This is a simple approach, more robust parsing might be needed for complex cases
            lines = generated_text.splitlines()
            csv_data = [line for line in lines if ',' in line]
            if csv_data:
                return "\n".join(csv_data)
            return generated_text # Fallback if no clear CSV lines found
        
        return generated_text

    except openai.AuthenticationError:
        return "Error: OpenAI API key is invalid or not provided. Please set the OPENAI_API_KEY environment variable or replace 'YOUR_OPENAI_API_KEY'."
    except openai.APIError as e:
        return f"Error communicating with OpenAI API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    print("E-commerce Product Description Generator")
    print("-------------------------------------")

    product_name = input("Enter product name: ")
    features_input = input("Enter features (comma-separated): ")
    features = [f.strip() for f in features_input.split(',') if f.strip()]

    benefits_input = input("Enter benefits (comma-separated): ")
    benefits = [b.strip() for b in benefits_input.split(',') if b.strip()]

    while True:
        try:
            price = float(input("Enter price: "))
            break
        except ValueError:
            print("Invalid price. Please enter a numeric value.")

    while True:
        output_format = input("Enter desired output format (json, markdown, csv): ").lower()
        if output_format in ["json", "markdown", "csv"]:
            break
        else:
            print("Invalid format. Please choose from 'json', 'markdown', 'csv'.")

    print("\nGenerating description...")
    description = generate_product_description(product_name, features, benefits, price, output_format)
    
    print(f"\nGenerated Description ({output_format.upper()}):\n")
    if output_format == "json":
        try:
            # Attempt to pretty print JSON if it's valid
            json_output = json.loads(description)
            print(json.dumps(json_output, indent=2))
        except json.JSONDecodeError:
            print(description) # Print raw if not valid JSON
    else:
        print(description)

    print("\nDone.")
