import os
import random
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from trl import SFTTrainer
import torch

CONFIG = {
    "OPENAI_API_KEY": "YOUR_OPENAI_API_KEY",
    "BASE_LLM_MODEL_PATH": "./finetuned_ecommerce_llm",
    "BASE_LLM_NAME": "facebook/opt-125m",
}

SHOPIFY_API_DOC = """
Shopify Product API Documentation:
Endpoint: /admin/api/2023-10/products.json
Methods: GET, POST, PUT, DELETE

GET /admin/api/2023-10/products.json
    Description: Retrieve a list of products.
    Parameters:
        limit (int): Amount of products to retrieve. (default: 50)
        fields (string): Comma-separated list of fields to include.
    Example Response:
        [{"id": 123, "title": "Example Product", "price": "10.00"}]

POST /admin/api/2023-10/products.json
    Description: Create a new product.
    Body:
        product (object): Product object.
            title (string, required): The name of the product.
            body_html (string): The description of the product.
            vendor (string): The vendor of the product.
            product_type (string): The type of product.
            tags (string): A comma-separated list of tags.
    Example Request:
        {"product": {"title": "New T-Shirt", "body_html": "Comfortable cotton t-shirt", "vendor": "MyBrand"}}

PUT /admin/api/2023-10/products/{product_id}.json
    Description: Update an existing product.
    Path Parameters:
        product_id (int, required): The ID of the product.
    Body:
        product (object): Product object with fields to update.
    Example Request:
        {"product": {"title": "Updated T-Shirt Price"}}

DELETE /admin/api/2023-10/products/{product_id}.json
    Description: Delete a product.
    Path Parameters:
        product_id (int, required): The ID of the product.
"""

def parse_api_call(api_call_string):
    try:
        parts = api_call_string.split("(", 1)
        if len(parts) < 2:
            return {"error": "Invalid API call format"}
        
        platform_method = parts[0]
        if "." not in platform_method:
            return {"error": "Invalid API call format: missing platform.method"}

        platform, method = platform_method.split(".", 1)
        
        args_str = parts[1].rstrip(")")
        args = {}
        if args_str:
            for arg_pair in args_str.split(", "):
                if "=" in arg_pair:
                    key, value = arg_pair.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if value.startswith("'") and value.endswith("'"):
                        args[key] = value[1:-1]
                    elif value.lower() == "true":
                        args[key] = True
                    elif value.lower() == "false":
                        args[key] = False
                    else:
                        try:
                            args[key] = int(value)
                        except ValueError:
                            try:
                                args[key] = float(value)
                            except ValueError:
                                args[key] = value
        return {"platform": platform.strip(), "method": method.strip(), "args": args}
    except Exception as e:
        return {"error": f"Parsing failed: {e}"}

def simulate_powerful_llm_response(prompt):
    if "create a new product" in prompt.lower() and "shopify" in prompt.lower():
        product_names = ["Vintage T-Shirt", "Leather Wallet", "Handmade Soap", "Organic Coffee Beans"]
        descriptions = ["Comfortable and stylish.", "Durable and elegant.", "Natural ingredients.", "Rich aroma."]
        vendors = ["Fashionista", "LeatherCraft", "EcoBeauty", "BeanBrew"]
        
        title = random.choice(product_names)
        body_html = random.choice(descriptions)
        vendor = random.choice(vendors)

        instruction = f"Create a new product on Shopify called '{title}' with a description '{body_html}' from the vendor '{vendor}'."
        api_call = f"shopify.create_product(title='{title}', body_html='{body_html}', vendor='{vendor}')"
        return [
            {"instruction": instruction, "api_call": api_call},
        ]
    elif "list products" in prompt.lower() and "shopify" in prompt.lower():
        instruction = "List all products on Shopify."
        api_call = "shopify.get_products()"
        return [
            {"instruction": instruction, "api_call": api_call}
        ]
    elif "update product" in prompt.lower() and "shopify" in prompt.lower():
        product_id = random.randint(1000, 9999)
        new_price = round(random.uniform(10.0, 100.0), 2)
        instruction = f"Update the price of product with ID {product_id} to {new_price} on Shopify."
        api_call = f"shopify.update_product(product_id={product_id}, price={new_price})"
        return [
            {"instruction": instruction, "api_call": api_call}
        ]
    elif "delete product" in prompt.lower() and "shopify" in prompt.lower():
        product_id = random.randint(1000, 9999)
        instruction = f"Delete product with ID {product_id} from Shopify."
        api_call = f"shopify.delete_product(product_id={product_id})"
        return [
            {"instruction": instruction, "api_call": api_call}
        ]
    return []

def generate_synthetic_data(num_samples=100):
    print("Generating synthetic instruction-API data...")
    data = []
    base_prompt_template = """
    Given the following API documentation for Shopify:
    {api_doc}

    Generate a real-world use case instruction and the corresponding API call.
    Instruction:
    API Call:
    """
    
    in_context_examples = [
        {"instruction": "Create a new product named 'Summer Dress' with body 'Light and airy.' from vendor 'FashionWear'.",
         "api_call": "shopify.create_product(title='Summer Dress', body_html='Light and airy.', vendor='FashionWear')"},
        {"instruction": "Get a list of all products.",
         "api_call": "shopify.get_products()"},
    ]

    for _ in range(num_samples):
        current_api_doc = SHOPIFY_API_DOC
        
        prompt = base_prompt_template.format(api_doc=current_api_doc)
        
        num_examples = random.randint(0, min(len(in_context_examples), 2))
        for i in range(num_examples):
            example = random.choice(in_context_examples)
            prompt += f"\nInstruction: {example['instruction']}\nAPI Call: {example['api_call']}"

        generated_pairs = simulate_powerful_llm_response(prompt)
        data.extend(generated_pairs)
    
    print(f"Generated {len(data)} synthetic samples.")
    return Dataset.from_list(data)

def finetune_llm(dataset):
    print("Starting LLM finetuning...")

    model_name = CONFIG["BASE_LLM_NAME"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    def formatting_prompts_func(examples):
        output_texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            api_call = examples["api_call"][i]
            text = f"### Instruction:\n{instruction}\n### API Call:\n{api_call}{tokenizer.eos_token}"
            output_texts.append(text)
        return output_texts

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        max_seq_length=512,
        args=training_args,
    )

    trainer.train()
    
    finetuned_model_path = CONFIG["BASE_LLM_MODEL_PATH"]
    trainer.model.save_pretrained(finetuned_model_path)
    tokenizer.save_pretrained(finetuned_model_path)
    print(f"Finetuning complete. Model saved to {finetuned_model_path}")
    return finetuned_model_path

def simulate_api_execution(parsed_api_call):
    platform = parsed_api_call.get("platform")
    method = parsed_api_call.get("method")
    args = parsed_api_call.get("args", {})

    print(f"Simulating API call: Platform={platform}, Method={method}, Args={args}")

    if platform == "shopify":
        if method == "create_product":
            product_title = args.get("title", "Unknown Product")
            return {"status": "success", "message": f"Product '{product_title}' created successfully (simulated).", "product_id": random.randint(10000, 99999)}
        elif method == "get_products":
            return {"status": "success", "message": "Retrieved 3 products (simulated).", "products": [{"id": 1, "title": "Sample 1"}, {"id": 2, "title": "Sample 2"}]}
        elif method == "update_product":
            product_id = args.get("product_id")
            return {"status": "success", "message": f"Product {product_id} updated successfully (simulated)."}
        elif method == "delete_product":
            product_id = args.get("product_id")
            return {"status": "success", "message": f"Product {product_id} deleted successfully (simulated)."}
    return {"status": "error", "message": f"Unsupported API call (simulated): {platform}.{method}"}

def run_ecommerce_assistant(model_path, num_interaction=5):
    print("Loading finetuned LLM for inference...")
    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        torch_dtype=torch.bfloat16,
        device=0 if torch.cuda.is_available() else -1
    )
    print("E-commerce Assistant Ready. Type your commands.")

    for i in range(num_interaction):
        user_input = input(f"\n[{i+1}/{num_interaction}] Your command: ")
        if user_input.lower() == "exit":
            break

        prompt = f"### Instruction:\n{user_input}\n### API Call:\n"
        
        generation_output = generator(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            eos_token_id=generator.tokenizer.eos_token_id
        )
        
        generated_text = generation_output[0]["generated_text"]
        
        try:
            api_call_start = generated_text.rfind("### API Call:\n") + len("### API Call:\n")
            api_call_string = generated_text[api_call_start:].strip()
            if generator.tokenizer.eos_token in api_call_string:
                api_call_string = api_call_string.split(generator.tokenizer.eos_token)[0].strip()
        except Exception:
            api_call_string = "Error: Could not extract API call."
        
        print(f"Generated API Call: {api_call_string}")

        if "Error" not in api_call_string:
            parsed_call = parse_api_call(api_call_string)
            if "error" not in parsed_call:
                api_response = simulate_api_execution(parsed_call)
                print(f"API Response: {api_response['message']}")
            else:
                print(f"API Parsing Error: {parsed_call['error']}")
        else:
            print("API call generation failed.")

def main():
    synthetic_dataset = generate_synthetic_data(num_samples=50)
    
    finetuned_model_path = CONFIG["BASE_LLM_MODEL_PATH"]
    if not os.path.exists(finetuned_model_path) or not os.path.isdir(finetuned_model_path):
        print("Finetuned model not found. Attempting to finetune a base model.")
        print("WARNING: Finetuning can be resource-intensive and time-consuming.")
        try:
            finetuned_model_path = finetune_llm(synthetic_dataset)
        except Exception as e:
            print(f"Finetuning failed (likely due to missing GPU or large model). Using base model for inference simulation: {e}")
            finetuned_model_path = CONFIG["BASE_LLM_NAME"]
    else:
        print(f"Using existing finetuned model at: {finetuned_model_path}")

    run_ecommerce_assistant(finetuned_model_path, num_interaction=3)

if __name__ == "__main__":
    main()