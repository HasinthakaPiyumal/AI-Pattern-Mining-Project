import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
import random

def generate_synthetic_instruction_data(num_samples: int):
    data = []
    for i in range(num_samples):
        instruction_types = [
            "What is the status of my order {order_id}?",
            "Can you tell me the shipping date for order {order_id}?",
            "Summarize the return policy for {product_category} in bullet points.",
            "What are the steps to return a product?",
            "I want to change my shipping address for order {order_id}.",
            "How do I contact customer support?",
            "What is your refund policy?",
            "Tell me about your privacy policy.",
            "Can I track my package with order {order_id}?",
            "What is your warranty policy for electronics?"
        ]
        response_templates = [
            "The status of your order {order_id} is: {status}.",
            "Your order {order_id} is expected to ship on {shipping_date}.",
            "Here is a summary of the return policy for {product_category}:\n- Item must be returned within 30 days.\n- Original packaging required.\n- Receipt is mandatory.",
            "To return a product, please follow these steps:\n1. Fill out the return form.\n2. Package the item securely.\n3. Ship it to our return center.",
            "Please provide your new shipping address and we will update order {order_id} for you.",
            "You can contact customer support via live chat on our website or by calling 1-800-555-0123.",
            "Our refund policy states that refunds are processed within 5-7 business days after the returned item is received and inspected.",
            "Our privacy policy details how we collect, use, and protect your personal information. You can find the full policy on our website.",
            "Yes, you can track your package. For order {order_id}, your tracking number is {tracking_id}.",
            "The warranty for electronics typically covers manufacturing defects for one year from the purchase date. Please refer to the product manual for specifics."
        ]

        order_id = f"ORD{random.randint(10000, 99999)}"
        product_category = random.choice(["electronics", "apparel", "home goods"])
        status = random.choice(["Processing", "Shipped", "Delivered", "Cancelled"])
        shipping_date = f"2023-12-{random.randint(1, 30):02d}"
        tracking_id = f"TRK{random.randint(100000000, 999999999)}"

        instruction = random.choice(instruction_types).format(order_id=order_id, product_category=product_category)
        response = random.choice(response_templates).format(
            order_id=order_id, status=status, shipping_date=shipping_date,
            product_category=product_category, tracking_id=tracking_id
        )
        data.append({"instruction": instruction, "response": response})
    return data

def format_data_for_ift(data):
    formatted_data = []
    for item in data:
        formatted_data.append(f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}")
    return formatted_data

def load_base_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def setup_sft_trainer(model, tokenizer, train_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        report_to="none",
    )

    sft_config = SFTConfig(max_seq_length=512)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        args=training_args,
        packing=True,
        **sft_config.to_dict(),
    )
    return trainer

def run_finetuning(trainer):
    trainer.train()

def save_finetuned_model(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def mock_order_database(order_id):
    orders = {
        "ORD12345": {"status": "Shipped", "shipping_date": "2023-12-20", "items": "Laptop, Mouse"},
        "ORD67890": {"status": "Processing", "shipping_date": "N/A", "items": "Keyboard"},
        "ORD11223": {"status": "Delivered", "shipping_date": "2023-12-10", "items": "Monitor"}
    }
    return orders.get(order_id, {"status": "Not Found", "shipping_date": "N/A", "items": "N/A"})

def mock_policy_database(policy_type):
    policies = {
        "return": "Our return policy allows returns within 30 days of purchase with original packaging and receipt.",
        "refund": "Refunds are processed within 5-7 business days after item inspection.",
        "warranty": "Electronics warranty covers manufacturing defects for one year."
    }
    return policies.get(policy_type, "Policy not found.")

if __name__ == "__main__":
    # I. Data Preparation Module
    print("Generating synthetic instruction data...")
    synthetic_data = generate_synthetic_instruction_data(num_samples=100)
    
    print("Formatting data for IFT...")
    formatted_ift_data = format_data_for_ift(synthetic_data)
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame({"text": formatted_ift_data}))
    
    # II. Instruction Finetuning (IFT) Module
    model_name = "distilgpt2"  # A small model for demonstration
    output_dir = "./intelli_desk_finetuned_model"

    print(f"Loading base model and tokenizer: {model_name}...")
    model, tokenizer = load_base_model_and_tokenizer(model_name)

    print("Setting up SFT Trainer...")
    trainer = setup_sft_trainer(model, tokenizer, train_dataset, output_dir)

    # Due to resource constraints and the nature of code generation, 
    # we will only set up the trainer, not run a full training here.
    # Uncomment the line below to run finetuning if you have the resources.
    # print("Running finetuning (this may take a while)...\n")
    # run_finetuning(trainer)
    # print("Finetuning complete!")
    
    # print(f"Saving finetuned model to {output_dir}...")
    # save_finetuned_model(model, tokenizer, output_dir)

    print("\n--- Finetuning setup complete. (Training step skipped for demo) ---\n")

    # III. Knowledge Retrieval Module (Simulated)
    print("Demonstrating Knowledge Retrieval (Simulated):\n")

    # Simulate order lookup
    order_id_to_lookup = "ORD12345"
    order_info = mock_order_database(order_id_to_lookup)
    print(f"Order {order_id_to_lookup} info: {order_info}")

    order_id_to_lookup = "ORD99999" # Non-existent order
    order_info = mock_order_database(order_id_to_lookup)
    print(f"Order {order_id_to_lookup} info: {order_info}")

    # Simulate policy lookup
    policy_type_to_lookup = "return"
    policy_text = mock_policy_database(policy_type_to_lookup)
    print(f"Return Policy: {policy_text}")

    policy_type_to_lookup = "shipping" # Non-existent policy
    policy_text = mock_policy_database(policy_type_to_lookup)
    print(f"Shipping Policy: {policy_text}")

    print("\nIntelliDesk architecture demonstration complete.")
