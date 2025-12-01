import torch
import os
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def generate_synthetic_data(num_samples=100, file_path="synthetic_training_data.txt"):
    """Generates synthetic customer queries and corresponding custom e-commerce commands."""
    data = []
    customer_queries = [
        "What is the status of my order?",
        "Can I track my recent purchase?",
        "Where is my package?",
        "I want to return an item.",
        "How do I initiate a return?",
        "What is your return policy?",
        "I need help with my account settings.",
        "How do I change my shipping address?",
        "Can I update my payment method?",
        "Show me my previous orders."
    ]
    platform_commands = [
        "check_order_status()",
        "track_shipment()",
        "track_shipment()",
        "initiate_return()",
        "initiate_return()",
        "show_return_policy()",
        "access_account_settings()",
        "update_shipping_address()",
        "update_payment_method()",
        "list_past_orders()"
    ]

    for i in range(num_samples):
        query_idx = i % len(customer_queries)
        # For behavior cloning, we want the model to output the command given the query.
        # We format it as 'Customer: [query] Agent: [command]<EOS>'
        data.append(f"Customer: {customer_queries[query_idx]} Agent: {platform_commands[query_idx]}<|endoftext|>")
    
    # Save to a text file for TextDataset
    with open(file_path, "w") as f:
        for item in data:
            f.write(item + "\n")
    print(f"Generated {num_samples} synthetic data samples to {file_path}")

def train_chatbot(model_output_dir="./chatbot_model_behavior_cloning", data_file="synthetic_training_data.txt", num_samples=1000, num_train_epochs=3):
    """Trains a GPT-2 model using behavior cloning on synthetic e-commerce customer support data."""

    # 1. Generate synthetic data if not already present or specified
    if not os.path.exists(data_file):
        generate_synthetic_data(num_samples=num_samples, file_path=data_file)

    # 2. Load Tokenizer and Model
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add a padding token if it doesn't exist, which is good practice for batching
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    # 3. Prepare Dataset and Data Collator
    # TextDataset expects a text file where each line is a training example
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_file,
        block_size=128 # Max sequence length
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False # Causal Language Modeling
    )

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        save_steps=10_000, # Only save at the end for this small demo
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=50,
    )

    # 5. Initialize and Run Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete. Model saved to", model_output_dir)

    # Save the tokenizer with the model
    tokenizer.save_pretrained(model_output_dir)

def load_fine_tuned_model(model_path="./chatbot_model_behavior_cloning"):
    """Loads the fine-tuned GPT-2 model and tokenizer."""
    if not os.path.exists(model_path):
        print(f"Error: Model directory '{model_path}' not found. Please run training first.")
        return None, None
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval() # Set model to evaluation mode
    print(f"Model and tokenizer loaded from {model_path}")
    return tokenizer, model

def generate_command(query, tokenizer, model, max_length=50):
    """Generates an e-commerce platform command based on a customer query using the fine-tuned model."""
    if tokenizer is None or model is None:
        print("Model or tokenizer not loaded. Cannot generate command.")
        return ""

    # Format the input similar to how it was trained
    input_text = f"Customer: {query} Agent:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output tokens
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id, # Use EOS token for padding
            do_sample=True, # Use sampling for more diverse outputs
            top_k=50, # Consider top 50 tokens
            top_p=0.95, # Nucleus sampling
            temperature=0.7 # Less deterministic
        )

    # Decode the generated output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    
    # Extract only the command part. This requires careful parsing based on your training format.
    # We trained with 'Agent: [command]<|endoftext|>', so we look for 'Agent:' and then the end token.
    try:
        command_start_idx = generated_text.find("Agent:") + len("Agent:")
        command_end_idx = generated_text.find("<|endoftext|>", command_start_idx)
        
        if command_end_idx == -1: # If EOS token not found, take until end or max length
            predicted_command = generated_text[command_start_idx:].strip()
        else:
            predicted_command = generated_text[command_start_idx:command_end_idx].strip()
            
        # Further refinement to ensure it's a valid-looking command
        if not predicted_command.endswith(')'):
            # Heuristic: if it looks like a command but is cut off, try to complete it
            if '(' in predicted_command and not ')' in predicted_command:
                 predicted_command += ')' # Simple completion
            
        return predicted_command
    except Exception as e:
        print(f"Error parsing generated text: {e}")
        return generated_text # Return raw generated text on error

def main():
    parser = argparse.ArgumentParser(description="Customer Support Chatbot using Behavior Cloning.")
    parser.add_argument("mode", choices=["train", "inference"], help="Mode to run the chatbot in: 'train' or 'inference'")
    parser.add_argument("--model_dir", type=str, default="./chatbot_model_behavior_cloning", help="Directory to save/load the fine-tuned model.")
    parser.add_argument("--data_file", type=str, default="synthetic_training_data.txt", help="Path to the synthetic training data file.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic samples to generate if data_file does not exist.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")

    args = parser.parse_args()

    if args.mode == "train":
        train_chatbot(model_output_dir=args.model_dir, data_file=args.data_file, num_samples=args.num_samples, num_train_epochs=args.epochs)
    elif args.mode == "inference":
        tokenizer, model = load_fine_tuned_model(model_path=args.model_dir)
        if tokenizer and model:
            print("\n--- Chatbot Interaction (type 'exit' to quit) ---")
            while True:
                user_query = input("You: ")
                if user_query.lower() == 'exit':
                    break
                
                command = generate_command(user_query, tokenizer, model)
                print(f"Chatbot (predicted command): {command}")
                # In a real application, you would now execute this 'command' against the e-commerce platform API
        print("Exiting Chatbot.")

if __name__ == "__main__":
    main()