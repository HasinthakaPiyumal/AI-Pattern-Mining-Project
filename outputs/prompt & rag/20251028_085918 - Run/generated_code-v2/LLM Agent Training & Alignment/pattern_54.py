
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset

# 1. Data Collection and Preprocessing (Simulated)
# Simulate a dataset of customer support interactions
demonstrations = [
    {"customer_query": "Hi, what's the status of my order 12345?", "agent_response": "Sure, let me check. ACTION:CHECK_ORDER(ORDER_ID='12345') Your order 12345 is currently being processed."},
    {"customer_query": "I want to return item XYZ. How do I do that?", "agent_response": "No problem. ACTION:INITIATE_RETURN(PRODUCT_SKU='XYZ') Please fill out the return form on our website."},
    {"customer_query": "Can I change my delivery address to 123 Main St?", "agent_response": "Yes, I can update that for you. ACTION:UPDATE_PROFILE(FIELD='address', VALUE='123 Main St') Address updated successfully."},
    {"customer_query": "Where is my package for order 98765?", "agent_response": "Let me find that for you. ACTION:CHECK_ORDER(ORDER_ID='98765') Your package for order 98765 is out for delivery."},
    {"customer_query": "I need help with product ABC. It's not working.", "agent_response": "I understand. Can you describe the issue in more detail?"},
]

# Convert to Hugging Face Dataset format
dataset = Dataset.from_list(demonstrations)

# 2. Language Model (LLM) Selection and Loading
model_name = "distilgpt2" # A smaller, faster model for demonstration
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add a pad token if the tokenizer doesn't have one (common for GPT-like models)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Or use eos_token
    model.resize_token_embeddings(len(tokenizer))

# 3. Behavior Cloning (Supervised Fine-tuning)
# Preprocess function to tokenize and prepare data for CausalLM
def preprocess_function(examples):
    # Concatenate customer query and agent response as input for causal language modeling
    # The model learns to predict the agent_response given the customer_query
    inputs = [f"Customer: {q} Agent: {a}{tokenizer.eos_token}" for q, a in zip(examples["customer_query"], examples["agent_response"])]
    tokenized_inputs = tokenizer(inputs, truncation=True, max_length=128, padding="max_length")

    # For causal LMs, labels are usually the same as input_ids, but shifted internally
    # We don't need to explicitly create `labels` if `input_ids` are passed and attention_mask is present,
    # the Trainer will handle the shifting for causal language modeling loss.
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,             # Number of training epochs
    per_device_train_batch_size=2,  # Batch size per device during training
    per_device_eval_batch_size=2,   # Batch size for evaluation
    warmup_steps=10,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,              # Strength of weight decay
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",          # Save checkpoint every epoch
    learning_rate=5e-5,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train the model
print("Starting model training...")
trainer.train()
print("Model training complete.")

# 4. Inference (Example Usage)
def generate_agent_response(customer_query, fine_tuned_model, query_tokenizer, max_length=150):
    input_text = f"Customer: {customer_query} Agent: "
    input_ids = query_tokenizer.encode(input_text, return_tensors="pt")

    # Ensure input_ids are on the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tuned_model.to(device)
    input_ids = input_ids.to(device)

    # Generate response
    output_ids = fine_tuned_model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=query_tokenizer.pad_token_id,
        do_sample=True, # Use sampling for more varied responses
        top_k=50,       # Sample from top 50 most likely next tokens
        top_p=0.95,     # Sample from tokens that sum up to 95% probability
        temperature=0.7 # Control randomness of predictions
    )

    generated_text = query_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the agent's part of the response
    agent_response_start = generated_text.find("Agent: ")
    if agent_response_start != -1:
        return generated_text[agent_response_start + len("Agent: "):]
    return generated_text

# Test the fine-tuned model
print("\n--- Testing Fine-tuned Chatbot ---")
new_customer_query1 = "My order 11223 seems delayed."
response1 = generate_agent_response(new_customer_query1, model, tokenizer)
print(f"Customer: {new_customer_query1}\nChatbot: {response1}")

new_customer_query2 = "I want to update my email."
response2 = generate_agent_response(new_customer_query2, model, tokenizer)
print(f"Customer: {new_customer_query2}\nChatbot: {response2}")

new_customer_query3 = "Can I return item GHI?"
response3 = generate_agent_response(new_customer_query3, model, tokenizer)
print(f"Customer: {new_customer_query3}\nChatbot: {response3}")
