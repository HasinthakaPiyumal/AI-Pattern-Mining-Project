import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
import random

# --- 1. Data Layer: Generate Synthetic E-commerce Dataset ---

def generate_synthetic_data(num_samples=100):
    products = [
        "Laptop Pro X", "Smartphone Z1", "Smartwatch Ultra", "Wireless Earbuds 2.0",
        "Gaming PC Elite", "4K Monitor Curved", "Ergonomic Keyboard", "Gaming Mouse RGB",
        "Webcam HD", "USB-C Hub", "Portable SSD", "Router Wi-Fi 6",
        "Noise-Cancelling Headphones", "Bluetooth Speaker", "E-reader Paperwhite",
        "Drone Explorer", "Action Camera 4K", "VR Headset Pro", "Fitness Tracker",
        "Electric Toothbrush", "Air Fryer", "Robot Vacuum", "Smart Thermostat",
        "Coffee Maker Espresso", "Blender High-Speed"
    ]
    categories = [
        "Electronics", "Gaming", "Home Appliances", "Health & Personal Care", "Smart Home"
    ]
    user_behaviors = [
        "viewed", "added to cart", "searched for", "previously bought", "liked"
    ]

    data = []
    for i in range(num_samples):
        user_id = f"user_{i}"
        history_items = random.sample(products, random.randint(1, 5))
        history_behavior = random.choice(user_behaviors)
        target_product = random.choice(list(set(products) - set(history_items)))

        history_str = ", ".join([f"{history_behavior} {item}" for item in history_items])
        
        instruction = f"Given user history: {history_str}. Recommend products:"
        response = target_product
        
        data.append({"instruction": instruction, "response": response})
    
    return pd.DataFrame(data)

# Generate a smaller dataset for quick demonstration
synthetic_df = generate_synthetic_data(num_samples=200)

# Format data for SFTTrainer
def format_data(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"}

formatted_data = synthetic_df.apply(format_data, axis=1)
dataset = Dataset.from_pandas(formatted_data[['text']])

print("--- Synthetic Dataset Sample ---")
print(dataset[0])

# --- 2. Model Layer: Base LLM and LoRA Configuration ---

model_name = "facebook/opt-125m"  # A small model for demonstration
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Set pad token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 # Use bfloat16 for better efficiency if supported
)

# Prepare model for LoRA (usually done with 4-bit/8-bit quantization for QLoRA, but good practice)
# For this example, we'll focus on LoRA without quantization for simplicity, but it's a useful function.
model = prepare_model_for_kbit_training(model) 

# LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA attention dimension
    lora_alpha=16,  # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "v_proj"],  # Target query and value matrices for LoRA
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",  # Do not train bias terms
    task_type="CAUSAL_LM", # Causal Language Modeling task
)

# Get LoRA model
model = get_peft_model(model, lora_config)
print("--- LoRA Model Architecture ---")
model.print_trainable_parameters() # Show trainable parameters

# --- 3. Training Workflow ---

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # Keep epochs low for demonstration
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2, # Simulate larger batch size
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none", # Disable reporting to W&B etc. for simplicity
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=128, # Limit sequence length for efficiency
)

print("--- Starting Fine-tuning ---")
trainer.train()

print("--- Fine-tuning Complete ---")

# Save the fine-tuned LoRA adapters
trainer.model.save_pretrained("./fine_tuned_lora_adapters")
print("LoRA adapters saved to ./fine_tuned_lora_adapters")

# --- 4. Inference/Recommendation Demonstration ---

# Load the base model again
inference_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)

# Load the LoRA adapters and merge them with the base model for inference
from peft import PeftModel
inference_model = PeftModel.from_pretrained(inference_model, "./fine_tuned_lora_adapters")
inference_model = inference_model.merge_and_unload()

inference_model.eval() # Set model to evaluation mode

print("--- Demonstrating Inference ---")

def generate_recommendation(user_history_str, model, tokenizer, max_new_tokens=20):
    prompt = f"### Instruction:\nGiven user history: {user_history_str}. Recommend products:\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate a response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, do_sample=True, top_p=0.9, temperature=0.7)
    
    # Decode and extract only the generated part after the prompt
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # Basic post-processing to get the product name
    if "\n" in response:
        response = response.split("\n")[0]
    response = response.strip()

    return response

# Example user history for inference
user_history_example = "viewed Gaming PC Elite, searched for RGB Gaming Mouse, liked 4K Monitor Curved"
recommended_product = generate_recommendation(user_history_example, inference_model, tokenizer)

print(f"User History: {user_history_example}")
print(f"Recommended Product: {recommended_product}")

user_history_example_2 = "previously bought Smartwatch Ultra, viewed Fitness Tracker"
recommended_product_2 = generate_recommendation(user_history_example_2, inference_model, tokenizer)

print(f"\nUser History: {user_history_example_2}")
print(f"Recommended Product: {recommended_product_2}")
