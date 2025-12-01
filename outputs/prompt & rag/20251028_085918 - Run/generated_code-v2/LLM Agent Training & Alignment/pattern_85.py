import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Data Simulation/Loading
# Simulate e-commerce data
product_descriptions = [
    "A sleek black smartphone with 128GB storage and a high-resolution camera.",
    "Comfortable running shoes with breathable mesh and responsive cushioning.",
    "Stylish denim jeans, slim fit, perfect for casual wear.",
    "Organic coffee beans, dark roast, ethically sourced from Ethiopia.",
    "Noise-cancelling headphones with long battery life and superb audio quality."
]

user_interaction_logs = [
    {"user_id": "user_1", "product_id": "prod_A", "interaction": "bought"},
    {"user_id": "user_1", "product_id": "prod_B", "interaction": "viewed"},
    {"user_id": "user_2", "product_id": "prod_C", "interaction": "bought"},
    {"user_id": "user_3", "product_id": "prod_A", "interaction": "viewed"},
    {"user_id": "user_2", "product_id": "prod_D", "interaction": "added_to_cart"},
]

# Create a simple dataset for fine-tuning. In a real scenario, this would be more complex
# and involve mapping user preferences to product features.
data = []
for i, desc in enumerate(product_descriptions):
    data.append({"text": f"Product description: {desc} User feedback: interested."})

# For demonstration, let's create a more direct instruction-based dataset for fine-tuning
# This simulates a user's past interaction and what they might be interested in next.
training_texts = [
    "User previously bought 'A sleek black smartphone'. Recommend similar: 'High-end smartphone with advanced features'.",
    "User viewed 'Comfortable running shoes'. Recommend sports apparel: 'Performance running socks'.",
    "User added 'Stylish denim jeans' to cart. Recommend matching accessories: 'Leather belt'.",
    "User is interested in 'Organic coffee beans'. Recommend brewing equipment: 'French press coffee maker'."
]

# Convert to Hugging Face Dataset format
training_dataset = Dataset.from_pandas(pd.DataFrame(training_texts, columns=["text"]))

# 2. Base LLM Selection and Loading
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Using a smaller model for faster demonstration

# Configure 4-bit quantization
bits_and_bytes_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Set pad_token for casual LMs

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bits_and_bytes_config,
    device_map="auto"
)

# 3. Efficient Fine-tuning (LoRA Implementation)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA attention dimension
    lora_alpha=16,  # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",  # Do not fine-tune bias weights
    task_type="CAUSAL_LM", # Causal Language Modeling task
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Display trainable parameters

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1, # For demonstration, a single epoch
    per_device_train_batch_size=2, # Reduce batch size for memory efficiency
    gradient_accumulation_steps=1, # No accumulation for simplicity
    optim="paged_adamw_8bit", # Use 8-bit AdamW optimizer
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False, # Set to True if your GPU supports fp16 and you have memory issues
    bf16=True, # Use bfloat16 for computation if supported
    max_steps=50, # Limit steps for quick demo
    warmup_steps=5,
    lr_scheduler_type="constant",
)

# Supervised Fine-tuning Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=training_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=256, # Max sequence length for training
)

# Start training (commented out for a quick executable demo, uncomment to run actual fine-tuning)
# trainer.train()

# 4. Recommendation Generation/Inference
print("\n--- Inference Example ---\n")

def generate_recommendation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

user_query_1 = "User previously bought 'Noise-cancelling headphones'. Recommend audio accessories:"
recommendation_1 = generate_recommendation(user_query_1)
print(f"User Query 1: {user_query_1}")
print(f"Recommendation 1: {recommendation_1.strip()}")

user_query_2 = "User viewed 'Stylish denim jeans'. Recommend clothing items:"
recommendation_2 = generate_recommendation(user_query_2)
print(f"User Query 2: {user_query_2}")
print(f"Recommendation 2: {recommendation_2.strip()}")
