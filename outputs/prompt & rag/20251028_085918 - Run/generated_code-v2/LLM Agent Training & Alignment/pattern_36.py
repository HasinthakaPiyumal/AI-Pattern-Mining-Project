
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# 1. Data Preparation
# Synthetic E-commerce data
synthetic_data = [
    {"query": "I'm looking for a comfortable pair of running shoes.", "recommendation": "Product: Nike Air Zoom Pegasus 38, known for its cushioning and responsive design. Ideal for daily runs."},
    {"query": "Suggest a good book on artificial intelligence for beginners.", "recommendation": "Product: 'AI Superpowers: China, Silicon Valley, and the New World Order' by Kai-Fu Lee. A great introduction to AI's impact and future."},
    {"query": "What's a durable smartphone with a good camera?", "recommendation": "Product: Samsung Galaxy S23 Ultra, features a robust build and a versatile camera system with high-resolution sensors."},
    {"query": "I need a laptop for programming and occasional gaming.", "recommendation": "Product: Dell XPS 15, offers powerful processors and dedicated graphics options, suitable for both coding and light gaming."},
    {"query": "Find me a stylish watch for daily wear.", "recommendation": "Product: Fossil Gen 6 Smartwatch, combines classic design with smart features for everyday use."}
]

# Convert to datasets.Dataset
dataset = Dataset.from_list(synthetic_data)

# 2. Base LLM and Tokenizer Loading
model_name = "facebook/opt-125m"  # Using a small LLM for demonstration
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token to eos_token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Format input for causal language modeling: "query -> recommendation"
    texts = [f"{ex['query']} -> {ex['recommendation']}{tokenizer.eos_token}" for ex in examples]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# 3. Efficient Fine-tuning (LoRA) Configuration
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # LoRA attention dimension
    lora_alpha=16,  # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none",
    task_type="CAUSAL_LM", # Causal Language Modeling
)

model = get_peft_model(model, lora_config)

# 4. Training Setup
training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",
    per_device_train_batch_size=2, # Small batch size for demonstration
    gradient_accumulation_steps=1, # No accumulation for simplicity
    warmup_steps=10, # Minimal warmup steps
    max_steps=50, # Very few steps for quick demonstration
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(), # Enable mixed precision if CUDA is available
    logging_steps=10,
    save_steps=50,
    overwrite_output_dir=True,
    report_to="none" # Disable reporting to external services
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 5. Fine-tuning Execution
trainer.train()

# Save the fine-tuned LoRA adapters
model.save_pretrained("./lora_finetuned_adapters")

# 6. Inference/Recommendation Function
def get_product_recommendation(user_query, model_path="./lora_finetuned_adapters", base_model_name="facebook/opt-125m"):
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    inference_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if inference_tokenizer.pad_token is None:
        inference_tokenizer.pad_token = inference_tokenizer.eos_token

    # Load LoRA adapters
    from peft import PeftModel
    model_to_infer = PeftModel.from_pretrained(base_model, model_path)
    model_to_infer.eval()

    # Prepare input
    input_text = f"{user_query} -> "
    inputs = inference_tokenizer(input_text, return_tensors="pt")

    # Generate recommendation
    with torch.no_grad():
        outputs = model_to_infer.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50, # Generate up to 50 new tokens
            num_return_sequences=1,
            pad_token_id=inference_tokenizer.eos_token_id, # Ensure pad_token_id is set
        )

    decoded_output = inference_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the recommendation part after the '->'
    recommendation_start = decoded_output.find(input_text)
    if recommendation_start != -1:
        extracted_recommendation = decoded_output[recommendation_start + len(input_text):].strip()
        # Further refine to remove potential remnants of other inputs if present after EOS token
        eos_index = extracted_recommendation.find(inference_tokenizer.eos_token)
        if eos_index != -1:
            extracted_recommendation = extracted_recommendation[:eos_index].strip()
        return extracted_recommendation
    return "No recommendation found."

# Example usage of the inference function
if __name__ == "__main__":
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete. LoRA adapters saved.")

    user_input = "I need a new pair of headphones for working out."
    recommendation = get_product_recommendation(user_input)
    print(f"\nUser Query: {user_input}")
    print(f"Recommendation: {recommendation}")

    user_input_2 = "Suggest a thrilling mystery novel."
    recommendation_2 = get_product_recommendation(user_input_2)
    print(f"\nUser Query: {user_input_2}")
    print(f"Recommendation: {recommendation_2}")
