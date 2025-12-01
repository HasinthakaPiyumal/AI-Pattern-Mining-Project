import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# 1. Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
# Use a smaller model if Mistral-7B is too large for the environment
# MODEL_NAME = "facebook/opt-125m"

# 2. Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Ensure tokenizer has a pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

# 3. Prepare Dummy Medical Dataset
# In a real-world scenario, this would be a large dataset of specialized medical texts
medical_data = [
    "Patient presents with severe abdominal pain and fever, suspecting appendicitis. Medical history includes hypertension.",
    "Diagnosis: Type 2 Diabetes Mellitus. Prescribed Metformin. Advised lifestyle changes.",
    "Recent studies indicate that drug X significantly reduces symptoms of condition Y in elderly patients.",
    "Differential diagnosis for chest pain includes angina, myocardial infarction, and pericarditis.",
    "Summarize patient's visit: 65-year-old male, shortness of breath, elevated troponin levels, admitted for observation."
]

# Convert text to tokenized input IDs
def tokenize_function(examples):
    return tokenizer(examples, truncation=True, max_length=128)

# Create a dummy Dataset object
tokenized_medical_data = tokenize_function(medical_data)

dummy_dataset = Dataset.from_dict({
    'input_ids': tokenized_medical_data['input_ids'],
    'attention_mask': tokenized_medical_data['attention_mask'],
    'labels': tokenized_medical_data['input_ids'].copy() # For causal LM, labels are typically input_ids
})

# 4. LoRA Configuration
lora_config = LoraConfig(
    r=8,  # LoRA attention dimension
    lora_alpha=16, # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "v_proj"], # Modules to apply LoRA to
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none", # Bias type
    task_type="CAUSAL_LM", # Task type for causal language modeling
)

# 5. Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./lora_finetuned_medical_llm",
    per_device_train_batch_size=2, # Small batch size for demonstration
    gradient_accumulation_steps=1, # Accumulate gradients for effective larger batch size
    warmup_steps=10, # Number of warmup steps for learning rate scheduler
    max_steps=50, # Small number of steps for demonstration
    learning_rate=2e-4, # Learning rate
    fp16=False, # Use bfloat16 for Mistral, if available and supported by GPU
    bf16=True, 
    logging_steps=10,
    optim="paged_adamw_8bit", # Optimized AdamW for memory efficiency
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dummy_dataset,
    tokenizer=tokenizer,
)

# 8. Fine-tune the model
trainer.train()

# 9. Save the LoRA adapters
trainer.model.save_pretrained("./lora_finetuned_medical_llm_adapters")

print("LoRA fine-tuning complete. Adapters saved to ./lora_finetuned_medical_llm_adapters")
print("You can load these adapters and merge them with the base model for inference.")

# Example of loading and merging adapters for inference (optional)
# from peft import PeftModel, PeftConfig

# peft_model_id = "./lora_finetuned_medical_llm_adapters"
# config = PeftConfig.from_pretrained(peft_model_id)
# base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16)
# model = PeftModel.from_pretrained(base_model, peft_model_id)
# model.eval()

# prompt = "Patient needs information about diabetes management:"
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda") # Move to GPU if available
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
