import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

# --- Configuration Constants ---
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Example LLM, requires Hugging Face authentication or local download
DATA_PATH = "./simulated_user_data.csv"
LORA_ADAPTER_PATH = "./lora_adapters"

# --- 1. Data Ingestion and Preprocessing (Simulated) ---
def generate_simulated_data(num_samples=100):
    data = {
        "user_id": [i % 20 for i in range(num_samples)],
        "browsed_category": ["Electronics", "Books", "Clothing", "Home Goods", "Sports"] * (num_samples // 5),
        "liked_products": [
            "Smartphone", "Laptop", "Novel", "T-shirt", "Blender",
            "Running Shoes", "Smartwatch", "Cookbook", "Jeans", "Vacuum Cleaner"
        ] * (num_samples // 10),
        "prompt": ["" for _ in range(num_samples)],
        "completion": ["" for _ in range(num_samples)]
    }
    df = pd.DataFrame(data)

    for i in range(num_samples):
        user_id = df.loc[i, "user_id"]
        browsed = df.loc[i, "browsed_category"]
        liked = df.loc[i, "liked_products"]
        df.loc[i, "prompt"] = f"User {user_id} browsed {browsed} and liked {liked}. Recommend new products: "
        # Simulate a completion/recommendation for fine-tuning
        df.loc[i, "completion"] = f"Product A, Product B, Product C"

    return df

# Generate and save simulated data
simulated_df = generate_simulated_data()
simulated_df.to_csv(DATA_PATH, index=False)

# Load data and prepare for LLM
df = pd.read_csv(DATA_PATH)
dataset = Dataset.from_pandas(df)

# --- 2. Base LLM Selection and Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token # Or use a specific pad token if available

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# --- 3. Efficient Fine-tuning Module (QLoRA) ---
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# --- 4. Fine-tuning Process (using SFTTrainer) ---
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # Reduced for demonstration
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=100, # Save less frequently for demo
    logging_steps=10, # Log less frequently for demo
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1, # Train for num_train_epochs
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none"
)

def formatting_prompts_func(examples):
    prompts = [p + c + tokenizer.eos_token for p, c in zip(examples["prompt"], examples["completion"])]
    return {"text": prompts}

# Apply the formatting function to the dataset
formatted_dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=list(dataset.features))

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,
)

# Commented out actual training for immediate API setup. Uncomment to fine-tune.
# trainer.train()

# Save LoRA adapters after (simulated) training
# trainer.model.save_pretrained(LORA_ADAPTER_PATH)

# --- 5. Inference and Recommendation API (FastAPI) ---

app = FastAPI()

# Load model for inference (assuming adapters are saved and loaded)
def load_inference_model():
    if not os.path.exists(LORA_ADAPTER_PATH):
        # Fallback for demonstration if adapters are not trained/saved
        print("LoRA adapters not found. Using base model for inference. Please run fine-tuning first if you want custom recommendations.")
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
        return base_model, AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Loading fine-tuned model from {LORA_ADAPTER_PATH}")
    # Re-initialize the base model with quantization config
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    # Load the trained LoRA adapters
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# This will be called once when the FastAPI app starts
# model_inference, tokenizer_inference = load_inference_model()

# Placeholder for loaded model and tokenizer (replace with actual loading in a real app)
# For this example, we'll assume a global model_inference and tokenizer_inference that gets loaded
# if the adapters exist. If not, the `generate_recommendations` function will just use a base model logic.
model_inference = None
tokenizer_inference = None

@app.on_event("startup")
async def startup_event():
    global model_inference, tokenizer_inference
    model_inference, tokenizer_inference = load_inference_model()
    model_inference.eval()

class RecommendationRequest(BaseModel):
    user_id: int
    browsed_category: str
    liked_products: str

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    input_text = f"User {request.user_id} browsed {request.browsed_category} and liked {request.liked_products}. Recommend new products: "
    
    if model_inference is None or tokenizer_inference is None:
        return {"error": "Model not loaded. Please ensure adapters exist or fine-tuning is run.", "recommendations": []}

    inputs = tokenizer_inference(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model_inference.device)
    
    with torch.no_grad():
        outputs = model_inference.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=1, # Simple greedy decoding for demo
            do_sample=True, # Allow sampling for varied output
            temperature=0.7,
            top_k=50,
            no_repeat_ngram_size=2, # Prevent repetitive phrases
            eos_token_id=tokenizer_inference.eos_token_id
        )
    
    # Decode the output, skipping the input prompt part
    generated_text = tokenizer_inference.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    # Simple parsing of recommendations (can be improved based on LLM output format)
    recommendations = [item.strip() for item in generated_text.split(",") if item.strip()]

    return {"user_id": request.user_id, "recommendations": recommendations}

if __name__ == "__main__":
    # To run the API:
    # uvicorn recommendation_engine:app --host 0.0.0.0 --port 8000
    print("To run the API, use: uvicorn recommendation_engine:app --host 0.0.0.0 --port 8000")
    print("Note: Actual fine-tuning is commented out. Uncomment trainer.train() and ensure MODEL_NAME points to a valid LLM and you have necessary authentication/local files.")
    print(f"Simulated data saved to {DATA_PATH}")

    # Example of how to run fine-tuning if desired (uncomment trainer.train() above first)
    # from peft import PeftModel
    # print("Starting simulated fine-tuning process...")
    # trainer.train()
    # trainer.model.save_pretrained(LORA_ADAPTER_PATH)
    # print(f"LoRA adapters saved to {LORA_ADAPTER_PATH}")

    # Small test for data generation
    # df_test = generate_simulated_data(num_samples=5)
    # print(df_test[["prompt", "completion"]].iloc[0].to_dict())

