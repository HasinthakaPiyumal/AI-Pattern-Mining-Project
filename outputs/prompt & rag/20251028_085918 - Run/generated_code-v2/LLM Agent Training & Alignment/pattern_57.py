import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

# --- Configuration --- #
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
OUTPUT_DIR = "./fine_tuned_llm"

# --- FastAPI App Initialization --- #
app = FastAPI()

# --- Global variables for model and tokenizer --- #
tokenizer = None
model = None

# --- Data Preprocessing (Dummy Data for Demonstration) --- #
def generate_dummy_data():
    data = [
        {"prompt": "What should I buy based on my last purchase: a smart watch?", "completion": "You might like wireless earbuds or a fitness tracker."}, 
        {"prompt": "I need a new laptop. My budget is around $1000 and I prefer a lightweight model.", "completion": "Consider the Dell XPS 13 or MacBook Air M1/M2."}, 
        {"prompt": "Show me some summer dresses.", "completion": "Check out our floral maxi dresses or linen sundresses."}, 
        {"prompt": "I'm looking for a gift for a tech enthusiast. Budget is flexible.", "completion": "How about a VR headset, a drone, or a smart home device?"},
        {"prompt": "My recent purchase was a gaming console. What accessories do you recommend?", "completion": "You might enjoy a high-refresh-rate monitor, a gaming headset, or an extra controller."},
        {"prompt": "I want to redecorate my living room. Any furniture recommendations?", "completion": "Explore our modern sofas, minimalist coffee tables, or accent chairs."},
        {"prompt": "I bought running shoes. What else do I need for my fitness journey?", "completion": "Consider moisture-wicking activewear, a smart water bottle, or a GPS running watch."}
    ]
    df = pd.DataFrame(data)
    df["text"] = df.apply(lambda row: f"### Human: {row['prompt']}\n### Assistant: {row['completion']}", axis=1)
    return Dataset.from_pandas(df)

# --- Model Loading and LoRA Setup --- #
def load_base_model_and_tokenizer():
    global tokenizer, model
    
    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Common target modules
    )

    model = get_peft_model(model, peft_config)
    print(f"Model loaded and prepared for LoRA fine-tuning. Trainable parameters: {model.print_trainable_parameters()}")

# --- Fine-tuning Function --- #
def fine_tune_model(train_dataset):
    global model, tokenizer
    
    if model is None or tokenizer is None:
        load_base_model_and_tokenizer()

    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1, # For demonstration, use a small number of epochs
        per_device_train_batch_size=2, # Reduce batch size for memory
        gradient_accumulation_steps=1, # Adjust based on GPU memory
        optim="paged_adamw_32bit",
        save_steps=10, # Save checkpoints frequently
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True, # Use float16 for faster training
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1, # Train for num_train_epochs
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none", # wandb can be enabled here by setting 'wandb'
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=LoraConfig(
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            r=LORA_R,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        ),
        tokenizer=tokenizer,
        args=training_arguments,
        dataset_text_field="text",
        max_seq_length=512,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete. Saving adapter model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")

# --- Recommendation Generation --- #
class RecommendationRequest(BaseModel):
    user_context: str

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    global model, tokenizer

    if model is None or tokenizer is None:
        # Attempt to load fine-tuned model first, then base if not found
        if os.path.exists(OUTPUT_DIR):
            print(f"Loading fine-tuned model from {OUTPUT_DIR}")
            # Load base model first with bnb_config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            
            # Load LoRA adapters
            model = get_peft_model(base_model, LoraConfig.from_pretrained(OUTPUT_DIR))
            print("Fine-tuned model loaded successfully.")
        else:
            print("Fine-tuned model not found. Loading base model (without adapters).")
            load_base_model_and_tokenizer() # This will load the base and apply new LoRA config if not trained

    if model is None or tokenizer is None:
        return {"error": "Model not loaded. Please ensure the model is fine-tuned or manually loaded."}

    prompt = f"### Human: {request.user_context}\n### Assistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda") # Ensure input is on GPU

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100, 
            do_sample=True, 
            top_p=0.05, # Reduce top_p for more focused recommendations
            temperature=0.7, 
            eos_token_id=tokenizer.eos_token_id
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    assistant_start = response_text.find("### Assistant:")
    if assistant_start != -1:
        recommendation = response_text[assistant_start + len("### Assistant:"):].strip()
    else:
        recommendation = response_text # Fallback if structure is not as expected

    return {"recommendations": recommendation}

# --- API Endpoint to Trigger Fine-tuning --- #
@app.post("/train_model")
async def trigger_fine_tuning():
    global model, tokenizer
    if model is not None and tokenizer is not None and os.path.exists(OUTPUT_DIR):
        return {"message": "Model already fine-tuned and loaded."}
    
    print("Triggering model fine-tuning...")
    train_dataset = generate_dummy_data()
    fine_tune_model(train_dataset)
    return {"message": "Fine-tuning initiated. Check logs for progress. Model will be saved after completion."}

# --- Main Entry Point --- #
if __name__ == "__main__":
    # Optional: Initial model loading or fine-tuning upon startup
    # For a real application, fine-tuning would be a separate, scheduled job
    # and the FastAPI app would only load the *already fine-tuned* model.
    
    # To avoid fine-tuning on every startup, we'll try to load existing adapters.
    # If adapters exist, load them. Otherwise, load the base model.
    if os.path.exists(OUTPUT_DIR):
        print(f"Found existing fine-tuned model at {OUTPUT_DIR}. Loading adapters...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = get_peft_model(base_model, LoraConfig.from_pretrained(OUTPUT_DIR))
        model.eval() # Set model to evaluation mode for inference
        print("Fine-tuned model adapters loaded successfully.")
    else:
        print("No fine-tuned model found. The base model will be loaded upon the first recommendation request or explicit training trigger.")

    # Start the FastAPI server
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
