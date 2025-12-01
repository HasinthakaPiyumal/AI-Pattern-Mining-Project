import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import torch
import gradio as gr
import os

# --- 1. Data Preparation and Curation Module (Simulated) ---
def prepare_data():
    # Simulate a small dataset of rare disease Q&A pairs
    data = [
        {"question": "What are the common symptoms of Fabry disease?", "answer": "Fabry disease often presents with acroparesthesias, angiokeratomas, hypohidrosis, corneal verticillata, and gastrointestinal issues. Renal and cardiac complications develop later."}, 
        {"question": "How is Huntington's disease diagnosed?", "answer": "Huntington's disease is diagnosed primarily through genetic testing to detect the expanded CAG repeat in the HTT gene. Clinical evaluation of motor, cognitive, and psychiatric symptoms is also crucial."}, 
        {"question": "What are the treatment options for Cystic Fibrosis?", "answer": "Treatment for Cystic Fibrosis focuses on managing symptoms and preventing complications. It includes airway clearance techniques, mucolytics, bronchodilators, antibiotics for infections, pancreatic enzyme replacement, and CFTR modulator therapies."}, 
        {"question": "Tell me about Duchenne Muscular Dystrophy inheritance pattern.", "answer": "Duchenne Muscular Dystrophy (DMD) is an X-linked recessive disorder. It primarily affects males, who inherit the mutated gene from their mothers. Females are typically carriers and usually asymptomatic."}
    ]
    
    df = pd.DataFrame(data)
    
    # Format for SFTTrainer
    # SFTTrainer can take a column directly for text, or format as prompts
    # For this example, we'll create a 'text' column for instruction-like finetuning
    df['text'] = df.apply(lambda row: f"### Question: {row['question']}\n### Answer: {row['answer']}", axis=1)
    
    # Convert pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    print("\n--- Data Preparation Complete ---")
    print(f"Dataset size: {len(dataset)} examples")
    print("Sample data point:")
    print(dataset[0]['text'])
    
    return dataset

# --- 2. LLM Finetuning Module (Domain-Specific Finetuning - DSF) ---
def finetune_llm(dataset, model_output_dir="./rare_disease_llm"):
    model_name = "gpt2"  # Using a smaller model for demonstration

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # gpt2 does not have a pad token by default
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=2,  # Small batch size for demo
        num_train_epochs=3,             # Few epochs for demo
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,                 # Save checkpoint every 100 steps
        save_total_limit=2,             # Only keep the last 2 checkpoints
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        gradient_accumulation_steps=1,
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text", # Column in dataset containing text for training
        args=training_args,
        max_seq_length=512,        # Max sequence length for inputs
    )

    print("\n--- Starting LLM Finetuning ---")
    # Start training
    trainer.train()

    # Save the finetuned model and tokenizer
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"\n--- Finetuning Complete. Model saved to {model_output_dir} ---")
    
    return model_output_dir

# --- 3. Inference and Deployment Module (Clinical Decision Support Interface) ---
def run_inference_app(model_path):
    print("\n--- Loading Finetuned Model for Inference ---")
    # Load the finetuned model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token # Ensure pad_token is set for generation
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval() # Set model to evaluation mode
        if torch.cuda.is_available():
            model.to("cuda")
        print(f"Model and tokenizer loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure the model has been finetuned and saved correctly.")
        return

    def predict(query):
        # Format the query as expected by the finetuned model
        formatted_query = f"### Question: {query}\n### Answer:"
        
        inputs = tokenizer(formatted_query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=200, # Max length for the generated answer
                num_return_sequences=1,
                do_sample=True,     # Use sampling for more diverse answers
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Post-process to extract only the answer part
        # The model might repeat the prompt, so we look for the 