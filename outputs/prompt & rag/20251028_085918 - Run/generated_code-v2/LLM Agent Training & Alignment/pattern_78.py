import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
import uvicorn
from fastapi import FastAPI
import gradio as gr


def prepare_medical_data():
    data = [
        {"context": "Patient has mild fever and cough.", "question": "What are common remedies for mild fever and cough?", "answer": "For mild fever and cough, rest, hydration, and over-the-counter medications like paracetamol can help. If symptoms worsen, consult a doctor."},
        {"context": "Patient is allergic to penicillin and has a skin rash.", "question": "What could be causing a rash if I'm allergic to penicillin?", "answer": "A rash could be due to many reasons, including other allergies, irritants, or skin conditions. Given your penicillin allergy, it's crucial to consult a doctor to identify the cause and ensure it's not related to another medication."},
        {"context": "Patient with diabetes asking about diet.", "question": "What kind of diet should I follow for type 2 diabetes?", "answer": "A balanced diet rich in whole grains, lean proteins, fruits, and vegetables is recommended for type 2 diabetes. Limiting processed foods, sugary drinks, and unhealthy fats is also important. Consult a dietitian for a personalized plan."},
        {"context": "Patient needs information about flu vaccination.", "question": "When is the best time to get a flu shot?", "answer": "The best time to get a flu shot is typically in the early fall, before flu activity begins to increase. However, it's never too late to get vaccinated during flu season."},
        {"context": "Patient experiencing stomach pain and nausea.", "question": "What could cause sudden stomach pain and nausea?", "answer": "Sudden stomach pain and nausea can be caused by various factors, including food poisoning, gastroenteritis, indigestion, or more serious conditions. If symptoms are severe or persist, seek medical attention."},
    ]
    df = pd.DataFrame(data)
    
    # Format data for SFTTrainer
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append({"text": f"### Context:\n{row['context']}\n### Question:\n{row['question']}\n### Answer:\n{row['answer']}"})
    
    return Dataset.from_pandas(pd.DataFrame(formatted_data))


def load_and_finetune_model(dataset):
    model_name = "NousResearch/Llama-2-7b-chat-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train()

    trainer.model.save_pretrained("./fine_tuned_llama_medical")
    tokenizer.save_pretrained("./fine_tuned_llama_medical")

    return model, tokenizer


class MedicalChatbot:
    def __init__(self):
        model_path = "./fine_tuned_llama_medical"
        base_model_name = "NousResearch/Llama-2-7b-chat-hf"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        self.model = get_peft_model(base_model, LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        ))
        self.model.load_state_dict(torch.load(f"{model_path}/adapter_model.bin"), strict=False)
        self.model.eval()

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate_response(self, query: str, patient_profile: str = "") -> str:
        full_prompt = f"### Context:\n{{patient_profile}}\n### Question:\n{{query}}\n### Answer:\n"
        
        sequences = self.pipeline(
            full_prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=512,
        )
        generated_text = sequences[0]["generated_text"]
        
        # Extract only the answer part
        answer_start = generated_text.find("### Answer:\n")
        if answer_start != -1:
            return generated_text[answer_start + len("### Answer:\n"):].strip()
        return generated_text.strip()


# --- FastAPI Application ---
app = FastAPI()
chatbot_instance = None

@app.on_event("startup")
async def startup_event():
    global chatbot_instance
    print("Loading and fine-tuning model (this may take some time for actual training, here it's symbolic)...")
    dataset = prepare_medical_data()
    _, _ = load_and_finetune_model(dataset) # Perform symbolic training and save adapter
    chatbot_instance = MedicalChatbot() # Load the fine-tuned model for inference
    print("Model loaded and ready.")

@app.post("/predict/")
async def predict(query: str, patient_profile: str = ""):
    if chatbot_instance is None:
        return {"error": "Model not loaded yet. Please wait.", "response": ""}
    response = chatbot_instance.generate_response(query, patient_profile)
    return {"response": response}


# --- Gradio Interface ---
def gradio_interface(query, patient_profile):
    import requests
    url = "http://127.0.0.1:8000/predict/"
    headers = {'Content-Type': 'application/json'}
    data = {"query": query, "patient_profile": patient_profile}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "Error: FastAPI backend is not running. Please ensure the server is started with 'uvicorn medical_chatbot_app:app --reload'."
    except requests.exceptions.RequestException as e:
        return f"Error during prediction: {e}"


if __name__ == "__main__":
    # This part needs to be run in two separate steps:
    # 1. Start the FastAPI server: uvicorn medical_chatbot_app:app --host 0.0.0.0 --port 8000
    # 2. Run the Gradio interface in a separate Python script or block after the FastAPI is up.
    # For a single file, we simulate this by running Gradio if a specific condition is met,
    # but typically, you'd run FastAPI as a service and Gradio as a client.
    print("To run this application:")
    print("1. Start the FastAPI server in one terminal: `uvicorn medical_chatbot_app:app --host 0.0.0.0 --port 8000`")
    print("2. After the FastAPI server is running, open another terminal and run this script directly to launch the Gradio UI.")

    # Only launch Gradio if this script is run directly and not within uvicorn (which imports it)
    # This is a heuristic and might not be robust in all deployment scenarios.
    if "__main__" == __name__ and not any("uvicorn." in arg for arg in sys.argv):
        import sys
        if not "gradio" in sys.modules:
             # This block will be executed when running the script directly, not via uvicorn
            print("Launching Gradio interface. Ensure FastAPI is running on http://127.0.0.1:8000")
            
            demo = gr.Interface(
                fn=gradio_interface,
                inputs=[
                    gr.Textbox(lines=2, label="Medical Question", placeholder="e.g., What are the side effects of XYZ medication?"),
                    gr.Textbox(lines=3, label="Patient Profile (Optional)", placeholder="e.g., 45-year-old female with history of hypertension and penicillin allergy.")
                ],
                outputs=gr.Textbox(label="Personalized Medical Information"),
                title="Personalized Medical Information Chatbot",
                description="Get personalized medical information based on your queries and profile. Disclaimer: This chatbot provides general information and should not replace professional medical advice."
            )
            demo.launch(share=False)

