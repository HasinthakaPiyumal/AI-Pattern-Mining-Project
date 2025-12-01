
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import time

# --- 1. Data Preparation and Augmentation (Mock) ---
# This is a highly simplified mock of a healthcare Q&A dataset
sample_qa_data = [
    {"question": "What are the symptoms of the common cold?", "answer": "Symptoms include runny nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, and low-grade fever."},
    {"question": "How to treat a minor cut?", "answer": "Clean the wound with mild soap and water, apply an antiseptic, and cover it with a sterile bandage."},
    {"question": "What is diabetes?", "answer": "Diabetes is a chronic condition that affects how your body turns food into energy. It results in too much sugar in the blood."},
    {"question": "Can you prescribe medication for my headache?", "answer": "As an AI, I cannot provide medical diagnoses or prescribe medication. Please consult a healthcare professional."},
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "Tell me about complex neurological diseases.", "answer": "This is a complex medical query that requires professional medical advice. Please consult a neurologist for accurate information and diagnosis."},
    {"question": "How do I schedule an appointment with Dr. Smith?", "answer": "Please visit our website's 'Appointments' section or call our clinic at [Phone Number] during business hours."},
]

def prepare_finetuning_dataset(data: list, abstention_ratio: float = 0.2) -> list:
    augmented_data = []
    num_to_abstain = int(len(data) * abstention_ratio)
    abstention_indices = random.sample(range(len(data)), num_to_abstain)

    for i, item in enumerate(data):
        if i in abstention_indices:
            # Simulate replacing ground truth with 'I don't know' for finetuning
            augmented_data.append({"question": item["question"], "answer": "I don't know. Please consult a healthcare professional for specific medical advice."})
        else:
            augmented_data.append(item)
    print(f"Mock: Prepared a finetuning dataset with {num_to_abstain} abstention examples.")
    return augmented_data

# --- 2. LLM Finetuning Module (Mock) ---
class MockTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": [list(map(ord, text))], "attention_mask": [[1]*len(text)]}

class MockModel:
    def __call__(self, input_ids, attention_mask):
        # Mock model output: simple concatenation or a placeholder response
        return "Mocked LLM Response based on input."

def load_finetuned_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", lora_adapter_path: str = None):
    print(f"Mock: Loading finetuned LLM '{model_name}' with LoRA adapter '{lora_adapter_path}'...")
    # In a real scenario, this would load a Hugging Face model and tokenizer
    # e.g., from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if lora_adapter_path: model = PeftModel.from_pretrained(model, lora_adapter_path)

    # Return mock objects instead
    return MockModel(), MockTokenizer()

# --- 3. Retrieval Augmented Generation (RAG) Module (Mock) ---
medical_knowledge_base = {
    "common cold": "The common cold is a viral infection of your nose and throat. Symptoms often include runny nose, sore throat, cough, and congestion. Treatment usually involves rest and over-the-counter remedies.",
    "minor cut treatment": "For a minor cut, clean the wound with mild soap and water. Apply gentle pressure with a clean cloth or bandage for a few minutes to stop bleeding. Then, apply an antiseptic and cover with a sterile bandage.",
    "diabetes": "Diabetes is a group of diseases that result in too much sugar in the blood (high blood glucose). There are several types, including Type 1, Type 2, and gestational diabetes. Management involves diet, exercise, and often medication.",
    "appointment scheduling": "To schedule an appointment, please visit our official clinic website, navigate to the 'Appointments' section, or call our dedicated scheduling line during business hours. Have your patient ID ready."
}

def retrieve_context(query: str) -> str:
    print(f"Mock: Retrieving context for query: '{query}'")
    # Simulate semantic search in a vector DB
    query_lower = query.lower()
    for keyword, context in medical_knowledge_base.items():
        if any(k in query_lower for k in keyword.split()):
            return context
    return "No relevant context found in knowledge base."

# --- LLM Inference with Controlled Abstention (Mock) ---
def generate_response(query: str, context: str, model, tokenizer) -> str:
    print(f"Mock: Generating response for query: '{query}' with context: '{context[:50]}...' ")

    # Abstention logic based on keywords or lack of context
    abstention_keywords = ["prescribe", "diagnose", "medical advice", "my symptoms", "am I sick", "what should I do"]
    if any(keyword in query.lower() for keyword in abstention_keywords) or \
       "no relevant context found" in context.lower() or \
       len(context) < 100: # Heuristic for insufficient context
        return "I'm sorry, as an AI, I cannot provide medical diagnoses, treatment recommendations, or prescribe medication. Please consult a qualified healthcare professional for personalized medical advice."

    # Simulate LLM generating a response using context
    if "common cold" in query.lower():
        return f"Based on the information, {context.lower().replace('the common cold is', 'A common cold is')}"
    elif "minor cut" in query.lower():
        return f"For a minor cut: {context}"
    elif "diabetes" in query.lower():
        return f"Regarding diabetes: {context}"
    elif "appointment" in query.lower():
        return f"To schedule an appointment: {context}"
    elif "capital of france" in query.lower():
         return "The capital of France is Paris."
    else:
        return "I can provide general health information. If your query is urgent or requires medical diagnosis, please contact a doctor."

# --- 4. Inference and Chatbot API Module ---

app = FastAPI()

# Global model and tokenizer (loaded once at startup)
finetuned_llm_model, finetuned_llm_tokenizer = None, None

@app.on_event("startup")
async def startup_event():
    global finetuned_llm_model, finetuned_llm_tokenizer
    print("Application startup: Loading models...")
    # Prepare a dummy dataset for finetuning simulation
    _ = prepare_finetuning_dataset(sample_qa_data)

    # Load the finetuned LLM (mocked)
    finetuned_llm_model, finetuned_llm_tokenizer = load_finetuned_llm(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        lora_adapter_path="./mock_lora_adapters" # Placeholder path
    )
    print("Models loaded successfully.")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    context_used: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_query = request.query
    print(f"Received query: {user_query}")

    # Step 1: Retrieve context using RAG (mocked)
    retrieved_context = retrieve_context(user_query)

    # Step 2: Generate response using the finetuned LLM with controlled abstention (mocked)
    chatbot_response = generate_response(user_query, retrieved_context, finetuned_llm_model, finetuned_llm_tokenizer)

    return ChatResponse(response=chatbot_response, context_used=retrieved_context)

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": finetuned_llm_model is not None}

# To run this application:
# 1. Save the code as healthcare_chatbot.py
# 2. Install dependencies: pip install fastapi uvicorn pydantic
# 3. Run from your terminal: uvicorn healthcare_chatbot:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs

# --- 5. MLOps (Conceptual Mentions in Architecture) ---
# Real MLOps integration would involve:
# - Using wandb for logging finetuning metrics.
# - Implementing custom evaluation scripts for abstention rate, hallucination rate.
# - Setting up Prometheus/Grafana for monitoring production API and model metrics.
# - Containerizing with Docker and deploying with Kubernetes or BentoML for scalable serving.
