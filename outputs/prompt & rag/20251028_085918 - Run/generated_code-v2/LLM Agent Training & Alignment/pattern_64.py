from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

# Mocking bitsandbytes, peft, and accelerate if not installed, for conceptual completeness
try:
    from peft import PeftModel, LoraConfig, get_peft_model
    _PEFT_INSTALLED = True
except ImportError:
    _PEFT_INSTALLED = False
    class PeftModel: pass
    class LoraConfig: pass
    def get_peft_model(*args, **kwargs): return args[0]

try:
    import accelerate
    _ACCELERATE_INSTALLED = True
except ImportError:
    _ACCELERATE_INSTALLED = False


class Product(BaseModel):
    id: str
    name: str
    description: str
    category: str

class ChatQuery(BaseModel):
    query: str
    user_id: str = None

app = FastAPI()

# --- Global Variables for Models and DBs ---
embedding_model: SentenceTransformer = None
product_collection = None
kb_collection = None
llm_model = None
llm_tokenizer = None

# --- Mock Data ---
mock_products_data = [
    {"id": "P001", "name": "Wireless Bluetooth Headphones", "description": "High-quality wireless headphones with noise-canceling feature and long battery life.", "category": "Electronics"},
    {"id": "P002", "name": "Ergonomic Office Chair", "description": "Comfortable office chair with adjustable lumbar support and breathable mesh.", "category": "Furniture"},
    {"id": "P003", "name": "Smartwatch with Heart Rate Monitor", "description": "Fitness tracker and smartwatch with GPS, heart rate, and sleep tracking.", "category": "Electronics"},
    {"id": "P004", "name": "Organic Green Tea", "description": "Premium organic green tea bags, rich in antioxidants.", "category": "Food & Beverages"},
    {"id": "P005", "name": "4K Ultra HD Smart TV", "description": "55-inch smart TV with stunning 4K resolution and built-in streaming apps.", "category": "Electronics"}
]

mock_faqs_data = [
    {"id": "F001", "question": "How do I track my order?", "answer": "You can track your order using the tracking number provided in your shipping confirmation email on our website's 'Track Order' page."},
    {"id": "F002", "question": "What is your return policy?", "answer": "We offer a 30-day return policy for most items, provided they are in their original condition. Please see our full return policy for details."},
    {"id": "F003", "question": "How can I contact customer support?", "answer": "You can contact customer support via live chat on our website, email at support@example.com, or by calling 1-800-123-4567."},
    {"id": "F004", "question": "Do you offer international shipping?", "answer": "Yes, we offer international shipping to many countries. Shipping costs and delivery times vary by destination."}
]

# --- LLM Placeholder/Mock --- 
class FineTunedLLMMock:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        # This simulates the fine-tuned aspect without actual fine-tuning here
        # In a real scenario, `model` would be the LoRA/QLoRA adapted model

    def generate_response(self, prompt: str, context: str = "") -> str:
        full_prompt = f"Given the following context: {context}\nBased on this, answer the question: {prompt}\nAnswer:"
        
        # For demonstration, a simple rule-based response combining context and prompt
        if "track order" in prompt.lower() and "tracking number" in context.lower():
            return "Please use the tracking number from your shipping confirmation email on our website's 'Track Order' page."
        if "return policy" in prompt.lower() and "30-day return" in context.lower():
            return "Our return policy allows returns within 30 days for most items in original condition. Refer to the full policy for details."
        if "contact support" in prompt.lower() and "live chat" in context.lower():
            return "You can reach customer support via live chat, email at support@example.com, or by calling 1-800-123-4567."
        if "recommend" in prompt.lower():
            if "headphones" in context.lower():
                return "Based on your interest, I recommend the Wireless Bluetooth Headphones with noise-canceling."
            if "office chair" in context.lower():
                return "For a comfortable workspace, consider the Ergonomic Office Chair with adjustable lumbar support."
            return f"I recommend checking out some of our best-selling items in {context}."
        return f"I'm sorry, I need more information or specific context to provide a better answer for '{prompt}'."


@app.on_event("startup")
async def startup_event():
    global embedding_model, product_collection, kb_collection, llm_model, llm_tokenizer

    # --- 1. Embedding Model Initialization ---
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- 2. ChromaDB Initialization ---
    client = chromadb.Client()
    product_collection = client.get_or_create_collection(name="products")
    kb_collection = client.get_or_create_collection(name="knowledge_base")

    # --- Populate ChromaDB with mock data ---
    # Products
    product_ids = [p["id"] for p in mock_products_data]
    product_descriptions = [f"{p['name']}: {p['description']}" for p in mock_products_data]
    product_embeddings = embedding_model.encode(product_descriptions).tolist()
    product_metadatas = mock_products_data

    if product_collection.count() == 0:
        product_collection.add(embeddings=product_embeddings, metadatas=product_metadatas, ids=product_ids)

    # Knowledge Base (FAQs)
    faq_ids = [f["id"] for f in mock_faqs_data]
    faq_questions_answers = [f"{f['question']} {f['answer']}" for f in mock_faqs_data]
    faq_embeddings = embedding_model.encode(faq_questions_answers).tolist()
    faq_metadatas = mock_faqs_data

    if kb_collection.count() == 0:
        kb_collection.add(embeddings=faq_embeddings, metadatas=faq_metadatas, ids=faq_ids)

    # --- 3. LLM Core & Efficient Fine-tuning (Conceptual Setup) ---
    # This part conceptually sets up an LLM, but doesn't perform actual fine-tuning
    # In a real scenario, `load_in_4bit=True` and LoRA config would be applied to a much larger model.
    
    model_name = "facebook/opt-125m" # A small model for demonstration purposes
    
    # Use CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # QLoRA specific configuration (conceptual, actual training is not performed here)
    bnb_config = None
    if device == "cuda": # Only apply quantization if running on GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    llm_model_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if bnb_config else None,
        torch_dtype=torch.bfloat16 if bnb_config else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    llm_model_base.eval()

    # Apply LoRA/QLoRA (conceptual): In a real app, this would be loading a PEFT adapted model
    if _PEFT_INSTALLED:
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16, 
            target_modules=["q_proj", "v_proj"], # Example target modules
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM"
        )
        llm_model = get_peft_model(llm_model_base, lora_config) # This just wraps the base model for a mock
    else:
        llm_model = llm_model_base

    llm_model = FineTunedLLMMock(llm_tokenizer, llm_model, device) # Wrap with our mock generator


# --- API Endpoints ---

@app.post("/recommendations/similar/{product_id}")
async def get_similar_products(product_id: str, num_results: int = 3):
    if product_collection is None:
        raise HTTPException(status_code=503, detail="Vector database not initialized")

    result = product_collection.get(ids=[product_id], include=['embeddings'])
    if not result['embeddings']:
        raise HTTPException(status_code=404, detail="Product not found.")
    
    query_embedding = result['embeddings'][0]
    
    # Query Chroma for similar products
    similar_products = product_collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results + 1, # +1 to exclude the product itself
        include=['metadatas']
    )
    
    recommendations = []
    for i, metadata in enumerate(similar_products['metadatas'][0]):
        if metadata['id'] != product_id:
            recommendations.append(Product(**metadata))
    
    return {"product_id": product_id, "recommendations": recommendations}


@app.post("/chatbot/query")
async def chat_with_bot(chat_query: ChatQuery):
    if kb_collection is None or llm_model is None or embedding_model is None:
        raise HTTPException(status_code=503, detail="Chatbot services not initialized")

    # Retrieve context from knowledge base
    query_embedding = embedding_model.encode(chat_query.query).tolist()
    retrieved_docs = kb_collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=['metadatas']
    )
    
    context = ""
    if retrieved_docs['metadatas'] and retrieved_docs['metadatas'][0]:
        context_metadata = retrieved_docs['metadatas'][0][0]
        context = f"Question: {context_metadata.get('question', '')} Answer: {context_metadata.get('answer', '')}"

    # Generate response using the (mock) fine-tuned LLM
    response = llm_model.generate_response(chat_query.query, context)
    
    return {"user_query": chat_query.query, "bot_response": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
