from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from vllm import LLM, SamplingParams
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
import uvicorn


app = FastAPI()

# --- LLM Inference Server (vLLM) Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
llm = LLM(model=MODEL_NAME)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

# --- RAG System Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Initialize ChromaDB (in-memory for this example, persist_directory for production)
# For a single file demonstration, we'll keep it simple without persistence
chroma_db = Chroma(embedding_function=embeddings)

def ingest_documents():
    documents = [
        Document(page_content="Our return policy allows returns within 30 days of purchase with a valid receipt.", metadata={"source": "policy"}),
        Document(page_content="To reset your password, visit the login page and click on 'Forgot Password'. Follow the instructions sent to your registered email.", metadata={"source": "faq"}),
        Document(page_content="Product X features a 10-hour battery life and a 13-inch Retina display.", metadata={"source": "product_x_spec"}),
        Document(page_content="Shipping typically takes 3-5 business days for domestic orders.", metadata={"source": "shipping_info"}),
        Document(page_content="You can contact customer support via live chat on our website or by calling 1-800-555-0199 during business hours.", metadata={"source": "contact"}),
    ]
    chroma_db.add_documents(documents)
ingest_documents()

# --- Chatbot API Models ---
class ChatRequest(BaseModel):
    user_message: str
    conversation_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    bot_message: str
    conversation_history: List[Dict[str, str]]

# --- Chatbot API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    conversation_history = request.conversation_history
    user_message = request.user_message

    # RAG Retrieval
    retrieved_docs = chroma_db.similarity_search(user_message, k=2)
    context_str = "\n".join([doc.page_content for doc in retrieved_docs])

    # Prompt Construction
    system_prompt = (
        "You are a helpful customer support assistant for a large enterprise. "
        "Answer questions concisely and accurately based on the provided context or conversation history. "
        "If you don't know the answer, state that you don't have enough information."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": f"Context: {context_str}\n\nUser: {user_message}"})

    # Format messages for Mistral-like instruct models
    # This is a basic example; for complex cases, use transformers tokenizer chat template
    prompt = """
{% for message in messages %}
{% if message['role'] == 'system' %}
{{ message['content'] }}
{% elif message['role'] == 'user' %}
[INST] {{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}
{{ message['content'] }}
{% endif %}
{% endfor %}
Assistant: """.format(messages=messages)

    try:
        outputs = llm.generate(prompt, sampling_params)
        bot_message = outputs[0].outputs[0].text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM inference failed: {e}")

    # Update conversation history
    new_conversation_history = conversation_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": bot_message}
    ]

    return ChatResponse(bot_message=bot_message, conversation_history=new_conversation_history)

if __name__ == "__main__":
    # To run this, you need a vLLM server running or configured to load the model locally.
    # For local execution without a separate vLLM server, ensure your environment can load the LLM.
    # This script assumes vLLM can load the model directly based on its configuration.
    uvicorn.run(app, host="0.0.0.0", port=8000)
