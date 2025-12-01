import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import requests

# 1. Configuration (`config.py` equivalent)
load_dotenv()

class Config:
    VLLM_API_URL: str = os.getenv("VLLM_API_URL", "http://localhost:8000/generate")
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

config = Config()

# 2. Data Models (`models.py` equivalent)
class ChatRequest(BaseModel):
    session_id: str = Field(..., example="user_123")
    message: str = Field(..., example="What is your return policy?")

class ChatResponse(BaseModel):
    session_id: str
    response: str

# 3. RAG Manager (`rag_manager.py` equivalent)
class RAGManager:
    def __init__(self, db_path: str, embedding_model_name: str):
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_collection(name="customer_support")
        except:
            self.collection = self.client.create_collection(name="customer_support")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        # Add some dummy documents for demonstration
        if self.collection.count() == 0:
            documents = [
                "Our return policy allows returns within 30 days of purchase with a valid receipt.",
                "Shipping usually takes 5-7 business days for standard delivery.",
                "You can track your order using the tracking number provided in your shipping confirmation email.",
                "For technical support, please visit our online help center or call us during business hours.",
                "We offer a 1-year warranty on all electronic products."
            ]
            metadatas = [
                {"source": "policy"},
                {"source": "shipping"},
                {"source": "tracking"},
                {"source": "support"},
                {"source": "warranty"}
            ]
            ids = [f"doc_{i}" for i in range(len(documents))]
            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
            print(f"Initialized ChromaDB with {len(documents)} documents.")

    def retrieve_documents(self, query: str, n_results: int = 2) -> list[str]:
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents']
        )
        return results['documents'][0] if results and results['documents'] else []

# 4. LLM Inference Service Client (`kv_cache_llm_client.py` equivalent)
class KVCCacheLLMClient:
    def __init__(self, vllm_api_url: str, llm_model_name: str):
        self.vllm_api_url = vllm_api_url
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        # In a real scenario, this would send a request to the vLLM server.
        # For this mock, we'll simulate a response.
        # The prompt is constructed to leverage vLLM's KV cache reuse internally.

        headers = {"Content-Type": "application/json"}
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stop": ["\nUser:", "\nAgent:"] # Example stop sequences for chat
        }

        try:
            # Mocking vLLM response for demonstration. In a real setup, vLLM would process.
            # response = requests.post(self.vllm_api_url, headers=headers, json=payload)
            # response.raise_for_status()
            # return response.json()['text'][0][len(prompt):].strip()

            # Simple mock response: just append a canned answer
            mock_llm_response = f"Hello! I am a smart customer support agent. How can I assist you today?"
            if "return policy" in prompt.lower():
                mock_llm_response = "Our return policy allows returns within 30 days of purchase with a valid receipt."
            elif "shipping" in prompt.lower():
                mock_llm_response = "Shipping typically takes 5-7 business days. You can track your order online."
            elif "warranty" in prompt.lower():
                mock_llm_response = "We offer a 1-year warranty on all electronic products."
            elif "hello" in prompt.lower() or "hi" in prompt.lower():
                mock_llm_response = "Hello! How can I help you today?"
            
            # Simulate KV cache reuse by just returning the generated part for the new query
            # This part would be handled by vLLM internally based on the full prompt
            return mock_llm_response

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to vLLM server: {e}")
            return "I am sorry, but I am currently unable to process your request. Please try again later."


# 5. Backend Server (`main.py`)
app = FastAPI()

# In-memory storage for conversation history (for demonstration)
conversation_history = {}

rag_manager = RAGManager(config.CHROMA_DB_PATH, config.EMBEDDING_MODEL_NAME)
llm_client = KVCCacheLLMClient(config.VLLM_API_URL, config.LLM_MODEL_NAME)

SYSTEM_INSTRUCTIONS = "You are a helpful customer support agent. Provide concise and accurate answers based on the provided context. If you don't know the answer, politely state that you cannot help with that specific query."

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message

    # Retrieve current conversation history
    history = conversation_history.get(session_id, [])

    # Retrieve relevant documents from RAG
    retrieved_docs = rag_manager.retrieve_documents(user_message)
    context = "\n\nRelevant Knowledge Base:\n" + "\n".join(retrieved_docs) if retrieved_docs else ""

    # Construct the full prompt, leveraging shared prefixes for KV cache reuse
    # The format below is designed for instruct models and to clearly delineate prefixes
    prompt_parts = [
        f"### System Instructions:\n{SYSTEM_INSTRUCTIONS}"
    ]
    
    if history:
        prompt_parts.append("\n\n### Conversation History:")
        for speaker, msg in history:
            prompt_parts.append(f"{speaker}: {msg}")
            
    if context:
        prompt_parts.append(context)
    
    prompt_parts.append(f"\n\n### User Query:\nUser: {user_message}")
    prompt_parts.append("Agent:") # Prompt the LLM to start its response as the Agent

    full_prompt = "\n".join(prompt_parts)

    # Get LLM response
    llm_response = llm_client.generate(full_prompt)

    # Update conversation history
    history.append(("User", user_message))
    history.append(("Agent", llm_response))
    conversation_history[session_id] = history

    return ChatResponse(session_id=session_id, response=llm_response)

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI application...")
    print(f"ChromaDB path: {config.CHROMA_DB_PATH}")
    print(f"Embedding model: {config.EMBEDDING_MODEL_NAME}")
    print(f"LLM model (tokenizer only): {config.LLM_MODEL_NAME}")
    print(f"Mock vLLM API URL: {config.VLLM_API_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8001)
