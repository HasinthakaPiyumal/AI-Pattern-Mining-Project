import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import os

# Mocking vLLM and ChromaDB for code generation purposes as they require external services/setup
# In a real application, you would initialize vllm.LLM and chromadb.PersistentClient

class MockVLLMResponse:
    def __init__(self, text):
        self.text = text

class MockVLLMEngine:
    def __init__(self, model: str):
        self.model = model

    async def generate(self, prompts: List[str], sampling_params):
        # Simulate LLM response, potentially based on prompt for demo
        results = []
        for prompt in prompts:
            if "order number" in prompt.lower() and "#12345" in prompt:
                response_text = "I see order #12345 is for a 'Smartwatch Pro'. Is there anything else I can help you with regarding this order?"
            elif "hi" in prompt.lower() or "hello" in prompt.lower():
                response_text = "Hello! How can I assist you with your order today?"
            elif "question about my order" in prompt.lower():
                response_text = "Certainly, what is your order number?"
            else:
                response_text = f"Thank you for your query. I am processing your request based on: '{prompt[-50:]}'"
            results.append([MockVLLMResponse(response_text)])
        return results

class MockChromaDBClient:
    def get_or_create_collection(self, name):
        return MockChromaDBCollection()

class MockChromaDBCollection:
    def query(self, query_texts: List[str], n_results: int):
        # Simulate RAG by returning a generic document
        if "order" in query_texts[0].lower():
            return {"documents": [["Customer support policy: For order-related queries, please provide your order number."]]}
        return {"documents": [["General company information: We are committed to providing excellent customer service."]]}

class MockSentenceTransformer:
    def encode(self, texts: List[str], convert_to_tensor: bool = False):
        # Simulate embedding by returning a list of mock vectors
        return [[0.1] * 384 for _ in texts] # Example dimension

# --- Configuration --- 
# Replace with actual model and vLLM endpoint in a real application
LLM_MODEL = os.getenv("LLM_MODEL", "mock-llm-model")
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/generate") # Not directly used by MockVLLMEngine
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

app = FastAPI(
    title="AI Customer Support Assistant",
    description="An AI assistant leveraging KV Cache Reuse for optimized LLM inference."
)

# Initialize Mock LLM Engine and RAG components
llm_engine = MockVLLMEngine(model=LLM_MODEL)
chroma_client = MockChromaDBClient()
embedding_model = MockSentenceTransformer()

# Initialize RAG collection (mocking for demonstration)
rag_collection = chroma_client.get_or_create_collection(name="customer_support_docs")

# In-memory conversation history storage
conversation_history: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str
    include_rag: bool = False

async def get_relevant_documents(query: str) -> List[str]:
    # In a real scenario, this would query a vector database
    # and return actual relevant documents.
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    results = rag_collection.query(
        query_texts=[query], # For mock, actual embedding is not used directly in query method
        n_results=1
    )
    return results.get("documents", [[]])[0] # Return the first list of documents

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message
    include_rag = request.include_rag

    # Retrieve conversation history for the session
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    current_history = conversation_history[session_id]

    # Append user's message to history
    current_history.append({"role": "user", "content": user_message})

    # Construct prompt for LLM
    # In a real LLM, we'd structure this based on the model's preferred chat format (e.g., Llama-2 chat template)
    # For this mock, we'll just concatenate.
    full_prompt_parts = []
    system_prompt = "You are a helpful customer support assistant. Always be polite and informative."
    full_prompt_parts.append(f"System: {system_prompt}")

    if include_rag:
        relevant_docs = await get_relevant_documents(user_message)
        if relevant_docs:
            full_prompt_parts.append("\n\nRelevant Information:\n" + "\n".join(relevant_docs))

    for entry in current_history:
        full_prompt_parts.append(f"{entry['role'].title()}: {entry['content']}")

    full_prompt_parts.append("Assistant:") # Prompt the assistant to respond
    llm_prompt = "\n".join(full_prompt_parts)

    try:
        # Mock SamplingParams as it's typically required by vLLM
        class MockSamplingParams:
            def __init__(self, temperature=0.7, top_p=0.9, max_tokens=256):
                self.temperature = temperature
                self.top_p = top_p
                self.max_tokens = max_tokens

        sampling_params = MockSamplingParams()
        
        # Call the mock LLM engine
        response_list = await llm_engine.generate(prompts=[llm_prompt], sampling_params=sampling_params)
        
        if not response_list or not response_list[0]:
            raise HTTPException(status_code=500, detail="LLM returned an empty response.")
            
        ai_response_text = response_list[0][0].text.strip()

        # Append AI's response to history
        current_history.append({"role": "assistant", "content": ai_response_text})

        return {"session_id": session_id, "response": ai_response_text, "history": current_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LLM inference: {str(e)}")

# To run the application:
# uvicorn main:app --host 0.0.0.0 --port 8000
