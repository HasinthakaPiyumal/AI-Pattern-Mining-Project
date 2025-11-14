from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import logging

# LangChain imports (mocked for simplicity in this single file, assuming installation)
# In a real application, you would import from langchain_core, langchain_community, etc.

# --- Configuration and Initialization ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(title="E-commerce Chatbot API", description="AI-powered customer support chatbot for an e-commerce platform.")

# In-memory storage for conversational history (for demonstration)
conversation_history: Dict[str, List[Dict[str, str]]] = {}

# --- Mock VLLM Client (simulating LLM inference) ---

class MockVLLMClient:
    """A mock client to simulate vllm's asynchronous LLM generation."""
    def __init__(self, model_name: str = "ecommerce-llm"): # Assuming a fine-tuned model
        self.model_name = model_name
        logging.info(f"MockVLLMClient initialized for model: {self.model_name}")

    async def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Simulates LLM response generation."""
        logging.info(f"Mock LLM received prompt: {prompt[:100]}...")
        # Simple rule-based mock responses for common e-commerce queries
        if "order status" in prompt.lower():
            return "Your order #12345 is currently being processed and is expected to ship within 2 business days."
        elif "return policy" in prompt.lower():
            return "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging."
        elif "contact support" in prompt.lower():
            return "You can reach our customer support team by emailing support@example.com or calling 1-800-123-4567."
        elif "product availability" in prompt.lower():
            return "Please provide the product name or ID, and I can check its current stock availability for you."
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! How can I assist you with your e-commerce needs today?"
        else:
            return f"I'm sorry, I don't have enough information to fully answer that. Could you please rephrase or provide more details? (Simulated LLM response based on '{prompt[:50]}...')"

mock_llm_client = MockVLLMClient()

# --- Mock Chroma Client (simulating Vector DB for RAG) ---

class MockChromaClient:
    """A mock client to simulate ChromaDB collection queries for RAG."""
    def __init__(self):
        logging.info("MockChromaClient initialized.")
        # Simulate a small knowledge base
        self.knowledge_base = [
            {"content": "Our shipping typically takes 3-5 business days for standard delivery.", "source": "FAQ_Shipping"},
            {"content": "Expedited shipping options are available at checkout for an additional fee.", "source": "FAQ_Shipping"},
            {"content": "We offer a 1-year warranty on all electronic products.", "source": "Policy_Warranty"},
            {"content": "To reset your password, visit the 'Forgot Password' link on the login page.", "source": "FAQ_Account"},
            {"content": "Product XYZ is currently out of stock, expected back in 2 weeks.", "source": "Product_Status"},
        ]

    async def query_collection(self, query_text: str, n_results: int = 2) -> List[Dict[str, str]]:
        """Simulates querying the vector database for relevant documents."""
        logging.info(f"Mock Chroma received query: {query_text[:50]}...")
        # Simple keyword-based matching for demonstration
        relevant_docs = []
        for doc in self.knowledge_base:
            if any(keyword in doc['content'].lower() for keyword in query_text.lower().split()):
                relevant_docs.append(doc)
        return relevant_docs[:n_results]

mock_chroma_client = MockChromaClient()

# --- LangChain-like Orchestration ---

class ChatbotChain:
    """A class simulating a LangChain-like runnable for the chatbot logic."""
    def __init__(self, llm_client: MockVLLMClient, knowledge_db_client: MockChromaClient):
        self.llm = llm_client
        self.knowledge_db = knowledge_db_client
        # Basic prompt template (in a real LangChain app, this would be more sophisticated)
        self.prompt_template = (
            "You are an AI assistant for an e-commerce platform. Your goal is to provide helpful and concise customer support."
            "Answer the user's question based on the provided context, if available. If you cannot find the answer, politely state that."
            "\n\nContext: {context}"
            "\n\nConversation History: {history}"
            "\n\nUser: {query}"
            "\n\nAssistant:"
        )
        logging.info("ChatbotChain initialized.")

    async def invoke(self, session_id: str, user_query: str) -> str:
        """Invokes the chatbot chain to get a response."""
        history = conversation_history.get(session_id, [])
        formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in history])

        # 1. Simulate RAG: Retrieve context from the knowledge base
        retrieved_docs = await self.knowledge_db.query_collection(user_query)
        context = "\n".join([doc['content'] for doc in retrieved_docs]) if retrieved_docs else "No relevant context found."

        # 2. Prepare the prompt for the LLM
        full_prompt = self.prompt_template.format(
            context=context,
            history=formatted_history,
            query=user_query
        )

        # 3. Get response from LLM
        llm_response = await self.llm.generate(full_prompt)

        # 4. Update conversation history
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": llm_response})
        conversation_history[session_id] = history

        return llm_response

chatbot_chain = ChatbotChain(llm_client=mock_llm_client, knowledge_db_client=mock_chroma_client)

# --- API Endpoints ---

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    history: List[Dict[str, str]]


@app.post("/chat", response_model=ChatResponse, summary="Send a message to the chatbot")
async def chat_with_bot(request: ChatRequest):
    """Processes a user message and returns a chatbot response, updating conversation history."""
    logging.info(f"Received chat request for session {request.session_id}: {request.message[:100]}...")
    try:
        response_message = await chatbot_chain.invoke(request.session_id, request.message)
        return ChatResponse(
            session_id=request.session_id,
            response=response_message,
            history=conversation_history.get(request.session_id, [])
        )
    except Exception as e:
        logging.error(f"Error processing chat request for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing your request.")


@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Returns a simple health status to indicate the API is running."""
    return {"status": "healthy", "message": "E-commerce Chatbot API is operational."}


@app.get("/history/{session_id}", response_model=List[Dict[str, str]], summary="Retrieve conversation history for a session")
async def get_history(session_id: str):
    """Retrieves the full conversation history for a given session ID."""
    history = conversation_history.get(session_id, [])
    if not history:
        raise HTTPException(status_code=404, detail=f"No conversation history found for session ID: {session_id}")
    return history


# --- Running the FastAPI application ---

if __name__ == "__main__":
    logging.info("Starting FastAPI application with Uvicorn...")
    # To run this, save it as main.py and execute: python main.py
    # Or directly with uvicorn: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
