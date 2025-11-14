
import os
from collections import deque
from typing import Dict, List, Any
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

# Mocking external libraries for a self-contained example
try:
    from sentence_transformers import SentenceTransformer
    from chromadb import Client, Settings
    from chromadb.utils import embedding_functions
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.language_models import BaseLLM
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    logger.warning("Some libraries (sentence_transformers, chromadb, langchain) not found. Using mock implementations.")
    # Mock implementations for environments without these libraries
    class MockSentenceTransformer:
        def encode(self, texts, **kwargs):
            return [[0.1] * 768 for _ in texts] # Dummy embedding

    class MockChromaClient:
        def __init__(self, *args, **kwargs):
            pass
        def get_or_create_collection(self, name, *args, **kwargs):
            return MockChromaCollection()

    class MockChromaCollection:
        def add(self, documents, metadatas, ids):
            logger.info(f"Mock ChromaDB: Added {len(documents)} documents.")
        def query(self, query_embeddings, n_results=1, *args, **kwargs):
            # Return dummy results
            return {"documents": [["Mocked knowledge base response"]], "metadatas": [[{"source": "mock"}]]}

    class MockLLM(BaseLLM):
        def _call(self, prompt: str, stop: List[str] = None, run_manager: Any = None) -> str:
            # Simple echo LLM
            logger.info(f"Mock LLM received prompt: {prompt[:100]}...")
            if "order status" in prompt.lower():
                return "Your order #12345 is currently being processed and is expected to ship within 2 business days."
            return f"Mock LLM response to: {prompt[:50]}... (Please integrate a real LLM)"
        def _llm_type(self) -> str:
            return "mock-llm"

    class MockLLMChain:
        def __init__(self, llm, prompt):
            self.llm = llm
            self.prompt = prompt
            self.output_parser = StrOutputParser()
        def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
            formatted_prompt = self.prompt.format(**input)
            response = self.llm._call(formatted_prompt)
            return {"text": self.output_parser.parse(response)}

    class SentenceTransformer: # Alias for consistent usage
        def __init__(self, model_name):
            self._model = MockSentenceTransformer()
        def encode(self, texts, **kwargs):
            return self._model.encode(texts, **kwargs)

    class Client: # Alias for consistent usage
        def __init__(self, *args, **kwargs):
            self._client = MockChromaClient(*args, **kwargs)
        def get_or_create_collection(self, name, *args, **kwargs):
            return self._client.get_or_create_collection(name, *args, **kwargs)

    class Settings:
        pass # Mock settings

    BaseLLM = object # Mock BaseLLM if not available
    LLMChain = MockLLMChain
    PromptTemplate = object # Mock PromptTemplate
    StrOutputParser = object # Mock StrOutputParser
    HumanMessage = object # Mock HumanMessage
    AIMessage = object # Mock AIMessage



load_dotenv() # Load environment variables

app = FastAPI(
    title="Adaptive Customer Support LLM Assistant",
    description="An LLM assistant with adaptive memory and processing strategies for e-commerce customer support."
)

# --- Configuration and Initialization ---

# Short-Term Memory (Conversational Context Manager)
# Using a dictionary to store context per session_id, with a deque for conversation turns
SESSION_CONTEXT: Dict[str, deque] = {}
MAX_SHORT_TERM_MEMORY_TURNS = int(os.getenv("MAX_SHORT_TERM_MEMORY_TURNS", "5"))

# Long-Term Memory (Knowledge Base & Customer History)
# Initialize embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
except Exception as e:
    logger.error(f"Could not load SentenceTransformer or ChromaDB embedding function: {e}. Using mock embedding.")
    embed_model = SentenceTransformer("mock") # Fallback to mock
    chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2") # Still need to pass a name

# Initialize ChromaDB client (in-memory for this example)
chroma_client = Client(Settings(allow_reset=True))

# Knowledge Base Collection (FAQs, product info)
kb_collection = chroma_client.get_or_create_collection(
    name="ecommerce_knowledge_base",
    embedding_function=chroma_ef
)

# Dummy Knowledge Base Data
dummy_kb_data = [
    {"id": "prod_faq_001", "content": "Our shipping typically takes 3-5 business days for standard delivery within the contiguous United States.", "metadata": {"type": "shipping", "product_id": None}},
    {"id": "prod_faq_002", "content": "You can track your order using the tracking number provided in your shipping confirmation email.", "metadata": {"type": "tracking", "product_id": None}},
    {"id": "prod_faq_003", "content": "Returns are accepted within 30 days of purchase, provided the item is unused and in its original packaging.", "metadata": {"type": "returns", "product_id": None}},
    {"id": "prod_faq_004", "content": "Product A is a high-performance gaming laptop with an RTX 4080 GPU and 32GB RAM.", "metadata": {"type": "product_info", "product_id": "product_A"}},
    {"id": "prod_faq_005", "content": "Product B is a comfortable ergonomic office chair designed for long hours of use.", "metadata": {"type": "product_info", "product_id": "product_B"}},
]

# Add dummy data to ChromaDB if not already populated
if kb_collection.count() == 0:
    kb_collection.add(
        documents=[item["content"] for item in dummy_kb_data],
        metadatas=[item["metadata"] for item in dummy_kb_data],
        ids=[item["id"] for item in dummy_kb_data]
    )
    logger.info("Dummy knowledge base loaded into ChromaDB.")


# Mock Relational Database for customer data
MOCK_CUSTOMER_DB: Dict[str, Dict[str, Any]] = {
    "customer_123": {"name": "Alice Smith", "email": "alice@example.com", "orders": ["order_12345", "order_67890"], "order_12345": {"status": "processing", "items": ["Product A"], "shipping_date": "2023-11-15"}},
    "customer_456": {"name": "Bob Johnson", "email": "bob@example.com", "orders": ["order_98765"], "order_98765": {"status": "shipped", "items": ["Product B"], "shipping_date": "2023-11-10"}}
}

def get_customer_data(customer_id: str) -> Dict[str, Any]:
    """Simulates fetching customer data from a relational DB."""
    return MOCK_CUSTOMER_DB.get(customer_id, {})

def get_order_details(customer_id: str, order_id: str) -> Dict[str, Any]:
    """Simulates fetching order details from a relational DB."""
    customer_data = get_customer_data(customer_id)
    if customer_data and order_id in customer_data:
        return customer_data[order_id]
    return {}

# Core LLM (using a mock or actual LLM)
# For a real application, replace MockLLM with an actual LLM integration (e.g., from langchain_openai)
# Example with OpenAI: llm = ChatOpenAI(model="gpt-4", temperature=0.7)
llm = MockLLM() # Using our mock LLM

# LLM prompt templates
SIMPLE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """You are a helpful e-commerce customer support assistant. Answer the user's question concisely.
    Question: {question}
    Answer:"""
)

RAG_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """You are a helpful e-commerce customer support assistant. Use the provided context and conversation history to answer the user's question.
    If the answer is not in the context, state that you don't know.

    Conversation History:
    {history}

    Context:
    {context}

    Question: {question}
    Answer:"""
)

# LLM Chains
simple_llm_chain = LLMChain(llm=llm, prompt=SIMPLE_PROMPT_TEMPLATE)
rag_llm_chain = LLMChain(llm=llm, prompt=RAG_PROMPT_TEMPLATE)

# --- Query Complexity Classifier ---

def classify_query_complexity(query: str) -> str:
    """Classifies query as simple, medium, or complex/personal.
    This is a rule-based simplification. A real system would use ML.
    """
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ["my order", "customer id", "account", "return status"]):
        return "personal_account_inquiry" # High complexity, often needs DB lookup
    elif any(keyword in query_lower for keyword in ["shipping", "delivery", "returns", "product info", "warranty"]):
        return "medium_knowledge_lookup" # Medium complexity, needs RAG
    else:
        return "simple_fact_lookup" # Simple, direct LLM

# --- Adaptive Processing Strategies ---

def get_short_term_history(session_id: str) -> str:
    """Retrieves formatted short-term conversation history for the current session."""
    history = SESSION_CONTEXT.get(session_id, deque())
    formatted_history = []
    for turn in history:
        if isinstance(turn, HumanMessage):
            formatted_history.append(f"Customer: {turn.content}")
        elif isinstance(turn, AIMessage):
            formatted_history.append(f"Assistant: {turn.content}")
    return "\n".join(formatted_history)


def update_short_term_history(session_id: str, human_message: str, ai_message: str):
    """Updates short-term memory with the latest conversational turn."""
    if session_id not in SESSION_CONTEXT:
        SESSION_CONTEXT[session_id] = deque(maxlen=MAX_SHORT_TERM_MEMORY_TURNS)
    SESSION_CONTEXT[session_id].append(HumanMessage(content=human_message))
    SESSION_CONTEXT[session_id].append(AIMessage(content=ai_message))
    logger.info(f"Session {session_id} history updated. Current length: {len(SESSION_CONTEXT[session_id])}")

def retrieve_long_term_context(query: str, n_results: int = 2) -> str:
    """Retrieves relevant documents from the vector database."""
    try:
        query_embedding = embed_model.encode([query]).tolist()
        results = kb_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas']
        )
        docs = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []

        context_parts = []
        for doc, meta in zip(docs, metadatas):
            context_parts.append(f"Content: {doc}\nMetadata: {meta}")
        return "\n---\n".join(context_parts)
    except Exception as e:
        logger.error(f"Error retrieving from ChromaDB: {e}")
        return ""


async def process_query(session_id: str, query: str, customer_id: str = None) -> str:
    """Main function to process a user query with adaptive strategies."""
    complexity = classify_query_complexity(query)
    logger.info(f"Query classified as: {complexity}")

    short_term_history = get_short_term_history(session_id)
    context = ""

    if complexity == "simple_fact_lookup":
        response = simple_llm_chain.invoke({"question": query})["text"]

    elif complexity == "medium_knowledge_lookup":
        retrieved_context = retrieve_long_term_context(query)
        context = f"Retrieved Knowledge:\n{retrieved_context}"
        response = rag_llm_chain.invoke({"question": query, "context": context, "history": short_term_history})["text"]

    elif complexity == "personal_account_inquiry":
        customer_info = {}
        if customer_id:
            customer_info = get_customer_data(customer_id)
            if "order status" in query.lower() and customer_info and customer_info.get("orders"):
                # Attempt to find an order ID, or use the first if ambiguous
                order_id_found = None
                # More sophisticated parsing needed for real scenarios
                for order_id in customer_info["orders"]:
                    if order_id in query.lower(): # Simple check for order ID presence in query
                        order_id_found = order_id
                        break
                if not order_id_found and customer_info.get("orders"): # Default to first order if none specified
                    order_id_found = customer_info["orders"][0]

                if order_id_found:
                    order_details = get_order_details(customer_id, order_id_found)
                    context += f"\nCustomer Order Details (Order {order_id_found}): {order_details}"
                else:
                    context += "\nNo specific order ID found in query or customer history."
            else:
                context += f"\nCustomer Info: {customer_info}"
        else:
            context += "\nNo customer ID provided for personal inquiry."

        # Even for personal inquiries, RAG can be useful for general policies related to accounts
        retrieved_context = retrieve_long_term_context(query)
        if retrieved_context:
            context += f"\nRetrieved Knowledge: {retrieved_context}"

        response = rag_llm_chain.invoke({"question": query, "context": context, "history": short_term_history})["text"]

    else:
        response = "I'm sorry, I don't understand the complexity of your query." # Fallback

    update_short_term_history(session_id, query, response)
    return response

# --- Efficient Fine-tuning Module (Conceptual) ---
class FineTuningModule:
    def __init__(self, model_path: str = "./fine_tuned_llm"):
        self.model_path = model_path
        logger.info(f"FineTuningModule initialized. Model path: {self.model_path}")

    def prepare_data(self, interaction_logs: List[Dict]) -> Any:
        """Simulates data preparation for fine-tuning using Hugging Face Datasets."""
        logger.info(f"Preparing {len(interaction_logs)} interaction logs for fine-tuning...")
        # In a real scenario, this would involve tokenization, formatting for LoRA/QLoRA, etc.
        # Example: datasets.Dataset.from_list(interaction_logs)
        return {"status": "data_prepared", "count": len(interaction_logs)}

    def fine_tune_model(self, prepared_data: Any):
        """Simulates fine-tuning the LLM using LoRA/QLoRA with TRL/Accelerate."""
        logger.info("Starting simulated fine-tuning process...")
        # This would involve loading the base LLM, setting up LoRA adapters, training loop, etc.
        # from trl import SFTTrainer
        # from peft import LoraConfig
        # trainer = SFTTrainer(model=llm, train_dataset=prepared_data, ...)
        # trainer.train()
        logger.info("Simulated fine-tuning complete. Model saved to {self.model_path}.")
        # In a real system, the new fine-tuned model would then be loaded for inference
        return {"status": "fine_tuned", "model_path": self.model_path}

fine_tuning_module = FineTuningModule()

# --- API Gateway (FastAPI) ---

class ChatRequest(BaseModel):
    session_id: str
    query: str
    customer_id: str = None # Optional, for personal account inquiries

class ChatResponse(BaseModel):
    session_id: str
    response: str
    query_complexity: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request for session {request.session_id} with query: {request.query}")
    try:
        # Re-classify complexity here for the response object, though it's done inside process_query too
        complexity = classify_query_complexity(request.query)
        response_text = await process_query(request.session_id, request.query, request.customer_id)
        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            query_complexity=complexity
        )
    except Exception as e:
        logger.exception(f"Error processing chat request for session {request.session_id}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Adaptive Customer Support LLM Assistant is running."}

# Example endpoint to trigger a simulated fine-tuning process (for demonstration)
@app.post("/simulate-finetune")
async def simulate_finetune_endpoint():
    # In a real scenario, interaction logs would be collected over time
    dummy_logs = [
        {"user": "My order is late.", "assistant": "Please provide your order ID.", "label": "late_order"},
        {"user": "Where is product A?", "assistant": "Product A is a gaming laptop...", "label": "product_query"},
    ]
    prepared_data = fine_tuning_module.prepare_data(dummy_logs)
    finetune_result = fine_tuning_module.fine_tune_model(prepared_data)
    return {"message": "Simulated fine-tuning process initiated.", "result": finetune_result}


if __name__ == "__main__":
    # To run this application:
    # 1. pip install fastapi uvicorn python-dotenv loguru sentence-transformers chromadb langchain
    #    (Note: langchain might require specific sub-packages like langchain-openai)
    # 2. python your_file_name.py
    # 3. Access the API at http://127.0.0.1:8000/docs

    # Example usage (if running directly without uvicorn command):
    logger.info("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
