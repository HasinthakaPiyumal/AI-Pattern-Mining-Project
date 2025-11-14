
import collections
from typing import Dict, Any, List

from fastapi import FastAPI
from pydantic import BaseModel

# Mock Libraries/Components for demonstration
# In a real application, these would be actual database connections, LLM clients, etc.

# --- Mock Databases/Data ---

customer_orders_db = {
    "user_123": [
        {"order_id": "ORD001", "product": "Laptop", "status": "Shipped", "tracking": "TRK12345"},
        {"order_id": "ORD002", "product": "Mouse", "status": "Delivered", "tracking": "TRK67890"},
    ],
    "user_456": [
        {"order_id": "ORD003", "product": "Keyboard", "status": "Processing", "tracking": None},
    ]
}

product_knowledge_base_data = [
    {"id": "prod_faq_1", "text": "Our return policy allows returns within 30 days of purchase for a full refund."},
    {"id": "prod_faq_2", "text": "To track your order, please use the tracking number provided in your shipping confirmation email."},
    {"id": "prod_faq_3", "text": "For technical support, please visit our support page or contact our dedicated technical team."},
    {"id": "prod_faq_4", "text": "The warranty for electronic items is typically one year from the date of purchase."},
    {"id": "prod_faq_5", "text": "How can I reset my password? You can reset your password by clicking on 'Forgot Password' on the login page."},
    {"id": "prod_faq_6", "text": "What payment methods do you accept? We accept Visa, Mastercard, American Express, PayPal, and Apple Pay."},
]

# --- In-memory Chroma DB Simulation (requires 'chromadb' and 'sentence-transformers') ---
# For a true in-memory experience, you might need to install these:
# pip install chromadb sentence-transformers

try:
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.Client()
    kb_collection = client.get_or_create_collection(name="product_knowledge_base")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Add documents to Chroma KB
    for item in product_knowledge_base_data:
        kb_collection.add(
            documents=[item["text"]],
            metadatas=[{"source": "FAQ", "id": item["id"]}],
            ids=[item["id"]]
        )

    def get_embeddings(texts: List[str]) -> List[List[float]]:
        return embedding_model.encode(texts).tolist()

    CHROMA_AVAILABLE = True
except ImportError:
    print("Warning: chromadb or sentence_transformers not installed. Using simple keyword search for knowledge base.")
    CHROMA_AVAILABLE = False


# --- Query Classifier ---

class QueryClassifier:
    def classify_query(self, query: str) -> str:
        query_lower = query.lower()
        if "order status" in query_lower or "track my order" in query_lower:
            return "simple_order_status"
        elif "return policy" in query_lower or "warranty" in query_lower or "payment method" in query_lower or "reset password" in query_lower:
            return "simple_faq"
        elif "compare" in query_lower or "troubleshoot" in query_lower or "recommend" in query_lower:
            return "medium"
        else:
            return "complex"


# --- Memory Systems ---

class MemorySystems:
    def __init__(self):
        self.short_term_memory: Dict[str, collections.deque] = {}

    def get_short_term_context(self, user_id: str, limit: int = 5) -> List[str]:
        return list(self.short_term_memory.get(user_id, collections.deque(maxlen=limit)))

    def update_short_term_context(self, user_id: str, message: str, limit: int = 5):
        if user_id not in self.short_term_memory:
            self.short_term_memory[user_id] = collections.deque(maxlen=limit)
        self.short_term_memory[user_id].append(message)

    def get_customer_orders(self, user_id: str) -> List[Dict[str, Any]]:
        return customer_orders_db.get(user_id, [])

    def retrieve_knowledge(self, query: str, top_k: int = 2) -> List[str]:
        if CHROMA_AVAILABLE:
            query_embedding = get_embeddings([query])[0]
            results = kb_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            if results and results["documents"]:
                return [doc for sublist in results["documents"] for doc in sublist]
            return []
        else:
            # Fallback to simple keyword search if Chroma is not available
            relevant_docs = []
            query_lower = query.lower()
            for item in product_knowledge_base_data:
                if query_lower in item["text"].lower():
                    relevant_docs.append(item["text"])
            return relevant_docs


# --- LLM Processing Strategies ---

class LLMProcessingStrategies:
    def _mock_llm_response(self, prompt: str) -> str:
        # This mocks an LLM call. In a real app, this would call OpenAI, HuggingFace, etc.
        if "order status" in prompt:
            return "Based on your query, I can provide order status information. Please provide your order ID."
        elif "return policy" in prompt:
            return "Our return policy states that items can be returned within 30 days. You can find more details on our website."
        elif "no direct answer" in prompt:
            return "I understand this is a complex issue. Let me summarize your query for a human agent. Would you like to proceed with human assistance?"
        elif "product comparison" in prompt:
            return "To help you compare products, could you tell me which products you are interested in?"
        return f"(LLM Mock Response): I processed your request: '{prompt}'."

    def handle_simple_query_faq(self, query: str, relevant_knowledge: List[str]) -> str:
        if relevant_knowledge:
            return relevant_knowledge[0]
        return self._mock_llm_response(f"Query: {query}. Looking for simple FAQ answer.")

    def handle_simple_query_order_status(self, user_id: str, query: str, orders: List[Dict[str, Any]]) -> str:
        if orders:
            response = "Here are your recent orders:\n"
            for order in orders:
                status = order.get("status", "N/A")
                tracking = order.get("tracking", "N/A")
                response += f"- Order ID: {order['order_id']}, Product: {order['product']}, Status: {status}, Tracking: {tracking}\n"
            return response
        return "I couldn't find any recent orders for your account. Please ensure you are logged in with the correct user ID."

    def handle_medium_query(self, user_id: str, query: str, short_term_context: List[str], relevant_knowledge: List[str]) -> str:
        context_str = " ".join(short_term_context)
        knowledge_str = " ".join(relevant_knowledge)
        prompt = (
            f"User ID: {user_id}\n"
            f"Current Query: {query}\n"
            f"Conversation History: {context_str}\n"
            f"Relevant Knowledge: {knowledge_str}\n"
            f"Please provide a helpful and concise answer based on the above information."
        )
        return self._mock_llm_response(prompt)

    def handle_complex_query(self, user_id: str, query: str, short_term_context: List[str], relevant_knowledge: List[str]) -> str:
        context_str = " ".join(short_term_context)
        knowledge_str = " ".join(relevant_knowledge)
        summary_prompt = (
            f"Summarize the following complex customer issue for a human agent.\n"
            f"User ID: {user_id}\n"
            f"Query: {query}\n"
            f"Conversation History: {context_str}\n"
            f"Relevant Knowledge Retrieved: {knowledge_str}\n"
            f"Summary for human agent:"
        )
        summary = self._mock_llm_response(summary_prompt)
        return f"I understand this is a complex issue and requires more in-depth assistance. I've prepared a summary for a human agent: \n\n'{summary}'\n\nWould you like me to connect you with one of our specialists?"


# --- Adaptive Customer Support Agent (Orchestrator) ---

class AdaptiveCustomerSupportAgent:
    def __init__(self):
        self.classifier = QueryClassifier()
        self.memory = MemorySystems()
        self.llm_strategies = LLMProcessingStrategies()

    def process_customer_query(self, user_id: str, query: str) -> str:
        # 1. Classify Query
        query_type = self.classifier.classify_query(query)
        print(f"[Agent] Classified query as: {query_type}")

        # 2. Retrieve Memory
        short_term_context = self.memory.get_short_term_context(user_id)
        relevant_knowledge = self.memory.retrieve_knowledge(query)

        # 3. Select and Execute LLM Strategy
        response = "An error occurred."
        if query_type == "simple_faq":
            response = self.llm_strategies.handle_simple_query_faq(query, relevant_knowledge)
        elif query_type == "simple_order_status":
            customer_orders = self.memory.get_customer_orders(user_id)
            response = self.llm_strategies.handle_simple_query_order_status(user_id, query, customer_orders)
        elif query_type == "medium":
            response = self.llm_strategies.handle_medium_query(user_id, query, short_term_context, relevant_knowledge)
        elif query_type == "complex":
            response = self.llm_strategies.handle_complex_query(user_id, query, short_term_context, relevant_knowledge)
        else:
            response = self.llm_strategies._mock_llm_response(f"Could not process query type: {query_type} for '{query}'")

        # 4. Update Short-Term Memory with current query and response
        self.memory.update_short_term_context(user_id, f"User: {query}")
        self.memory.update_short_term_context(user_id, f"Agent: {response}")

        return response


# --- FastAPI Application ---

app = FastAPI(
    title="Adaptive Customer Support Agent API",
    description="An API for an intelligent customer support agent leveraging adaptive LLM strategies."
)

agent = AdaptiveCustomerSupportAgent()


class ChatRequest(BaseModel):
    user_id: str
    query: str


class ChatResponse(BaseModel):
    user_id: str
    query: str
    response: str
    context: List[str]


@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    response_text = agent.process_customer_query(request.user_id, request.query)
    current_context = agent.memory.get_short_term_context(request.user_id)
    return ChatResponse(
        user_id=request.user_id,
        query=request.query,
        response=response_text,
        context=current_context
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Customer Support Agent is running."}


# To run this application:
# 1. Save the code as `customer_support_agent.py`.
# 2. Install necessary libraries: `pip install fastapi uvicorn pydantic chromadb sentence-transformers`
#    (If chromadb or sentence-transformers fail, the KB will use a keyword fallback.)
# 3. Run from your terminal: `uvicorn customer_support_agent:app --reload`
# 4. Access the API documentation at `http://127.0.0.1:8000/docs`
#    You can test the `/chat` endpoint from there.

