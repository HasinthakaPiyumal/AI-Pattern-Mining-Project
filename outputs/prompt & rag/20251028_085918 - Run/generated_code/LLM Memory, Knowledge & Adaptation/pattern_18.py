import os
from collections import deque
from enum import Enum
from typing import List, Dict, Tuple

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformers import SentenceTransformerEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence
from langchain.schema import StrOutputParser, Document
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

CHROMA_PATH = "./chroma_db"

# --- Data Models ---
class QueryType(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    HIGH = "high"

class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    user_query: str

# --- Memory Management ---
class ShortTermMemory:
    def __init__(self, max_length: int = 5):
        self.history: deque[ChatMessage] = deque(maxlen=max_length * 2) # Store user and assistant messages

    def add_message(self, role: str, content: str):
        self.history.append(ChatMessage(role=role, content=content))

    def get_history(self) -> List[Dict[str, str]]:
        # Convert ChatMessage objects to dicts for LLM input
        return [msg.dict() for msg in self.history]

    def clear_history(self):
        self.history.clear()

class LongTermMemory:
    def __init__(self):
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectordb = self._initialize_chroma()

    def _initialize_chroma(self):
        if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
            print("Initializing ChromaDB with dummy data...")
            # Dummy knowledge base for demonstration
            documents = [
                Document(page_content="Our product is an AI assistant called IntelliDesk. It helps with customer support.", metadata={"source": "product_info"}),
                Document(page_content="To reset your password, navigate to the login page and click 'Forgot Password'. Follow the instructions sent to your registered email.", metadata={"source": "faq"}),
                Document(page_content="Common error code E101 means 'Invalid API Key'. Please check your configuration.", metadata={"source": "troubleshooting"}),
                Document(page_content="Our service operates 24/7. For urgent issues, please contact our emergency line at 555-1234.", metadata={"source": "contact"}),
                Document(page_content="You can find our privacy policy on our website's footer section.", metadata={"source": "policy"}),
                Document(page_content="The premium subscription includes unlimited queries and priority support.", metadata={"source": "subscription"}),
            ]
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=CHROMA_PATH
            )
            vectordb.persist()
            return vectordb
        else:
            print("Loading existing ChromaDB...")
            return Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)

    def retrieve_docs(self, query: str, k: int = 3) -> List[Document]:
        return self.vectordb.similarity_search(query, k=k)

# --- Query Processing ---
class QueryClassifier:
    def classify(self, query: str, conversation_history: List[Dict[str, str]]) -> QueryType:
        query_lower = query.lower()

        # Simple queries
        simple_keywords = ["hello", "hi", "what is intellidesk", "thank you", "thanks", "bye", "goodbye", "hours", "operating times"]
        if any(keyword in query_lower for keyword in simple_keywords):
            return QueryType.SIMPLE

        # High complexity/sensitive queries
        high_keywords = ["account access", "billing dispute", "cancel subscription", "security issue", "report fraud", "human agent", "speak to a person"]
        if any(keyword in query_lower for keyword in high_keywords):
            return QueryType.HIGH

        # Medium complexity (default for anything not simple or high)
        return QueryType.MEDIUM

# --- LLM Interaction & Strategy ---
class LLMManager:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        formatted_history = []
        for msg in history:
            formatted_history.append(f"{msg['role'].capitalize()}: {msg['content']}")
        return "\n".join(formatted_history)

    def generate_response_simple(self, query: str, history: List[Dict[str, str]]) -> str:
        prompt_template = PromptTemplate.from_template(
            """You are a friendly customer support AI. Respond concisely and politely.
            \nConversation History:\n{history}\nUser: {query}\nAssistant:"""
        )
        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({"history": self._format_history(history), "query": query})

    def generate_response_medium(self, query: str, history: List[Dict[str, str]], retrieved_context: List[Document]) -> str:
        context_str = "\n".join([doc.page_content for doc in retrieved_context])
        history_str = self._format_history(history)

        prompt_template = PromptTemplate.from_template(
            """You are a helpful and knowledgeable customer support AI, IntelliDesk. \n"""
            """Use the following retrieved context to answer the user's question. If the answer isn't in the context, \n"""
            """say you don't know, but try to be helpful based on general knowledge if appropriate.\n"""
            """Always maintain a professional and friendly tone.\n\n"""
            """Retrieved Context:\n{context}\n\n"""
            """Conversation History:\n{history_str}\n\n"""
            """User: {query}\nAssistant:"""
        )
        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({
            "context": context_str,
            "history_str": history_str,
            "query": query
        })

    def generate_handover_summary(self, query: str, history: List[Dict[str, str]]) -> str:
        history_str = self._format_history(history)
        prompt_template = PromptTemplate.from_template(
            """The user has a complex or sensitive issue requiring human intervention. \n"""
            """Summarize the conversation history and the user's current query for a human agent. \n"""
            """Focus on the key problem and any important details. \n"""
            """Conversation History:\n{history_str}\n\n"""
            """Current User Query: {query}\n\n"""
            """Summary for Human Agent:"""
        )
        chain = prompt_template | self.llm | StrOutputParser()
        summary = chain.invoke({"history_str": history_str, "query": query})
        return f"I'm sorry, I cannot directly assist with this complex issue. I've prepared a summary for a human agent who will contact you shortly.\n\n{summary}"

# --- Main IntelliDesk Agent ---
class IntelliDeskAgent:
    def __init__(self):
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.query_classifier = QueryClassifier()
        self.llm_manager = LLMManager()

    def process_query(self, user_query: str) -> str:
        self.short_term_memory.add_message(role="user", content=user_query)
        conversation_history = self.short_term_memory.get_history()

        # The last message in history is the current user query, remove it for classification if needed
        # or handle its presence explicitly in classification logic.
        # For simplicity, we pass the full history to classify, which can look for the last user message.

        query_type = self.query_classifier.classify(user_query, conversation_history)
        print(f"Classified query as: {query_type.value}")

        agent_response = ""
        if query_type == QueryType.SIMPLE:
            agent_response = self.llm_manager.generate_response_simple(user_query, conversation_history)
        elif query_type == QueryType.MEDIUM:
            retrieved_context = self.long_term_memory.retrieve_docs(user_query)
            agent_response = self.llm_manager.generate_response_medium(user_query, conversation_history, retrieved_context)
        elif query_type == QueryType.HIGH:
            agent_response = self.llm_manager.generate_handover_summary(user_query, conversation_history)

        self.short_term_memory.add_message(role="assistant", content=agent_response)
        return agent_response

# --- FastAPI Application ---
app = FastAPI()
intellidesk_agent = IntelliDeskAgent()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = intellidesk_agent.process_query(request.user_query)
    return {"response": response}

@app.get("/health")
async def health_check():
    return {"status": "ok", "agent_ready": True}

# To run the application:
# 1. pip install fastapi uvicorn python-dotenv langchain openai chromadb sentence-transformers pydantic
# 2. Set your OPENAI_API_KEY environment variable.
# 3. Run: uvicorn main:app --reload
# 4. Access at http://127.0.0.1:8000/docs

# Example usage with curl:
# curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_query": "Hi, what is IntelliDesk?"}'
# curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_query": "How do I reset my password?"}'
# curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"user_query": "I need to speak to someone about my account access."}'