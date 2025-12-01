from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.language_models import BaseLLM


# --- Configuration ---
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" # This is a placeholder for vllm
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"

# --- LLM Simulation (vllm replacement for demonstration) ---
# In a real scenario, this would be an HTTP client calling a vllm server.
class VLLMSimulator(BaseLLM):
    model_name: str = LLM_MODEL

    def _call(self, prompt: str, stop: List[str] = None, **kwargs) -> str:
        # Simulate LLM response, acknowledging KV cache reuse is handled by actual vllm server
        if "question:" in prompt.lower():
            return f"[Simulated LLM Response from {self.model_name}, KV cache reused] I understand your question. Here is a helpful answer based on the context: {prompt.split('Context:')[-1].split('Question:')[0].strip()}. How else can I assist you?"
        return f"[Simulated LLM Response from {self.model_name}, KV cache reused] Hello! How can I help you today?"

    @property
    def _llm_type(self) -> str:
        return "vllm_simulator"


# --- Knowledge Base Setup ---
def setup_knowledge_base():
    documents = [
        "The product warranty covers manufacturing defects for one year from the purchase date.",
        "To reset your password, navigate to the login page and click 'Forgot Password'. Follow the instructions sent to your registered email.",
        "Our customer support hours are Monday to Friday, 9 AM to 5 PM EST.",
        "Shipping usually takes 3-5 business days for standard delivery within the continental US.",
        "Returns are accepted within 30 days of purchase, provided the item is in its original condition and packaging.",
        "For technical support, please visit our online help center or open a support ticket."
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.create_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_PATH)
    db.persist()
    return db


# --- FastAPI App Initialization ---
app = FastAPI()

# Global instances
embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
retriever = vectorstore.as_retriever()

# Initialize LLM (VLLMSimulator)
llm = VLLMSimulator()

# Setup Langchain Memory
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

# Prompt Template for RAG + Conversation
PROMPT_TEMPLATE = """
You are an AI-powered customer support assistant. Answer the user's question based on the provided context and conversation history.
If the answer is not in the context, politely state that you don't have enough information.

Conversation History:
{chat_history}

Context:
{context}

Question: {question}
Assistant:
"""

qa_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(PROMPT_TEMPLATE),
    memory=memory,
)


# --- API Endpoints ---
class ChatRequest(BaseModel):
    message: str


@app.on_event("startup")
async def startup_event():
    global vectorstore, retriever, embedding_function
    print("Setting up knowledge base...")
    db = setup_knowledge_base()
    vectorstore = db
    retriever = db.as_retriever()
    print("Knowledge base setup complete.")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message

    # Retrieve relevant documents from the knowledge base
    docs = retriever.get_relevant_documents(user_message)
    context = "\n".join([doc.page_content for doc in docs])

    # Get conversation history from memory (managed by LLMChain)
    # The LLMChain will inject `chat_history` into the prompt using the memory object.

    # Prepare the input for the LLMChain
    # The LLMChain's `apply` method will internally get `chat_history` from `memory`
    # and then construct the final prompt using the PROMPT_TEMPLATE.
    response = qa_chain.invoke({"question": user_message, "context": context})

    # The LLMChain updates its memory automatically after `invoke`

    return {"response": response["text"]}


# To run this application:
# 1. Install dependencies: pip install fastapi uvicorn chromadb langchain langchain-community sentence-transformers pydantic
# 2. Run the FastAPI app: uvicorn main:app --reload
# 3. Access the API at http://127.0.0.1:8000/docs
