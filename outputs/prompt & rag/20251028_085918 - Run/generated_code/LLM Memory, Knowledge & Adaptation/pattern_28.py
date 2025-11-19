from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import uvicorn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

# LangChain related imports (mocked or simplified for single file)
# In a real app, these would be proper imports
class FakeLLM:
    def invoke(self, prompt):
        if "billing" in prompt.lower():
            return "It seems like you have a billing inquiry. Please provide your account number for assistance."
        elif "technical" in prompt.lower():
            return "For technical support, could you describe the issue in more detail?"
        elif "service upgrade" in prompt.lower():
            return "We have several service upgrade options. What are you looking for?"
        elif "context" in prompt.lower() and "long" in prompt.lower():
            return "I've summarized our long conversation to focus on the key points you mentioned earlier."
        elif "non-parametric" in prompt.lower() and "data" in prompt.lower():
            return "Accessing our knowledge base, I found the following information: \"Non-parametric data allows for flexible, real-time updates.\""
        return f"Hello! I am your AI assistant. You said: '{prompt}'. How can I help further?"

class ConversationBufferMemory:
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []

class ChromaClient:
    def __init__(self, name="default"):
        self.collections = {}
        self.active_collection = name
        self.create_collection(name)

    def create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = {"documents": [], "embeddings": [], "metadatas": []}

    def get_collection(self, name=None):
        return self.collections.get(name or self.active_collection)

    def add(self, documents, metadatas=None):
        collection = self.get_collection()
        if collection:
            collection["documents"].extend(documents)
            # Mock embeddings
            collection["embeddings"].extend([[random.random() for _ in range(10)] for _ in documents])
            collection["metadatas"].extend(metadatas or [{} for _ in documents])

    def query(self, query_texts, n_results=1):
        collection = self.get_collection()
        if not collection or not collection["documents"]:
            return {"documents": [[]], "metadatas": [[]]}
        # Simple mock: return the first n_results documents that vaguely match
        results_docs = []
        results_metas = []
        for query_text in query_texts:
            found_docs = []
            found_metas = []
            for i, doc in enumerate(collection["documents"]):
                if query_text.lower() in doc.lower():
                    found_docs.append(doc)
                    found_metas.append(collection["metadatas"][i])
                if len(found_docs) >= n_results:
                    break
            results_docs.append(found_docs)
            results_metas.append(found_metas)
        return {"documents": results_docs, "metadatas": results_metas}

    def set_active_collection(self, name):
        if name in self.collections:
            self.active_collection = name
        else:
            raise ValueError(f"Collection {name} does not exist.")

# Mock Data Stores
customer_db: Dict[str, Dict[str, Any]] = {
    "cust123": {"name": "Alice Smith", "plan": "Premium", "history": ["billing inquiry 2 months ago", "tech support last week"]},
    "cust456": {"name": "Bob Johnson", "plan": "Basic", "history": ["service activation issue"]}
}

redis_client: Dict[str, Dict[str, List[Dict[str, str]]]] = {}

chroma_client = ChromaClient("telecom_knowledge")
chroma_client.add(
    documents=[
        "How to pay your bill: Visit our website and login to your account.",
        "Troubleshooting internet connection: Check your router lights and restart the device.",
        "Upgrade your data plan: Contact customer support or check online offers.",
        "Company policy on refunds: Refunds are processed within 5-7 business days for eligible services."
    ],
    metadatas=[
        {"type": "FAQ", "category": "billing"},
        {"type": "Troubleshooting", "category": "technical"},
        {"type": "Service", "category": "sales"},
        {"type": "Policy", "category": "billing"},
    ]
)

# Query Complexity Classifier
corpus = [
    ("What's my bill this month?", "billing"),
    ("My internet is not working.", "technical"),
    ("I want to upgrade my plan.", "sales"),
    ("How do I pay my invoice?", "billing"),
    ("My phone has no signal.", "technical"),
    ("Tell me about new offers.", "sales"),
    ("Simple question about my account.", "simple"),
    ("Complex problem with network configuration and multiple devices.", "complex"),
    ("What are the charges for my data usage?", "billing"),
    ("I can't access certain websites after the last update.", "technical"),
    ("Can I add a new line to my family plan?", "sales"),
    ("What is the average speed of my current plan?", "simple"),
    ("Detailed explanation of network latency issues in a fiber connection.", "complex"),
]

X_train = [item[0] for item in corpus]
y_train = [item[1] for item in corpus]

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)

classifier = LogisticRegression()
classifier.fit(X_train_vectors, y_train)

def classify_query(query: str) -> str:
    query_vector = vectorizer.transform([query])
    prediction = classifier.predict(query_vector)[0]
    return prediction

# Memory Management
class MemoryManager:
    def __init__(self):
        self.short_term_memories: Dict[str, ConversationBufferMemory] = {}

    def get_short_term_memory(self, session_id: str) -> ConversationBufferMemory:
        if session_id not in self.short_term_memories:
            self.short_term_memories[session_id] = ConversationBufferMemory()
        return self.short_term_memories[session_id]

    def get_long_term_memory(self, customer_id: str) -> Dict[str, Any]:
        return customer_db.get(customer_id, {})

    def augment_memory(self, query: str) -> List[str]:
        results = chroma_client.query(query_texts=[query], n_results=2)
        docs = results["documents"][0]
        return docs

    def long_context_management(self, conversation_history: List[Dict[str, str]]) -> str:
        if len(conversation_history) > 5:  # Simple threshold for summarization
            last_n_messages = conversation_history[-3:]
            summary = "Summarizing recent interaction: " + ". ".join([msg["content"] for msg in last_n_messages])
            return summary
        return " ".join([msg["content"] for msg in conversation_history])

# Knowledge Management
class NonParametricMemoryManager:
    def __init__(self, chroma_client: ChromaClient):
        self.chroma_client = chroma_client
        self.current_index_name = "telecom_knowledge"

    def get_knowledge(self, query: str) -> List[str]:
        results = self.chroma_client.query(query_texts=[query], n_results=3)
        return results["documents"][0]

    def update_knowledge(self, new_docs: List[str], metadatas: List[Dict[str, str]] = None):
        self.chroma_client.add(documents=new_docs, metadatas=metadatas)

    def hotswap_index(self, new_index_name: str):
        print(f"Performing index hotswap from {self.current_index_name} to {new_index_name}")
        self.chroma_client.create_collection(new_index_name)
        self.chroma_client.set_active_collection(new_index_name)
        self.current_index_name = new_index_name
        print(f"Index hotswapped to {self.current_index_name}")

# LLM Core Service
llm_model = FakeLLM()
memory_manager = MemoryManager()
non_parametric_manager = NonParametricMemoryManager(chroma_client)

def generate_llm_response(session_id: str, customer_id: str, query: str) -> str:
    st_memory = memory_manager.get_short_term_memory(session_id)
    st_memory.add_message("user", query)

    long_term_data = memory_manager.get_long_term_memory(customer_id)
    augmented_docs = memory_manager.augment_memory(query)

    # Combine short-term, long-term, and augmented knowledge
    context_elements = [
        f"Customer ID: {customer_id}",
        f"Customer Name: {long_term_data.get('name', 'N/A')}",
        f"Customer Plan: {long_term_data.get('plan', 'N/A')}",
        f"Customer History: {', '.join(long_term_data.get('history', []))}",
    ]
    if augmented_docs:
        context_elements.append(f"Relevant knowledge: {', '.join(augmented_docs)}")
    
    # Handle long context if needed
    full_conversation_context = st_memory.get_history()
    processed_context = memory_manager.long_context_management(full_conversation_context)

    # Combine all into a comprehensive prompt for the LLM
    combined_context = " ".join(context_elements) + f"\nConversation history: {processed_context}\nUser query: {query}"

    llm_response = llm_model.invoke(combined_context)
    st_memory.add_message("assistant", llm_response)

    return llm_response

# FastAPI Application
app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    customer_id: str
    query: str

class KnowledgeUpdateRequest(BaseModel):
    documents: List[str]
    metadatas: List[Dict[str, str]] = None

class IndexHotswapRequest(BaseModel):
    new_index_name: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    query_type = classify_query(request.query)
    print(f"Classified query as: {query_type}")

    # Dynamic strategy based on query_type (simplified)
    if query_type == "simple":
        response = f"Detected simple query. \n{generate_llm_response(request.session_id, request.customer_id, request.query)}"
    elif query_type == "complex":
        response = f"Detected complex query, engaging advanced reasoning. \n{generate_llm_response(request.session_id, request.customer_id, request.query)}"
    else: # billing, technical, sales, etc.
        response = generate_llm_response(request.session_id, request.customer_id, request.query)

    return {"response": response, "query_classification": query_type}

@app.post("/knowledge/update")
async def update_knowledge_endpoint(request: KnowledgeUpdateRequest):
    try:
        non_parametric_manager.update_knowledge(request.documents, request.metadatas)
        return {"status": "success", "message": "Knowledge base updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/hotswap")
async def hotswap_index_endpoint(request: IndexHotswapRequest):
    try:
        non_parametric_manager.hotswap_index(request.new_index_name)
        return {"status": "success", "message": f"Index hotswapped to {request.new_index_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customer_memory/{customer_id}")
async def get_customer_memory(customer_id: str):
    long_term_data = memory_manager.get_long_term_memory(customer_id)
    return {"customer_id": customer_id, "memory": long_term_data}

@app.get("/session_history/{session_id}")
async def get_session_history(session_id: str):
    st_memory = memory_manager.get_short_term_memory(session_id)
    return {"session_id": session_id, "history": st_memory.get_history()}

# Placeholder for LLM Fine-tuning (MLOps part)
def efficient_llm_fine_tuning():
    print("Simulating efficient LLM fine-tuning process...")
    print("Using PEFT (e.g., LoRA) with 'datasets' for data preparation.")
    print("MLflow/Weights & Biases would track experiments and model versions.")
    print("Model deployed via Kubernetes after successful fine-tuning.")

# Example of how to run the fine-tuning (not part of FastAPI)
# if __name__ == "__main__":
#     efficient_llm_fine_tuning()


# Simple Streamlit UI (can be run separately or integrated via API calls)
import streamlit as st
import requests

if __name__ == "__main__":
    st.title("Intelligent Customer Support LLM Assistant")

    st.sidebar.header("Configuration")
    fastapi_url = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")
    customer_id = st.sidebar.text_input("Customer ID", "cust123")
    session_id = st.sidebar.text_input("Session ID", "sess" + str(random.randint(1000, 9999)))

    st.header("Customer Chat")
    user_query = st.text_input("Your query:", key="user_query")
    if st.button("Send Query"):
        if user_query:
            payload = {"session_id": session_id, "customer_id": customer_id, "query": user_query}
            try:
                response = requests.post(f"{fastapi_url}/chat", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.write(f"**Assistant ({result['query_classification']} query):** {result['response']}")
                    st.session_state.chat_history = st.session_state.get('chat_history', []) + \
                                                     [{'role': 'user', 'content': user_query}, {'role': 'assistant', 'content': result['response']}]
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server. Please ensure it is running.")

    st.subheader("Chat History")
    for msg in st.session_state.get('chat_history', [])[::-1]:
        st.text(f"{msg['role'].capitalize()}: {msg['content']}")

    st.sidebar.header("Knowledge Management (Agent/Manager View)")
    new_doc_content = st.sidebar.text_area("New Knowledge Document Content")
    new_doc_metadata_str = st.sidebar.text_input("New Knowledge Document Metadata (e.g., {'type': 'FAQ'})")
    if st.sidebar.button("Add Knowledge"):
        try:
            metadatas = eval(new_doc_metadata_str) if new_doc_metadata_str else {}
            payload = {"documents": [new_doc_content], "metadatas": [metadatas]}
            response = requests.post(f"{fastapi_url}/knowledge/update", json=payload)
            if response.status_code == 200:
                st.sidebar.success("Knowledge added!")
            else:
                st.sidebar.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.sidebar.error(f"Invalid metadata format or API error: {e}")

    new_index_name = st.sidebar.text_input("New Index Name for Hotswap", "new_telecom_knowledge")
    if st.sidebar.button("Hotswap Knowledge Index"):
        payload = {"new_index_name": new_index_name}
        response = requests.post(f"{fastapi_url}/knowledge/hotswap", json=payload)
        if response.status_code == 200:
            st.sidebar.success(f"Index hotswapped to {new_index_name}!")
        else:
            st.sidebar.error(f"Error: {response.status_code} - {response.text}")

    st.sidebar.header("Diagnostics")
    if st.sidebar.button("Show Customer Memory"):
        response = requests.get(f"{fastapi_url}/customer_memory/{customer_id}")
        if response.status_code == 200:
            st.sidebar.json(response.json())
        else:
            st.sidebar.error("Failed to retrieve customer memory.")

    if st.sidebar.button("Show Session History"):
        response = requests.get(f"{fastapi_url}/session_history/{session_id}")
        if response.status_code == 200:
            st.sidebar.json(response.json())
        else:
            st.sidebar.error("Failed to retrieve session history.")

    if st.sidebar.button("Run LLM Fine-tuning Simulation"):
        efficient_llm_fine_tuning()
        st.sidebar.info("LLM fine-tuning simulation completed. Check console for output.")


# To run the FastAPI server:
# uvicorn customer_support_llm_assistant:app --reload
# To run the Streamlit UI:
# streamlit run customer_support_llm_assistant.py
# Ensure both are running to interact. Start FastAPI first.
