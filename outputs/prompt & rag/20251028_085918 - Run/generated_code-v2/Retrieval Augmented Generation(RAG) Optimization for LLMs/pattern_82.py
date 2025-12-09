import streamlit as st
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import hashlib
import json
import requests
from collections import deque


class KVNode:
    def __init__(self, document_id: str, kv_tensor: np.ndarray = None):
        self.document_id = document_id
        self.kv_tensor = kv_tensor
        self.children = {}

class KnowledgeTreeKVLoader:
    def __init__(self):
        self.root = KVNode("root")
        self.cache_hits = 0
        self.cache_misses = 0

    def _simulate_kv_tensor_generation(self, document_content: str, sequence_hash: str) -> np.ndarray:
        # Simulate KV tensor generation based on document content and its preceding sequence
        # In a real scenario, this would involve an LLM processing the document and its context
        seed_str = document_content + sequence_hash
        seed = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
        np.random.seed(seed)
        return np.random.rand(1, 128)  # Dummy KV tensor: (batch, hidden_dim)

    def get_or_create_kv_tensors(self, document_ids: list, all_documents_content: dict) -> list:
        current_node = self.root
        sequence_kv_tensors = []
        current_path_ids = deque()

        for doc_id in document_ids:
            current_path_ids.append(doc_id)
            path_hash = hashlib.sha256("- ".join(current_path_ids).encode('utf-8')).hexdigest()

            if doc_id not in current_node.children:
                self.cache_misses += 1
                doc_content = all_documents_content.get(doc_id, "")
                kv_tensor = self._simulate_kv_tensor_generation(doc_content, path_hash)
                current_node.children[doc_id] = KVNode(doc_id, kv_tensor)
            else:
                self.cache_hits += 1

            current_node = current_node.children[doc_id]
            sequence_kv_tensors.append(current_node.kv_tensor)

        return sequence_kv_tensors

class DummyEmbeddingModel:
    def encode(self, texts: list) -> np.ndarray:
        return np.array([np.random.rand(768) for _ in texts])

class DummyVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.id_to_content = {}
        self.next_id = 0
        self.embedding_model = DummyEmbeddingModel()

    def add_documents(self, contents: list):
        new_documents = []
        for content in contents:
            doc_id = f"doc_{self.next_id}"
            self.next_id += 1
            new_documents.append((doc_id, content))
            self.id_to_content[doc_id] = content
        
        new_embeddings = self.embedding_model.encode([c for _, c in new_documents])
        self.documents.extend([d_id for d_id, _ in new_documents])
        self.embeddings.extend(new_embeddings)

    def similarity_search(self, query: str, k: int = 3) -> list:
        query_embedding = self.embedding_model.encode([query])[0]
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            similarities.append((similarity, self.documents[i]))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [doc_id for _, doc_id in similarities[:k]]

class DummyLLM:
    def generate_response(self, prompt: str) -> str:
        # Simulate LLM response generation
        if "troubleshooting" in prompt.lower():
            return f"Based on the retrieved information, for troubleshooting the issue described: {prompt.split('Query:')[-1].strip()} you should check the power supply and connection cables first."
        elif "installation" in prompt.lower():
            return f"To install the product mentioned in your query: {prompt.split('Query:')[-1].strip()}, please follow the steps outlined in the manual on page 5."
        return f"I have processed your query: {prompt.split('Query:')[-1].strip()} and retrieved relevant documents. Based on this, I can tell you that a generic response is needed."

class RAGSystem:
    def __init__(self):
        self.vector_store = DummyVectorStore()
        self.kv_cache = KnowledgeTreeKVLoader()
        self.llm = DummyLLM()
        self.all_documents = {}

    def add_documents_to_rag(self, contents: list):
        self.vector_store.add_documents(contents)
        for doc_id, content in zip(self.vector_store.documents[-len(contents):], contents):
            self.all_documents[doc_id] = content

    def process_query(self, query: str) -> str:
        retrieved_doc_ids = self.vector_store.similarity_search(query, k=3)
        retrieved_contents = {doc_id: self.all_documents[doc_id] for doc_id in retrieved_doc_ids}

        # Get or create KV tensors using the Knowledge Tree
        # For this simulation, the actual KV tensors are not directly passed to DummyLLM, 
        # but their generation/retrieval from the tree is demonstrated.
        # In a real RAG, these would influence the LLM's internal state.
        _ = self.kv_cache.get_or_create_kv_tensors(retrieved_doc_ids, retrieved_contents)

        # Construct prompt for LLM
        context_text = "\n".join([f"Document {i+1} (ID: {doc_id}): {retrieved_contents[doc_id]}" 
                                   for i, doc_id in enumerate(retrieved_doc_ids)])
        prompt = f"Context:\n{context_text}\n\nQuery: {query}\n\nAnswer:"

        # Generate response using LLM
        llm_response = self.llm.generate_response(prompt)
        return llm_response

# FastAPI Application
app = FastAPI()
rag_system = RAGSystem()

# Populate with some dummy documents
dummy_docs = [
    "The power supply unit (PSU) for Model X requires 650W. Ensure proper ventilation.",
    "Troubleshooting guide for slow performance on Model X: Check RAM usage and background processes.",
    "Installation instructions for Model X graphics card: Insert into PCIe slot, secure with screw, connect power cables.",
    "General FAQ for Product Y: What are the minimum system requirements? RAM: 8GB, CPU: i5 equivalent.",
    "Troubleshooting network connectivity for Product Y: Verify Wi-Fi settings and router configuration.",
    "Warranty information for all products: 2-year limited warranty from date of purchase."
]
rag_system.add_documents_to_rag(dummy_docs)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    kv_cache_hits: int
    kv_cache_misses: int

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    response_text = rag_system.process_query(request.query)
    return {
        "response": response_text,
        "kv_cache_hits": rag_system.kv_cache.cache_hits,
        "kv_cache_misses": rag_system.kv_cache.cache_misses
    }

# Streamlit UI
st.set_page_config(page_title="Customer Support Assistant")
st.title("Intelligent Customer Support Assistant")

query_input = st.text_area("Enter your customer's query here:", height=150)

if st.button("Get Assistance"):
    if query_input:
        try:
            # Make a request to the FastAPI backend
            fastapi_url = "http://localhost:8000/query" # Assuming FastAPI runs on 8000
            payload = {"query": query_input}
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(fastapi_url, data=json.dumps(payload), headers=headers)
            response.raise_for_status() 
            
            response_data = response.json()
            
            st.subheader("Assistant's Response:")
            st.write(response_data["response"])
            st.markdown(f"<small>KV Cache Hits: {response_data['kv_cache_hits']}, KV Cache Misses: {response_data['kv_cache_misses']}</small>", unsafe_allow_html=True)
            
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Please ensure it is running at http://localhost:8000.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error during API call: {e}")
    else:
        st.warning("Please enter a query.")

st.sidebar.subheader("How to Run:")
st.sidebar.markdown(
    "1. Save this code as `support_assistant_app.py`\n"
    "2. Install dependencies: `pip install streamlit fastapi uvicorn numpy pydantic requests`\n"
    "3. Run the FastAPI backend: `uvicorn support_assistant_app:app --host 0.0.0.0 --port 8000`\n"
    "4. In a separate terminal, run the Streamlit frontend: `streamlit run support_assistant_app.py`"
)
