import os
import logging
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# --- Configuration and Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INDEX_DIR = "./indexes"
CURRENT_INDEX_FILE = os.path.join(INDEX_DIR, "current_index.faiss")
KNOWLEDGE_BASE_PATH = os.path.join(INDEX_DIR, "knowledge_base.csv")

os.makedirs(INDEX_DIR, exist_ok=True)

# --- 1. Data Ingestion and Indexing Pipeline ---

class DataLoader:
    def __init__(self, data_path: str = KNOWLEDGE_BASE_PATH):
        self.data_path = data_path
        self.knowledge_base = []

    def load_data(self):
        # Simulate loading data from various sources
        # In a real scenario, this would involve scraping, API calls, DB queries
        logger.info(f"Loading simulated data from {self.data_path}")
        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
            self.knowledge_base = df.to_dict(orient='records')
        else:
            # Dummy initial data
            self.knowledge_base = [
                {"id": "prod_001", "type": "product", "content": "Product A is a high-quality smartphone with a 6.1-inch display and 128GB storage. Price: $799. Available in Black, White, Blue."}, 
                {"id": "prod_002", "type": "product", "content": "Product B is a wireless earbud set with noise cancellation and 24-hour battery life. Price: $149. Color: Black."}, 
                {"id": "policy_001", "type": "policy", "content": "Our return policy allows returns within 30 days of purchase for a full refund. Items must be in original condition."}, 
                {"id": "shipping_001", "type": "shipping", "content": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days and costs $15."}, 
                {"id": "faq_001", "type": "faq", "content": "How can I track my order? You can track your order using the tracking number provided in your shipping confirmation email on our website."}, 
                {"id": "promo_001", "type": "promotion", "content": "Get 10% off all accessories this week! Use code ACC10 at checkout. Offer valid until next Sunday."}, 
            ]
            df = pd.DataFrame(self.knowledge_base)
            df.to_csv(self.data_path, index=False)
        logger.info(f"Loaded {len(self.knowledge_base)} knowledge items.")
        return self.knowledge_base

    def chunk_data(self, data: List[Dict]) -> List[Dict]:
        # For simplicity, each item is a chunk. In reality, large texts would be split.
        logger.info(f"Chunking {len(data)} items.")
        return data

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded Embedding Model: {model_name}")

    def encode(self, texts: List[str]) -> np.ndarray:
        logger.info(f"Encoding {len(texts)} texts.")
        return self.model.encode(texts, show_progress_bar=False)

class IndexManager:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.metadata = [] # Stores original content alongside embeddings
        self.load_index(CURRENT_INDEX_FILE)

    def create_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_index.add(embeddings)
        self.metadata = metadata
        logger.info(f"Created new FAISS index with {self.faiss_index.ntotal} vectors.")

    def save_index(self, path: str = CURRENT_INDEX_FILE):
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, path)
            # Save metadata alongside the index
            pd.DataFrame(self.metadata).to_csv(path + ".metadata.csv", index=False)
            logger.info(f"FAISS index and metadata saved to {path}")
        else:
            logger.warning("No index to save.")

    def load_index(self, path: str = CURRENT_INDEX_FILE):
        if os.path.exists(path) and os.path.exists(path + ".metadata.csv"):
            self.faiss_index = faiss.read_index(path)
            self.metadata = pd.read_csv(path + ".metadata.csv").to_dict(orient='records')
            logger.info(f"FAISS index and metadata loaded from {path} with {self.faiss_index.ntotal} vectors.")
        else:
            logger.warning(f"No existing index found at {path}. A new index will be created upon first indexing.")
            self.faiss_index = None # Ensure it's None if file doesn't exist
            self.metadata = []

    def hotswap_index(self, new_index_path: str):
        try:
            new_faiss_index = faiss.read_index(new_index_path)
            new_metadata = pd.read_csv(new_index_path + ".metadata.csv").to_dict(orient='records')
            self.faiss_index = new_faiss_index
            self.metadata = new_metadata
            logger.info(f"Index hotswapped successfully from {new_index_path}. New index has {self.faiss_index.ntotal} vectors.")
            # Optionally, update the CURRENT_INDEX_FILE to point to the new one
            self.save_index(CURRENT_INDEX_FILE) # Save the new one as the current one
        except Exception as e:
            logger.error(f"Error during index hotswap: {e}")

# --- Indexing Service (FastAPI) ---

from fastapi import FastAPI, HTTPException
import uvicorn
import threading

app = FastAPI()

embedding_model = EmbeddingModel()
index_manager = IndexManager(embedding_dim=embedding_model.model.get_sentence_embedding_dimension())

@app.post("/index_data")
async def index_data():
    logger.info("Indexing data request received.")
    try:
        data_loader = DataLoader()
        raw_data = data_loader.load_data()
        chunks = data_loader.chunk_data(raw_data)

        if not chunks:
            raise ValueError("No data to index.")

        texts_to_encode = [chunk["content"] for chunk in chunks]
        embeddings = embedding_model.encode(texts_to_encode)

        new_index_name = f"index_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.faiss"
        new_index_path = os.path.join(INDEX_DIR, new_index_name)

        temp_index_manager = IndexManager(embedding_dim=embedding_model.model.get_sentence_embedding_dimension())
        temp_index_manager.create_index(embeddings, chunks)
        temp_index_manager.save_index(new_index_path)

        # Now hotswap the main index_manager with the newly created one
        index_manager.hotswap_index(new_index_path)

        return {"status": "success", "message": f"New index created and hotswapped: {new_index_name}", "indexed_items": len(chunks)}
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_knowledge_base")
async def update_knowledge_base(new_item: Dict):
    logger.info(f"Updating knowledge base with new item: {new_item}")
    try:
        # Load existing data
        data_loader = DataLoader()
        current_kb = data_loader.load_data()
        
        # Add new item (assign a simple ID if not provided)
        if "id" not in new_item:
            new_item["id"] = f"manual_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}_{len(current_kb)}"
        current_kb.append(new_item)

        # Save updated knowledge base
        updated_df = pd.DataFrame(current_kb)
        updated_df.to_csv(KNOWLEDGE_BASE_PATH, index=False)
        logger.info("Knowledge base updated successfully. Triggering re-indexing...")

        # Trigger re-indexing after updating the knowledge base
        await index_data() # This will create a new index and hotswap it

        return {"status": "success", "message": "Knowledge base updated and new index hotswapped.", "new_item_id": new_item["id"]}
    except Exception as e:
        logger.error(f"Failed to update knowledge base or re-index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 2. Chatbot Core (RAG System) ---

class RetrievalModule:
    def __init__(self, index_manager: IndexManager, embedding_model: EmbeddingModel):
        self.index_manager = index_manager
        self.embedding_model = embedding_model
        logger.info("Retrieval Module initialized.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index_manager.faiss_index is None:
            logger.warning("No FAISS index available for retrieval. Please index data first.")
            return []

        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index_manager.faiss_index.search(query_embedding, top_k)

        retrieved_chunks = []
        for i, dist in zip(indices[0], distances[0]):
            if i != -1: # Ensure a valid index was found
                chunk = self.index_manager.metadata[i]
                retrieved_chunks.append({"content": chunk["content"], "score": float(dist), "id": chunk.get("id", "N/A")})
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: '{query}'")
        return retrieved_chunks

class GenerationModule:
    def __init__(self, llm_model_name: str = "mock_LLM"):
        self.llm_model_name = llm_model_name
        logger.info(f"Generation Module initialized with {llm_model_name}.")

    def generate_response(self, query: str, retrieved_chunks: List[Dict]) -> str:
        if not retrieved_chunks:
            return "I'm sorry, I couldn't find relevant information to answer your question. Could you please rephrase it?"
        
        context = "\n".join([chunk["content"] for chunk in retrieved_chunks])
        
        # Simulate LLM response generation
        prompt = f"Based on the following information, answer the question: '{query}'\n\nContext:\n{context}\n\nAnswer:"
        
        # In a real scenario, this would call an LLM API or a local model
        # For this example, we'll generate a simple mock response
        mock_response = f"(Simulated LLM response based on context) For your query about '{query}', I found: {context[:200]}..."
        logger.info(f"Generated mock response for query: '{query}'")
        return mock_response

class Chatbot:
    def __init__(self, retrieval_module: RetrievalModule, generation_module: GenerationModule):
        self.retrieval_module = retrieval_module
        self.generation_module = generation_module
        logger.info("Chatbot initialized.")

    def ask(self, query: str) -> str:
        retrieved_chunks = self.retrieval_module.retrieve(query)
        response = self.generation_module.generate_response(query, retrieved_chunks)
        return response

# --- Streamlit UI for Chatbot ---

import streamlit as st

# Initialize chatbot components (these instances will be shared across Streamlit sessions)
# This is crucial for the hotswapping to work across the FastAPI and Streamlit parts
retrieval_module = RetrievalModule(index_manager=index_manager, embedding_model=embedding_model)
generation_module = GenerationModule()
chatbot_instance = Chatbot(retrieval_module, generation_module)

def run_streamlit_ui():
    st.set_page_config(page_title="E-commerce Customer Support Chatbot")
    st.title("🛒 E-commerce Customer Support Chatbot")
    st.subheader("Powered by Index Hotswapping for real-time knowledge updates")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                response = chatbot_instance.ask(prompt)
                full_response = response
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.sidebar.title("System Status")
    st.sidebar.markdown(f"**Current Index:** {os.path.basename(CURRENT_INDEX_FILE)}")
    if index_manager.faiss_index:
        st.sidebar.markdown(f"**Vectors in Index:** {index_manager.faiss_index.ntotal}")
    else:
        st.sidebar.markdown("**Vectors in Index:** Not loaded")

    if st.sidebar.button("Trigger Manual Re-indexing (Hot-swap)"):
        st.sidebar.info("Triggering re-indexing... Check console for status.")
        # In a real app, this would call the FastAPI endpoint for re-indexing
        # For this combined file, we'll directly call the index_data function's logic
        try:
            data_loader = DataLoader()
            raw_data = data_loader.load_data()
            chunks = data_loader.chunk_data(raw_data)
            texts_to_encode = [chunk["content"] for chunk in chunks]
            embeddings = embedding_model.encode(texts_to_encode)

            new_index_name = f"index_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.faiss"
            new_index_path = os.path.join(INDEX_DIR, new_index_name)

            temp_index_manager = IndexManager(embedding_dim=embedding_model.model.get_sentence_embedding_dimension())
            temp_index_manager.create_index(embeddings, chunks)
            temp_index_manager.save_index(new_index_path)

            index_manager.hotswap_index(new_index_path)
            st.sidebar.success("Index re-indexed and hotswapped successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error during re-indexing: {e}")


def run_fastapi_server():
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import sys

    # Initial indexing upon startup if no index exists
    if index_manager.faiss_index is None or index_manager.faiss_index.ntotal == 0:
        logger.info("No initial index found or index is empty. Performing initial data indexing.")
        try:
            data_loader = DataLoader()
            raw_data = data_loader.load_data()
            chunks = data_loader.chunk_data(raw_data)
            texts_to_encode = [chunk["content"] for chunk in chunks]
            embeddings = embedding_model.encode(texts_to_encode)
            index_manager.create_index(embeddings, chunks)
            index_manager.save_index(CURRENT_INDEX_FILE)
            logger.info("Initial index created and saved.")
        except Exception as e:
            logger.error(f"Initial indexing failed: {e}")

    if len(sys.argv) > 1 and sys.argv[1] == "fastapi":
        run_fastapi_server()
    elif len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        run_streamlit_ui()
    else:
        logger.info("Please specify 'fastapi' to run the FastAPI server or 'streamlit' to run the Streamlit UI.")
        logger.info("Example: python your_script_name.py fastapi")
        logger.info("Example: streamlit run your_script_name.py streamlit")
