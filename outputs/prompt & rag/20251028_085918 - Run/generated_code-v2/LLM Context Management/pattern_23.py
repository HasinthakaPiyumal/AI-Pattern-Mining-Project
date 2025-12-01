import os
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Request
from pydantic import BaseModel

# --- Configuration Constants ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "distilgpt2"
FAISS_INDEX_DIR = "indices"
ACTIVE_INDEX_FILENAME = "active_index.faiss"
DOCUMENTS_JSON_PATH = "data/medical_docs.json"

# Ensure directories exist
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DOCUMENTS_JSON_PATH), exist_ok=True)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class QAResponse(BaseModel):
    answer: str
    sources: list[str]

class HotswapRequest(BaseModel):
    new_index_filename: str

class HotswapResponse(BaseModel):
    status: str
    message: str

# --- Data Loader and Chunker ---
class DataLoader:
    @staticmethod
    def load_documents(file_path: str) -> list[str]:
        if not os.path.exists(file_path):
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("documents", [])

    @staticmethod
    def chunk_document(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks

# --- Embedding Manager ---
class EmbeddingManager:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

# --- Index Manager ---
class IndexManager:
    def __init__(self, embedding_manager: EmbeddingManager, index_dir: str, active_index_filename: str):
        self.embedding_manager = embedding_manager
        self.index_dir = index_dir
        self._active_index_path = os.path.join(self.index_dir, active_index_filename)
        self._active_index = None
        self._active_doc_chunks = []
        self._load_active_index()

    def _load_active_index(self):
        if os.path.exists(self._active_index_path):
            try:
                loaded_data = np.load(self._active_index_path, allow_pickle=True)
                self._active_index = faiss.read_index(loaded_data["index_bytes"].item())
                self._active_doc_chunks = loaded_data["doc_chunks"].tolist()
                print(f"Loaded active index from {self._active_index_path} with {len(self._active_doc_chunks)} chunks.")
            except Exception as e:
                print(f"Error loading active index: {e}")
                self._active_index = None
                self._active_doc_chunks = []
        else:
            print("No active index found. Initialize with build_index.")

    def build_index(self, documents: list[str], output_filename: str) -> bool:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(DataLoader.chunk_document(doc))
        
        if not all_chunks:
            print("No chunks to build index.")
            return False

        embeddings = self.embedding_manager.get_embeddings(all_chunks)
        d = embeddings.shape[1]
        new_index = faiss.IndexFlatL2(d)
        new_index.add(embeddings)

        self.save_index(new_index, all_chunks, output_filename)
        print(f"New index built and saved to {output_filename} with {len(all_chunks)} chunks.")
        return True

    def save_index(self, index: faiss.Index, doc_chunks: list[str], filename: str):
        index_path = os.path.join(self.index_dir, filename)
        index_bytes = faiss.write_index(index)
        np.save(index_path, {"index_bytes": index_bytes, "doc_chunks": doc_chunks})

    def load_index_from_file(self, filename: str) -> tuple[faiss.Index, list[str]]:
        index_path = os.path.join(self.index_dir, filename)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        loaded_data = np.load(index_path, allow_pickle=True)
        loaded_index = faiss.read_index(loaded_data["index_bytes"].item())
        loaded_chunks = loaded_data["doc_chunks"].tolist()
        return loaded_index, loaded_chunks

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[str]:
        if self._active_index is None:
            return []
        
        D, I = self._active_index.search(np.array([query_embedding]), k)
        retrieved_chunks = [self._active_doc_chunks[idx] for idx in I[0] if idx != -1]
        return retrieved_chunks

    def hotswap_index(self, new_index_filename: str) -> tuple[bool, str]:
        try:
            new_index, new_doc_chunks = self.load_index_from_file(new_index_filename)
            # Atomically update references
            self._active_index = new_index
            self._active_doc_chunks = new_doc_chunks
            # Update the symlink/reference file if needed, or simply overwrite the active_index.faiss
            # For simplicity, we just loaded into memory. For persistence, you'd copy/rename files.
            print(f"Successfully hotswapped to {new_index_filename}")
            return True, f"Successfully hotswapped to {new_index_filename}"
        except FileNotFoundError:
            return False, f"Error: New index file {new_index_filename} not found."
        except Exception as e:
            return False, f"Error during hotswap: {e}"

# --- LLM Generator ---
class LLMGenerator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_answer(self, prompt: str, max_length: int = 250, num_return_sequences: int = 1) -> str:
        # Ensure the prompt is not too long for the model's context window
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True)
        generated_text = self.generator(inputs.input_ids[0].tolist(), 
                                        max_new_tokens=max_length - len(inputs.input_ids[0]), 
                                        num_return_sequences=num_return_sequences,
                                        pad_token_id=self.tokenizer.eos_token_id)[0]["generated_text"]
        
        # Remove the input prompt from the generated text if the generator includes it
        if isinstance(generated_text, str) and generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()

# --- RAG Pipeline ---
class RAGPipeline:
    def __init__(self, embedding_manager: EmbeddingManager, index_manager: IndexManager, llm_generator: LLMGenerator):
        self.embedding_manager = embedding_manager
        self.index_manager = index_manager
        self.llm_generator = llm_generator

    def run(self, query: str, k: int = 5) -> tuple[str, list[str]]:
        query_embedding = self.embedding_manager.get_embedding(query)
        retrieved_chunks = self.index_manager.search(query_embedding, k=k)

        context = "\n".join(retrieved_chunks)
        if not context:
            prompt = f"Answer the following question: {query}"
            sources = []
        else:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            sources = retrieved_chunks # In a real app, these would be source document IDs/titles
        
        answer = self.llm_generator.generate_answer(prompt)
        return answer, sources

# --- FastAPI Application --- 
app = FastAPI(title="Dynamic Medical Q&A System")

# Initialize components
embedding_manager = EmbeddingManager(EMBEDDING_MODEL_NAME)
index_manager = IndexManager(embedding_manager, FAISS_INDEX_DIR, ACTIVE_INDEX_FILENAME)
llm_generator = LLMGenerator(LLM_MODEL_NAME)
rag_pipeline = RAGPipeline(embedding_manager, index_manager, llm_generator)

# --- Dummy Data Generation (for initial setup) ---
def generate_dummy_medical_data(path):
    if os.path.exists(path):
        return
    sample_documents = [
        "Insulin is a hormone produced by the pancreas that helps regulate blood sugar. Type 1 diabetes occurs when the body does not produce insulin.",
        "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
        "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain. It also has antiplatelet effects, which can prevent blood clots.",
        "The COVID-19 pandemic, caused by the SARS-CoV-2 virus, led to widespread respiratory illness. Vaccines were developed rapidly to provide immunity.",
        "Common side effects of statins include muscle pain and liver dysfunction. Regular monitoring is recommended for patients taking statins."
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"documents": sample_documents}, f, indent=2)
    print(f"Generated dummy medical data at {path}")

# Initialize dummy data and an initial FAISS index if they don't exist
generate_dummy_medical_data(DOCUMENTS_JSON_PATH)
if index_manager._active_index is None:
    print("Building initial index...")
    medical_docs = DataLoader.load_documents(DOCUMENTS_JSON_PATH)
    if medical_docs:
        index_manager.build_index(medical_docs, ACTIVE_INDEX_FILENAME)
        index_manager._load_active_index() # Reload to ensure it's set as active
    else:
        print("No documents loaded to build initial index.")

# --- FastAPI Endpoints ---
@app.post("/qa", response_model=QAResponse)
async def answer_question(request: QueryRequest):
    answer, sources = rag_pipeline.run(request.query)
    return QAResponse(answer=answer, sources=sources)

@app.post("/hotswap_index", response_model=HotswapResponse)
async def hotswap_active_index(request: HotswapRequest):
    success, message = index_manager.hotswap_index(request.new_index_filename)
    status = "success" if success else "failure"
    return HotswapResponse(status=status, message=message)

# Example of how to run the application (for development)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# To build a new index offline (example script logic, can be put in scripts/build_index.py):
# from datetime import datetime
# def build_new_index_example():
#     print("Simulating new data collection and index build...")
#     # Imagine new_medical_docs contains updated/new information
#     new_medical_docs = DataLoader.load_documents(DOCUMENTS_JSON_PATH) + [
#         "Recent study shows that a new drug, 'MediCure', significantly reduces symptoms of advanced Parkinson's disease with minimal side effects.",
#         "The latest WHO guidelines recommend booster shots for certain demographics against emerging viral strains."
#     ]
#     new_index_name = f"new_index_{datetime.now().strftime('%Y%m%d%H%M%S')}.faiss"
#     index_manager.build_index(new_medical_docs, new_index_name)
#     print(f"New index ready for hotswap: {new_index_name}")
#     return new_index_name

# You can then call `hotswap_active_index` endpoint with this new_index_name
# via a separate admin tool or scheduled task. This example code is purely for demonstration.