"""
This module implements a Medical Literature Q&A System with Distractor-Aware Finetuning.
It demonstrates data preprocessing, document chunking, vector embedding, retrieval,
and a conceptual framework for distractor-aware LLM finetuning and RAG inference.
"""

import re
import random
from typing import List, Dict, Any

# --- Simulated Library Imports (as actual imports would require installation) ---
# For a real application, you would install these:
# pip install langchain chromadb sentence-transformers transformers

# Placeholder for langchain's RecursiveCharacterTextSplitter
class MockRecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # Simple split for demonstration. A real splitter is more sophisticated.
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
        return [c for c in chunks if c.strip()]

# Placeholder for sentence_transformers.SentenceTransformer
class MockSentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, sentences: List[str], convert_to_tensor: bool = False) -> List[List[float]]:
        # Simulate embeddings with random vectors for demonstration
        # In a real scenario, this would generate meaningful embeddings
        return [[random.uniform(-1, 1) for _ in range(384)] for _ in sentences]

# Placeholder for chromadb.Client and collections
class MockChromaDBCollection:
    def __init__(self, name: str):
        self.name = name
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        self.id_counter = 0

    def add(self, documents: List[str], embeddings: List[List[float]], metadatas: List[Dict], ids: List[str]):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def query(self, query_embeddings: List[List[float]], n_results: int = 10) -> Dict:
        # Simple cosine similarity simulation for demonstration
        results = []
        for q_emb in query_embeddings:
            similarities = []
            for i, doc_emb in enumerate(self.embeddings):
                # Simulate dot product for similarity
                dot_product = sum(q * d for q, d in zip(q_emb, doc_emb))
                similarities.append((dot_product, i))
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_indices = [idx for _, idx in similarities[:n_results]]
            results.append({
                "ids": [self.ids[i] for i in top_indices],
                "documents": [self.documents[i] for i in top_indices],
                "metadatas": [self.metadatas[i] for i in top_indices]
            })
        return results[0] if results else {"ids": [], "documents": [], "metadatas": []}

class MockChromaDBClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name: str) -> MockChromaDBCollection:
        if name not in self.collections:
            self.collections[name] = MockChromaDBCollection(name)
        return self.collections[name]

    def delete_collection(self, name: str):
        if name in self.collections:
            del self.collections[name]

# Placeholder for transformers models (conceptual)
class MockLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"MockLLM initialized: {model_name}")

    def finetune(self, dataset: List[Dict[str, Any]], epochs: int = 3, learning_rate: float = 2e-5):
        print(f"\n--- Simulating Distractor-Aware Finetuning for {self.model_name} ---")
        print(f"Training on {len(dataset)} examples for {epochs} epochs.")
        print("This step involves feeding the LLM with contexts containing both golden and distractor documents.")
        print("The LLM learns to identify relevant information and ignore irrelevant parts.")
        print("Finetuning complete (simulated).")

    def generate_answer(self, query: str, context: List[str]) -> str:
        # Simulate answer generation based on context.
        # A real LLM would process the context and query to produce a coherent answer.
        context_str = "\n".join(context)
        
        # Simple heuristic to simulate distractor-awareness:
        # If a keyword from the query appears in a context chunk, prioritize that chunk.
        relevant_snippets = []
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
        
        for i, chunk in enumerate(context):
            if any(keyword in chunk.lower() for keyword in query_keywords):
                relevant_snippets.append(f"[Relevant Snippet {i+1}]: {chunk}")
            else:
                pass # Disregard distractor-like content conceptually

        if not relevant_snippets:
            return f"Based on the provided information, I couldn't find a direct answer to '{query}'. (Processed {len(context)} documents, no strong matches found for keywords)."
        
        simulated_answer = f"Based on the relevant information from the medical literature, regarding \"{query}\":\n\n" \
                           + "\n".join(relevant_snippets[:2]) \
                           + "\n\n(A real LLM would synthesize a more coherent answer, discerning relevant facts from the provided context chunks, even amidst distractors.)"
        
        return simulated_answer


# --- Configuration --- #
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistral-7b-instruct"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 10 # Number of documents to retrieve for RAG

# --- 1. Data Ingestion and Preprocessing (Simulated) ---
class TextPreprocessor:
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s.,;]', '', text) # Keep basic punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# Simulated Medical Literature (for demonstration)
medical_corpus = [
    "Glioblastoma is an aggressive type of cancer that can occur in the brain or spinal cord. Glioblastoma can be very difficult to treat and a cure is often not possible.",
    "Treatment for glioblastoma typically involves surgery, followed by radiation therapy and chemotherapy. Temozolomide is a common chemotherapy drug used.",
    "Alzheimer's disease is a progressive neurological disorder that causes the brain to shrink and brain cells to die. It is the most common cause of dementia.",
    "Symptoms of Alzheimer's include memory loss, cognitive decline, and behavioral changes. There is no cure, but treatments can help manage symptoms.",
    "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy.",
    "Type 2 diabetes is more common and often develops in adults. Lifestyle changes and medication like metformin are common treatments.",
    "A new study investigates the efficacy of immunotherapy drugs in treating advanced lung cancer. Patients showed promising responses.",
    "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle modifications are often the first line of treatment."
]

# Prepare chunks
text_preprocessor = TextPreprocessor()
chunker = MockRecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

processed_chunks = []
for i, doc in enumerate(medical_corpus):
    cleaned_doc = text_preprocessor.clean_text(doc)
    chunks = chunker.split_text(cleaned_doc)
    for j, chunk in enumerate(chunks):
        processed_chunks.append({
            "id": f"doc_{i}_chunk_{j}",
            "content": chunk,
            "metadata": {"source_doc_id": i, "chunk_idx": j}
        })

# --- 2. Retrieval System (Vector Database & Retriever) ---
embedding_model = MockSentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = MockChromaDBClient()
collection = chroma_client.get_or_create_collection("medical_literature")

doc_ids = [c["id"] for c in processed_chunks]
doc_contents = [c["content"] for c in processed_chunks]
doc_metadatas = [c["metadata"] for c in processed_chunks]
doc_embeddings = embedding_model.encode(doc_contents)

collection.add(documents=doc_contents, embeddings=doc_embeddings, metadatas=doc_metadatas, ids=doc_ids)

def retrieve_documents(query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
    query_embedding = embedding_model.encode([query])
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    retrieved_docs = []
    for i in range(len(results["ids"])):
        retrieved_docs.append({
            "id": results["ids"][i],
            "content": results["documents"][i],
            "metadata": results["metadatas"][i]
        })
    return retrieved_docs


# --- 3. LLM Finetuning (Distractor-Aware Finetuning Data Preparation) ---

# Simulated training data for distractor-aware finetuning
# Format: (query, golden_document_chunk_id, expected_answer_keyword)
# In a real scenario, this would come from a carefully curated dataset.
finetuning_qa_data = [
    ("What are the treatments for glioblastoma?", "doc_1_chunk_0", "surgery, radiation, chemotherapy, temozolomide"),
    ("Tell me about Alzheimer's disease symptoms.", "doc_3_chunk_0", "memory loss, cognitive decline, behavioral changes"),
    ("What is type 2 diabetes and its treatments?", "doc_5_chunk_0", "metformin, lifestyle changes"),
    ("What is glioblastoma?", "doc_0_chunk_0", "aggressive cancer, brain, spinal cord"),
]

def generate_distractor_aware_finetuning_dataset(
    qa_data: List[tuple],
    all_chunks: List[Dict[str, Any]],
    retriever_func: Any,
    num_distractors: int = 3
) -> List[Dict[str, Any]]:
    dataset = []
    all_chunk_ids = {chunk["id"]: chunk["content"] for chunk in all_chunks}

    print(f"\n--- Generating Distractor-Aware Finetuning Dataset with {num_distractors} distractors per example ---")

    for query, golden_chunk_id, expected_answer_keyword in qa_data:
        golden_chunk_content = all_chunk_ids.get(golden_chunk_id)
        if not golden_chunk_content:
            print(f"Warning: Golden chunk ID {golden_chunk_id} not found. Skipping.")
            continue

        # Retrieve a mix of relevant and potentially irrelevant documents
        retrieved_for_distractors = retriever_func(query, k=TOP_K_RETRIEVAL * 2) # Retrieve more to find distractors

        context_chunks = [golden_chunk_content] # Start with the golden chunk
        added_distractor_ids = set([golden_chunk_id])

        # Add distractors (chunks that are not the golden chunk)
        for doc in retrieved_for_distractors:
            if doc["id"] not in added_distractor_ids and len(context_chunks) < num_distractors + 1:
                context_chunks.append(doc["content"])
                added_distractor_ids.add(doc["id"])
            if len(context_chunks) == num_distractors + 1: # Golden + num_distractors
                break
        
        # Shuffle the context to ensure the LLM learns to identify relevant parts regardless of position
        random.shuffle(context_chunks)

        dataset.append({
            "query": query,
            "context": context_chunks,
            "answer": expected_answer_keyword, # For finetuning, this could be the full answer
                                               # or an instruction to extract from context.
        })
        print(f"  Generated example for query: '{query}' with {len(context_chunks)-1} distractors.")

    print("Dataset generation complete.")
    return dataset


# Initialize the (mock) LLM
llm = MockLLM(LLM_MODEL_NAME)

# Generate the finetuning dataset
distractor_aware_dataset = generate_distractor_aware_finetuning_dataset(
    finetuning_qa_data, 
    processed_chunks, 
    retrieve_documents,
    num_distractors=3
)

# Simulate LLM Finetuning
llm.finetune(distractor_aware_dataset, epochs=3)


# --- 4. RAG System (Inference with Distractor-Aware LLM) ---

def medical_qa_system(query: str) -> str:
    print(f"\n--- Processing Query: '{query}' ---")
    
    # 1. Retrieve documents
    retrieved_docs = retrieve_documents(query, k=TOP_K_RETRIEVAL)
    retrieved_contents = [doc["content"] for doc in retrieved_docs]
    
    print(f"Retrieved {len(retrieved_docs)} documents.")
    # print("Retrieved Contents (first 200 chars each):")
    # for i, content in enumerate(retrieved_contents):
    #     print(f"  Doc {i+1}: {content[:200]}...")

    # 2. Generate answer using the distractor-aware LLM
    answer = llm.generate_answer(query, retrieved_contents)
    
    return answer


# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- Medical Literature Q&A System Demo ---")
    
    # Example Query 1: Focus on Glioblastoma treatment (should filter distractors)
    query1 = "What are the recommended treatments for glioblastoma?"
    answer1 = medical_qa_system(query1)
    print(f"\nQuery: {query1}\nAnswer: {answer1}")

    # Example Query 2: Focus on Alzheimer's symptoms
    query2 = "Describe the symptoms of Alzheimer's disease."
    answer2 = medical_qa_system(query2)
    print(f"\nQuery: {query2}\nAnswer: {answer2}")
    
    # Example Query 3: General medical query, might hit some distractors from setup
    query3 = "What is hypertension?"
    answer3 = medical_qa_system(query3)
    print(f"\nQuery: {query3}\nAnswer: {answer3}")

    # Example Query 4: A query where a direct golden chunk might not be in the small corpus
    query4 = "What are new immunotherapies for lung cancer?"
    answer4 = medical_qa_system(query4)
    print(f"\nQuery: {query4}\nAnswer: {answer4}")

    # Clean up (for mock ChromaDB)
    chroma_client.delete_collection("medical_literature")
    print("\nCleaned up mock ChromaDB collection.")

