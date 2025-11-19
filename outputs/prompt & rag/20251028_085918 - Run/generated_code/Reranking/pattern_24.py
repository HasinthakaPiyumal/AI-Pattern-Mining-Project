
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

app = FastAPI()

# --- 1. Knowledge Base Simulation ---
# In a real-world scenario, this would be a large, indexed database of medical texts.
medical_knowledge_base = [
    "Aspirin is commonly used for pain relief, fever reduction, and anti-inflammatory purposes. It can also be prescribed to prevent blood clots.",
    "Diabetes mellitus is a chronic metabolic disease characterized by high blood glucose levels. Type 1 diabetes is an autoimmune condition, while Type 2 diabetes is often associated with insulin resistance and lifestyle factors.",
    "Hypertension, or high blood pressure, significantly increases the risk of heart disease, stroke, and kidney disease. Lifestyle modifications and medications are common treatments.",
    "The COVID-19 pandemic is caused by the SARS-CoV-2 virus. Symptoms range from mild respiratory illness to severe pneumonia. Vaccination is a key preventive measure.",
    "Antibiotics are medications that fight bacterial infections. They do not work against viral infections like the common cold or flu. Overuse can lead to antibiotic resistance.",
    "Asthma is a chronic respiratory condition where airways narrow and swell, producing extra mucus. This can make breathing difficult and trigger coughing, wheezing, and shortness of breath.",
    "The human heart has four chambers: two atria and two ventricles. It pumps blood throughout the body, delivering oxygen and nutrients to tissues and removing waste products.",
    "Cancer is a disease in which some of the body's cells grow uncontrollably and spread to other parts of the body. There are many types of cancer, and treatments vary widely.",
    "Vaccines stimulate the body's immune system to protect against specific infections. They have significantly reduced the incidence of many infectious diseases worldwide.",
    "Stroke occurs when the blood supply to part of your brain is interrupted or reduced, depriving brain tissue of oxygen and nutrients. Brain cells begin to die within minutes.",
    "Migraines are severe headaches often accompanied by throbbing pain or a pulsing sensation, usually on one side of the head. They are often accompanied by nausea, vomiting, and extreme sensitivity to light and sound."
]

# --- 2. Embedding Model and Vector Store ---
# Load a pre-trained sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for the knowledge base
kb_embeddings = embedding_model.encode(medical_knowledge_base, convert_to_tensor=True)

# Create a FAISS index
dimension = kb_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension) # Using Inner Product for cosine similarity
faiss_index.add(kb_embeddings.cpu().numpy())

# --- 3. Simulated Language Model (LLM) ---
def simulated_llm_response(prompt: str) -> str:
    # In a real application, this would be an API call to OpenAI, Cohere, HuggingFace, etc.
    # For demonstration, we'll just echo the prompt and add a generic answer.
    if "COVID-19" in prompt:
        return "The prompt mentioned COVID-19. Based on the provided context, the COVID-19 pandemic is caused by the SARS-CoV-2 virus, and vaccination is a key preventive measure. Please consult a healthcare professional for specific medical advice."
    elif "Aspirin" in prompt:
        return "The prompt mentioned Aspirin. Based on the provided context, Aspirin is used for pain relief, fever reduction, anti-inflammatory purposes, and preventing blood clots. Always follow medical guidance."
    elif "Diabetes" in prompt:
        return "The prompt mentioned Diabetes. Based on the provided context, Diabetes mellitus is a chronic metabolic disease characterized by high blood glucose levels, with Type 1 and Type 2 variants. Consult a doctor for diagnosis and treatment."
    return f"I am a simulated AI. Based on the information provided: {prompt}. Always consult a medical professional for accurate diagnoses and treatment plans."

# --- 4. Retrieval and Reranking Logic ---
def retrieve_and_rerank_documents(query: str, top_k: int = 5) -> List[str]:
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # FAISS search: D for distances, I for indices
    distances, indices = faiss_index.search(query_embedding.cpu().numpy().reshape(1, -1), top_k * 2) # Retrieve more for reranking

    # Get the raw retrieved documents
    retrieved_docs_with_scores = []
    for i, idx in enumerate(indices[0]):
        if idx < len(medical_knowledge_base):
            retrieved_docs_with_scores.append({
                "doc": medical_knowledge_base[idx],
                "score": distances[0][i] # FAISS IP index gives similarity score directly
            })
    
    # Simple reranking: Here, we just rely on the FAISS similarity score, which is already a good reranker.
    # In a more advanced scenario, you'd use a dedicated cross-encoder reranker model (e.g., from `transformers`).
    # For this example, we'll just sort by score (descending for IP/cosine similarity).
    reranked_docs = sorted(retrieved_docs_with_scores, key=lambda x: x['score'], reverse=True)[:top_k]

    return [item['doc'] for item in reranked_docs]

# --- 5. Conditional Retrieval Logic ---
def should_retrieve(query: str) -> bool:
    # Simple heuristic: If the query contains medical keywords, retrieve.
    # In a real system, this could involve a classification model or keyword extraction.
    medical_keywords = ["symptom", "treatment", "cure", "disease", "medication", "diagnosis", "what is", "how to treat", "causes of"]
    for keyword in medical_keywords:
        if keyword in query.lower():
            return True
    # Also, if the query is very short, it might be a general question that doesn't need grounding.
    if len(query.split()) < 3:
        return False
    return True

# --- FastAPI Endpoint ---
class MedicalQuery(BaseModel):
    query: str

@app.post("/ask_medical_question")
async def ask_medical_question(medical_query: MedicalQuery):
    query = medical_query.query
    context_documents = []

    if should_retrieve(query):
        context_documents = retrieve_and_rerank_documents(query, top_k=3)
        print(f"Retrieved documents for query '{query}': {context_documents}")
    else:
        print(f"Skipping retrieval for query '{query}' based on conditional logic.")

    # Construct the prompt for the LLM
    if context_documents:
        context_str = "\n\nContext:\n" + "\n".join([f"- {doc}" for doc in context_documents])
        full_prompt = f"Answer the following medical question accurately and based on the provided context. If the context does not contain enough information, state that.\n\nQuestion: {query}{context_str}\n\nAnswer:"
    else:
        # If no retrieval, or retrieval was skipped, ask the LLM without specific grounding
        full_prompt = f"Answer the following medical question accurately. State if you are unable to provide a precise answer.\n\nQuestion: {query}\n\nAnswer:"
    
    # Get response from the simulated LLM
    llm_answer = simulated_llm_response(full_prompt)

    return {
        "query": query,
        "grounding_documents": context_documents,
        "answer": llm_answer,
        "source_attribution": [doc for doc in context_documents] # Simple attribution, could be more granular
    }

# To run this application:
# 1. Save the code as `medical_qa_system.py`
# 2. Install necessary libraries: `pip install fastapi uvicorn sentence-transformers faiss-cpu pydantic`
# 3. Run from your terminal: `uvicorn medical_qa_system:app --reload`
# 4. Access the API documentation at http://127.0.0.1:8000/docs
