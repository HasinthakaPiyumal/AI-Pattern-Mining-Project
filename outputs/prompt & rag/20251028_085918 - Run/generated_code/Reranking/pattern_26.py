import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import numpy as np

# --- 1. Simulate Medical Knowledge Base ---
medical_documents = [
    "Symptoms of influenza include fever, cough, sore throat, muscle aches, and fatigue. It is caused by the influenza virus.",
    "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. Type 1 diabetes is an autoimmune condition, while type 2 diabetes is often lifestyle-related.",
    "Hypertension, or high blood pressure, significantly increases the risk of heart disease and stroke. Lifestyle changes and medication are common treatments.",
    "The common cold is a viral infection of the nose and throat. Symptoms include runny nose, sneezing, and congestion, usually milder than flu.",
    "Migraine is a severe type of headache often accompanied by throbbing pain, sensitivity to light and sound, and nausea. Triggers can vary widely.",
    "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon. It typically causes pain in the lower right abdomen.",
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm, fever, chills, and difficulty breathing.",
    "Asthma is a chronic lung disease that inflames and narrows the airways, causing recurring periods of wheezing, chest tightness, shortness of breath, and coughing.",
    "COVID-19 symptoms can range from mild to severe, and include fever, cough, fatigue, loss of taste or smell, and shortness of breath. It is caused by the SARS-CoV-2 virus."
]

# Load a pre-trained sentence transformer model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedding_model()

# Pre-compute embeddings for the knowledge base
corpus_embeddings = embedder.encode(medical_documents, convert_to_tensor=True)

# --- 2. Load Language Model for Generation and Zero-shot Reranking ---
# Using a smaller T5 model for quicker inference in a demo setup
@st.cache_resource
def load_llm_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-small", device=0 if torch.cuda.is_available() else -1)

llm_pipeline = load_llm_pipeline()

# --- Core Functions ---

def retrieve_documents(query: str, top_k: int = 3) -> list[str]:
    """Retrieves top_k most similar documents from the knowledge base."""
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    retrieved_docs = []
    for score, idx in zip(top_results[0], top_results[1]):
        retrieved_docs.append(f"Document (Score: {score:.2f}): {medical_documents[idx]}")
    return retrieved_docs

def zero_shot_rerank_documents(query: str, documents: list[str]) -> list[str]:
    """Reranks documents based on their relevance to the query using the LLM's understanding.
       This is a simplified approach, a dedicated cross-encoder would be more robust.
    """
    reranked_scores = []
    for doc_text in documents:
        # Simple prompt to assess relevance. A more sophisticated approach would involve NLI.
        prompt = f"Given the query: '{query}', how relevant is the following document? '{doc_text}'\nRelevance score (0-10):"
        response = llm_pipeline(prompt, max_new_tokens=10, num_return_sequences=1)
        try:
            # Try to parse a score, default to a lower score if not parsable
            score_str = response[0]['generated_text'].split(':')[-1].strip()
            score = float(score_str) if score_str.replace('.', '', 1).isdigit() else 5.0
        except (IndexError, ValueError):
            score = 5.0 # Default score if LLM doesn't return a clear number
        reranked_scores.append((score, doc_text))

    # Sort by score in descending order
    reranked_scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in reranked_scores]

def conditionally_trigger_retrieval(user_query: str) -> bool:
    """A placeholder for logic to decide if knowledge retrieval is needed.
       For this demo, we'll always trigger retrieval if the query is not empty.
    """
    return bool(user_query.strip())

def generate_response_with_grounding(query: str, grounded_docs: list[str]) -> str:
    """Generates a response using the LLM, grounded by retrieved documents."""
    context = "\n".join(grounded_docs)
    prompt = f"Context from medical knowledge base:\n{context}\n\nBased on the above context, answer the following query: {query}\n\nAssistant:"
    
    response = llm_pipeline(prompt, max_new_tokens=200, num_return_sequences=1, do_sample=False, temperature=0.7)
    return response[0]['generated_text']

# --- Streamlit Application UI ---
st.set_page_config(layout="wide")
st.title("🩺 Medical Diagnosis Assistant with Dynamic Knowledge Grounding")
st.markdown("This assistant uses dynamic retrieval and LLM reranking to provide grounded medical information.")

user_query = st.text_area("Enter patient symptoms, medical history, or a specific medical query:", height=100)

if st.button("Get Diagnosis/Information") and user_query:
    st.subheader("Processing Request...")
    
    if conditionally_trigger_retrieval(user_query):
        st.info("🔍 Retrieving relevant medical documents...")
        initial_retrieved_docs = retrieve_documents(user_query, top_k=5)
        
        if initial_retrieved_docs:
            st.success(f"Found {len(initial_retrieved_docs)} potential documents.")
            st.expander("Initial Retrieved Documents").write("\n".join(initial_retrieved_docs))

            st.info("✨ Reranking documents for optimal relevance (Zero-shot LM reranking)...")
            reranked_docs = zero_shot_rerank_documents(user_query, initial_retrieved_docs)
            
            # Limit to top 3 reranked docs for grounding to keep prompt size manageable
            grounding_docs = reranked_docs[:3]
            st.expander("Top Reranked Documents for Grounding").write("\n".join(grounding_docs))

            st.info("🤖 Generating response with grounded knowledge...")
            llm_response = generate_response_with_grounding(user_query, grounding_docs)
            
            st.subheader("Generated Medical Information:")
            st.write(llm_response)
            
            st.subheader("Sources (from knowledge base):")
            for i, doc in enumerate(grounding_docs):
                st.markdown(f"- {doc}")
        else:
            st.warning("No relevant documents found in the knowledge base.")
            st.info("🤖 Generating response without external grounding (may be less accurate)...")
            # Fallback to pure LLM if no docs found
            llm_response = llm_pipeline(f"Answer the following medical query: {user_query}", max_new_tokens=150, num_return_sequences=1, do_sample=False)[0]['generated_text']
            st.write(llm_response)
    else:
        st.warning("No query provided or retrieval not triggered.")

# --- Notes on further development ---
st.sidebar.subheader("Developer Notes")
st.sidebar.info(
    "**Knowledge Base:** In a real application, this would integrate with external APIs (e.g., PubMed) or internal databases."
    "\n\n**Trained Reranker:** A dedicated model (e.g., based on BERT, fine-tuned on relevance scores) would provide more robust reranking."
    "\n\n**Conditional Triggering:** More sophisticated logic using keyword extraction, entity recognition, or a small classification model could determine when to activate retrieval."
    "\n\n**LLM Choice:** For production, larger, more capable LLMs (e.g., GPT-3.5/4, Llama 2 70B, other Flan-T5 variants) would be used."
)
