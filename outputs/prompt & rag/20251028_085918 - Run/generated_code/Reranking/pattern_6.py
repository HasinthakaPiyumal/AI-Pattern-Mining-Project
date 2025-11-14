
import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import threading

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.linear_model import LogisticRegression
import numpy as np
import uuid
import time

# --- 1. Global Objects/Models Initialization ---

# ChromaDB Client
chroma_client = chromadb.Client()
collection_name = "medical_knowledge_base"
try:
    db_collection = chroma_client.get_or_create_collection(name=collection_name)
except Exception as e:
    st.error(f"Error initializing ChromaDB collection: {e}")
    st.stop()

# Embedding Model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = load_embedding_model()

# Reranker Model
@st.cache_resource
def load_reranker_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker_model = load_reranker_model()

# Language Model
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base") # Using base for better quality, can be 'small'
    model = AutoModelForCausalLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model
llm_tokenizer, llm_model = load_llm()

# --- 2. Knowledge Base Setup ---

dummy_medical_docs = [
    {"id": "doc1", "content": "A fever is a temporary increase in your body temperature, often due to an illness. It's a common sign of a viral or bacterial infection.", "source": "CDC"},
    {"id": "doc2", "content": "Headaches are a common condition that most people will experience many times in their lives. They can range from mild to severe, and some types include tension headaches, migraines, and cluster headaches.", "source": "Mayo Clinic"},
    {"id": "doc3", "content": "The flu (influenza) is a contagious respiratory illness caused by influenza viruses that infect the nose, throat, and sometimes the lungs. It can cause mild to severe illness, and at times can lead to death.", "source": "WHO"},
    {"id": "doc4", "content": "Hypertension, also known as high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.", "source": "NIH"},
    {"id": "doc5", "content": "Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream.", "source": "CDC"},
    {"id": "doc6", "content": "Common cold symptoms usually include a runny nose, sore throat, cough, congestion, slight body aches or a mild headache, and sneezing. It's caused by a virus.", "source": "WebMD"},
    {"id": "doc7", "content": "Antibiotics are medicines that fight bacterial infections. They work by killing bacteria or preventing them from reproducing. They are not effective against viruses, such as those that cause colds or flu.", "source": "NHS"},
    {"id": "doc8", "content": "Vaccines stimulate your immune system to produce antibodies, just as it would if you were exposed to the disease. After getting vaccinated, you develop immunity to that disease, without having to get the illness first.", "source": "CDC"},
    {"id": "doc9", "content": "Asthma is a chronic lung disease that inflames and narrows the airways. Asthma causes recurring periods of wheezing (a whistling sound when you breathe), chest tightness, shortness of breath, and coughing.", "source": "NHLBI"},
    {"id": "doc10", "content": "Allergies occur when your immune system reacts to a foreign substance — such as pollen, bee venom, pet dander or a food — that doesn't cause a reaction in most people. Antibodies identify a particular allergen as harmful.", "source": "ACAAI"}
]

def load_and_process_documents():
    if db_collection.count() == 0: # Only add if collection is empty
        st.write("Initializing knowledge base with dummy medical documents...")
        documents = [doc["content"] for doc in dummy_medical_docs]
        metadatas = [{"source": doc["source"], "id": doc["id"]} for doc in dummy_medical_docs]
        ids = [doc["id"] for doc in dummy_medical_docs]

        # Generate embeddings
        embeddings = embedding_model.encode(documents).tolist()

        # Add to ChromaDB
        db_collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        st.success(f"Added {len(documents)} documents to ChromaDB.")
    else:
        st.info(f"ChromaDB already contains {db_collection.count()} documents.")


# --- 3. Conditional Retrieval Module ---

# Dummy training data for conditional retrieval
conditional_retrieval_queries = [
    "What is a fever?", "How to treat a headache?", "Tell me a joke.",
    "What is the capital of France?", "Symptoms of flu.", "When to take antibiotics?",
    "What is your name?", "Explain hypertension.", "Who is the president of USA?",
    "What is the structure of DNA?", "Causes of diabetes."
]
conditional_retrieval_labels = [
    1, 1, 0, # Medical (1), General (0)
    0, 1, 1,
    0, 1, 0,
    0, 1
]

medical_keywords = [
    "fever", "headache", "flu", "hypertension", "diabetes", "symptoms", 
    "treat", "medicine", "antibiotics", "vaccines", "asthma", "allergies", 
    "disease", "medical", "health", "condition", "diagnosis"
]

def extract_features(query):
    # A very simplistic feature extractor: check for medical keywords
    return [1 if any(keyword in query.lower() for keyword in medical_keywords) else 0]

X_train = np.array([extract_features(q) for q in conditional_retrieval_queries])
y_train = np.array(conditional_retrieval_labels)

# Train a simple Logistic Regression model for conditional retrieval
conditional_retriever_model = LogisticRegression()
conditional_retriever_model.fit(X_train, y_train)

def should_retrieve(query: str) -> bool:
    features = np.array([extract_features(query)])
    prediction = conditional_retriever_model.predict(features)[0]
    return prediction == 1

# --- 4. Retrieval Module ---

def retrieve_documents(query: str, n_results: int = 5) -> list:
    query_embedding = embedding_model.encode([query]).tolist()
    results = db_collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    # Each result is {'documents': [doc1, doc2], 'metadatas': [meta1, meta2], ...}
    # Flatten the results into a list of dictionaries
    retrieved_docs = []
    if results['documents'] and results['metadatas']:
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
    return retrieved_docs

# --- 5. Reranking Module (Zero-Shot LM Reranking) ---

def rerank_documents(query: str, documents: list, top_k: int = 3) -> list:
    if not documents:
        return []

    # Prepare pairs for the cross-encoder: [(query, doc_content), ...]
    pairs = [[query, doc["content"]] for doc in documents]
    
    # Get scores from the cross-encoder
    scores = reranker_model.predict(pairs)

    # Combine documents with their scores and sort
    doc_scores = []
    for i, doc in enumerate(documents):
        doc_scores.append({"doc": doc, "score": scores[i]})
    
    # Sort by score in descending order
    reranked_docs = sorted(doc_scores, key=lambda x: x["score"], reverse=True)

    return [item["doc"] for item in reranked_docs[:top_k]]

# --- 6. InContext RALM Module ---

def generate_answer(query: str, context: list) -> str:
    prompt_template = """
    Answer the following question based on the provided context. 
    If the answer is not available in the context, state that you don't have enough information.

    Context:
    {context_str}

    Question: {query}
    Answer:
    """

    context_str = ""
    if context:
        for i, doc in enumerate(context):
            context_str += f"Document {i+1} (Source: {doc['metadata']['source']}): {doc['content']}\n"
    else:
        context_str = "No specific medical context found. Relying on general knowledge if possible."

    full_prompt = prompt_template.format(context_str=context_str, query=query)

    # Generate response from the LLM
    inputs = llm_tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=4, # Use beam search for better quality
        early_stopping=True
    )
    # Decode the generated text, skipping the prompt part
    generated_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Flan-T5 often repeats the prompt or structure. Extract only the answer part.
    # This is a heuristic and might need adjustment based on specific LLM behavior.
    if "Answer:" in generated_text:
        answer_start_index = generated_text.rfind("Answer:") + len("Answer:")
        extracted_answer = generated_text[answer_start_index:].strip()
    else:
        extracted_answer = generated_text.strip()
        
    # Further clean up if the LLM output includes the question again or context
    if query.lower() in extracted_answer.lower():
        extracted_answer = extracted_answer.replace(query, "").strip()
    if "context:" in extracted_answer.lower(): # Basic cleanup for context repetition
         extracted_answer = extracted_answer.split("context:", 1)[0].strip()

    return extracted_answer

# --- Streamlit Application (UI and Orchestration) ---

st.set_page_config(page_title="Medical Information Assistant", layout="wide")
st.title("🩺 Medical Information Assistant")
st.markdown("Ask any medical question and get answers augmented with external knowledge.")

# Initialize the knowledge base once
if 'kb_loaded' not in st.session_state:
    load_and_process_documents()
    st.session_state.kb_loaded = True

user_query = st.text_input("Enter your medical query here:", "What are the symptoms of flu?")

if st.button("Get Answer"):
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        st.info("Processing your query...")
        start_time = time.time()
        
        answer = ""
        retrieved_doc_contents = []
        source_attributions = []

        # 1. Conditional Retrieval
        needs_retrieval_flag = should_retrieve(user_query)
        st.write(f"Conditional Retrieval: {'Retrieval needed' if needs_retrieval_flag else 'No retrieval needed (using general LM knowledge)'}")

        context_for_lm = []
        if needs_retrieval_flag:
            # 2. Retrieval
            with st.spinner("Retrieving relevant documents..."):
                retrieved_documents = retrieve_documents(user_query, n_results=10) # Retrieve more to allow reranker to choose
            
            if retrieved_documents:
                # 3. Reranking
                with st.spinner("Reranking documents..."):
                    reranked_documents = rerank_documents(user_query, retrieved_documents, top_k=3)
                
                context_for_lm = reranked_documents
                retrieved_doc_contents = [doc['content'] for doc in reranked_documents]
                source_attributions = [doc['metadata']['source'] for doc in reranked_documents]
            else:
                st.warning("No relevant documents found in the knowledge base.")

        # 4. InContext RALM
        with st.spinner("Generating answer..."):
            answer = generate_answer(user_query, context_for_lm)
        
        end_time = time.time()
        st.success(f"Answer generated in {end_time - start_time:.2f} seconds.")

        st.subheader("Generated Answer:")
        st.write(answer)

        if retrieved_doc_contents:
            st.subheader("Retrieved Documents (Context Used):")
            for i, doc_content in enumerate(retrieved_doc_contents):
                st.markdown(f"**Document {i+1} (Source: {source_attributions[i]})**")
                st.text(doc_content)

# --- How to run ---
st.markdown("""
--- 
**How to run this application:**

1.  Save this code as `medical_assistant.py`.
2.  Install necessary libraries:
    `pip install streamlit fastapi uvicorn chromadb sentence-transformers transformers scikit-learn numpy torch`
    (Note: `torch` is needed by `transformers` and `sentence-transformers`)
3.  Run the Streamlit application from your terminal:
    `streamlit run medical_assistant.py`

This script runs the entire logic within the Streamlit application for simplicity and adherence to the 'single code file' request.
If you were to use FastAPI as a separate backend service, you would define the core logic within FastAPI endpoints and run FastAPI with `uvicorn medical_assistant:app --reload --port 8000`, and then have a separate Streamlit app make API calls to it.
""")
