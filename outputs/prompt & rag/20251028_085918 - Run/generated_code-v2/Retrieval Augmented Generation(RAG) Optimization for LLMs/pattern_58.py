import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline, set_seed

# --- Configuration ---
KNOWLEDGE_BASE_DIR = "./medical_knowledge_base"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATIVE_MODEL_NAME = "google/flan-t5-small"
TOP_K_DOCUMENTS = 5

# --- 1. Medical Knowledge Base Simulation ---
def load_medical_documents(knowledge_base_dir):
    documents = []
    if not os.path.exists(knowledge_base_dir):
        os.makedirs(knowledge_base_dir)
        # Create some dummy medical documents if the directory is empty
        with open(os.path.join(knowledge_base_dir, "clinical_guideline_1.txt"), "w") as f:
            f.write("Clinical guideline for Type 2 Diabetes: Initial treatment often involves lifestyle changes, metformin. Regular blood glucose monitoring is crucial. Consider SGLT2 inhibitors or GLP-1 receptor agonists if glycemic targets are not met.")
        with open(os.path.join(knowledge_base_dir, "drug_info_metformin.txt"), "w") as f:
            f.write("Metformin: Oral biguanide used to treat type 2 diabetes. Common side effects include gastrointestinal upset. Contraindicated in severe renal impairment.")
        with open(os.path.join(knowledge_base_dir, "research_paper_heart_failure.txt"), "w") as f:
            f.write("Recent research on heart failure management highlights the importance of ACE inhibitors, beta-blockers, and mineralocorticoid receptor antagonists. Fluid restriction and sodium limitation are also key.")
        with open(os.path.join(knowledge_base_dir, "patient_case_study_pneumonia.txt"), "w") as f:
            f.write("Patient case study: 65-year-old male presenting with fever, cough, and shortness of breath. Chest X-ray showed lobar infiltrate consistent with bacterial pneumonia. Treated with amoxicillin-clavulanate. History of COPD.")
        with open(os.path.join(knowledge_base_dir, "drug_info_amoxicillin.txt"), "w") as f:
            f.write("Amoxicillin-clavulanate: Broad-spectrum antibiotic used for bacterial infections like pneumonia and sinusitis. Common side effects include diarrhea. Allergic reactions possible.")
        with open(os.path.join(knowledge_base_dir, "clinical_guideline_hypertension.txt"), "w") as f:
            f.write("Clinical guideline for Hypertension: First-line treatments include thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers. Lifestyle modifications such as diet and exercise are fundamental.")

    for filename in os.listdir(knowledge_base_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(knowledge_base_dir, filename)
            with open(filepath, "r") as f:
                documents.append({"id": filename, "content": f.read()})
    return documents

# --- 2. Embedding and Vector Store (ChromaDB) ---
@st.cache_resource
def initialize_chroma_db(documents, db_path, embedding_model_name):
    st.spinner("Initializing medical knowledge base...")
    client = chromadb.PersistentClient(path=db_path)
    collection_name = "medical_knowledge"

    try:
        collection = client.get_or_create_collection(name=collection_name)
    except Exception as e:
        st.error(f"Error creating/getting ChromaDB collection: {e}. Please ensure ChromaDB is accessible and try again.")
        return None, None

    embedding_model = SentenceTransformer(embedding_model_name)
    
    if collection.count() == 0:
        st.info("Embedding and storing documents in ChromaDB. This might take a moment...")
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        embeddings = embedding_model.encode(contents).tolist()
        collection.add(embeddings=embeddings, documents=contents, ids=ids)
        st.success(f"Successfully embedded and stored {len(documents)} documents.")
    else:
        st.info(f"ChromaDB collection '{collection_name}' already contains {collection.count()} documents.")

    return collection, embedding_model

# --- 3. Generative Model Loading ---
@st.cache_resource
def load_generative_model(model_name):
    st.spinner(f"Loading generative model: {model_name}...")
    set_seed(42)
    try:
        generator = pipeline("text2text-generation", model=model_name)
        st.success(f"Generative model {model_name} loaded successfully.")
        return generator
    except Exception as e:
        st.error(f"Error loading generative model {model_name}: {e}. Please check model name and internet connection.")
        return None

# --- 4. Retrieval Module ---
def retrieve_documents(query, collection, embedding_model, top_k):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=['documents']
    )
    return [doc for doc in results['documents'][0]]

# --- 5. Prompt Engineering for RAGToken Simulation ---
def construct_rag_prompt(patient_query, retrieved_docs):
    context_str = ""
    for i, doc in enumerate(retrieved_docs):
        context_str += f"\nDocument {i+1}: {doc}"
    
    prompt = (
        f"You are an AI-powered Medical Diagnostic Assistant. Your goal is to provide a comprehensive diagnostic hypothesis, "
        f"suggest relevant tests, and outline potential treatment plans based on the patient's information and the provided medical knowledge. "
        f"Synthesize information from the documents provided below, referencing them implicitly or explicitly as needed to build a coherent and detailed response. "
        f"Prioritize accuracy and clinical relevance.\n\n"
        f"Patient Information: {patient_query}\n\n"
        f"Medical Knowledge:\n{context_str}\n\n"
        f"Based on the patient information and the medical knowledge above, provide a diagnostic hypothesis, suggested tests, and a treatment plan:"
    )
    return prompt

# --- Streamlit UI and Orchestration ---
st.set_page_config(layout="wide", page_title="AI Medical Diagnostic Assistant")

st.title("🩺 AI Medical Diagnostic Assistant")
st.markdown("---\nProvide patient symptoms and medical history to get a diagnostic hypothesis, suggested tests, and potential treatment plans, synthesized from a medical knowledge base.")

# Load medical documents
medical_documents = load_medical_documents(KNOWLEDGE_BASE_DIR)
if not medical_documents:
    st.warning(f"No medical documents found in '{KNOWLEDGE_BASE_DIR}'. Please add some .txt files or restart to use dummy data.")
    st.stop()

# Initialize ChromaDB
chroma_collection, sentence_transformer_model = initialize_chroma_db(
    medical_documents, CHROMA_DB_PATH, EMBEDDING_MODEL_NAME
)
if chroma_collection is None or sentence_transformer_model is None:
    st.error("Failed to initialize ChromaDB or embedding model. Please check logs.")
    st.stop()

# Load Generative Model
generator = load_generative_model(GENERATIVE_MODEL_NAME)
if generator is None:
    st.error("Failed to load generative model. Please check logs.")
    st.stop()

patient_symptoms = st.text_area(
    "Enter Patient Symptoms and Medical History:",
    "68-year-old male with a history of hypertension and recent onset of shortness of breath, fatigue, and swollen ankles. Occasional dry cough.",
    height=150
)

if st.button("Get Diagnostic Assistance"):
    if patient_symptoms:
        with st.spinner("Analyzing patient data and synthesizing medical knowledge..."):
            # 1. Retrieve relevant documents
            retrieved_chunks = retrieve_documents(
                patient_symptoms,
                chroma_collection,
                sentence_transformer_model,
                TOP_K_DOCUMENTS
            )
            
            if not retrieved_chunks:
                st.warning("Could not retrieve relevant medical documents. Please try a different query.")
            else:
                st.subheader("\n--- Retrieved Medical Knowledge ---\n")
                for i, chunk in enumerate(retrieved_chunks):
                    st.text(f"Document {i+1}: {chunk[:200]}...") # Displaying a snippet

                # 2. Construct RAG-inspired prompt
                rag_prompt = construct_rag_prompt(patient_symptoms, retrieved_chunks)
                
                # 3. Generate response
                # The max_new_tokens is crucial for controlling response length
                # num_beams for more coherent output, but increases computation
                generated_response = generator(rag_prompt, max_new_tokens=500, num_beams=5, early_stopping=True)[0]['generated_text']
                
                st.subheader("\n--- Diagnostic Hypothesis & Treatment Plan ---\n")
                st.write(generated_response)
    else:
        st.warning("Please enter patient symptoms and medical history.")
