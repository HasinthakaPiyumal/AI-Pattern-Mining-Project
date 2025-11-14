import streamlit as st
import os
from dotenv import load_dotenv

# Langchain and related imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Transformers for reranking and local LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForSequenceClassification
import torch

# --- Configuration ---
load_dotenv()
DATA_DIR = "medical_docs"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_LLM_MODEL_NAME = "google/flan-t5-base" # Using a smaller model for demonstration

# --- 1. Knowledge Base Management Module ---

@st.cache_resource
def load_and_process_documents():
    """Loads, chunks, and returns documents from the DATA_DIR."""
    # Create dummy medical documents if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(DATA_DIR, "doc1.txt")):
        with open(os.path.join(DATA_DIR, "doc1.txt"), "w") as f:
            f.write("Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain. It can also be used to treat inflammatory conditions such as arthritis, and as an antiplatelet agent to prevent blood clots. Common side effects include stomach upset, heartburn, and drowsiness. Source: Medical Encyclopedia.")
        with open(os.path.join(DATA_DIR, "doc2.txt"), "w") as f:
            f.write("Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. Type 1 diabetes is an autoimmune condition where the body does not produce insulin. Type 2 diabetes, more common, occurs when the body either doesn't produce enough insulin or doesn't use insulin effectively. Treatment often involves diet, exercise, medication, and sometimes insulin injections. Source: WHO Guidelines on Diabetes.")
        with open(os.path.join(DATA_DIR, "doc3.txt"), "w") as f:
            f.write("Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle changes (diet, exercise) and medications (e.g., diuretics, ACE inhibitors) are common treatments. Regular monitoring is crucial. Source: American Heart Association.")
        st.info(f"Created dummy medical documents in the '{DATA_DIR}' directory.")

    documents = []
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, file_name)
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

@st.cache_resource
def get_embedding_model():
    """Initializes and returns the HuggingFace Embeddings model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_vector_store(documents, embeddings):
    """Initializes or loads the ChromaDB vector store."""
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        st.success("Loading existing ChromaDB.")
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    else:
        st.info("Creating and persisting new ChromaDB from documents. This may take a moment.")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
        vectorstore.persist()
        st.success("ChromaDB created and persisted.")
    return vectorstore

# --- 2. Query Processing Pipeline ---

@st.cache_resource
def get_reranker_model():
    """Loads and returns the cross-encoder reranker tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
    return tokenizer, model

@st.cache_resource
def get_local_llm():
    """Loads and returns a local HuggingFace LLM via HuggingFacePipeline."""
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device=0 if torch.cuda.is_available() else -1, # Use GPU if available, else CPU
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 # Use float16 for GPU if possible
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def rerank_documents(query, retrieved_docs, reranker_tokenizer, reranker_model, top_n=5):
    """Reranks retrieved documents based on relevance to the query using a cross-encoder."""
    if not retrieved_docs:
        return []
    
    # Prepare sentences for reranking
    sentences = [doc.page_content for doc in retrieved_docs]
    
    # Create features for the cross-encoder model
    features = reranker_tokenizer([query] * len(sentences), sentences, padding=True, truncation=True, return_tensors="pt")

    # Move features to GPU if available
    if torch.cuda.is_available():
        features = {k: v.to("cuda") for k, v in features.items()}

    with torch.no_grad():
        scores = reranker_model(**features).logits.squeeze().cpu()
        
    # Sort retrieved_docs based on reranked scores
    reranked_docs_with_scores = sorted(zip(retrieved_docs, scores.tolist()), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in reranked_docs_with_scores[:top_n]]
    
    return reranked_docs

def get_medical_answer(query, vectorstore, llm, reranker_tokenizer, reranker_model):
    """Processes a medical query, retrieves, reranks, and generates an answer with sources."""
    st.markdown("🔍 *Performing initial document retrieval...*")
    
    # Initial Document Retrieval: Retrieve more documents for better reranking potential
    retrieved_docs = vectorstore.similarity_search(query, k=10)

    if not retrieved_docs:
        return "No relevant information found in the knowledge base.", []

    # Reranking Module
    st.markdown(f"🔄 *Reranking {len(retrieved_docs)} retrieved documents...*")
    reranked_docs = rerank_documents(query, retrieved_docs, reranker_tokenizer, reranker_model, top_n=5)
    
    if not reranked_docs:
        return "No relevant information found after reranking.", []

    # Context Augmentation (InContext RALM) and LLM Inference
    # Customizing the prompt to include sources and guide the LLM
    qa_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Provide factual information based *only* on the given context. List the sources explicitly at the end of your answer, referencing the 'Source' part from the context. If multiple pieces of context refer to the same source, list it once. For example: 'Source: Clinical Guidelines A, Medical Journal B'.
    
    Context: {context}
    
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)

    # Combine reranked document content for the LLM context
    context_string = "\n\n".join([doc.page_content for doc in reranked_docs])

    # Create the final prompt for the LLM
    final_prompt = QA_CHAIN_PROMPT.format(context=context_string, question=query)
    
    st.markdown("🧠 *Generating answer with Language Model...*")
    response = llm.invoke(final_prompt)
    
    # Extract unique sources from the reranked documents' metadata
    sources = list(set([doc.metadata.get("source", "N/A") for doc in reranked_docs]))
    
    return response.strip(), sources


# --- Streamlit Frontend ---

def main():
    st.set_page_config(page_title="Medical Information Assistant", layout="wide")
    st.title("👨‍⚕️ Medical Information Assistant for Clinicians")
    st.markdown("This assistant provides accurate and up-to-date medical information by leveraging a knowledge base and advanced retrieval techniques.")
    st.markdown("--- Make sure the `medical_docs` directory exists and contains some `.txt` files or let the app create dummy ones. ---")

    # Load resources
    with st.spinner("Initializing system: Loading documents, embeddings, reranker, and LLM... This might take a few minutes on first run."):
        documents = load_and_process_documents()
        embeddings = get_embedding_model()
        vectorstore = get_vector_store(documents, embeddings)
        reranker_tokenizer, reranker_model = get_reranker_model()
        llm = get_local_llm()
    st.success("System ready! Ask your medical question below.")

    user_query = st.text_area("Enter your medical query here:", height=100, placeholder="e.g., What are the treatments for hypertension?")

    if st.button("Get Information", use_container_width=True):
        if user_query:
            with st.spinner("Processing your medical query..."):
                answer, sources = get_medical_answer(user_query, vectorstore, llm, reranker_tokenizer, reranker_model)
                
                st.subheader("Generated Answer:")
                st.markdown(answer)

                if sources:
                    st.subheader("Sources:")
                    for source in sources:
                        st.markdown(f"- `{source}`")
                else:
                    st.info("No specific sources were identified for this answer from the retrieved context.")
        else:
            st.warning("Please enter a medical query to get information.")

    st.markdown("--- Powered by LangChain, Transformers, and Streamlit ---")

if __name__ == "__main__":
    main()
