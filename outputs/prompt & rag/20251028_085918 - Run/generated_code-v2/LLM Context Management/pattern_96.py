import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import shutil

# Configuration
DATA_DIR = "data/medical_docs"
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Embedding Model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

embeddings = get_embedding_model()

# Initialize LLM (replace with your actual API key and desired model)
# st.secrets could be used here for secure API key management in Streamlit Cloud
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

# Function to load and chunk documents
def load_and_chunk_documents(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith((".txt", ".md")):
            file_path = os.path.join(data_path, filename)
            try:
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Function to create or update ChromaDB
def create_or_update_vectordb(documents):
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        st.info("Loading existing vector database...")
        vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        # For simplicity, we'll re-add documents if any change is detected or new ones are added
        # A more robust solution would check for new/modified documents and only update those.
        if documents:
            existing_docs = vectordb.get(include=['metadatas'])['metadatas'] # Get some info to check if new docs exist
            new_doc_paths = set([doc.metadata['source'] for doc in documents])
            existing_doc_paths = set([meta['source'] for meta in existing_docs])

            if new_doc_paths != existing_doc_paths or len(documents) > len(existing_docs):
                st.warning("Changes detected in knowledge base or new documents added. Re-indexing...")
                # Delete existing and recreate for full refresh
                shutil.rmtree(CHROMA_PERSIST_DIR)
                vectordb = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_PERSIST_DIR)
            else:
                st.info("No new documents or changes detected. Using existing index.")
    else:
        st.info("Creating new vector database...")
        vectordb = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_PERSIST_DIR)
    vectordb.persist()
    return vectordb


# RAG Chain setup
@st.cache_resource
def get_rag_chain(vectordb):
    retriever = vectordb.as_retriever()
    
    system_prompt = (
        "You are a helpful medical assistant for clinicians. "
        "Use the following retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
        "Provide the source documents from which you extracted the information at the end of your answer." 
        "Each source document starts with 'Source: ' followed by its filename."
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


# --- Streamlit UI --- 
st.set_page_config(page_title="Medical Knowledge Assistant", layout="wide")
st.title("🩺 Medical Knowledge Assistant for Clinicians")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Clinician Interface", "Administrator Interface"])
    st.markdown("---")
    st.info("This assistant provides verifiable medical information using a human-readable and writable knowledge base.")

# Load and process documents globally once
all_documents = load_and_chunk_documents(DATA_DIR)
vectordb = create_or_update_vectordb(all_documents)
rag_chain = get_rag_chain(vectordb)

if page == "Clinician Interface":
    st.header("Clinician Interface: Query Medical Information")
    
    query = st.text_area("Enter your medical query here:", height=150)
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Fetching medical information..."):
                try:
                    response = rag_chain.invoke({"input": query})
                    st.subheader("Answer:")
                    st.write(response["answer"])
                    
                    st.subheader("Sources:")
                    if response.get("context"):
                        unique_sources = set()
                        for doc in response["context"]:
                            if 'source' in doc.metadata:
                                unique_sources.add(os.path.basename(doc.metadata['source']))
                        for source in unique_sources:
                            st.markdown(f"- **{source}**")
                    else:
                        st.info("No specific sources were retrieved for this query.")
                except Exception as e:
                    st.error(f"An error occurred: {e}. Please ensure your OPENAI_API_KEY is set correctly.")
        else:
            st.warning("Please enter a query.")

elif page == "Administrator Interface":
    st.header("Administrator Interface: Manage Knowledge Base")

    st.subheader("Upload New Medical Documents (.txt, .md)")
    uploaded_files = st.file_uploader("Choose files", type=["txt", "md"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded {uploaded_file.name}")
        st.experimental_rerun() # Rerun to reflect new files and trigger re-index logic

    st.subheader("Current Medical Documents")
    current_docs = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    if current_docs:
        for doc in current_docs:
            st.write(f"- {doc}")
    else:
        st.info("No medical documents found. Upload some to get started!")

    st.subheader("Re-index Knowledge Base")
    st.info("Clicking 'Re-index' will reprocess all documents and update the search index. This is useful after adding, removing, or editing documents directly in the 'data/medical_docs' folder.")
    if st.button("Re-index Knowledge Base Now"):
        with st.spinner("Re-indexing documents..."):
            # Clear existing ChromaDB and re-create
            if os.path.exists(CHROMA_PERSIST_DIR):
                shutil.rmtree(CHROMA_PERSIST_DIR)
            
            all_documents = load_and_chunk_documents(DATA_DIR)
            vectordb = create_or_update_vectordb(all_documents) # This will now create a new one
            st.session_state["rag_chain"] = get_rag_chain(vectordb) # Update cached chain
            st.success("Knowledge base re-indexed successfully!")
            st.experimental_rerun() # Rerun to ensure everything is fresh
