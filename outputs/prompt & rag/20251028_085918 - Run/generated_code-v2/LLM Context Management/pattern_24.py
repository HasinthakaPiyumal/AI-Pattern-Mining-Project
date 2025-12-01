import streamlit as st
import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# --- Configuration ---
KNOWLEDGE_BASE_DIR = "./medical_knowledge_base"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-3.5-turbo" # Example LLM. Can be replaced with a local transformers model.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY") # Get from environment or replace

# --- Knowledge Base Management ---
def update_knowledge_base():
    """Loads documents, splits them, embeds them, and stores/updates ChromaDB."""
    # Ensure knowledge base directory exists
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        st.info(f"Created knowledge base directory: `{KNOWLEDGE_BASE_DIR}`. Please add .txt or .md files here.")
        return None

    # 1. Load documents from the directory
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="**/*.txt", # Also consider adding "**/*.md" if markdown files are expected
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True} # Added autodetect for robustness
    )
    docs = []
    try:
        docs = loader.load()
    except Exception as e:
        st.error(f"Error loading documents from `{KNOWLEDGE_BASE_DIR}`: {e}")
        return None

    if not docs:
        st.warning("No .txt documents found in the knowledge base directory. Add files to update the index.")
        return None

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Create embeddings and store in ChromaDB
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Clear existing ChromaDB for a full refresh (simple update strategy)
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        st.info(f"Cleared existing ChromaDB at `{CHROMA_DB_PATH}` for a full refresh.")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH
    )
    vectorstore.persist()
    st.success(f"Knowledge base updated successfully with {len(splits)} chunks.")
    return vectorstore

def get_vectorstore():
    """Retrieves an existing ChromaDB instance."""
    if not os.path.exists(CHROMA_DB_PATH):
        return None
    
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
    return vectorstore

# --- RAG Pipeline ---
def get_rag_chain():
    """Constructs and returns the RAG chain for querying."""
    vectorstore = get_vectorstore()
    if not vectorstore:
        st.error("Vector store not initialized. Please update the knowledge base first.")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant chunks

    # Define the prompt template for the LLM
    template = """You are a helpful medical assistant. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Keep the answer concise and evidence-based.

    Context: {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the Language Model
    # For local transformers model (e.g., Llama 2), replace ChatOpenAI with HuggingFacePipeline:
    # from langchain_community.llms import HuggingFacePipeline
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="path/to/your/local/llama2", # e.g., "meta-llama/Llama-2-7b-chat-hf"
    #     task="text-generation",
    #     pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.1, "device": 0}, # Adjust device as needed
    # )
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        st.warning("OpenAI API Key is not set. Please set the OPENAI_API_KEY environment variable or replace 'YOUR_OPENAI_API_KEY' in the script.")
        return None
    llm = ChatOpenAI(model_name=LLM_MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0.0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Define the RAG chain that also returns sources
    rag_chain_with_sources = (
        {"context": retriever, "question": RunnablePassthrough()} # Retriever gets the docs
        | RunnableLambda(
            lambda x: {
                "answer": (
                    prompt | llm | StrOutputParser()
                ).invoke({"context": format_docs(x["context"]), "question": x["question"]}),
                "sources": x["context"] # Pass the raw documents as sources
            }
        )
    )
    
    return rag_chain_with_sources

# --- Streamlit UI ---
st.set_page_config(page_title="AI Medical Info System", layout="wide")

st.title("👨‍⚕️ AI-Powered Medical Information System")
st.markdown("This system provides evidence-based medical information by retrieving from a human-readable/writable knowledge base.")

# Ensure knowledge base directory exists on app start
if not os.path.exists(KNOWLEDGE_BASE_DIR):
    os.makedirs(KNOWLEDGE_BASE_DIR)

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Doctor Interface", "Knowledge Base Management"])

with tab1:
    st.header("Medical Query Assistant")
    query = st.text_area("Enter patient symptoms or a medical question:", height=150)

    if st.button("Get Information", key="query_button"):
        if not query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Retrieving and synthesizing information..."):
                rag_chain = get_rag_chain()
                if rag_chain:
                    try:
                        result = rag_chain.invoke(query)
                        st.subheader("Generated Response:")
                        st.markdown(result["answer"])

                        st.subheader("Sources (Click to expand):")
                        if result["sources"]:
                            for i, doc in enumerate(result["sources"]):
                                with st.expander(f"Source {i+1}: {os.path.basename(doc.metadata.get('source', 'Unknown'))}"):
                                    st.write(doc.page_content)
                                    if 'source' in doc.metadata:
                                        st.caption(f"File: {doc.metadata['source']}")
                        else:
                            st.info("No specific sources were retrieved for this query.")

                    except Exception as e:
                        st.error(f"An error occurred during retrieval or generation: {e}")
                        st.info("Please ensure your OpenAI API key is correctly set and the knowledge base is updated.")
                else:
                    st.error("RAG system not fully initialized. Please update the knowledge base in the 'Knowledge Base Management' tab and ensure the LLM API key is set.")

with tab2:
    st.header("Knowledge Base Management for Medical Experts")
    st.info(f"To update the medical knowledge base, place your medical articles, clinical guidelines, and research papers (as .txt files) into the `{KNOWLEDGE_BASE_DIR}` directory. You can then click the button below to re-index the content.")
    
    st.markdown("---")
    st.subheader("Update Knowledge Base Index")
    st.write("Clicking this will re-process all documents in the knowledge base directory and rebuild the search index. This is how new information is added or updated.")
    
    if st.button("Update Medical Knowledge Base Index", key="update_kb_button"):
        with st.spinner("Updating knowledge base... This might take a moment depending on the number and size of documents."):
            vectorstore = update_knowledge_base()
            if vectorstore:
                st.success("Knowledge base index updated successfully!")
            else:
                st.warning("Knowledge base update completed, but there might be no documents or an error occurred. Check the messages above.")

    st.markdown("---")
    st.subheader("Current Knowledge Base Status")
    vectorstore_status = get_vectorstore()
    if vectorstore_status:
        st.success(f"ChromaDB is active at `{CHROMA_DB_PATH}` with {vectorstore_status._collection.count()} entries.")
        # Optional: Display files in the knowledge base directory
        st.subheader("Files in Knowledge Base Directory")
        if os.path.exists(KNOWLEDGE_BASE_DIR):
            files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(('.txt', '.md'))]
            if files:
                for f in files:
                    st.write(f"- {f}")
            else:
                st.info("No .txt or .md files found in the knowledge base directory yet.")
    else:
        st.warning(f"ChromaDB is not initialized or found at `{CHROMA_DB_PATH}`. Please update the knowledge base first.")

# Instructions for running the app
st.sidebar.header("How to Run")
st.sidebar.markdown("1. Save this code as `medical_info_system.py`.")
st.sidebar.markdown("2. Install required libraries:")
st.sidebar.code("pip install streamlit langchain-community langchain-text-splitters sentence-transformers chromadb langchain-openai")
st.sidebar.markdown(f"3. Create a directory named `{KNOWLEDGE_BASE_DIR}` in the same location as `medical_info_system.py`.")
st.sidebar.markdown("4. Place your medical text documents (.txt files) into the `medical_knowledge_base` directory.")
st.sidebar.markdown("5. Set your OpenAI API key as an environment variable `OPENAI_API_KEY` or replace `YOUR_OPENAI_API_KEY` in the script.")
st.sidebar.code("export OPENAI_API_KEY='your_key_here' # For Linux/macOS")
st.sidebar.code("$env:OPENAI_API_KEY='your_key_here' # For Windows PowerShell")
st.sidebar.markdown("6. Run the Streamlit application from your terminal:")
st.sidebar.code("streamlit run medical_info_system.py")
st.sidebar.markdown("7. Go to the 'Knowledge Base Management' tab and click 'Update Medical Knowledge Base Index'.")
st.sidebar.markdown("8. Switch to the 'Doctor Interface' tab and ask a medical question!")
