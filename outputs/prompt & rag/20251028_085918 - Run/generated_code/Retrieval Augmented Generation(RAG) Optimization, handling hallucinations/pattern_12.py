
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Data Simulation for Knowledge Base ---
MEDICAL_KNOWLEDGE_BASE = [
    "Aspirin is commonly used as an analgesic to relieve minor aches and pains, as an antipyretic to reduce fever, and as an anti-inflammatory agent. It also has antiplatelet effects, which are useful in preventing blood clots.",
    "Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). It is characterized by insulin resistance or insufficient insulin production. Management often involves diet, exercise, and medication.",
    "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle changes and various medications are typically prescribed.",
    "Ciprofloxacin is an antibiotic used to treat a variety of bacterial infections. It belongs to a class of drugs called fluoroquinolones. Side effects can include nausea, diarrhea, and dizziness. It should not be taken with antacids.",
    "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. Symptoms range from mild to severe, and can include fever, cough, fatigue, and loss of taste or smell. Vaccination is highly effective in preventing severe illness.",
    "The recommended treatment for acute appendicitis is typically surgical removal of the appendix (appendectomy). This can be performed open or laparoscopically. Early diagnosis is crucial to prevent rupture.",
    "Migraine is a severe headache often accompanied by symptoms such as throbbing pain on one side of the head, sensitivity to light and sound, and nausea. Triggers vary widely among individuals, and treatment can involve acute pain relief and preventative medications.",
    "Insulin is a hormone produced by the pancreas that allows your body to use sugar (glucose) from carbohydrates for energy or to store glucose for future use. It plays a key role in type 1 diabetes management and sometimes in type 2 diabetes."
]

def setup_vector_store():
    """Sets up the Chroma vector store with simulated medical knowledge."""
    # Create a simple text file from the knowledge base
    with open("medical_docs.txt", "w") as f:
        for doc in MEDICAL_KNOWLEDGE_BASE:
            f.write(doc + "\n\n")

    loader = TextLoader("medical_docs.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Initialize Chroma DB
    if not os.path.exists(VECTOR_DB_DIR):
        os.makedirs(VECTOR_DB_DIR)
    
    # This will create and persist the collection if it doesn't exist, 
    # or load it if it does.
    db = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_DB_DIR)
    st.success(f"Vector store initialized with {len(docs)} documents.")
    return db

# --- RAG Chain Definition ---
def initialize_rag_chain(vector_store):
    """Initializes the LangChain RAG pipeline."""
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        return None

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, api_key=OPENAI_API_KEY)

    # Define the prompt template for the LLM
    prompt = ChatPromptTemplate.from_template(
        """You are a highly knowledgeable medical AI assistant. Answer the user's question based only on the provided context.
        If you cannot find the answer in the context, clearly state that you don't have enough information.
        Provide detailed and accurate information, citing the context where appropriate.

        Context: {context}
        Question: {input}
        Answer:"""
    )

    # Create a document chain to combine retrieved documents with the prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create the RAG retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    st.success("RAG chain initialized.")
    return retrieval_chain

# --- Streamlit Frontend ---
st.set_page_config(page_title="CDSS RAG System", layout="wide")
st.title("🩺 Clinical Decision Support System (CDSS) powered by RAG")
st.markdown("\n--- Guidance ---\nThis system provides evidence-based answers to medical queries by integrating external knowledge. \nIt uses Retrieval-Augmented Generation (RAG) to ensure factual accuracy and mitigate hallucinations. \n\n**Disclaimer**: This system is for informational purposes only and should not replace professional medical advice.")

# Initialize vector store and RAG chain only once
if "vector_store" not in st.session_state:
    st.session_state.vector_store = setup_vector_store()

if "rag_chain" not in st.session_state and st.session_state.vector_store is not None:
    st.session_state.rag_chain = initialize_rag_chain(st.session_state.vector_store)

# User input
query = st.text_area("Enter your medical query here:", height=100)

if st.button("Get Medical Insight"):
    if not query:
        st.warning("Please enter a query.")
    elif st.session_state.rag_chain is None:
        st.error("RAG system not fully initialized. Check API key and vector store setup.")
    else:
        with st.spinner("Retrieving and generating insight..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": query})
                
                st.subheader("Generated Medical Insight:")
                st.write(response["answer"])
                
                st.subheader("Retrieved Sources (Context):")
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Source {i+1}:** {doc.page_content}")
                    # You might want to add metadata if available, e.g., doc.metadata
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please ensure your OpenAI API key is correct and you have an active internet connection.")

# Optional: Clean up created files/directories on exit (or manually)
# For a more robust solution, manage persistence outside of a simple script run.
