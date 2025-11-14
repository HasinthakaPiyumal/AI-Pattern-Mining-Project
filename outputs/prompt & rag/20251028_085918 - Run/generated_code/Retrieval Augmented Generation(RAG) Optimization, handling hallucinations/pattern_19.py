
import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration --- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
DATA_DIR = "./medical_data"

# --- Helper Functions --- #
def get_embedding_model():
    """Initializes and returns the embedding model."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def load_and_split_documents(data_dir):
    """Loads documents from the specified directory and splits them into chunks."""
    documents = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def initialize_vector_store(embedding_model, data_dir):
    """Initializes or loads the Chroma vector store."""
    if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
        st.sidebar.info("Creating new vector store. This may take a moment...")
        os.makedirs(data_dir, exist_ok=True)
        
        # Simulate some medical data files
        with open(os.path.join(data_dir, "diabetes_guideline.txt"), "w") as f:
            f.write("""Diabetes Mellitus Type 2 Guidelines: Key recommendations include regular blood glucose monitoring, dietary management (low carb, high fiber), regular exercise, and medication such as Metformin. HbA1c target usually below 7%. Complications include neuropathy, nephropathy, and retinopathy. Early diagnosis and intervention are crucial.
            Patient Case 1: A 55-year-old male with new onset fatigue, polydipsia, and polyuria. Fasting glucose 210 mg/dL. HbA1c 8.5%. Diagnosis: Type 2 Diabetes.
            """)
        with open(os.path.join(data_dir, "hypertension_research.txt"), "w") as f:
            f.write("""Recent research in hypertension suggests that lifestyle modifications, including reduced sodium intake, increased potassium, regular physical activity, and weight management, are first-line treatments. Pharmacological interventions often start with ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics. Blood pressure target is typically <130/80 mmHg for most adults. Uncontrolled hypertension significantly increases the risk of stroke and heart attack.
            Patient Case 2: A 62-year-old female with blood pressure readings consistently above 140/90 mmHg. No other significant symptoms. Family history of hypertension.
            """)
        with open(os.path.join(data_dir, "common_cold_treatment.txt"), "w") as f:
            f.write("""The common cold is a viral infection of the nose and throat. Treatment is primarily symptomatic, focusing on rest, hydration, and over-the-counter medications like pain relievers (ibuprofen, acetaminophen) and decongestants. Antibiotics are ineffective against viral colds. Symptoms usually resolve within 7-10 days. Differentiate from flu or allergies.
            Patient Case 3: A 30-year-old with runny nose, sore throat, and mild cough for 3 days. No fever. Feeling generally well otherwise.
            """)

        docs = load_and_split_documents(data_dir)
        vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=CHROMA_DB_PATH)
        st.sidebar.success("Vector store created and initialized!")
    else:
        st.sidebar.info("Loading existing vector store...")
        vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
        st.sidebar.success("Vector store loaded!")
    return vectordb

# --- RAG Chain and LLM Setup --- #
def setup_rag_chain(retriever, llm):
    """Sets up the RAG chain for question answering."""
    rag_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a highly knowledgeable Medical Diagnostic Assistant. Your task is to provide a differential diagnosis and reasoning based on the patient's symptoms and the provided medical context. Always prioritize safety and accuracy. If you are uncertain or need more information, clearly state it."),
            ("human", "Patient Symptoms: {question}\n\nMedical Context: {context}\n\nBased on the above, provide a differential diagnosis, reasoning, and suggested next steps (e.g., further tests, specialist referral). If you cannot confidently make a diagnosis, state what additional information would be helpful."),
        ]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Stuff all retrieved documents into the context
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={"prompt": rag_prompt_template}
    )
    return qa_chain

# --- Adaptive Decision-Making & Caching (Simplified) --- #
# A simple in-memory cache for frequently accessed guidelines or common queries
cache = {}

def adaptive_diagnose(qa_chain, symptoms):
    """Performs diagnosis with adaptive decision-making and caching.
    
    For simplicity, adaptive decision here checks LLM's output for uncertainty and suggests more info.
    """
    st.session_state.retrieved_docs = []
    st.session_state.diagnostic_output = ""
    st.session_state.adaptive_note = ""

    # Check cache first for common queries (simplified)
    if symptoms.lower() in cache:
        st.session_state.diagnostic_output = f"(From cache) {cache[symptoms.lower()]}"
        st.session_state.adaptive_note = "This query was resolved quickly using cached information."
        return

    try:
        response = qa_chain.invoke({"query": symptoms})
        llm_output = response["result"]
        retrieved_docs = response["source_documents"]

        st.session_state.retrieved_docs = [doc.page_content for doc in retrieved_docs]
        st.session_state.diagnostic_output = llm_output

        # Simulate adaptive decision-making
        if any(keyword in llm_output.lower() for keyword in ["uncertain", "further information", "additional tests", "clarification needed"]):
            st.session_state.adaptive_note = "The model expressed some uncertainty or suggested further information is needed for a more definitive diagnosis. This indicates an opportunity for adaptive retrieval or user interaction."
        else:
            st.session_state.adaptive_note = "The model provided a confident diagnosis and next steps based on the available information."
        
        # Add to cache (simplified: only if confident and not already there)
        if "uncertain" not in llm_output.lower():
             cache[symptoms.lower()] = llm_output # Cache the full confident response

    except Exception as e:
        st.error(f"An error occurred during diagnosis: {e}")
        st.session_state.diagnostic_output = "Could not process the request due to an error. Please try again or refine your symptoms."

# --- Streamlit UI --- #
st.set_page_config(page_title="Medical Diagnostic Assistant", layout="wide")
st.title("🩺 Medical Diagnostic Assistant (RAG System)")
st.markdown("This assistant leverages a Retrieval-Augmented Generation (RAG) system to provide diagnostic insights based on patient symptoms and a comprehensive medical knowledge base.")

# Sidebar for Configuration
st.sidebar.header("Configuration")
with st.sidebar.expander("OpenAI API Key"):    
    if OPENAI_API_KEY is None:
        st.warning("OPENAI_API_KEY not found in environment variables. Please enter it here or set it in your .env file.")
        user_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if user_api_key:
            os.environ["OPENAI_API_KEY"] = user_api_key
            OPENAI_API_KEY = user_api_key
            st.success("API Key set!")
        else:
            st.error("OpenAI API Key is required to run the application.")
            st.stop()
    else:
        st.success("OpenAI API Key loaded from environment variables.")


# Initialize components only once
if 'vectordb' not in st.session_state:
    embedding_model = get_embedding_model()
    st.session_state.vectordb = initialize_vector_store(embedding_model, DATA_DIR)
    st.session_state.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY)
    st.session_state.retriever = st.session_state.vectordb.as_retriever()
    st.session_state.qa_chain = setup_rag_chain(st.session_state.retriever, st.session_state.llm)
    st.session_state.retrieved_docs = []
    st.session_state.diagnostic_output = ""
    st.session_state.adaptive_note = ""

# Main application UI
patient_symptoms = st.text_area(
    "Enter Patient Symptoms (e.g., 'A 55-year-old male with new onset fatigue, polydipsia, and polyuria. Fasting glucose 210 mg/dL.')",
    height=150
)

if st.button("Get Diagnostic Suggestion"):    
    if patient_symptoms and OPENAI_API_KEY:
        with st.spinner("Analyzing symptoms and generating diagnosis..."):
            adaptive_diagnose(st.session_state.qa_chain, patient_symptoms)
    elif not patient_symptoms:
        st.warning("Please enter patient symptoms to get a diagnosis.")
    elif not OPENAI_API_KEY:
        st.error("OpenAI API Key is not set. Please configure it in the sidebar.")


st.subheader("Diagnostic Suggestion")
if st.session_state.diagnostic_output:
    st.markdown(st.session_state.diagnostic_output)

    st.subheader("\nAdaptive Decision Insight")
    st.info(st.session_state.adaptive_note)

    with st.expander("Show Retrieved Medical Context"):        
        if st.session_state.retrieved_docs:
            for i, doc_content in enumerate(st.session_state.retrieved_docs):
                st.write(f"**Document {i+1}:**")
                st.markdown(doc_content)
                st.markdown("---")
        else:
            st.write("No specific documents were retrieved for this query (possibly cached response).")

st.markdown("""
**How to run this application:**
1.  Save the code as `medical_diagnostic_assistant.py`.
2.  Create a `.env` file in the same directory and add your OpenAI API key: `OPENAI_API_KEY="your_openai_api_key_here"`
3.  Install necessary libraries: `pip install streamlit langchain-community langchain-text-splitters sentence-transformers chromadb langchain-openai python-dotenv`
4.  Run from your terminal: `streamlit run medical_diagnostic_assistant.py`
""")
