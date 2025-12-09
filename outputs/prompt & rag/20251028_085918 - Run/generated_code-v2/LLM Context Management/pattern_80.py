import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Ensure the knowledge_base directory exists and has sample files
KNOWLEDGE_BASE_DIR = "knowledge_base"
if not os.path.exists(KNOWLEDGE_BASE_DIR):
    os.makedirs(KNOWLEDGE_BASE_DIR)

    # Create sample medical texts
    with open(os.path.join(KNOWLEDGE_BASE_DIR, "common_cold.txt"), "w") as f:
        f.write("""
Common Cold:
Symptoms: Runny nose, sneezing, sore throat, cough, mild fatigue, low-grade fever (optional).
Causes: Viral infection (rhinoviruses, coronaviruses).
Treatment: Rest, fluids, over-the-counter medications for symptom relief (pain relievers, decongestants, cough suppressants). Antibiotics are not effective.
Duration: Typically 7-10 days.
""")

    with open(os.path.join(KNOWLEDGE_BASE_DIR, "influenza.txt"), "w") as f:
        f.write("""
Influenza (Flu):
Symptoms: Fever, body aches, chills, fatigue, cough, sore throat, headache. Symptoms are typically more severe than a common cold.
Causes: Influenza virus.
Treatment: Antiviral drugs (e.g., oseltamivir) if started early, rest, fluids, over-the-counter symptom relief. Vaccination is recommended for prevention.
Complications: Pneumonia, bronchitis, sinus infections.
""")

    with open(os.path.join(KNOWLEDGE_BASE_DIR, "strep_throat.txt"), "w") as f:
        f.write("""
Strep Throat (Streptococcal Pharyngitis):
Symptoms: Sudden onset of sore throat, pain when swallowing, fever, red and swollen tonsils (sometimes with white patches or streaks of pus), tiny red spots on the roof of the mouth (petechiae). Headache, stomach ache, nausea, or vomiting can occur, especially in children. Cough and runny nose are usually absent.
Causes: Bacterial infection (Streptococcus pyogenes).
Treatment: Antibiotics (e.g., penicillin, amoxicillin) to prevent complications like rheumatic fever.
Diagnosis: Rapid strep test or throat culture.
""")

    with open(os.path.join(KNOWLEDGE_BASE_DIR, "diabetes_type2.txt"), "w") as f:
        f.write("""
Type 2 Diabetes:
Symptoms: Increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, frequent infections.
Causes: Insulin resistance and/or insufficient insulin production. Often linked to genetics, lifestyle (diet, exercise), and obesity.
Treatment: Lifestyle modifications (diet, exercise), oral medications (e.g., metformin), insulin injections in some cases. Regular monitoring of blood glucose levels.
Complications: Heart disease, nerve damage, kidney damage, eye damage.
""")

    st.success("Sample medical knowledge base created in the 'knowledge_base/' directory.")


# --- Knowledge Base Indexing and Retrieval --- 
@st.cache_resource
def load_and_index_knowledge_base():
    st.info("Loading and indexing knowledge base...")
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    faiss_index_path = "faiss_index"
    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(faiss_index_path)

    st.success("Knowledge base loaded and indexed!")
    return vectorstore.as_retriever()


# --- Language Model (LLM) Integration ---
@st.cache_resource
def initialize_llm():
    st.info("Initializing Language Model (Flan-T5-small). This may take a moment...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    st.success("Language Model initialized!")
    return llm


# --- Streamlit UI --- 
st.set_page_config(page_title="Medical Diagnostic Assistant", layout="wide")
st.title("🩺 Medical Diagnostic Assistant")
st.markdown("This AI assistant provides evidence-based diagnostic support using a human-readable/writable knowledge base.")

# Initialize LLM and Retriever in session state
if "retriever" not in st.session_state:
    st.session_state.retriever = load_and_index_knowledge_base()
if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm()

retriever = st.session_state.retriever
llm = st.session_state.llm

# Re-index button
if st.button("Re-index Knowledge Base (for updates)"):
    with st.spinner("Re-indexing knowledge base... This might take a moment."):
        st.session_state.retriever = load_and_index_knowledge_base()
    st.success("Knowledge Base re-indexed successfully!")


st.subheader("Enter Patient Symptoms or Medical Query")
symptoms_query = st.text_area(
    "Describe the patient's symptoms or ask a medical question:",
    height=150,
    placeholder="e.g., 'Patient has sudden sore throat, fever, difficulty swallowing, and no cough.'"
)

if st.button("Get Diagnosis") and symptoms_query:
    if not retriever or not llm:
        st.error("System not fully initialized. Please wait a moment or try re-indexing.")
    else:
        with st.spinner("Generating diagnosis..."):
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            response = qa_chain({"query": symptoms_query})

            st.subheader("AI's Diagnostic Suggestion and Treatment")
            st.write(response["result"])

            st.subheader("Source Documents (Evidence)")
            if response["source_documents"]:
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Source {i+1}:** `{doc.metadata.get('source', 'Unknown')}`")
                    st.code(doc.page_content, language='text')
            else:
                st.info("No specific source documents found for this query.")
elif st.button("Get Diagnosis") and not symptoms_query:
    st.warning("Please enter patient symptoms or a medical query to get a diagnosis.")
