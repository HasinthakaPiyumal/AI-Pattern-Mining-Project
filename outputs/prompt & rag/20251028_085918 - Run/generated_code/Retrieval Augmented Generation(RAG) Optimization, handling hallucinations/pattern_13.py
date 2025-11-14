import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# --- Configuration --- #
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"
CHROMA_PERSIST_DIR = "./chroma_db"

# --- Dummy Medical Knowledge Base --- #
dummy_medical_documents = [
    "Pneumonia symptoms include cough, fever, shortness of breath, and chest pain. Diagnosis often involves a chest X-ray and sputum culture. Treatment typically involves antibiotics.",
    "Diabetes Mellitus Type 2 is characterized by insulin resistance. Symptoms can include increased thirst, frequent urination, and unexplained weight loss. Management often involves lifestyle changes, oral medications, and sometimes insulin.",
    "Hypertension (high blood pressure) is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. It's often asymptomatic. Treatment includes lifestyle changes and medication.",
    "Migraine is a severe headache disorder often accompanied by symptoms such as throbbing pain on one side of the head, nausea, vomiting, and extreme sensitivity to light and sound. Triggers can vary widely among individuals.",
    "Influenza, commonly known as the flu, is a contagious respiratory illness caused by influenza viruses. Symptoms include fever, cough, sore throat, muscle aches, and fatigue. Vaccination is recommended annually for prevention.",
    "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to symptoms like wheezing, shortness of breath, chest tightness, and coughing. It's often managed with inhalers and avoiding triggers.",
    "Hypothyroidism occurs when the thyroid gland doesn't produce enough thyroid hormones. Symptoms include fatigue, weight gain, constipation, and cold sensitivity. Treatment involves thyroid hormone replacement therapy.",
    "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon. Symptoms typically include sudden pain that begins around your navel and shifts to your lower right abdomen, nausea, vomiting, and loss of appetite. Surgical removal of the appendix is the standard treatment."
]

# --- RAG System Initialization --- #
@st.cache_resource
def initialize_rag_system():
    st.write("Initializing RAG system... This may take a moment.")

    # 1. Embeddings Model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 2. ChromaDB (Vector Store)
    # Check if the database already exists and has data
    try:
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        if vectorstore._collection.count() == 0: # Check if collection is empty
            st.write("Populating ChromaDB with medical knowledge...")
            vectorstore.add_texts(texts=dummy_medical_documents)
            vectorstore.persist()
            st.write(f"Added {len(dummy_medical_documents)} documents to ChromaDB.")
        else:
            st.write(f"ChromaDB already contains {vectorstore._collection.count()} documents.")
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        st.write("Attempting to re-initialize and populate ChromaDB...")
        vectorstore = Chroma.from_texts(texts=dummy_medical_documents, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)
        vectorstore.persist()
        st.write(f"Successfully re-initialized and populated ChromaDB with {len(dummy_medical_documents)} documents.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 documents

    # 3. LLM Model (using Hugging Face Transformers pipeline)
    # Load tokenizer and model for the LLM
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.bfloat16) # Use bfloat16 for efficiency if supported

    # Create a text generation pipeline
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        device=0 if torch.cuda.is_available() else -1, # Use GPU if available
        trust_remote_code=True # For some models, this might be needed
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # 4. Prompt Template
    template = """You are a Medical Diagnostic Assistant. Provide diagnostic insights and treatment recommendations based ONLY on the following medical context and the patient's symptoms. If the information is not in the context, state that you cannot provide a definitive diagnosis and recommend consulting a healthcare professional.

Context:
{context}

Patient Symptoms: {question}

Diagnosis and Recommendations:"""
    prompt = PromptTemplate.from_template(template)

    # 5. RAG Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    st.success("RAG system initialized!")
    return rag_chain

# --- Streamlit Application --- #
def main():
    st.set_page_config(page_title="Medical Diagnostic Assistant", layout="wide")
    st.title("🩺 Medical Diagnostic Assistant")

    st.markdown(
        """This AI assistant provides diagnostic insights and treatment recommendations based on a knowledge base and your input. 
        **Always consult a qualified healthcare professional for medical advice.**
        """
    )

    rag_chain = initialize_rag_system()

    patient_symptoms = st.text_area(
        "Describe the patient's symptoms (e.g., 'cough, fever, shortness of breath, chest pain'):",
        height=150
    )

    if st.button("Get Diagnosis and Recommendations"): 
        if patient_symptoms:
            with st.spinner("Analyzing symptoms and retrieving medical knowledge..."):
                try:
                    response = rag_chain.invoke(patient_symptoms)
                    st.subheader("AI Diagnosis and Recommendations:")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}. Please try again.")
                    st.write("Ensure your LLM model is correctly loaded and accessible.")
        else:
            st.warning("Please enter some symptoms to get a diagnosis.")

    st.markdown("---")
    st.info("Disclaimer: This tool is for informational purposes only and does not constitute medical advice. Consult a healthcare professional for any health concerns.")

if __name__ == "__main__":
    main()
