
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Set your OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it in a .env file or directly.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)

# --- Knowledge Base Simulation (for demonstration) ---
# In a real application, this would be a comprehensive medical knowledge base
# populated with actual medical documents, guidelines, etc.
# For this example, we'll use a simple in-memory ChromaDB with mock documents.

medical_docs = [
    "Symptoms of influenza include fever, cough, sore throat, muscle aches, and fatigue. It is caused by influenza viruses.",
    "Common cold symptoms are milder than flu and usually include a runny nose, sneezing, and sore throat, without high fever.",
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid. Symptoms include cough with phlegm, fever, chills, and difficulty breathing.",
    "Diabetes mellitus is a chronic condition that affects how your body turns food into energy. Symptoms include increased thirst, frequent urination, and unexplained weight loss.",
    "Hypertension, or high blood pressure, often has no symptoms. Regular check-ups are essential for diagnosis.",
    "A migraine is a severe headache often accompanied by throbbing pain, sensitivity to light and sound, nausea, and vomiting.",
    "Appendicitis is an inflammation of the appendix. Symptoms typically include sudden pain that begins on the right side of the lower abdomen, nausea, vomiting, and loss of appetite.",
    "Urinary tract infections (UTIs) commonly cause frequent urination, pain or burning during urination, and cloudy urine."
]

# Initialize embeddings for ChromaDB
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a ChromaDB instance from the mock documents
# This will store the embeddings of our simulated medical knowledge
vectorstore = Chroma.from_texts(texts=medical_docs, embedding=embeddings_model)

# Create a retriever for the knowledge base
retriever = vectorstore.as_retriever()

# --- Langchain Chains for Enhanced Reasoning ---

# 1. Initial Hypothesis Generation (Chain-of-Thought Step 1)
# This prompt encourages the LLM to think step-by-step about possible conditions.
initial_hypothesis_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are a helpful medical diagnostic assistant. Your goal is to generate initial differential diagnoses based on patient symptoms. Think step-by-step."),
    HumanMessage("Patient symptoms: {symptoms}\nMedical history: {history}\n\nBased on these, what are the top 3-5 most probable conditions? For each, briefly explain why you consider it, and suggest one crucial follow-up question or test.")
])
initial_hypothesis_chain = LLMChain(llm=llm, prompt=initial_hypothesis_prompt)

# 2. Evidence Gathering and Refinement (RAG + CoT Step 2)
# This chain uses retrieved information to refine hypotheses and verify facts.
refinement_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are a medical reasoning assistant. Your task is to refine a given differential diagnosis based on patient context and provided medical knowledge. Evaluate the initial hypotheses against the medical facts and provide a more confident assessment. Explain your reasoning thoroughly."),
    HumanMessage("Patient Symptoms: {symptoms}\nMedical History: {history}\nInitial Hypotheses: {hypotheses}\n\nRetrieved Medical Knowledge:\n{medical_facts}\n\nBased on all this information, critically analyze each initial hypothesis. For each, state if it's supported, less likely, or needs more investigation, and why. Then, propose a most probable diagnosis with detailed justification.")
])
refinement_chain = LLMChain(llm=llm, prompt=refinement_prompt)

# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="Enhanced Medical Diagnostic Assistant")
st.title("🧠 Enhanced Medical Diagnostic Assistant")
st.markdown("This assistant uses advanced LLM reasoning (Chain-of-Thought, RAG, simulated verification) to help generate differential diagnoses.")

with st.sidebar:
    st.header("Patient Information")
    patient_symptoms = st.text_area("Enter Patient Symptoms (e.g., 'fever, cough, body aches, fatigue for 3 days')", height=150)
    medical_history = st.text_area("Enter Medical History (e.g., 'no known allergies, healthy adult male')", height=100)
    process_button = st.button("Get Diagnostic Assistance")

if process_button and patient_symptoms:
    st.subheader("Diagnostic Process & Reasoning")

    with st.spinner("Step 1: Generating Initial Hypotheses..."):
        # Step 1: Generate initial hypotheses
        initial_output = initial_hypothesis_chain.run(symptoms=patient_symptoms, history=medical_history)
        st.markdown("#### Initial Hypotheses (Chain-of-Thought)")
        st.write(initial_output)

    # Extract key terms for retrieval from initial hypotheses or symptoms
    # For a more robust system, a separate LLM call could extract these
    search_query = patient_symptoms + " " + initial_output[:100] # Use a portion of output as well

    with st.spinner("Step 2: Retrieving Medical Knowledge for Verification..."):
        # Step 2: Retrieve relevant medical knowledge
        retrieved_docs = retriever.get_relevant_documents(search_query)
        medical_facts = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        st.markdown("#### Retrieved Medical Knowledge (RAG)")
        if medical_facts:
            st.text_area("Relevant Medical Facts:", medical_facts, height=200, disabled=True)
        else:
            st.info("No specific medical facts retrieved for this query. This could indicate sparse knowledge base or unusual symptoms.")

    with st.spinner("Step 3: Refining Diagnosis and Verifying..."):
        # Step 3: Refine diagnosis using retrieved knowledge and perform verification
        final_diagnosis_output = refinement_chain.run(
            symptoms=patient_symptoms,
            history=medical_history,
            hypotheses=initial_output,
            medical_facts=medical_facts
        )
        st.markdown("#### Refined Diagnosis & Verification (Enhanced Reasoning)")
        st.write(final_diagnosis_output)

    st.success("Diagnostic process complete!")
elif process_button and not patient_symptoms:
    st.warning("Please enter patient symptoms to proceed.")

st.markdown("""
---
**Disclaimer:** This is an AI-powered assistant and should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for any medical concerns.
""")
