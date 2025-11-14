import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pandas as pd
import nltk
# nltk.download('punkt') # Uncomment if running for the first time
# nltk.download('stopwords') # Uncomment if running for the first time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import re

# --- Configuration ---
# Set your OpenAI API key here or as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Ensure NLTK data is available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')


# --- 1. Patient Data Ingestion & Preprocessing ---
def preprocess_patient_data(symptoms: str, history: str, lab_results: str) -> dict:
    """
    Standardizes and structures patient data for LLM input.
    """
    processed_data = {
        "symptoms": symptoms.strip(),
        "medical_history": history.strip(),
        "lab_results": lab_results.strip(),
        "combined_text": f"Symptoms: {symptoms}\nMedical History: {history}\nLab Results: {lab_results}"
    }
    # Basic text cleaning (e.g., lowercasing, removing extra spaces)
    for key in ["symptoms", "medical_history", "lab_results", "combined_text"]:
        processed_data[key] = re.sub(r'\s+', ' ', processed_data[key].lower()).strip()

    return processed_data

# --- 2. Knowledge Base Integration (RAG) ---
# Initialize embedding model
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a dummy Chroma vector store for demonstration
# In a real application, this would be populated with extensive medical literature.
# We'll use in-memory for simplicity.
docs = [
    {"text": "Appendicitis symptoms include acute abdominal pain, nausea, vomiting, and fever. Diagnosis often involves physical exam, blood tests (elevated white blood cell count), and imaging (ultrasound or CT scan)."},
    {"text": "Diabetes Mellitus Type 2 is characterized by insulin resistance. Symptoms include frequent urination, increased thirst, and fatigue. Diagnosis involves blood glucose tests (HbA1c). Treatment includes diet, exercise, and medication like metformin."},
    {"text": "Myocardial infarction (heart attack) is caused by blockage of blood flow to the heart muscle. Symptoms include chest pain radiating to the arm, shortness of breath. Diagnosis with ECG and cardiac enzyme tests. Treatment involves angioplasty or bypass surgery."},
    {"text": "Migraine headaches are severe headaches often accompanied by nausea, vomiting, and sensitivity to light and sound. Triggers can include certain foods, stress, and hormonal changes. Treatment involves pain relievers and preventative medications."},
    {"text": "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. Diagnosis with chest X-ray. Treatment with antibiotics for bacterial pneumonia."}
]
doc_texts = [d["text"] for d in docs]
metadatas = [{"source": "medical_guidelines"} for _ in docs]

vectorstore = Chroma.from_texts(
    texts=doc_texts,
    embedding=embeddings_model
    # persist_directory="./chroma_db" # Uncomment to persist the vector store
)
retriever = vectorstore.as_retriever()


# --- 3. Core LLM Reasoning Engine ---
llm = ChatOpenAI(model="gpt-4", temperature=0.7) # Using GPT-4 as an example powerful LLM

# --- Chain-of-Thought (CoT) Prompt Template ---
cot_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a highly experienced medical diagnostic AI assistant. Analyze the provided patient data step-by-step to arrive at a primary diagnosis, differential diagnoses, and initial treatment recommendations. Explain your reasoning clearly and concisely."),
        ("human", "Patient Data:\n\nSymptoms: {symptoms}\nMedical History: {medical_history}\nLab Results: {lab_results}\n\nBased on this information, please:\n1. Provide a step-by-step Chain-of-Thought (CoT) reasoning process.\n2. State the primary diagnosis.\n3. List differential diagnoses with brief justifications.\n4. Suggest initial treatment recommendations.\n5. Indicate your confidence level (1-100%).")
    ]
)

# --- Chain-of-Verification (CoVe) and Faithful Reasoning Prompt Template ---
cove_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a medical verification AI. Given a proposed diagnosis and reasoning, along with relevant medical facts, evaluate the consistency, accuracy, and completeness of the diagnosis. Identify any inconsistencies or suggest improvements/alternative interpretations. If the diagnosis seems plausible and well-supported, confirm it."),
        ("human", "Patient Data:\n{patient_data}\n\nProposed Diagnosis and Reasoning:\n{initial_diagnosis_reasoning}\n\nRelevant Medical Facts:\n{context}\n\nBased on these facts, please verify the proposed diagnosis. Point out any discrepancies or areas for improvement. Confirm if it's well-supported.")
    ]
)

# --- Main Diagnosis Function ---
def run_diagnosis(symptoms: str, history: str, lab_results: str):
    """
    Orchestrates the LLM reasoning, verification, and aggregation.
    """
    st.subheader("🤖 AI Analysis")
    st.write("---")

    # 1. Preprocess Patient Data
    patient_data = preprocess_patient_data(symptoms, history, lab_results)
    st.markdown("**Processed Patient Data:**")
    st.json(patient_data)
    st.write("---")

    # 2. Initial Diagnosis Generation (Chain-of-Thought)
    st.markdown("**Step 1: Generating Initial Diagnosis (Chain-of-Thought)**")
    with st.spinner("LLM is thinking... generating initial diagnostic hypotheses and reasoning."):
        cot_chain = cot_prompt_template | llm
        initial_diagnosis_response = cot_chain.invoke(patient_data)
        initial_diagnosis_text = initial_diagnosis_response.content
    st.write("Initial Diagnosis & Reasoning:")
    st.markdown(initial_diagnosis_text)
    st.write("---")

    # 3. Self-Correction & Verification (Chain-of-Verification)
    st.markdown("**Step 2: Verifying Diagnosis with Medical Knowledge (RAG & CoVe)**")
    with st.spinner("Retrieving relevant medical facts and performing verification..."):
        # Retrieve context based on the initial diagnosis and patient data
        # Using a combination of initial diagnosis and patient data for retrieval
        retrieval_query = f"Patient condition: {patient_data['combined_text']}. Possible diagnosis: {initial_diagnosis_text.split('1. Provide a step-by-step Chain-of-Thought (CoT) reasoning process.')[-1].split('2. State the primary diagnosis.')[0]}"
        docs = retriever.invoke(retrieval_query)
        context = "\n".join([doc.page_content for doc in docs])

        if not context:
            st.warning("No relevant medical facts retrieved from the knowledge base. Verification might be limited.")
            context = "No specific medical facts found in the knowledge base relevant to this query for verification."

        cove_chain = cove_prompt_template | llm
        verification_response = cove_chain.invoke({
            "patient_data": patient_data['combined_text'],
            "initial_diagnosis_reasoning": initial_diagnosis_text,
            "context": context
        })
        verification_text = verification_response.content
    st.write("Verification Feedback:")
    st.markdown(verification_text)
    st.write("---")

    # 4. Robust Aggregation (Simplified - could involve multiple LLM calls for self-consistency)
    # For this example, we'll combine the initial diagnosis and verification feedback.
    # In a full implementation, multiple CoT paths might be generated and then aggregated.
    st.markdown("**Step 3: Aggregating Results & Final Recommendations**")
    final_output = f"**Initial AI Assessment:**\n{initial_diagnosis_text}\n\n---\n\n**Verification & Refinement:**\n{verification_text}\n\n---\n\n**Overall Conclusion & Actionable Insights:**\nBased on the initial assessment and subsequent verification against relevant medical knowledge, here is the refined understanding and recommended next steps for the healthcare professional. This section would synthesize the verified diagnosis, consolidate treatment options, and highlight any remaining uncertainties or areas requiring further investigation."

    # A more sophisticated aggregation would analyze both responses and create a new summary.
    # For now, we present both for transparency.
    st.write("Final AI-Assisted Recommendation (Synthesized from Initial Diagnosis and Verification):")
    st.markdown(final_output)
    st.success("Analysis Complete!")

# --- Streamlit Frontend ---
st.set_page_config(layout="wide", page_title="MediReason AI Diagnostic Assistant")

st.title("👨‍⚕️ MediReason AI: Advanced Diagnostic & Treatment Assistant")
st.markdown("""
This AI system assists healthcare professionals in diagnosing complex cases and generating personalized treatment recommendations.
It employs advanced LLM reasoning techniques like Chain-of-Thought (CoT) for structured analysis and Chain-of-Verification (CoVe) with a medical knowledge base (RAG) for enhanced reliability and self-correction.
""")

st.header("Patient Information Input")

with st.form("patient_data_form"):
    st.subheader("Symptoms")
    symptoms_input = st.text_area("List all patient symptoms (e.g., 'acute abdominal pain, nausea, fever for 2 days').", height=150)

    st.subheader("Medical History")
    history_input = st.text_area("Provide relevant medical history (e.g., 'no significant past medical history, allergic to penicillin').", height=150)

    st.subheader("Lab Results & Imaging Findings")
    lab_results_input = st.text_area("Enter key lab results and imaging findings (e.g., 'WBC 15,000, CT scan shows inflamed appendix').", height=150)

    submitted = st.form_submit_button("Get AI Diagnosis & Treatment Recommendations")

if submitted:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set your OPENAI_API_KEY environment variable or hardcode it in the script to run the LLM.")
    elif not symptoms_input and not history_input and not lab_results_input:
        st.warning("Please enter some patient data to get a diagnosis.")
    else:
        run_diagnosis(symptoms_input, history_input, lab_results_input)

st.sidebar.header("About MediReason AI")
st.sidebar.markdown("""
This application demonstrates the **Enhanced LLM Reasoning and Reliability** pattern by integrating:
- **Chain-of-Thought (CoT)**: For structured, step-by-step diagnostic reasoning.
- **Retrieval Augmented Generation (RAG)**: Using a medical knowledge base (ChromaDB) to ground LLM outputs.
- **Chain-of-Verification (CoVe)**: A verification step to cross-check initial diagnoses against retrieved facts.

**Note:** This is a simplified prototype for demonstration. Do not use for actual medical advice. Always consult with qualified healthcare professionals.
""")