import os
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import FakeListLLM
from langchain_core.documents import Document

import streamlit as st

# --- Configuration --- #
# Set your OpenAI API key as an environment variable or replace 'os.getenv("OPENAI_API_KEY")' directly
# For local testing without a real LLM, set USE_FAKE_LLM = True
USE_FAKE_LLM = False

# --- Knowledge Base Setup --- #
# In-memory Chroma for demonstration
# In a real application, you'd persist this or connect to a dedicated Chroma server.
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Dummy medical literature and patient data
medical_literature_docs = [
    Document(page_content="Symptoms of common cold include runny nose, sore throat, cough, and congestion. It is caused by viruses.", metadata={"source": "medical_textbook"}),
    Document(page_content="Influenza, or flu, is a contagious respiratory illness caused by influenza viruses. Symptoms include fever, body aches, headache, and fatigue.", metadata={"source": "medical_textbook"}),
    Document(page_content="Diabetes mellitus is a chronic metabolic disease characterized by high blood glucose levels. Type 1 diabetes is an autoimmune condition; Type 2 is often lifestyle-related.", metadata={"source": "medical_textbook"}),
    Document(page_content="Hypertension, or high blood pressure, is a common condition that increases the risk of heart disease and stroke. Lifestyle changes and medication are common treatments.", metadata={"source": "medical_textbook"}),
    Document(page_content="Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm, fever, chills, and difficulty breathing.", metadata={"source": "medical_textbook"}),
]

# Initialize Chroma with dummy medical literature
vectorstore = Chroma.from_documents(documents=medical_literature_docs, embedding=embeddings)

def add_patient_data_to_vectorstore(patient_info: Dict[str, Any]):
    patient_docs = [
        Document(page_content=f"Patient Name: {patient_info.get('name')}", metadata={"source": "patient_record"}),
        Document(page_content=f"Age: {patient_info.get('age')}, Gender: {patient_info.get('gender')}", metadata={"source": "patient_record"}),
        Document(page_content=f"Chief Complaint: {patient_info.get('chief_complaint')}", metadata={"source": "patient_record"}),
        Document(page_content=f"Symptoms: {', '.join(patient_info.get('symptoms', []))}", metadata={"source": "patient_record"}),
        Document(page_content=f"Lab Results: {patient_info.get('lab_results')}", metadata={"source": "patient_record"}),
        Document(page_content=f"Medical History: {patient_info.get('medical_history')}", metadata={"source": "patient_record"}),
    ]
    vectorstore.add_documents(patient_docs)

# --- Reasoning Engine Setup --- #
if USE_FAKE_LLM:
    llm = FakeListLLM(responses=["Based on the provided information, it seems like a common cold. Rest and hydration are recommended.", "Considering the symptoms and lab results, influenza is a strong possibility. Antiviral medication might be considered.", "The patient's high blood glucose levels indicate diabetes, likely Type 2 given the history."])
else:
    llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

medical_qa_prompt_template = """You are a highly intelligent medical diagnostic assistant. Your goal is to provide accurate medical diagnoses and treatment recommendations based on the provided context and patient information.

Context:
{context}

Patient Information:
{patient_info_str}

Question: {question}

Based on the context and patient information, provide a concise diagnosis and recommended course of action. If you cannot determine a clear diagnosis, state what further information is needed. Focus on evidence-based answers.
"""

MEDICAL_QA_PROMPT = PromptTemplate(template=medical_qa_prompt_template, input_variables=["context", "patient_info_str", "question"])

def get_rag_chain(retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": MEDICAL_QA_PROMPT}
    )
    return qa_chain

# --- FastAPI Backend --- #
app = FastAPI()

class PatientInput(BaseModel):
    name: str
    age: int
    gender: str
    chief_complaint: str
    symptoms: List[str]
    lab_results: str = ""
    medical_history: str = ""
    clinical_question: str

@app.post("/diagnose")
async def diagnose_patient(patient_input: PatientInput):
    # Add patient data to the vector store temporarily for this query
    patient_info_dict = patient_input.model_dump(exclude={
        "clinical_question"
    })
    add_patient_data_to_vectorstore(patient_info_dict)

    # Convert patient_info_dict to a string for the prompt
    patient_info_str = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in patient_info_dict.items()])

    # Retrieve relevant documents (medical literature + patient data)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # Note: For actual hybrid retrieval, you'd integrate keyword search here.

    qa_chain = get_rag_chain(retriever)

    # Formulate the final question for the LLM
    result = qa_chain.invoke({
        "query": patient_input.clinical_question, # The query for retrieval
        "question": patient_input.clinical_question, # The question for the LLM
        "patient_info_str": patient_info_str
    })

    # Remove patient data for this specific session if using persistent vectorstore
    # (not strictly necessary for in-memory, but good practice)
    # For a real system, patient data would be handled more carefully with user sessions.

    return {"diagnosis": result["result"], "sources": [doc.metadata for doc in result["source_documents"]]}

# --- Streamlit Frontend --- #
def streamlit_app():
    st.set_page_config(page_title="Medical Diagnostic Assistant")
    st.title("🧠 Medical Diagnostic Assistant")
    st.markdown("An AI assistant to help clinicians with diagnoses and treatment recommendations.")

    st.header("Patient Information")
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=0, max_value=120)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col2:
            chief_complaint = st.text_area("Chief Complaint (main reason for visit)", height=50)
            symptoms_input = st.text_area("Symptoms (comma-separated)", height=100)
            medical_history = st.text_area("Medical History (e.g., pre-existing conditions)", height=100)
            lab_results = st.text_area("Lab Results (e.g., 'Blood pressure 140/90, Glucose 180 mg/dL')", height=100)

        clinical_question = st.text_area("Clinical Question (e.g., 'What is the likely diagnosis?' or 'What treatment options are suitable?')", height=100)

        submit_button = st.form_submit_button("Get Diagnosis and Recommendations")

        if submit_button:
            if not name or not chief_complaint or not clinical_question:
                st.error("Please fill in Patient Name, Chief Complaint, and Clinical Question.")
            else:
                symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
                patient_data = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "chief_complaint": chief_complaint,
                    "symptoms": symptoms,
                    "lab_results": lab_results,
                    "medical_history": medical_history,
                    "clinical_question": clinical_question,
                }

                try:
                    # Assuming FastAPI is running on http://127.0.0.1:8000
                    import requests
                    response = requests.post("http://127.0.0.1:8000/diagnose", json=patient_data)
                    response.raise_for_status()
                    result = response.json()

                    st.subheader("Diagnosis & Recommendations")
                    st.write(result["diagnosis"])

                    st.subheader("Sources")
                    for source in result["sources"]:
                        st.write(f"- {source.get('source', 'N/A')}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Please ensure it's running (run 'python medical_diagnostic_assistant.py --run_backend').")
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")

# --- Main Execution --- #
if __name__ == "__main__":
    import sys
    if "--run_backend" in sys.argv:
        st.write("Starting FastAPI backend...") # This will not be displayed in FastAPI, but useful for context
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif "--run_frontend" in sys.argv or len(sys.argv) == 1:
        st.write("Starting Streamlit frontend...") # This will not be displayed in Streamlit console
        streamlit_app()
    else:
        st.error("Invalid argument. Use '--run_backend' to start the FastAPI server or '--run_frontend' (or no argument) to start the Streamlit app.")
