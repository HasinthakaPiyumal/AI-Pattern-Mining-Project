
import streamlit as st
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import json

# LangChain components (simulated/simplified for a single file)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- 1. Medical Knowledge Base (Simplified for demonstration) ---
MEDICAL_DOCUMENTS = [
    "Symptoms of common cold include runny nose, sore throat, cough, congestion, slight body aches or mild headache, sneezing, low-grade fever, and a general feeling of being unwell (malaise).",
    "Influenza (flu) symptoms often come on suddenly and can include fever or feeling feverish/chills, cough, sore throat, runny or stuffy nose, muscle or body aches, headaches, and fatigue. Some people may have vomiting and diarrhea, though this is more common in children.",
    "For bacterial pneumonia, common treatments include antibiotics like amoxicillin, azithromycin, or doxycycline. Rest, fluids, and pain relievers are also recommended.",
    "Diabetes type 2 is characterized by high blood sugar due to insulin resistance or insufficient insulin production. Symptoms include increased thirst, frequent urination, increased hunger, fatigue, and blurred vision. Management involves diet, exercise, and medication like metformin.",
    "Hypertension (high blood pressure) often has no symptoms. Regular blood pressure checks are crucial. Treatments include lifestyle changes (diet, exercise) and medications such as ACE inhibitors, ARBs, diuretics, and beta-blockers.",
    "Appendicitis typically presents with pain that begins around the navel and then shifts to the lower right abdomen. Other symptoms may include nausea, vomiting, loss of appetite, low-grade fever, and constipation or diarrhea."
]

# --- 2. RALM Core Components ---

# Embedding Model Initialization
# This will download the model the first time it's run.
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# ChromaDB Initialization with in-memory client for simplicity
# For persistent storage, use: Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
vectorstore = Chroma.from_texts(MEDICAL_DOCUMENTS, embeddings_model)
retriever = vectorstore.as_retriever()

# Simulated LLM for demonstration purposes
class SimulatedLLM:
    def invoke(self, prompt: str) -> str:
        # Simulate a delay and provide a predefined response structure
        import time
        time.sleep(0.5) # Simulate processing time

        # Extract the query and context from the prompt
        # This is a very basic parsing; a real LLM would understand structure.
        query_start = prompt.find("User Query:")
        context_start = prompt.find("Context:")
        if query_start != -1 and context_start != -1:
            user_query = prompt[query_start + len("User Query:"):context_start].strip()
            context = prompt[context_start + len("Context:"):].strip()
        else:
            user_query = "N/A"
            context = prompt # Fallback if structure not found

        if "cold" in user_query.lower() or "cough" in user_query.lower():
            return f"Based on the retrieved information and your symptoms, it is likely a common cold. \n\n**Evidence from Knowledge Base:**\n{context}\n\n**Recommendation:** Rest, fluids, and over-the-counter cold remedies. If symptoms worsen, consult a doctor."
        elif "flu" in user_query.lower() or "fever" in user_query.lower() and "aches" in user_query.lower():
            return f"The symptoms suggest influenza. \n\n**Evidence from Knowledge Base:**\n{context}\n\n**Recommendation:** Get plenty of rest, stay hydrated. Antiviral drugs may be prescribed if caught early. Consult a doctor for proper diagnosis and treatment."
        elif "pneumonia" in user_query.lower():
             return f"Symptoms are consistent with pneumonia. \n\n**Evidence from Knowledge Base:**\n{context}\n\n**Recommendation:** Antibiotics (if bacterial), rest, and fluids. Medical consultation is essential."
        elif "diabetes" in user_query.lower():
             return f"Based on the query and context, consider Type 2 Diabetes. \n\n**Evidence from Knowledge Base:**\n{context}\n\n**Recommendation:** Consult an endocrinologist for diagnosis and management plan including diet, exercise, and medication."
        elif "blood pressure" in user_query.lower():
             return f"Your query relates to hypertension. \n\n**Evidence from Knowledge Base:**\n{context}\n\n**Recommendation:** Monitor blood pressure regularly. Adopt a healthy lifestyle. Consult a doctor for medication options if needed."
        elif "appendicitis" in user_query.lower() or "lower right abdomen pain" in user_query.lower():
             return f"Immediate medical attention is advised due to potential appendicitis. \n\n**Evidence from Knowledge Base:**\n{context}\n\n**Recommendation:** Seek emergency medical care immediately for diagnosis and potential surgery."
        else:
            return f"I can provide information based on the knowledge base. For a definitive diagnosis, please consult a healthcare professional. \n\n**Evidence from Knowledge Base:**\n{context}"

simulated_llm = SimulatedLLM()

# LangChain RAG Chain setup
rag_prompt = ChatPromptTemplate.from_template(
    """You are a helpful healthcare diagnostic assistant. Use the following pieces of retrieved context to answer the user's question. If you don't know the answer, just say that you don't have enough information. Do not try to make up an answer.

User Query: {question}

Context: {context}

Diagnostic Suggestion and Recommendations:
"""
)

# Simplified Context Assessment: always perform retrieval for now
def context_assessment(query: str) -> bool:
    # In a real scenario, this would involve NLP to determine if retrieval is needed
    # For this demo, we always assume retrieval is beneficial.
    return True

def dynamic_knowledge_retrieval(query: str, num_results: int = 3) -> List[str]:
    # Adapt retrieval based on an assumed 'complexity' (here, just fixed num_results)
    docs = retriever.invoke(query, k=num_results)
    return [doc.page_content for doc in docs]

# --- 3. FastAPI Backend ---
app = FastAPI(
    title="Healthcare Diagnostic Assistant API",
    description="API for an Adaptive and Optimized Retrieval-Augmented Language Model (RALM) for healthcare diagnostics.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    patient_symptoms: str

class DiagnosticResponse(BaseModel):
    diagnostic_suggestion: str
    evidence: List[str]
    recommendations: str

@app.post("/diagnose", response_model=DiagnosticResponse)
async def diagnose(request: QueryRequest):
    query = request.patient_symptoms

    if context_assessment(query):
        retrieved_context = dynamic_knowledge_retrieval(query)
        context_str = "\n".join([f"- {doc}" for doc in retrieved_context])
    else:
        retrieved_context = []
        context_str = "No specific medical context retrieved based on query complexity."

    # Create the prompt for the LLM
    full_prompt = rag_prompt.format(question=query, context=context_str)

    # Invoke the simulated LLM
    llm_response_raw = simulated_llm.invoke(full_prompt)

    # Parse the simulated LLM response (this would be more robust with a real LLM)
    diagnostic_suggestion = ""
    recommendations = ""
    
    # Simple parsing logic for the simulated LLM output
    if "\n\n**Evidence from Knowledge Base:**\n" in llm_response_raw:
        parts = llm_response_raw.split("\n\n**Evidence from Knowledge Base:**\n", 1)
        diagnostic_and_reco = parts[0]
        evidence_section = parts[1]

        # Further split diagnostic and recommendation if present
        if "\n\n**Recommendation:**" in diagnostic_and_reco:
            diag_parts = diagnostic_and_reco.split("\n\n**Recommendation:**", 1)
            diagnostic_suggestion = diag_parts[0].strip()
            recommendations = diag_parts[1].strip()
        else:
            diagnostic_suggestion = diagnostic_and_reco.strip()
            recommendations = "Consult a healthcare professional for definitive advice."

        # Reconstruct evidence from retrieved_context, as the LLM's 'evidence' is just a copy of the input context
        evidence_list = retrieved_context

    else: # Fallback for less structured simulated responses
        diagnostic_suggestion = llm_response_raw.split("\n\n**Recommendation:**")[0].strip() if "\n\n**Recommendation:**" in llm_response_raw else llm_response_raw.strip()
        recommendations = llm_response_raw.split("\n\n**Recommendation:**")[1].strip() if "\n\n**Recommendation:**" in llm_response_raw else "Consult a healthcare professional for definitive advice."
        evidence_list = retrieved_context


    return DiagnosticResponse(
        diagnostic_suggestion=diagnostic_suggestion,
        evidence=evidence_list,
        recommendations=recommendations
    )

# --- 4. Streamlit Frontend ---

def run_streamlit_app():
    st.set_page_config(page_title="Healthcare Diagnostic Assistant")
    st.title("🩺 Healthcare Diagnostic Assistant")
    st.markdown("--- Developed using Adaptive and Optimized RALMs ---")

    st.sidebar.header("About")
    st.sidebar.info(
        "This assistant provides evidence-based diagnostic support and treatment recommendations "
        "by leveraging dynamically retrieved, up-to-date medical knowledge. It uses an "
        "Adaptive and Optimized Retrieval-Augmented Language Model (RALM) architecture." 
        "Please note: This is a prototype and should NOT be used for actual medical diagnosis."
    )
    st.sidebar.header("How to Use")
    st.sidebar.markdown(
        "1. Enter patient symptoms in the text area.\n"
        "2. Click 'Get Diagnosis' to receive a diagnostic suggestion, supporting evidence, and recommendations."
    )

    st.header("Patient Symptoms and Query")
    symptoms_input = st.text_area(
        "Describe the patient's symptoms or your diagnostic query here:",
        height=150,
        placeholder="e.g., 'Patient has severe cough, high fever, and body aches for 3 days.' or 'What are the treatments for Type 2 Diabetes?'"
    )

    if st.button("Get Diagnosis"):
        if not symptoms_input:
            st.warning("Please enter some symptoms or a query to get a diagnosis.")
        else:
            with st.spinner("Analyzing symptoms and retrieving medical knowledge..."):
                try:
                    import requests
                    # Assuming FastAPI is running on http://127.0.0.1:8000
                    backend_url = "http://127.0.0.1:8000/diagnose"
                    response = requests.post(backend_url, json={"patient_symptoms": symptoms_input})
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    
                    diagnostic_output = response.json()

                    st.subheader("Diagnostic Suggestion")
                    st.info(diagnostic_output.get("diagnostic_suggestion", "No specific suggestion."))

                    st.subheader("Recommendations")
                    st.success(diagnostic_output.get("recommendations", "No specific recommendations."))

                    st.subheader("Evidence from Knowledge Base")
                    for i, evidence_item in enumerate(diagnostic_output.get("evidence", [])):
                        st.markdown(f"**{i+1}.** {evidence_item}")
                    if not diagnostic_output.get("evidence"):
                        st.info("No specific evidence found in the knowledge base for this query.")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Make sure the backend is running at http://127.0.0.1:8000.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    st.sidebar.markdown("## Application Setup")
    app_mode = st.sidebar.radio("Choose application mode:", ["Run Streamlit Frontend", "Run FastAPI Backend"]) # Added an option to differentiate

    if app_mode == "Run Streamlit Frontend":
        run_streamlit_app()
    elif app_mode == "Run FastAPI Backend":
        st.sidebar.warning("To run the FastAPI backend, you need to execute this script using uvicorn from your terminal:\n\n`uvicorn healthcare_diagnostic_assistant:app --reload`\n\nThis Streamlit app cannot start the FastAPI server directly within its process due to execution model differences.")
        st.write("FastAPI Backend Mode selected. Please run the backend using uvicorn as instructed in the sidebar.")

