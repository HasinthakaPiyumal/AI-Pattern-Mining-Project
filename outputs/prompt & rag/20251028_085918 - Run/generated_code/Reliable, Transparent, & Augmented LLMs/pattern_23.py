import os
from typing import List, Dict, Any, Optional

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import streamlit as st
import guardrails as gr

# --- Environment Setup (Mock for single file) ---
# In a real application, you would load these from .env
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# --- Pydantic Models for Data Validation ---
class PatientData(BaseModel):
    symptoms: str
    medical_history: str
    lab_results: str

class DiagnosticSuggestion(BaseModel):
    diagnosis: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    confidence_qualitative: str
    short_reasoning: str

class FullDiagnosisResponse(BaseModel):
    id: str
    initial_suggestion: DiagnosticSuggestion
    detailed_reasoning: str
    supporting_evidence: List[str]

class ReasoningResponse(BaseModel):
    reasoning: str

class EvidenceResponse(BaseModel):
    evidence: List[str]


# --- Guardrails AI Schema for Confidence Scoring ---
# This schema defines the expected output structure for LLM-generated confidence.
confidence_guard_rail = gr.Guard.from_string(
    validators=["llm-output-quality"],
    prompt_kwargs={},
    output_schema="""
<define name="confidence_schema" type="object">
    <string name="confidence_qualitative" description="Qualitative assessment of confidence (e.g., 'High', 'Medium', 'Low')" />
    <float name="confidence_score" description="Numerical confidence score between 0.0 and 1.0" />
</define>
"""
)

# --- ChromaDB and Embedding Model Setup ---
# Using an in-memory ChromaDB for simplicity in a single file example
client = chromadb.Client()
try:
    knowledge_collection = client.create_collection(name="medical_knowledge")
except Exception as e:
    # If collection already exists, get it
    knowledge_collection = client.get_collection(name="medical_knowledge")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    return embedding_model.encode(texts).tolist()

# --- Simulate Medical Knowledge Base Population ---
def populate_knowledge_base():
    docs = [
        "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness. Serious outcomes of flu infection can result in hospitalization or death. Symptoms include fever, cough, sore throat, muscle aches, and fatigue. Diagnosis is often clinical, but can be confirmed with rapid flu tests or PCR. Treatment includes antivirals like oseltamivir.",
        "Type 2 Diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). The body either doesn't produce enough insulin, or it resists the effects of insulin. Symptoms include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision. Diagnosis involves blood tests like A1C. Management includes diet, exercise, and medication like metformin.",
        "Myocardial Infarction (heart attack) occurs when blood flow to a part of your heart is blocked, usually by a blood clot. The interruption of blood flow can damage or destroy part of the heart muscle. Symptoms include chest pain that may spread to your arm, neck, jaw, or back, shortness of breath, cold sweat, nausea, lightheadedness. Diagnosis often involves ECG and blood tests for cardiac markers (e.g., troponin). Treatment includes aspirin, nitrates, beta-blockers, and reperfusion therapy.",
        "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon on the lower right side of your abdomen. Symptoms often include sudden pain that begins around your navel and shifts to your lower right abdomen, nausea, vomiting, loss of appetite, low-grade fever, constipation or diarrhea. Diagnosis is based on physical exam, blood tests (elevated white blood cell count), and imaging (ultrasound or CT scan). Treatment is usually surgical removal of the appendix (appendectomy).",
        "Urinary Tract Infection (UTI) is an infection in any part of your urinary system. It's more common in women. Symptoms include a strong, persistent urge to urinate, a burning sensation when urinating, passing frequent, small amounts of urine, cloudy urine, strong-smelling urine, pelvic pain in women. Diagnosis is typically with a urine test. Treatment involves antibiotics.",
        "Migraine is a type of headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can cause significant pain for hours to days. Diagnosis is clinical. Treatment includes pain relievers, triptans, and preventative medications."
    ]
    ids = [f"doc_{i}" for i in range(len(docs))]
    embeddings = get_embeddings(docs)
    knowledge_collection.add(documents=docs, embeddings=embeddings, ids=ids)
    st.success("Medical knowledge base populated (in-memory).")


# --- Mock LLM Service ---
class MockLLMService:
    _diagnosis_counter = 0
    _diagnoses_store: Dict[str, FullDiagnosisResponse] = {}

    def _generate_mock_llm_response(self, prompt: str) -> Dict[str, Any]:
        # Simulate LLM processing and response based on prompt keywords
        # In a real scenario, this would call an actual LLM API

        diagnosis_text = """Hypothetical Diagnosis: Unspecified condition. This is a placeholder diagnosis. 
Based on the input, a deeper analysis with real medical data and an LLM would be required to provide an accurate diagnosis.
The LLM would analyze the symptoms, history, and lab results in context of the retrieved medical knowledge.
"""
        reasoning_text = """This reasoning is a placeholder. In a real system, the LLM would articulate its thought process,
connecting the patient's symptoms and history to medical conditions, and referencing the retrieved knowledge base.
For example, if the symptoms strongly align with influenza, the reasoning would detail that alignment.
"""
        evidence_list = [
            "Evidence 1: Placeholder for retrieved document snippet matching symptoms.",
            "Evidence 2: Placeholder for retrieved document snippet matching lab results.",
            "Evidence 3: Placeholder for relevant medical guideline or research paper excerpt."
        ]

        # Simple keyword-based mock for demonstration
        if "flu" in prompt.lower() or "influenza" in prompt.lower():
            diagnosis_text = "Probable Influenza. The patient's symptoms (fever, cough, body aches) are highly suggestive of influenza. Given the flu season, this is a strong possibility."
            reasoning_text = "The combination of fever, cough, and body aches points towards a viral respiratory infection like influenza. Lab results (if available and indicative) would further support this. Differential diagnoses would include common cold or other viral infections, but the severity suggested by 'body aches' leans towards flu."
            evidence_list[0] = "Retrieved: Influenza (flu) is a contagious respiratory illness... Symptoms include fever, cough, sore throat, muscle aches, and fatigue."
        elif "diabetes" in prompt.lower() or "high blood sugar" in prompt.lower():
            diagnosis_text = "Suspected Type 2 Diabetes. Patient presents with increased thirst and fatigue, which are common symptoms. Further lab tests (A1C) would be crucial."
            reasoning_text = "Increased thirst and fatigue are classic signs of elevated blood glucose. Medical history (e.g., family history of diabetes, lifestyle factors) would be important here. Without actual lab results indicating high blood sugar, this remains a suspicion."
            evidence_list[0] = "Retrieved: Type 2 Diabetes is a chronic condition... Symptoms include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision."
        elif "chest pain" in prompt.lower() or "heart" in prompt.lower():
            diagnosis_text = "Potential Myocardial Infarction. Urgent cardiac evaluation is recommended due to reported chest pain and shortness of breath. This requires immediate medical attention."
            reasoning_text = "Acute onset chest pain, especially radiating or accompanied by shortness of breath, is a red flag for cardiac events. An ECG and cardiac enzyme tests are critical for diagnosis. This is a medical emergency."
            evidence_list[0] = "Retrieved: Myocardial Infarction (heart attack) occurs when blood flow to a part of your heart is blocked... Symptoms include chest pain that may spread to your arm, neck, jaw, or back, shortness of breath."
        elif "abdominal pain" in prompt.lower() and "lower right" in prompt.lower():
            diagnosis_text = "Possible Appendicitis. The reported lower right abdominal pain warrants further investigation, especially if associated with nausea and fever."
            reasoning_text = "Classic presentation of appendicitis involves periumbilical pain migrating to the right lower quadrant. Physical exam for rebound tenderness and imaging (ultrasound/CT) are essential for confirmation."
            evidence_list[0] = "Retrieved: Appendicitis is an inflammation of the appendix... Symptoms often include sudden pain that begins around your navel and shifts to your lower right abdomen."
        elif "urinate" in prompt.lower() or "burning" in prompt.lower() and "urine" in prompt.lower():
            diagnosis_text = "Likely Urinary Tract Infection (UTI). Frequent and painful urination are key indicators."
            reasoning_text = "Dysuria (painful urination) and frequency are hallmark symptoms of a UTI. A urine dipstick test and culture would confirm the diagnosis."
            evidence_list[0] = "Retrieved: Urinary Tract Infection (UTI) is an infection in any part of your urinary system... Symptoms include a strong, persistent urge to urinate, a burning sensation when urinating."
        elif "headache" in prompt.lower() and "throbbing" in prompt.lower():
            diagnosis_text = "Consistent with Migraine. Severe, throbbing headache, potentially with light/sound sensitivity, aligns with migraine symptoms."
            reasoning_text = "The description of a severe, throbbing headache, often unilateral, with associated symptoms like sensitivity to light/sound, is highly characteristic of a migraine attack. Exclusion of other headache types is important."
            evidence_list[0] = "Retrieved: Migraine is a type of headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound."


        # Simulate confidence scoring using Guardrails (mocked LLM output)
        llm_confidence_output = {
            "confidence_qualitative": "Medium" if "unspecified" in diagnosis_text else "High",
            "confidence_score": 0.6 if "unspecified" in diagnosis_text else 0.85
        }

        # In a real scenario, you'd call the LLM and pass its raw output to guard_rail.parse()
        # validated_output = confidence_guard_rail.parse(llm_output_for_confidence_scoring)
        # For this mock, we're directly using our simulated structured output
        validated_output = confidence_guard_rail.parse(llm_confidence_output)

        return {
            "diagnosis": diagnosis_text,
            "reasoning": reasoning_text,
            "confidence": validated_output,
            "evidence": evidence_list
        }

    def diagnose_and_reason(self, patient_data: PatientData, retrieved_knowledge: List[str]) -> FullDiagnosisResponse:
        MockLLMService._diagnosis_counter += 1
        diagnosis_id = f"diag_{MockLLMService._diagnosis_counter}"

        # Create a prompt for the mock LLM
        prompt = f"Patient Symptoms: {patient_data.symptoms}. Medical History: {patient_data.medical_history}. Lab Results: {patient_data.lab_results}. \n\nRelevant Medical Knowledge: {'. '.join(retrieved_knowledge)}\n\nProvide a diagnosis, short reasoning, and rate your confidence (qualitative and numerical between 0.0 and 1.0) for the diagnosis based on the provided information."

        mock_llm_output = self._generate_mock_llm_response(prompt)

        initial_suggestion = DiagnosticSuggestion(
            diagnosis=mock_llm_output["diagnosis"],
            confidence_score=mock_llm_output["confidence"]["confidence_score"],
            confidence_qualitative=mock_llm_output["confidence"]["confidence_qualitative"],
            short_reasoning=mock_llm_output["reasoning"]
        )

        full_response = FullDiagnosisResponse(
            id=diagnosis_id,
            initial_suggestion=initial_suggestion,
            detailed_reasoning=mock_llm_output["reasoning"], # Reusing for simplicity, could be more detailed
            supporting_evidence=mock_llm_output["evidence"]
        )
        MockLLMService._diagnoses_store[diagnosis_id] = full_response
        return full_response

    def get_detailed_reasoning(self, diagnosis_id: str) -> ReasoningResponse:
        diagnosis = MockLLMService._diagnoses_store.get(diagnosis_id)
        if not diagnosis:
            raise HTTPException(status_code=404, detail="Diagnosis not found")
        return ReasoningResponse(reasoning=diagnosis.detailed_reasoning)

    def get_supporting_evidence(self, diagnosis_id: str) -> EvidenceResponse:
        diagnosis = MockLLMService._diagnoses_store.get(diagnosis_id)
        if not diagnosis:
            raise HTTPException(status_code=404, detail="Diagnosis not found")
        return EvidenceResponse(evidence=diagnosis.supporting_evidence)


llm_service = MockLLMService()

# --- FastAPI Application ---
app = FastAPI(
    title="Medical Diagnostic Assistant API",
    description="API for an AI-powered Medical Diagnostic Assistant with progressive disclosure."
)

@app.post("/diagnose", response_model=FullDiagnosisResponse)
async def diagnose_patient(patient_data: PatientData):
    # Retrieve relevant medical knowledge using RAG
    query = f"{patient_data.symptoms} {patient_data.medical_history} {patient_data.lab_results}"
    results = knowledge_collection.query(query_embeddings=get_embeddings([query]), n_results=3, include=['documents'])
    retrieved_knowledge = results['documents'][0] if results['documents'] else []

    # Simulate LLM diagnosis and reasoning
    diagnosis_response = llm_service.diagnose_and_reason(patient_data, retrieved_knowledge)

    # In a real app, integrate Langsmith/TruLens for logging here:
    print(f"[Monitoring] New diagnosis request: {diagnosis_response.id}")
    return diagnosis_response

@app.get("/diagnosis/{diagnosis_id}/reasoning", response_model=ReasoningResponse)
async def get_reasoning(diagnosis_id: str):
    return llm_service.get_detailed_reasoning(diagnosis_id)

@app.get("/diagnosis/{diagnosis_id}/evidence", response_model=EvidenceResponse)
async def get_evidence(diagnosis_id: str):
    return llm_service.get_supporting_evidence(diagnosis_id)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("👨‍⚕️ AI-Powered Medical Diagnostic Assistant")
st.markdown("This assistant helps healthcare professionals by providing diagnostic support with explainable AI.")

# Populate knowledge base on first run or when explicitly requested
if "kb_populated" not in st.session_state:
    with st.spinner("Populating medical knowledge base (in-memory)..."):
        populate_knowledge_base()
        st.session_state.kb_populated = True

st.header("Patient Information")

with st.form("patient_form"):
    symptoms = st.text_area("Symptoms (e.g., 'fever, cough, body aches')", height=100)
    medical_history = st.text_area("Medical History (e.g., 'no known allergies, hypertension controlled with medication')", height=100)
    lab_results = st.text_area("Relevant Lab Results (e.g., 'WBC 12.5, CRP 8.2')", height=100)
    submitted = st.form_submit_button("Get Diagnosis")

    if submitted:
        if not symptoms and not medical_history and not lab_results:
            st.warning("Please provide some patient information to get a diagnosis.")
        else:
            patient_data = PatientData(
                symptoms=symptoms,
                medical_history=medical_history,
                lab_results=lab_results
            )

            with st.spinner("Analyzing patient data and retrieving medical knowledge..."):
                # Directly call the service logic instead of making HTTP request
                # for a simpler single-file Streamlit/FastAPI integration demo.
                # In a real deployment, Streamlit would call the FastAPI endpoints.
                query = f"{patient_data.symptoms} {patient_data.medical_history} {patient_data.lab_results}"
                results = knowledge_collection.query(query_embeddings=get_embeddings([query]), n_results=3, include=['documents'])
                retrieved_knowledge = results['documents'][0] if results['documents'] else []
                
                try:
                    diagnosis_response: FullDiagnosisResponse = llm_service.diagnose_and_reason(patient_data, retrieved_knowledge)
                    st.session_state.current_diagnosis = diagnosis_response
                except HTTPException as e:
                    st.error(f"Error during diagnosis: {e.detail}")
                    st.session_state.current_diagnosis = None

if "current_diagnosis" in st.session_state and st.session_state.current_diagnosis:
    st.header("Diagnostic Assistant Output")
    diagnosis = st.session_state.current_diagnosis

    st.subheader("Initial Suggestion")
    st.success(f"**Diagnosis:** {diagnosis.initial_suggestion.diagnosis}")
    st.info(f"**Confidence:** {diagnosis.initial_suggestion.confidence_qualitative} ({diagnosis.initial_suggestion.confidence_score:.2f})")

    st.subheader("Progressive Disclosure")
    with st.expander("Show Short Reasoning Path"):
        st.markdown(diagnosis.initial_suggestion.short_reasoning)

    with st.expander("Show Detailed Reasoning"):
        # Directly call the service method
        detailed_reasoning = llm_service.get_detailed_reasoning(diagnosis.id)
        st.markdown(detailed_reasoning.reasoning)

    with st.expander("Show Supporting Evidence"):
        # Directly call the service method
        supporting_evidence = llm_service.get_supporting_evidence(diagnosis.id)
        for i, evidence in enumerate(supporting_evidence.evidence):
            st.markdown(f"- {evidence}")

st.markdown("""
**How to run this application:**

1.  Save the code as `medical_diagnostic_assistant.py`.
2.  Install required libraries:
    `pip install pandas chromadb sentence-transformers fastapi uvicorn pydantic streamlit guardrails-ai`
3.  To run the FastAPI server (in a separate terminal if you want to test API directly):
    `uvicorn medical_diagnostic_assistant:app --reload`
4.  To run the Streamlit UI:
    `streamlit run medical_diagnostic_assistant.py`

Note: For this single-file demonstration, the Streamlit app directly calls the `MockLLMService` methods. 
In a production environment, the Streamlit UI would typically make HTTP requests to the deployed FastAPI backend.
""")


# --- Evaluation & Monitoring (Placeholder) ---
# In a real application, you'd integrate with Langsmith/TruLens/Wandb here.
# For example, after a diagnosis, log the inputs, outputs, confidence, and retrieved documents.
# This could be done within the MockLLMService or the FastAPI endpoints.

def log_diagnosis_event(diagnosis_id: str, patient_data: PatientData, response: FullDiagnosisResponse):
    print(f"[Evaluation Log - {diagnosis_id}]")
    print(f"  Patient Symptoms: {patient_data.symptoms[:50]}...")
    print(f"  Diagnosis: {response.initial_suggestion.diagnosis}")
    print(f"  Confidence: {response.initial_suggestion.confidence_score}")
    print(f"  Reasoning Snippet: {response.initial_suggestion.short_reasoning[:50]}...")
    print("----------------------------------")

# Example of integrating the logging (can be called within FastAPI/Streamlit logic)
# if 'current_diagnosis' in st.session_state and st.session_state.current_diagnosis:
#    log_diagnosis_event(st.session_state.current_diagnosis.id, patient_data_used_for_diagnosis, st.session_state.current_diagnosis)
