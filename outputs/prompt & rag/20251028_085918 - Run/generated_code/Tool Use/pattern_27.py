import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import StructuredTool


# --- Pydantic Models for Data Validation ---
class PatientInput(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient")
    symptoms: str = Field(..., description="Comma-separated list of patient symptoms")
    medical_history: str = Field(
        default="", description="Relevant medical history of the patient"
    )
    lab_results: str = Field(
        default="", description="Relevant lab test results (e.g., 'high glucose, normal WBC')"
    )
    imaging_report_summary: str = Field(
        default="", description="Summary of diagnostic imaging reports"
    )
    current_medications: str = Field(
        default="", description="Comma-separated list of current medications"
    )
    allergies: str = Field(default="", description="Comma-separated list of patient allergies")


class DiagnosisOutput(BaseModel):
    diagnosis: str = Field(..., description="The likely medical diagnosis")
    confidence: float = Field(
        ..., description="Confidence score (0.0-1.0) for the diagnosis"
    )
    justification: str = Field(
        ..., description="Reasoning and evidence supporting the diagnosis"
    )


class TreatmentOutput(BaseModel):
    recommendations: List[str] = Field(
        ..., description="List of recommended treatment steps"
    )
    justification: str = Field(
        ..., description="Explanation for the recommended treatments"
    )
    follow_up_instructions: str = Field(
        default="", description="Instructions for patient follow-up"
    )


# --- Mock External Tools and Services ---

class MockEHRConnector:
    def get_patient_history(self, patient_id: str) -> dict:
        if patient_id == "P001":
            return {
                "id": "P001",
                "name": "Alice Smith",
                "age": 45,
                "conditions": ["Hypertension", "Type 2 Diabetes"],
                "medications": ["Metformin", "Lisinopril"],
                "allergies": ["Penicillin"],
                "last_visit": "2023-10-26",
            }
        return {"id": patient_id, "name": "Unknown", "history": "No record found."}


class MockMedicalKnowledgeBase:
    def search_disease_info(self, query: str) -> str:
        if "fever" in query.lower() and "cough" in query.lower():
            return "Common cold: Viral infection causing runny nose, sore throat, cough, and fever. Treatment is symptomatic."
        if "chest pain" in query.lower() and "shortness of breath" in query.lower():
            return "Angina/Heart Attack: Severe chest pain, shortness of breath, radiating pain. Requires immediate medical attention."
        return "No specific information found for the query."

    def search_guidelines(self, disease: str) -> str:
        if "hypertension" in disease.lower():
            return "Hypertension guidelines: Lifestyle changes, medication (e.g., ACE inhibitors, diuretics). Regular monitoring required."
        return "No specific guidelines found for the disease."


class MockImagingAnalysisService:
    def analyze_image_report(self, report_summary: str) -> str:
        if "nodule in lung" in report_summary.lower():
            return "Potential lung nodule detected. Recommend further investigation (e.g., CT scan with contrast) to rule out malignancy."
        if "clear chest x-ray" in report_summary.lower():
            return "Chest X-ray appears normal, no acute cardiopulmonary findings."
        return "Imaging analysis inconclusive or no significant findings."


class MockDrugInteractionDatabase:
    def check_interactions(self, drugs: List[str]) -> str:
        if "Metformin" in drugs and "Lisinopril" in drugs:
            return "No significant interactions between Metformin and Lisinopril for most patients. Monitor kidney function."
        if "Aspirin" in drugs and "Warfarin" in drugs:
            return "Major interaction: Increased risk of bleeding. Concurrent use generally contraindicated. Consult physician."
        return "No critical drug interactions found for the given list."

    def check_allergies(self, drugs: List[str], allergies: List[str]) -> str:
        if "Penicillin" in drugs and "Penicillin" in allergies:
            return "Critical allergy interaction: Penicillin should not be administered to patients with Penicillin allergy."
        return "No immediate allergy concerns found."


# --- Initialize Mock Services ---
ehr_connector = MockEHRConnector()
medical_kb = MockMedicalKnowledgeBase()
imaging_service = MockImagingAnalysisService()
drug_db = MockDrugInteractionDatabase()


# --- LangChain Tools ---

# EHR Tool
def _get_patient_history(patient_id: str) -> str:
    history = ehr_connector.get_patient_history(patient_id)
    return str(history)


ehr_tool = StructuredTool.from_function(
    func=_get_patient_history,
    name="get_patient_history",
    description="Retrieve comprehensive medical history for a given patient ID from the EHR system.",
    args_schema=BaseModel,
    return_direct=False
)


# Medical Knowledge Base Tools
def _search_disease_info(query: str) -> str:
    return medical_kb.search_disease_info(query)


disease_info_tool = StructuredTool.from_function(
    func=_search_disease_info,
    name="search_disease_info",
    description="Search the medical knowledge base for information about diseases or symptoms.",
    args_schema=BaseModel,
    return_direct=False
)


def _search_guidelines(disease: str) -> str:
    return medical_kb.search_guidelines(disease)


guidelines_tool = StructuredTool.from_function(
    func=_search_guidelines,
    name="search_guidelines",
    description="Search for treatment guidelines for a specific disease from the medical knowledge base.",
    args_schema=BaseModel,
    return_direct=False
)


# Imaging Analysis Tool
def _analyze_imaging_report(report_summary: str) -> str:
    return imaging_service.analyze_image_report(report_summary)


imaging_analysis_tool = StructuredTool.from_function(
    func=_analyze_imaging_report,
    name="analyze_imaging_report",
    description="Analyze a diagnostic imaging report summary to identify potential findings or recommendations.",
    args_schema=BaseModel,
    return_direct=False
)


# Drug Interaction Tool
def _check_drug_interactions(drugs: List[str]) -> str:
    return drug_db.check_interactions(drugs)


drug_interaction_tool = StructuredTool.from_function(
    func=_check_drug_interactions,
    name="check_drug_interactions",
    description="Check for potential drug-drug interactions among a list of medications.",
    args_schema=BaseModel,
    return_direct=False
)


def _check_drug_allergies(drugs: List[str], allergies: List[str]) -> str:
    return drug_db.check_allergies(drugs, allergies)


drug_allergy_tool = StructuredTool.from_function(
    func=_check_drug_allergies,
    name="check_drug_allergies",
    description="Check for potential drug-allergy interactions.",
    args_schema=BaseModel,
    return_direct=False
)


all_tools = [
    ehr_tool,
    disease_info_tool,
    guidelines_tool,
    imaging_analysis_tool,
    drug_interaction_tool,
    drug_allergy_tool,
]


# --- LangChain Agent Setup ---

# Placeholder for a real LLM
# In a real application, you'd configure a proper OpenAI or other LLM instance.
# For local testing without an API key, you could use a mocked LLM or a local model.
llm = OpenAI(temperature=0, openai_api_key="sk-YOUR_OPENAI_API_KEY_HERE") # Replace with your actual key or use a local LLM


agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a highly skilled medical AI assistant. Your goal is to assist in diagnosis and treatment recommendations based on patient data using available tools.\n" \
         "Always use the tools provided to gather all necessary information before making a final recommendation.\n" \
         "If you need patient history, use the 'get_patient_history' tool with the patient_id.\n" \
         "If you need disease information or guidelines, use 'search_disease_info' and 'search_guidelines'.\n" \
         "If imaging reports are provided, use 'analyze_imaging_report'.\n" \
         "If current medications and allergies are provided, use 'check_drug_interactions' and 'check_drug_allergies'.\n" \
         "Provide a clear diagnosis, its confidence, justification, and actionable treatment recommendations.\n" \
         "Always specify drug dosages if you recommend medication (e.g., 'Drug X, 5mg once daily').\n" \
         "If a patient has an allergy, ensure your recommendations avoid that allergen."
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, all_tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI Medical Assistant")
st.title("🩺 AI-Powered Medical Diagnosis and Treatment Recommendation")

st.markdown(
    "This AI assistant uses specialized tools to help diagnose conditions and recommend treatments based on patient data."
)

with st.sidebar:
    st.header("Patient Information")
    patient_id = st.text_input("Patient ID", value="P001")
    symptoms = st.text_area(
        "Symptoms (comma-separated)",
        value="Fever, cough, sore throat, body aches",
    )
    medical_history = st.text_area(
        "Medical History",
        value="Has a history of seasonal allergies. No major chronic diseases besides what's in EHR.",
    )
    lab_results = st.text_area(
        "Lab Results (e.g., 'high glucose, normal WBC')",
        value="White blood cell count normal, CRP slightly elevated",
    )
    imaging_report_summary = st.text_area(
        "Imaging Report Summary (e.g., 'Clear chest X-ray')",
        value="Clear chest X-ray, no signs of pneumonia",
    )
    current_medications = st.text_area(
        "Current Medications (comma-separated)",
        value="Paracetamol, Vitamin C",
    )
    allergies = st.text_area(
        "Allergies (comma-separated)", value="No known drug allergies"
    )

    process_button = st.button("Get Diagnosis and Treatment")

st.header("AI Assistant's Findings")

if process_button:
    try:
        # Validate input with Pydantic
        patient_data = PatientInput(
            patient_id=patient_id,
            symptoms=symptoms,
            medical_history=medical_history,
            lab_results=lab_results,
            imaging_report_summary=imaging_report_summary,
            current_medications=current_medications,
            allergies=allergies,
        )

        st.write("### Input Data:")
        st.json(patient_data.dict())
        st.markdown("--- interstate ---")

        # Construct the prompt for the agent
        agent_input = (
            f"Patient ID: {patient_data.patient_id}. "
            f"Symptoms: {patient_data.symptoms}. "
            f"Medical History: {patient_data.medical_history}. "
            f"Lab Results: {patient_data.lab_results}. "
            f"Imaging Report Summary: {patient_data.imaging_report_summary}. "
            f"Current Medications: {patient_data.current_medications}. "
            f"Allergies: {patient_data.allergies}. "
            "Please provide a comprehensive medical diagnosis and treatment recommendations."
        )

        with st.spinner("AI is analyzing patient data and consulting medical tools..."):
            response = agent_executor.invoke({"input": agent_input})

        st.write("### AI Agent Response:")
        st.write(response["output"])

        # Attempt to parse into Pydantic models (can be improved with Pydantic output parsers)
        # For simplicity, we'll just display the raw text output for now,
        # as the agent's output format might vary if not strictly constrained.
        # A more robust solution would involve a PydanticOutputParser for the final step.
        # Example: diagnosis_result = DiagnosisOutput.parse_raw(response["output"])

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning(
            "Please ensure all required fields are filled and the OpenAI API key is valid."
        )

st.markdown("--- interstate ---")
st.info(
    "Disclaimer: This is an AI-powered assistant for informational purposes only and "
    "should not be used as a substitute for professional medical advice, diagnosis, or treatment."
)
