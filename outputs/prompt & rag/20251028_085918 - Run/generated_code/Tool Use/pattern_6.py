"""AI-Powered Clinical Decision Support System (CDSS) with Dynamic Tool Orchestration and Personalized Learning in Healthcare.

This application demonstrates a modular architecture leveraging FastAPI for API exposure and LangChain for LLM orchestration and tool integration. It includes mocked healthcare tools, a basic personalized learning module, and a placeholder for hallucination detection.

To run this application:
1.  Install necessary libraries: `pip install fastapi uvicorn langchain langchain-openai pydantic`
2.  Set your OpenAI API key as an environment variable: `export OPENAI_API_KEY="your_api_key_here"` (or use a different LLM provider and configure accordingly).
3.  Run the FastAPI application: `uvicorn cdss_app:app --reload`
4.  Access the API documentation at `http://127.0.0.1:8000/docs`
"""

import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# --- 1. Pydantic Models for FastAPI Request/Response --- 

class PatientInfo(BaseModel):
    patient_id: str = Field(..., example="P001")
    age: int = Field(..., example=55)
    gender: str = Field(..., example="Male")
    medical_history: List[str] = Field(default_factory=list, example=["Hypertension", "Type 2 Diabetes"])
    current_medications: List[str] = Field(default_factory=list, example=["Lisinopril", "Metformin"])
    allergies: List[str] = Field(default_factory=list, example=["Penicillin"])

class DiagnosisRequest(BaseModel):
    patient_info: PatientInfo
    symptoms: List[str] = Field(..., example=["shortness of breath", "chest pain"])
    additional_context: Optional[str] = Field(None, example="Patient reported symptoms started yesterday.")

class PrescriptionRequest(BaseModel):
    patient_id: str = Field(..., example="P001")
    medication_name: str = Field(..., example="Amoxicillin")
    dosage: str = Field(..., example="250mg three times a day")
    duration: str = Field(..., example="7 days")
    reason: str = Field(..., example="Bacterial infection")

class MedicalQueryRequest(BaseModel):
    query: str = Field(..., example="What are the latest guidelines for managing atrial fibrillation?")
    patient_id: Optional[str] = Field(None, example="P001")

class CDSSResponse(BaseModel):
    recommendation: str
    diagnosis_confidence: Optional[float] = Field(None, description="Confidence score for the diagnosis, if applicable.")
    tools_used: List[str] = Field(default_factory=list, description="List of tools invoked by the LLM agent.")
    warnings: List[str] = Field(default_factory=list, description="Any warnings or disclaimers.")

# --- 2. Mock Specialized Healthcare Tools --- 

@tool
def get_patient_history(patient_id: str) -> Dict[str, Any]:
    """Fetches comprehensive medical history for a given patient ID from the EHR system.
    Input should be a string representing the patient_id.
    """
    print(f"[TOOL] Fetching patient history for: {patient_id}")
    mock_db = {
        "P001": {
            "age": 55,
            "gender": "Male",
            "medical_history": ["Hypertension", "Type 2 Diabetes", "Previous MI (5 years ago)"],
            "current_medications": ["Lisinopril", "Metformin", "Aspirin"],
            "allergies": ["Penicillin"],
            "lab_results": {"cholesterol": "high", "glucose": "controlled"}
        },
        "P002": {
            "age": 32,
            "gender": "Female",
            "medical_history": ["Asthma"],
            "current_medications": ["Albuterol"],
            "allergies": ["Sulfa drugs"],
            "lab_results": {}
        }
    }
    history = mock_db.get(patient_id, {"error": "Patient not found"})
    return {"patient_id": patient_id, "history": history}

@tool
def search_medical_knowledge_base(query: str) -> str:
    """Searches a comprehensive medical knowledge base for guidelines, drug information, disease etiologies, or treatment protocols.
    Input should be a string representing the medical query.
    """
    print(f"[TOOL] Searching medical knowledge base for: {query}")
    if "hypertension guidelines" in query.lower():
        return "Latest hypertension guidelines recommend lifestyle modifications, ACE inhibitors/ARBs, calcium channel blockers, and thiazide diuretics as first-line options. Target BP < 130/80 mmHg for most adults."
    elif "drug interaction lisinopril metformin" in query.lower():
        return "Lisinopril and Metformin generally have no significant direct drug-drug interactions. Monitor renal function due to both affecting kidneys."
    elif "covid-19 treatment" in query.lower():
        return "Current guidelines for COVID-19 treatment include supportive care, antivirals like Paxlovid for eligible patients, and corticosteroids for severe cases."
    return f"No specific knowledge found for '{query}'. Please refine your query."

@tool
def analyze_diagnostic_imaging(image_id: str, patient_id: str) -> str:
    """Simulates analysis of a diagnostic imaging scan (e.g., X-ray, MRI, CT) for abnormalities.
    Input should be a comma-separated string of image_id and patient_id.
    Example: 'XRAY-12345,P001'
    """
    print(f"[TOOL] Analyzing imaging for patient {patient_id}, image: {image_id}")
    if "P001" in patient_id and "XRAY-12345" in image_id:
        return "Image analysis for XRAY-12345 (P001): Findings suggestive of mild cardiomegaly, no acute pulmonary infiltrates. Consistent with known hypertension."
    elif "P002" in patient_id and "CT-67890" in image_id:
        return "Image analysis for CT-67890 (P002): Normal brain CT, no acute intracranial pathology."
    return f"No specific imaging analysis result for image {image_id} and patient {patient_id}."

@tool
def verify_and_generate_prescription(medication_name: str, patient_id: str) -> str:
    """Verifies medication against patient allergies and current meds, then simulates prescription generation.
    Input should be a comma-separated string of medication_name and patient_id.
    Example: 'Amoxicillin,P001'
    """
    print(f"[TOOL] Verifying and generating prescription for {medication_name} for patient {patient_id}")
    patient_data = get_patient_history.invoke(patient_id)['history']
    if patient_data.get("error"): # Handle patient not found from mock
        return f"Error: {patient_data['error']} for prescription verification."

    allergies = [a.lower() for a in patient_data.get("allergies", [])]
    current_meds = [m.lower() for m in patient_data.get("current_medications", [])]

    if medication_name.lower() in allergies:
        return f"WARNING: Patient {patient_id} has an allergy to {medication_name}. Prescription NOT generated."
    if medication_name.lower() in current_meds:
        return f"WARNING: Patient {patient_id} is already on {medication_name}. Consider dosage adjustment or alternative. Prescription still generated for demonstration."
    
    # Simulate drug interaction check (simplified)
    if medication_name.lower() == "warfarin" and "aspirin" in current_meds:
         return f"WARNING: Potential interaction between Warfarin and Aspirin for patient {patient_id}. Increased bleeding risk. Prescription generated but advise caution."

    return f"Prescription for {medication_name} successfully generated for patient {patient_id}."

@tool
def get_differential_diagnosis(symptoms: str, patient_history_summary: str) -> str:
    """Provides a differential diagnosis based on a list of symptoms and a summary of patient history.
    Input should be a comma-separated string of symptoms and patient_history_summary.
    Example: 'fever,cough,fatigue,Patient is 35 year old male with no significant history.'
    """
    print(f"[TOOL] Getting differential diagnosis for symptoms: {symptoms}, history: {patient_history_summary}")
    if "chest pain" in symptoms.lower() and "shortness of breath" in symptoms.lower():
        if "previous mi" in patient_history_summary.lower() or "hypertension" in patient_history_summary.lower():
            return "Differential Diagnosis: Acute Coronary Syndrome, Angina, Pericarditis, Pulmonary Embolism. Given history, highly consider ACS."
        return "Differential Diagnosis: Angina, Pericarditis, Anxiety, Musculoskeletal pain, GERD, Pulmonary Embolism."
    elif "fever" in symptoms.lower() and "cough" in symptoms.lower():
        return "Differential Diagnosis: Viral URI, Influenza, Bronchitis, Pneumonia, COVID-19."
    return f"Differential Diagnosis for {symptoms}: Consider various respiratory or systemic conditions."

@tool
def search_and_summarize_medical_articles(query: str) -> str:
    """Searches real-time medical literature databases (e.g., PubMed) and summarizes relevant articles.
    Input should be a string representing the medical query.
    """
    print(f"[TOOL] Searching and summarizing medical articles for: {query}")
    if "new treatments type 2 diabetes" in query.lower():
        return "Summary of recent articles on Type 2 Diabetes: Recent studies highlight GLP-1 receptor agonists and SGLT2 inhibitors for cardiovascular and renal benefits beyond glycemic control. Emerging therapies include dual GIP/GLP-1 agonists."
    elif "atrial fibrillation management" in query.lower():
        return "Summary of recent articles on Atrial Fibrillation: Guidelines emphasize rhythm control vs. rate control strategies, newer oral anticoagulants (NOACs) over Warfarin for stroke prevention, and personalized risk assessment for intervention."
    return f"No recent articles directly matching '{query}' found or summarized."


# --- 3. Personalized Learning Module --- 

class PersonalizedLearning:
    def __init__(self):
        # Mock preferences. In a real system, this would be stored in a database.
        self.preferences = {
            "DrSmith": {
                "preferred_drug_class_hypertension": "ACE inhibitors",
                "avoid_drug_class": "NSAIDs for elderly patients",
                "diagnosis_verbosity": "concise"
            },
            "DrJones": {
                "preferred_drug_class_diabetes": "SGLT2 inhibitors",
                "diagnosis_verbosity": "detailed"
            }
        }

    def get_preferences(self, physician_id: Optional[str] = None) -> Dict[str, str]:
        """Retrieves personalized preferences for a given physician ID."""
        if physician_id and physician_id in self.preferences:
            print(f"[Personalized Learning] Retrieved preferences for {physician_id}")
            return self.preferences[physician_id]
        print("[Personalized Learning] No specific preferences found or physician_id not provided. Using default.")
        return {}

    def apply_preferences(self, recommendation: str, physician_id: Optional[str] = None) -> str:
        """Applies personalized preferences to a generated recommendation.
        This is a simplistic example, real application would involve more sophisticated prompt engineering or output filtering.
        """
        prefs = self.get_preferences(physician_id)
        if prefs.get("diagnosis_verbosity") == "concise":
            return f"[Concise Recommendation]: {recommendation.split('.')[0]}."
        return recommendation # Return as is if no specific verbosity preference

personalized_learning = PersonalizedLearning()

# --- 4. Hallucination Detection Module --- 

def detect_hallucination(text: str) -> List[str]:
    """Performs basic hallucination detection by checking for unverified claims or keywords.
    In a real system, this would involve cross-referencing with trusted knowledge bases, factual checks, or LLM-based verification.
    """
    warnings = []
    if "unverified claim" in text.lower() or "speculative diagnosis" in text.lower():
        warnings.append("Potential hallucination or unverified claim detected. Please cross-reference with medical literature.")
    if "no evidence" in text.lower():
         warnings.append("Recommendation lacks explicit evidence. Further validation recommended.")
    # Example: Check for nonsensical drug combinations based on a very simple rule
    if "ibuprofen" in text.lower() and "warfarin" in text.lower() and "bleeding risk" not in text.lower():
        warnings.append("Warning: Recommendation involving Ibuprofen and Warfarin did not explicitly mention increased bleeding risk. Review carefully.")
    return warnings

# --- 5. LLM Orchestrator (LangChain Agent) --- 

# Initialize LLM (Ensure OPENAI_API_KEY is set in your environment)
# For local development or other LLMs, you would configure it here.
# Example for a local LLM: from langchain_community.llms import Ollama; llm = Ollama(model="llama2")

try:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
except Exception as e:
    print(f"Warning: Could not initialize OpenAI LLM. Ensure OPENAI_API_KEY is set. Error: {e}")
    print("Using a mock LLM for demonstration purposes. Agent functionality will be limited.")
    class MockLLM(BaseModel):
        def invoke(self, prompt: str, **kwargs) -> Any:
            # Simple mock response based on tool calls in prompt
            if "get_patient_history" in prompt:
                return "Action: get_patient_history\nAction Input: P001"
            elif "search_medical_knowledge_base" in prompt:
                return "Action: search_medical_knowledge_base\nAction Input: hypertension guidelines"
            elif "get_differential_diagnosis" in prompt:
                return "Action: get_differential_diagnosis\nAction Input: fever,cough,Patient has a cold"
            elif "verify_and_generate_prescription" in prompt:
                return "Action: verify_and_generate_prescription\nAction Input: Amoxicillin,P001"
            return "Mock LLM response: Please provide a clear medical query for diagnosis or information."
        def with_structured_output(self, schema):
            # This is a simplification for a mock LLM to avoid complex schema handling
            return self
    llm = MockLLM()


# List of all tools available to the agent
available_tools = [
    get_patient_history,
    search_medical_knowledge_base,
    analyze_diagnostic_imaging,
    verify_and_generate_prescription,
    get_differential_diagnosis,
    search_and_summarize_medical_articles,
]

# Define the agent's prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly skilled AI-powered Clinical Decision Support System. Your goal is to provide accurate and evidence-based medical recommendations, diagnoses, and treatment plans. Utilize the available tools to gather information and process queries. Always prioritize patient safety and warn about potential drug interactions or allergies."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create the LangChain agent
agent = create_tool_calling_agent(llm, available_tools, prompt)

# Create the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=available_tools, verbose=True, handle_parsing_errors=True)

# --- 6. FastAPI Application --- 

app = FastAPI(
    title="AI-Powered Clinical Decision Support System",
    description="A system enhancing medical diagnosis, treatment planning, and patient management using LLM orchestration and specialized tools.",
    version="1.0.0"
)

@app.post("/diagnose", response_model=CDSSResponse)
async def diagnose_patient(request: DiagnosisRequest, physician_id: Optional[str] = None):
    """Provides a diagnosis and treatment recommendation based on patient information and symptoms."""
    try:
        # Construct a detailed query for the LLM agent
        patient_summary = f"Patient ID: {request.patient_info.patient_id}, Age: {request.patient_info.age}, Gender: {request.patient_info.gender}. " \
                          f"Medical History: {', '.join(request.patient_info.medical_history)}. " \
                          f"Current Medications: {', '.join(request.patient_info.current_medications)}. " \
                          f"Allergies: {', '.join(request.patient_info.allergies)}. "
        
        query = f"Diagnose and recommend treatment for a patient with the following symptoms: {', '.join(request.symptoms)}. " \
                f"Patient Information: {patient_summary}. " \
                f"Additional Context: {request.additional_context or 'None'}. " \
                f"Please use available tools to gather all necessary information."
        
        # Invoke the LangChain agent
        raw_agent_output = agent_executor.invoke({"input": query})
        recommendation = raw_agent_output.get("output", "Could not generate a specific recommendation.")
        
        # Apply personalized learning
        final_recommendation = personalized_learning.apply_preferences(recommendation, physician_id)

        # Perform hallucination detection
        warnings = detect_hallucination(final_recommendation)

        return CDSSResponse(
            recommendation=final_recommendation,
            diagnosis_confidence=None, # Placeholder for actual confidence score from LLM/tools
            tools_used=[tool.name for tool in available_tools], # Simplified: list all tools, ideally only used ones
            warnings=warnings
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error during diagnosis: {str(e)}")

@app.post("/prescribe", response_model=CDSSResponse)
async def prescribe_medication(request: PrescriptionRequest, physician_id: Optional[str] = None):
    """Verifies and generates a prescription for a given medication and patient."""
    try:
        query = f"Verify and generate a prescription for {request.medication_name} (Dosage: {request.dosage}, Duration: {request.duration}, Reason: {request.reason}) for Patient ID: {request.patient_id}. Check for allergies and interactions."
        
        raw_agent_output = agent_executor.invoke({"input": query})
        recommendation = raw_agent_output.get("output", "Could not generate prescription.")
        
        final_recommendation = personalized_learning.apply_preferences(recommendation, physician_id)
        warnings = detect_hallucination(final_recommendation)

        return CDSSResponse(
            recommendation=final_recommendation,
            tools_used=[tool.name for tool in available_tools], 
            warnings=warnings
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error during prescription: {str(e)}")

@app.post("/medical_query", response_model=CDSSResponse)
async def general_medical_query(request: MedicalQueryRequest, physician_id: Optional[str] = None):
    """Handles general medical inquiries using the LLM and available tools."""
    try:
        query_text = f"Medical query: {request.query}"
        if request.patient_id:
            query_text += f" (Context for Patient ID: {request.patient_id})"
        
        raw_agent_output = agent_executor.invoke({"input": query_text})
        recommendation = raw_agent_output.get("output", "Could not find information for your query.")

        final_recommendation = personalized_learning.apply_preferences(recommendation, physician_id)
        warnings = detect_hallucination(final_recommendation)

        return CDSSResponse(
            recommendation=final_recommendation,
            tools_used=[tool.name for tool in available_tools], 
            warnings=warnings
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error during medical query: {str(e)}")

