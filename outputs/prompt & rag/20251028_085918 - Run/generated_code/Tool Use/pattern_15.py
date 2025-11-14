
import os
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# --- 1. Pydantic Models for Tool Inputs ---

class MedicalKnowledgeBaseInput(BaseModel):
    query: str = Field(description="The medical question or topic to search for.")
    patient_context: Optional[str] = Field(None, description="Optional patient-specific context for the query.")

class EHRIntegrationInput(BaseModel):
    patient_id: str = Field(description="The unique identifier for the patient.")
    data_type: str = Field(description="Type of data to retrieve (e.g., 'medical_history', 'lab_results', 'medications', 'allergies').")

class MedicalImageAnalysisInput(BaseModel):
    image_url: str = Field(description="URL or path to the medical image for analysis.")
    analysis_type: str = Field(description="Type of analysis to perform (e.g., 'X-ray_chest', 'MRI_brain', 'CT_abdomen').")

class DrugInteractionInput(BaseModel):
    current_medications: List[str] = Field(description="List of current medications the patient is taking.")
    allergies: Optional[List[str]] = Field(None, description="Optional list of patient allergies.")

class DifferentialDiagnosisInput(BaseModel):
    symptoms: List[str] = Field(description="List of patient symptoms.")
    medical_history_summary: Optional[str] = Field(None, description="Summary of relevant patient medical history.")
    lab_results_summary: Optional[str] = Field(None, description="Summary of relevant lab results.")

class TreatmentPlanInput(BaseModel):
    diagnosis: str = Field(description="The confirmed or suspected diagnosis.")
    patient_id: str = Field(description="The unique identifier for the patient.")
    goals: Optional[str] = Field(None, description="Specific treatment goals or considerations.")

class MedicalCalculatorInput(BaseModel):
    calculation_type: str = Field(description="Type of medical calculation (e.g., 'BMI', 'GFR', 'drug_dosage').")
    parameters: dict = Field(description="Dictionary of parameters required for the calculation (e.g., {'weight_kg': 70, 'height_cm': 175} for BMI).")

# --- 2. Specialized Tool Functions ---

@tool("medical_knowledge_base", args_schema=MedicalKnowledgeBaseInput)
def medical_knowledge_base_tool(query: str, patient_context: Optional[str] = None) -> str:
    """Queries external medical databases for relevant information based on patient conditions, diseases, or drug queries."""
    print(f"\n[TOOL CALL] Querying medical knowledge base for: '{query}' with context: '{patient_context or 'None'}'")
    # Placeholder: In a real system, this would integrate with APIs like PubMed, UpToDate, or an internal RAG system.
    return f"Retrieved information for '{query}'. For example, clinical guidelines suggest standard treatment protocols. (Placeholder response)"

@tool("ehr_integration", args_schema=EHRIntegrationInput)
def ehr_integration_tool(patient_id: str, data_type: str) -> str:
    """Securely retrieves specific patient data from Electronic Health Records (EHR) systems."""
    print(f"\n[TOOL CALL] Retrieving '{data_type}' for patient ID: '{patient_id}' from EHR.")
    # Placeholder: In a real system, this would involve secure API calls to a compliant EHR system.
    if data_type == "medical_history":
        return f"Patient {patient_id} medical history: Hypertension (diagnosed 5 years ago), Type 2 Diabetes (controlled). (Placeholder response)"
    elif data_type == "lab_results":
        return f"Patient {patient_id} latest lab results: HbA1c 6.5%, Cholesterol LDL 120 mg/dL. (Placeholder response)"
    elif data_type == "medications":
        return f"Patient {patient_id} current medications: Lisinopril 10mg daily, Metformin 500mg twice daily. (Placeholder response)"
    elif data_type == "allergies":
        return f"Patient {patient_id} allergies: Penicillin. (Placeholder response)"
    return f"Could not retrieve {data_type} for patient {patient_id}. (Placeholder response)"

@tool("medical_image_analysis", args_schema=MedicalImageAnalysisInput)
def medical_image_analysis_tool(image_url: str, analysis_type: str) -> str:
    """Analyzes medical images (e.g., X-rays, MRIs, CT scans) to detect anomalies or provide preliminary reports."""
    print(f"\n[TOOL CALL] Analyzing medical image at '{image_url}' for type: '{analysis_type}'.")
    # Placeholder: Integrates with CV models (e.g., PyTorch, TensorFlow, Hugging Face Vision Transformers).
    if "X-ray_chest" in analysis_type.lower():
        return f"Analysis of {analysis_type} from {image_url}: Suggests mild cardiomegaly, no acute pulmonary infiltrate. (Placeholder response)"
    elif "mri_brain" in analysis_type.lower():
        return f"Analysis of {analysis_type} from {image_url}: Shows no significant abnormalities. (Placeholder response)"
    return f"Analysis for {analysis_type} from {image_url} completed with no specific findings. (Placeholder response)"

@tool("drug_interaction_checker", args_schema=DrugInteractionInput)
def drug_interaction_checker_tool(current_medications: List[str], allergies: Optional[List[str]] = None) -> str:
    """Checks for potential adverse drug-drug or drug-allergy interactions."""
    print(f"\n[TOOL CALL] Checking drug interactions for medications: {current_medications} with allergies: {allergies or 'None'}")
    # Placeholder: Integrates with a drug interaction API or database.
    if "Lisinopril" in current_medications and "Ibuprofen" in current_medications:
        return "Potential interaction: Lisinopril and Ibuprofen can reduce the antihypertensive effect of Lisinopril and increase risk of renal dysfunction. (Placeholder response)"
    if "Penicillin" in (allergies or []) and "Amoxicillin" in current_medications:
        return "Critical allergy interaction: Patient has penicillin allergy; Amoxicillin is a penicillin derivative. (Placeholder response)"
    return "No significant drug-drug or drug-allergy interactions detected. (Placeholder response)"

@tool("differential_diagnosis_generator", args_schema=DifferentialDiagnosisInput)
def differential_diagnosis_generator_tool(symptoms: List[str], medical_history_summary: Optional[str] = None, lab_results_summary: Optional[str] = None) -> str:
    """Suggests a list of possible diagnoses based on patient symptoms, medical history, and lab results."""
    print(f"\n[TOOL CALL] Generating differential diagnosis for symptoms: {symptoms}")
    # Placeholder: Could use rule-based systems, knowledge graphs, or fine-tuned medical LLMs.
    if "fever" in symptoms and "cough" in symptoms and "shortness of breath" in symptoms:
        return "Possible diagnoses: Pneumonia, Bronchitis, Influenza, COVID-19. (Placeholder response)"
    elif "headache" in symptoms and "neck stiffness" in symptoms:
        return "Possible diagnoses: Meningitis, Tension Headache, Migraine. (Placeholder response)"
    return "Based on provided information, a differential diagnosis could include general viral infection, fatigue. (Placeholder response)"

@tool("treatment_plan_suggestor", args_schema=TreatmentPlanInput)
def treatment_plan_suggestor_tool(diagnosis: str, patient_id: str, goals: Optional[str] = None) -> str:
    """Recommends personalized treatment plans, including medication, therapies, or lifestyle changes."""
    print(f"\n[TOOL CALL] Suggesting treatment plan for diagnosis: '{diagnosis}' for patient '{patient_id}' with goals: '{goals or 'None'}'")
    # Placeholder: Aligned with clinical guidelines and patient specifics.
    if "Pneumonia" in diagnosis:
        return f"Treatment plan for Patient {patient_id} with Pneumonia: Antibiotics (e.g., Azithromycin), supportive care (rest, fluids), monitor respiratory status. (Placeholder response)"
    elif "Hypertension" in diagnosis:
        return f"Treatment plan for Patient {patient_id} with Hypertension: Lifestyle modifications (diet, exercise), medication (e.g., Lisinopril), regular blood pressure monitoring. (Placeholder response)"
    return f"Suggesting general supportive care and further investigation for diagnosis: {diagnosis}. (Placeholder response)"

@tool("medical_calculator", args_schema=MedicalCalculatorInput)
def medical_calculator_tool(calculation_type: str, parameters: dict) -> str:
    """Performs common medical calculations (e.g., BMI, GFR, drug dosages based on weight)."""
    print(f"\n[TOOL CALL] Performing medical calculation: '{calculation_type}' with parameters: {parameters}")
    if calculation_type.lower() == "bmi":
        weight_kg = parameters.get("weight_kg")
        height_cm = parameters.get("height_cm")
        if weight_kg and height_cm:
            height_m = height_cm / 100
            bmi = weight_kg / (height_m ** 2)
            return f"Calculated BMI: {bmi:.2f}. (Placeholder response)"
        return "Error: Missing weight_kg or height_cm for BMI calculation. (Placeholder response)"
    elif calculation_type.lower() == "gfr":
        # Simplified GFR calculation for demonstration
        creatinine = parameters.get("creatinine")
        age = parameters.get("age")
        gender = parameters.get("gender") # 'male' or 'female'
        if creatinine and age and gender:
            # This is a highly simplified placeholder. Real GFR uses complex formulas (MDRD, CKD-EPI)
            gfr = 186 * (creatinine**-1.154) * (age**-0.203)
            if gender == 'female':
                gfr *= 0.742
            return f"Estimated GFR: {gfr:.2f} mL/min/1.73 m^2. (Placeholder response)"
        return "Error: Missing creatinine, age, or gender for GFR calculation. (Placeholder response)"
    return f"Unknown calculation type: {calculation_type}. (Placeholder response)"

# --- 3. Tool Registry/Manager ---
all_tools = [
    medical_knowledge_base_tool,
    ehr_integration_tool,
    medical_image_analysis_tool,
    drug_interaction_checker_tool,
    differential_diagnosis_generator_tool,
    treatment_plan_suggestor_tool,
    medical_calculator_tool,
]

# --- 4. LLM Orchestrator (Agent) ---

# Initialize the LLM
# Ensure OPENAI_API_KEY is set in your environment variables
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are MediAssist AI, an intelligent clinical decision support system. Your role is to assist healthcare professionals by orchestrating specialized medical tools to provide comprehensive insights and recommendations. Always use the available tools to answer questions and provide relevant information. Be precise and provide actionable insights. If you need more information, ask follow-up questions."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_tool_calling_agent(llm, all_tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True, handle_parsing_errors=True)

# --- 5. Main Application Logic ---

def run_mediassist_ai():
    print("Welcome to MediAssist AI - Your Clinical Decision Support System!")
    print("How can I assist you today? (Type 'exit' to quit)")
    
    chat_history = []

    while True:
        user_input = input("\nHealthcare Professional: ")
        if user_input.lower() == 'exit':
            print("Exiting MediAssist AI. Goodbye!")
            break
        
        try:
            # The agent_executor takes the 'input' and 'chat_history' variables defined in the prompt
            response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            print(f"MediAssist AI: {response['output']}")
            # Update chat history for context in subsequent turns (simple implementation)
            chat_history.append(("human", user_input))
            chat_history.append(("ai", response['output']))
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or rephrase your request.")

if __name__ == "__main__":
    run_mediassist_ai()
