from typing import List
import os

from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# --- 1. Pydantic Models for Tool Inputs ---
class MedicalKnowledgeBaseInput(BaseModel):
    query: str = Field(description="medical query about diseases, symptoms, or treatments")

class MedicalImagingAnalysisInput(BaseModel):
    image_id: str = Field(description="ID of the medical image to analyze")
    patient_id: str = Field(description="ID of the patient associated with the image")

class EhrSystemInput(BaseModel):
    patient_id: str = Field(description="ID of the patient to retrieve EHR for")

class DrugInteractionCheckerInput(BaseModel):
    drugs: List[str] = Field(description="list of drug names to check for interactions")

class ClinicalTrialDatabaseInput(BaseModel):
    condition: str = Field(description="medical condition for which to find clinical trials")
    patient_age: int = Field(description="age of the patient")

class MedicalLiteratureSearchInput(BaseModel):
    query: str = Field(description="medical literature search query")


# --- 2. Simulated Specialized Tools (Functions) ---
def medical_knowledge_base_api(query: str) -> str:
    if "chest pain" in query.lower() and "shortness of breath" in query.lower():
        return "Possible conditions for chest pain and shortness of breath include Myocardial Infarction, Pulmonary Embolism, Pericarditis, and Aortic Dissection. Further investigation is needed."
    return f"Information for '{query}': Placeholder response from medical knowledge base."

def medical_imaging_analysis_tool(image_id: str, patient_id: str) -> str:
    if "xray_001" == image_id and "John Doe" == patient_id:
        return "X-ray (ID: xray_001) for John Doe shows signs of mild cardiomegaly. Recommend further cardiac evaluation."
    return f"Analysis for image {image_id} of patient {patient_id}: Placeholder imaging findings."

def ehr_system_api(patient_id: str) -> str:
    if "John Doe" == patient_id:
        return "EHR for John Doe: Age 65, History: Hypertension, Hyperlipidemia. Current Medications: Lisinopril 10mg QD, Atorvastatin 20mg QD. Recent Labs: Troponin elevated (0.5 ng/mL)."
    return f"EHR for patient {patient_id}: Placeholder patient data."

def drug_interaction_checker_api(drugs: List[str]) -> str:
    if "lisinopril" in [d.lower() for d in drugs] and "atorvastatin" in [d.lower() for d in drugs]:
        return "Checking interactions for Lisinopril and Atorvastatin: Generally safe to co-administer. Monitor for muscle pain with statins."
    return f"Drug interaction check for {', '.join(drugs)}: No significant interactions found (placeholder)."

def clinical_trial_database_api(condition: str, patient_age: int) -> str:
    if "myocardial infarction" in condition.lower() and patient_age >= 60:
        return "Relevant clinical trials for Myocardial Infarction in elderly patients: \nTrial A (Phase III, new anti-platelet), \nTrial B (Phase II, stem cell therapy)."
    return f"Clinical trials for {condition} (age {patient_age}): Placeholder list of trials."

def medical_literature_search_tool(query: str) -> str:
    if "elevated troponin" in query.lower() and "diagnosis" in query.lower():
        return "Literature search for 'elevated troponin diagnosis': \n- 'High-sensitivity troponin in the diagnosis of myocardial infarction' (Journal of Cardiology) \n- 'Causes of elevated troponin in non-ischemic conditions' (Circulation)."
    return f"Medical literature search for '{query}': Placeholder research papers."


# --- 3. LangChain Tools ---
tools = [
    Tool(
        name="MedicalKnowledgeBase",
        func=medical_knowledge_base_api,
        description="Provides information on diseases, symptoms, and treatments. Use this when you need general medical facts.",
        args_schema=MedicalKnowledgeBaseInput,
    ),
    Tool(
        name="MedicalImagingAnalysis",
        func=medical_imaging_analysis_tool,
        description="Analyzes medical images (e.g., X-rays, MRIs) and returns findings. Requires image_id and patient_id.",
        args_schema=MedicalImagingAnalysisInput,
    ),
    Tool(
        name="EHRSystem",
        func=ehr_system_api,
        description="Retrieves patient-specific electronic health record data including history, medications, and lab results. Requires patient_id.",
        args_schema=EhrSystemInput,
    ),
    Tool(
        name="DrugInteractionChecker",
        func=drug_interaction_checker_api,
        description="Checks for potential adverse drug interactions between a list of medications.",
        args_schema=DrugInteractionCheckerInput,
    ),
    Tool(
        name="ClinicalTrialDatabase",
        func=clinical_trial_database_api,
        description="Searches for relevant clinical trials based on a medical condition and patient age.",
        args_schema=ClinicalTrialDatabaseInput,
    ),
    Tool(
        name="MedicalLiteratureSearch",
        func=medical_literature_search_tool,
        description="Searches scientific medical literature (e.g., PubMed) for research papers and articles.",
        args_schema=MedicalLiteratureSearchInput,
    ),
]


# --- 4. Initialize LLM and AgentExecutor ---
# Ensure you have your OpenAI API key set as an environment variable (OPENAI_API_KEY)
# For local testing, you might use a placeholder or mock LLM, or a local model server.
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")

# Define the agent prompt
prompt_template = """You are an AI-powered diagnostic assistant for healthcare professionals. \n
Your goal is to assist in diagnosing patients, recommending treatments, checking drug interactions, and finding relevant clinical trials based on the user's query and patient information. \n
Use the following tools to gather information and provide a comprehensive response. Prioritize patient safety and evidence-based medicine. \n
TOOLS:
{tools}

FORMAT INSTRUCTIONS:
{format_instructions}

USER'S PATIENT QUERY:
{input}

{agent_scratchpad}"""

prompt = PromptTemplate.from_template(prompt_template)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# --- 5. Main Execution Logic ---
def run_diagnostic_assistant(query: str):
    print(f"\n--- Running Diagnostic Assistant for Query ---\nQuery: {query}")
    try:
        result = agent_executor.invoke({"input": query})
        print("\n--- Final Diagnostic Assistant Response ---")
        print(result["output"])
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Set your OpenAI API key here if not set as an environment variable
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

    # Example Query 1: Comprehensive patient analysis
    query1 = "Patient John Doe, 65, presents with sudden chest pain, shortness of breath, and elevated troponin levels. History of hypertension. Analyze his EHR, suggest possible diagnoses, check for drug interactions with current meds (lisinopril, atorvastatin), and find relevant clinical trials."
    run_diagnostic_assistant(query1)

    print("\n" + "="*80 + "\n")

    # Example Query 2: Simple medical knowledge lookup
    query2 = "What are common symptoms of pneumonia?"
    run_diagnostic_assistant(query2)

    print("\n" + "="*80 + "\n")

    # Example Query 3: Specific drug interaction check
    query3 = "Check for interactions between ibuprofen and warfarin."
    run_diagnostic_assistant(query3)

    print("\n" + "="*80 + "\n")

    # Example Query 4: Imaging analysis and general condition info
    query4 = "Analyze X-ray xray_001 for patient John Doe. Also, tell me about cardiomegaly."
    run_diagnostic_assistant(query4)
