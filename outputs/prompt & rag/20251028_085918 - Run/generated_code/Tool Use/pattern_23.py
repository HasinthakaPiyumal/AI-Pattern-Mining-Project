from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --- 1. Pydantic Models for Tool Inputs/Outputs ---
class PatientIDInput(BaseModel):
    patient_id: str = Field(description="The unique identifier for the patient.")

class MedicalQueryInput(BaseModel):
    query: str = Field(description="The medical query or question.")
    source: str = Field(description="The source to query (e.g., 'pubmed', 'uptodate', 'all').", default="all")

class ImageIDInput(BaseModel):
    image_id: str = Field(description="The unique identifier for the medical image (e.g., X-ray, MRI).")

class DrugsInput(BaseModel):
    drugs: List[str] = Field(description="A list of drug names to check for interactions.")

class ConditionInput(BaseModel):
    condition: str = Field(description="The medical condition for which to retrieve guidelines.")

class CodeInput(BaseModel):
    code: str = Field(description="The Python code to execute for statistical analysis.")

# --- 2. Mock Tool Functions ---
def _get_patient_ehr_data(patient_id: str) -> Dict[str, Any]:
    if patient_id == "P001":
        return {
            "patient_id": "P001",
            "name": "Alice Smith",
            "age": 45,
            "gender": "Female",
            "symptoms": ["cough", "fever", "fatigue"],
            "medications": ["Lisinopril", "Metformin"],
            "allergies": ["Penicillin"],
            "lab_results": {"WBC": 12.5, "CRP": 15.2, "Glucose": 110},
            "history": "Hypertension, Type 2 Diabetes"
        }
    elif patient_id == "P002":
         return {
            "patient_id": "P002",
            "name": "Bob Johnson",
            "age": 60,
            "gender": "Male",
            "symptoms": ["chest pain", "shortness of breath"],
            "medications": ["Aspirin", "Atorvastatin"],
            "allergies": [],
            "lab_results": {"Troponin": 0.8, "CK-MB": 12.0},
            "history": "Coronary Artery Disease"
        }
    return {"error": f"Patient {patient_id} not found."}

def _query_medical_knowledge_base(query: str, source: str = "all") -> str:
    if "cough" in query.lower() and "fever" in query.lower():
        return f"Common causes for {query} include influenza, common cold, bronchitis, and pneumonia. Refer to latest CDC guidelines for differential diagnosis. (Source: {source})"
    elif "chest pain" in query.lower():
        return f"Chest pain can indicate various conditions including myocardial infarction, angina, GERD, or pleurisy. Immediate cardiac evaluation is crucial. (Source: {source})"
    return f"No specific information found for '{query}' in {source}."

def _analyze_medical_image(image_id: str) -> str:
    if image_id == "XRAY001":
        return "X-ray analysis for XRAY001: Evidence of bilateral lower lobe infiltrates, suggestive of pneumonia."
    elif image_id == "MRI002":
        return "MRI analysis for MRI002: Lesion detected in the left frontal lobe, consistent with glioblastoma. Further biopsy recommended."
    return f"No analysis available for image {image_id}."

def _check_drug_interactions(drugs: List[str]) -> str:
    drugs_lower = [d.lower() for d in drugs]
    if "lisinopril" in drugs_lower and "ibuprofen" in drugs_lower:
        return "WARNING: Lisinopril and Ibuprofen can increase the risk of kidney dysfunction and reduce antihypertensive effects. Monitor renal function closely."
    if "metformin" in drugs_lower and "contrast dye" in drugs_lower:
        return "WARNING: Metformin combined with iodine-containing contrast media can lead to lactic acidosis. Metformin should be withheld before and after contrast administration."
    return "No significant drug interactions found for the given medications."

def _get_clinical_guidelines(condition: str) -> str:
    if "pneumonia" in condition.lower():
        return "Clinical guidelines for Pneumonia: Recommend empirical antibiotic therapy based on local resistance patterns, consider severity scores (e.g., CURB-65), and ensure vaccination status is up-to-date."
    elif "diabetes" in condition.lower():
        return "Clinical guidelines for Type 2 Diabetes: Recommend lifestyle modifications, metformin as first-line, and consider GLP-1 receptor agonists or SGLT2 inhibitors for cardiovascular/renal benefits. Regular monitoring of HbA1c, blood pressure, and lipids."
    return f"No specific clinical guidelines found for '{condition}'."

def _execute_python_code(code: str) -> str:
    try:
        # WARNING: Using exec/eval directly is highly insecure in a production environment.
        # This is for demonstration only. A sandboxed environment is required for real apps.
        globals_dict = {}
        locals_dict = {}
        exec(code, globals_dict, locals_dict)
        return str(locals_dict.get('result', 'Code executed successfully, no explicit result variable.'))
    except Exception as e:
        return f"Error executing code: {e}"

# --- 3. LangChain Tools ---
get_patient_ehr_data = tool(args_schema=PatientIDInput)(_get_patient_ehr_data)
query_medical_knowledge_base = tool(args_schema=MedicalQueryInput)(_query_medical_knowledge_base)
analyze_medical_image = tool(args_schema=ImageIDInput)(_analyze_medical_image)
check_drug_interactions = tool(args_schema=DrugsInput)(_check_drug_interactions)
get_clinical_guidelines = tool(args_schema=ConditionInput)(_get_clinical_guidelines)
execute_python_code = tool(args_schema=CodeInput)(_execute_python_code)

tools = [
    get_patient_ehr_data,
    query_medical_knowledge_base,
    analyze_medical_image,
    check_drug_interactions,
    get_clinical_guidelines,
    execute_python_code
]

# --- 4. LLM Setup ---
# IMPORTANT: Replace with your actual API key or configure a local LLM.
# from dotenv import load_dotenv
# load_dotenv()
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- 5. Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent medical diagnostic assistant. Your goal is to help healthcare professionals by providing evidence-based insights using the available tools. Always prioritize patient safety and accuracy."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# --- 6. Agent Creation ---
agent = create_tool_calling_agent(llm, tools, prompt)

# --- 7. Agent Executor ---
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 8. CLI Loop ---
if __name__ == "__main__":
    print("\nIntelligent Medical Diagnostic Assistant (CLI)\n")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        user_query = input("\nDoctor's Query: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting diagnostic assistant. Goodbye!")
            break

        try:
            response = agent_executor.invoke({"input": user_query})
            print("\nAssistant's Response:")
            print(response["output"])
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again or refine your query.")
