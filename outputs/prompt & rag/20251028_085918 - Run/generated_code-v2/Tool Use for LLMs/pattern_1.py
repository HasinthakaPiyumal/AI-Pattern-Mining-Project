import os
from langchain.agents import AgentExecutor, tool, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# --- 2. Specialized Tools (APIs/Modules) ---

# Mock Medical Knowledge Base
medical_knowledge_base = {
    "fever": "A temporary increase in average body temperature, often due to an illness.",
    "headache": "Pain in the head or face, which can be throbbing, constant, sharp or dull.",
    "diabetes": "A chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces.",
    "hypertension": "A condition in which the force of the blood against the artery walls is too high.",
    "pneumonia": "An infection that inflames air sacs in one or both lungs, which may fill with fluid or pus."
}

@tool
def medical_kb_tool(query: str) -> str:
    """Searches a medical knowledge base for information on diseases, symptoms, or treatments. Input should be a string query."""
    query = query.lower()
    if query in medical_knowledge_base:
        return f"Information about {query}: {medical_knowledge_base[query]}"
    return f"No specific information found for '{query}' in the medical knowledge base."

# Mock Diagnostic Imaging Analysis Tool
@tool
def imaging_analysis_tool(image_type: str, patient_id: str = "") -> str:
    """Simulates analysis of a medical image (e.g., X-ray, MRI). Input 'image_type' (e.g., 'X-ray', 'MRI') and optionally 'patient_id'."""
    image_type = image_type.lower()
    if "x-ray" in image_type:
        return f"Simulated X-ray analysis for patient {patient_id}: Possible signs of pneumonia in the left lung. Further investigation recommended."
    elif "mri" in image_type:
        return f"Simulated MRI analysis for patient {patient_id}: No significant abnormalities detected in brain scan."
    return f"Cannot perform simulated analysis for image type '{image_type}'."

# Mock Lab Test Interpretation Tool
@tool
def lab_test_interpretation_tool(test_results: str, patient_id: str = "") -> str:
    """Interprets lab test results. Input 'test_results' as a summary string (e.g., 'high blood sugar', 'normal CBC') and optionally 'patient_id'."""
    test_results = test_results.lower()
    if "high blood sugar" in test_results or "elevated glucose" in test_results:
        return f"Simulated lab test interpretation for patient {patient_id}: Elevated glucose levels suggest potential pre-diabetes or diabetes. Recommend fasting blood sugar test and HBA1c."
    elif "normal cbc" in test_results:
        return f"Simulated lab test interpretation for patient {patient_id}: Complete Blood Count (CBC) results are within normal limits."
    elif "low hemoglobin" in test_results:
        return f"Simulated lab test interpretation for patient {patient_id}: Low hemoglobin levels indicate anemia. Further investigation into cause is recommended."
    return f"Cannot provide specific interpretation for '{test_results}'."

# Mock Drug Interaction Checker API
@tool
def drug_interaction_checker_tool(drugs: str) -> str:
    """Checks for potential drug-drug interactions. Input 'drugs' as a comma-separated string of drug names (e.g., 'ibuprofen, warfarin')."""
    drugs_list = [d.strip().lower() for d in drugs.split(',')]
    if "warfarin" in drugs_list and "ibuprofen" in drugs_list:
        return "Potential significant interaction: Ibuprofen can increase the anticoagulant effect of warfarin, leading to increased bleeding risk. AVOID concomitant use."
    elif "paracetamol" in drugs_list and "alcohol" in drugs_list:
        return "Moderate interaction: Excessive alcohol consumption with paracetamol can increase the risk of liver damage."
    elif len(drugs_list) > 1:
        return f"No significant interactions found between {', '.join(drugs_list)} in simulated data."
    return f"Please provide at least two drugs to check for interactions."

# Mock Electronic Health Record (EHR) System API
ehr_records = {
    "patient_001": {
        "name": "Alice Smith",
        "age": 45,
        "conditions": ["Hypertension", "Type 2 Diabetes"],
        "medications": ["Metformin", "Lisinopril"],
        "allergies": ["Penicillin"],
        "last_visit": "2023-10-26"
    },
    "patient_002": {
        "name": "Bob Johnson",
        "age": 60,
        "conditions": ["Coronary Artery Disease"],
        "medications": ["Aspirin", "Atorvastatin"],
        "allergies": [],
        "last_visit": "2023-11-15"
    }
}

@tool
def ehr_system_tool(patient_id: str) -> str:
    """Retrieves electronic health records for a given patient ID. Input should be a string patient_id (e.g., 'patient_001')."""
    if patient_id in ehr_records:
        record = ehr_records[patient_id]
        return (
            f"EHR for {record['name']} (ID: {patient_id}):\n"
            f"  Age: {record['age']}\n"
            f"  Conditions: {', '.join(record['conditions'])}\n"
            f"  Medications: {', '.join(record['medications'])}\n"
            f"  Allergies: {', '.join(record['allergies']) if record['allergies'] else 'None'}\n"
            f"  Last Visit: {record['last_visit']}"
        )
    return f"No EHR found for patient ID '{patient_id}'."

# List of all tools
tools = [
    medical_kb_tool,
    imaging_analysis_tool,
    lab_test_interpretation_tool,
    drug_interaction_checker_tool,
    ehr_system_tool
]

# --- 1. Core Component: Foundation Model (FM) Controller ---

# Initialize the LLM (using ChatOpenAI as a placeholder)
# Ensure OPENAI_API_KEY environment variable is set
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Define the prompt template for the agent
system_message = (
    "You are a highly intelligent and helpful Medical Diagnosis Assistant. "
    "Your primary goal is to assist doctors and patients in understanding medical conditions, "
    "interpreting diagnostic results, checking drug interactions, and retrieving patient history. "
    "You have access to several specialized medical tools. Use them wisely and logically to answer questions. "
    "Always consider the context of the user's query and provide comprehensive yet concise information. "
    "If a patient ID is mentioned, prioritize using the EHR tool to get patient context before making further decisions. "
    "If you need specific information that a tool can provide, use the tool. "
    "If a query asks for general medical information, use the medical_kb_tool. "
    "If an image analysis or lab test interpretation is mentioned, use the respective tools. "
    "If drug interactions are mentioned, use the drug_interaction_checker_tool. "
    "Always explain your reasoning and the steps you took to arrive at an answer."
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_message),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# Create the LangChain agent
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Main Application Loop ---

def run_diagnosis_assistant():
    print("Welcome to the Medical Diagnosis Assistant!")
    print("You can ask me medical questions, request diagnostic help, or query patient records.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Your query: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the Medical Diagnosis Assistant. Goodbye!")
            break
        try:
            response = agent_executor.invoke({"input": user_input})
            print("\nAssistant:" + response["output"] + "\n")
        except Exception as e:
            print(f"\nAssistant: An error occurred: {e}. Please try again.\n")

if __name__ == "__main__":
    # Set your OpenAI API key as an environment variable or uncomment the line below
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY environment variable not set. The LLM will not function.")
        print("Please set it or replace ChatOpenAI with a local LLM implementation.")
        # You can choose to exit or continue with a non-functional LLM
        # exit()

    run_diagnosis_assistant()
