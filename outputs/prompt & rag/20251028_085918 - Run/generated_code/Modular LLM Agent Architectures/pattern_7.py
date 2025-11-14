
import os
from dotenv import load_dotenv
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# --- 1. Pydantic Models for Tool Input Validation ---

class PatientIDInput(BaseModel):
    patient_id: str = Field(description="ID of the patient, e.g., 'P001'")

class MedicalQueryInput(BaseModel):
    query: str = Field(description="Medical query to search, e.g., 'symptoms of diabetes'")

class DrugInfoInput(BaseModel):
    drug_name: str = Field(description="Name of the drug to get information about, e.g., 'paracetamol'")

# --- 2. Simulated EHR System Tool ---

class EHRSystemSimulatorTool:
    def __init__(self):
        self.patient_data = {
            "P001": {
                "demographics": {"name": "Alice Smith", "age": 45, "gender": "Female"},
                "medical_history": ["Hypertension (diagnosed 5 years ago)", "Seasonal allergies"],
                "lab_results": [
                    {"date": "2023-10-20", "type": "Blood Pressure", "value": "140/90 mmHg"},
                    {"date": "2023-09-15", "type": "Cholesterol", "value": "LDL 150 mg/dL"}
                ]
            },
            "P002": {
                "demographics": {"name": "Bob Johnson", "age": 60, "gender": "Male"},
                "medical_history": ["Type 2 Diabetes (diagnosed 10 years ago)", "Coronary Artery Disease"],
                "lab_results": [
                    {"date": "2023-11-01", "type": "HbA1c", "value": "7.5%"},
                    {"date": "2023-08-10", "type": "ECG", "value": "Normal sinus rhythm"}
                ]
            }
        }

    def get_patient_demographics(self, patient_id: str) -> Dict[str, Any]:
        """Get demographic information for a given patient ID."""
        return self.patient_data.get(patient_id, {}).get("demographics", {})

    def get_patient_medical_history(self, patient_id: str) -> List[str]:
        """Get medical history for a given patient ID."""
        return self.patient_data.get(patient_id, {}).get("medical_history", [])

    def get_recent_lab_results(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get recent lab results for a given patient ID."""
        return self.patient_data.get(patient_id, {}).get("lab_results", [])

# --- 3. Simulated Medical Database Search Tool ---

class MedicalDatabaseSearchTool:
    def search_pubmed(self, query: str) -> str:
        """Searches a simulated PubMed for medical articles related to the query."""
        simulated_results = {
            "symptoms of diabetes": "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision. \nReference: \"Diabetes Mellitus: A Comprehensive Review\" (Journal of Clinical Endocrinology & Metabolism, 2022).",
            "treatment for hypertension": "First-line treatments often include lifestyle modifications (diet, exercise) and medications such as ACE inhibitors, ARBs, or diuretics. \nReference: \"Hypertension Management Guidelines\" (American Heart Association, 2023).",
            "side effects of metformin": "Common side effects include nausea, diarrhea, and abdominal discomfort. Less common but serious side effect is lactic acidosis. \nReference: \"Pharmacology of Metformin\" (British Journal of Clinical Pharmacology, 2021)."
        }
        return simulated_results.get(query.lower(), "No relevant articles found in simulated PubMed.")

    def get_drug_info(self, drug_name: str) -> str:
        """Retrieves information about a specific drug from a simulated database."""
        simulated_drug_info = {
            "paracetamol": "Paracetamol (Acetaminophen) is a pain reliever and fever reducer. \nDosage: 500-1000 mg every 4-6 hours as needed. \nSide effects: Rare at therapeutic doses, but liver damage can occur with overdose.",
            "lisinopril": "Lisinopril is an ACE inhibitor used to treat high blood pressure and heart failure. \nDosage: Typically 10-40 mg once daily. \nSide effects: Dizziness, dry cough, fatigue, headache."
        }
        return simulated_drug_info.get(drug_name.lower(), "No information found for this drug in the simulated database.")

# --- 4. Initialize LLM, Memory, and Tools ---

# Initialize OpenAI LLM
# Ensure OPENAI_API_KEY is set in your .env file or environment variables
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Short-term Conversation Memory
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)

# Initialize long-term patient context (simulated in-memory)
long_term_patient_context: Dict[str, Dict[str, Any]] = {
    "P001": {
        "notes": "Patient P001 has a history of mild hypertension and seasonal allergies. Requires regular monitoring of blood pressure. Sensitive to certain pollen types."
    },
    "P002": {
        "notes": "Patient P002 is a diabetic with a history of CAD. Strict adherence to medication and diet is crucial. Recent HbA1c suggests a need for review of diabetes management."
    }
}

# Instantiate the custom tools
ehr_simulator = EHRSystemSimulatorTool()
medical_db_search = MedicalDatabaseSearchTool()

# Wrap custom tools as LangChain Tools
tools = [
    Tool(
        name="GetPatientDemographics",
        func=ehr_simulator.get_patient_demographics,
        description="Useful for getting demographic information (name, age, gender) for a patient. Input should be a patient ID (e.g., 'P001').",
        args_schema=PatientIDInput
    ),
    Tool(
        name="GetPatientMedicalHistory",
        func=ehr_simulator.get_patient_medical_history,
        description="Useful for retrieving the medical history of a patient. Input should be a patient ID (e.g., 'P001').",
        args_schema=PatientIDInput
    ),
    Tool(
        name="GetRecentLabResults",
        func=ehr_simulator.get_recent_lab_results,
        description="Useful for getting recent laboratory test results for a patient. Input should be a patient ID (e.g., 'P001').",
        args_schema=PatientIDInput
    ),
    Tool(
        name="SearchPubMed",
        func=medical_db_search.search_pubmed,
        description="Useful for searching a medical database (like PubMed) for information on diseases, symptoms, treatments, etc. Input should be a medical query string.",
        args_schema=MedicalQueryInput
    ),
    Tool(
        name="GetDrugInfo",
        func=medical_db_search.get_drug_info,
        description="Useful for getting detailed information about a specific drug, including dosage and side effects. Input should be the drug name.",
        args_schema=DrugInfoInput
    )
]

# --- 5. Agent Orchestration --- 

# Define the agent prompt
# The prompt incorporates the `chat_history` from memory.
AGENT_PROMPT_TEMPLATE = """You are a Clinical Decision Support Agent assisting healthcare professionals. 
Your goal is to provide accurate and helpful information based on patient data and medical knowledge. 

Here is the current conversation history:
{chat_history}

If you need patient-specific information, remember to use the patient ID (e.g., 'P001', 'P002').

Relevant long-term patient context:
{long_term_context}

TOOLS:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

# Create a LangChain prompt template
prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor
# We will pass the long_term_context as part of the `inputs` to the `invoke` method.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

# --- 6. Main execution function ---

def run_agent(question: str, patient_id: str = None):
    """Runs the clinical decision support agent with a given question.
    Optionally provides patient_id for adding relevant long-term context.
    """
    print(f"\n--- Running Agent for: '{question}' ---")
    
    # Retrieve relevant long-term patient context if patient_id is provided
    current_long_term_context = "No specific patient context provided." 
    if patient_id and patient_id in long_term_patient_context:
        current_long_term_context = long_term_patient_context[patient_id].get("notes", "")
    
    # Prepare inputs for the agent executor
    inputs = {
        "input": question,
        "tools": tools,
        "tool_names": [tool.name for tool in tools],
        "long_term_context": current_long_term_context
    }
    
    try:
        response = agent_executor.invoke(inputs)
        print("\nAgent Final Answer:")
        print(response["output"])
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your input or the agent's reasoning.")

if __name__ == "__main__":
    # Example Usage
    # Ensure OPENAI_API_KEY is set in your .env file

    print("\n--- Clinical Decision Support Agent Initialized ---")
    print("You can ask questions like:\n")
    print("1. What are the demographics of patient P001?")
    print("2. What is the medical history for P002?")
    print("3. Search PubMed for 'treatment for hypertension'.")
    print("4. What are the side effects of paracetamol?")
    print("5. What are the recent lab results for patient P001 and what does 'LDL 150 mg/dL' signify?")

    while True:
        user_input = input("\nDoctor (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        # Simple heuristic to check for patient ID in input for long-term context
        patient_id_for_context = None
        if "P001" in user_input.upper():
            patient_id_for_context = "P001"
        elif "P002" in user_input.upper():
            patient_id_for_context = "P002"
            
        run_agent(user_input, patient_id=patient_id_for_context)

    print("\n--- Exiting Clinical Decision Support Agent --- ")
