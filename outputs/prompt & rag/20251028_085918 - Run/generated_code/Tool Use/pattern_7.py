import os
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# --- 1. Define Tool Input Schemas (using Pydantic) ---

class MedicalDatabaseSearchInput(BaseModel):
    query: str = Field(description="The medical query to search for (e.g., 'drug interactions for warfarin', 'symptoms of diabetes').")

class EHRRetrievalInput(BaseModel):
    patient_id: str = Field(description="The unique identifier for the patient whose EHR is to be retrieved.")
    data_type: str = Field(description="The specific type of data to retrieve from the EHR (e.g., 'medications', 'allergies', 'diagnoses', 'lab_results').")

class GuidelineSummarizerInput(BaseModel):
    topic: str = Field(description="The topic of the medical guideline to summarize (e.g., 'type 2 diabetes treatment', 'hypertension management').")

# --- 2. Define Mock Tool Functions ---

@tool(
    args_schema=MedicalDatabaseSearchInput,
    return_direct=False
)
def medical_database_search(query: str) -> str:
    """Searches a simulated medical database for information based on the provided query.
    Returns relevant medical information or an indication that no information was found.
    """
    print(f"[DEBUG] Searching medical database for: '{query}'")
    if "warfarin interactions" in query.lower():
        return "Warfarin interacts with many drugs, including NSAIDs, aspirin, and certain antibiotics, increasing the risk of bleeding."
    elif "type 2 diabetes symptoms" in query.lower():
        return "Common symptoms of Type 2 Diabetes include increased thirst, frequent urination, increased hunger, fatigue, and blurred vision."
    elif "medications for hypertension" in query.lower():
        return "Common medications for hypertension include ACE inhibitors (e.g., lisinopril), ARBs (e.g., losartan), diuretics (e.g., hydrochlorothiazide), beta-blockers (e.g., metoprolol), and calcium channel blockers (e.g., amlodipine)."
    else:
        return f"No specific information found in the medical database for '{query}'. Please try a different query."

@tool(
    args_schema=EHRRetrievalInput,
    return_direct=False
)
def ehr_retrieval(patient_id: str, data_type: str) -> str:
    """Retrieves specific data from a simulated Electronic Health Record (EHR) for a given patient.
    Returns the requested data or an error message if the patient/data is not found.
    """
    print(f"[DEBUG] Retrieving {data_type} for patient ID: {patient_id}")
    mock_ehr_data = {
        "patient_123": {
            "medications": "Lisinopril 10mg daily, Metformin 500mg twice daily",
            "allergies": "Penicillin",
            "diagnoses": "Hypertension, Type 2 Diabetes",
            "lab_results": "HbA1c: 7.2%, Blood Pressure: 140/90 mmHg"
        },
        "patient_456": {
            "medications": "Aspirin 81mg daily",
            "allergies": "None",
            "diagnoses": "Coronary Artery Disease",
            "lab_results": "Cholesterol: 220 mg/dL"
        }
    }

    patient_data = mock_ehr_data.get(patient_id)
    if not patient_data:
        return f"Patient with ID '{patient_id}' not found in EHR system."

    requested_data = patient_data.get(data_type)
    if not requested_data:
        return f"Data type '{data_type}' not found for patient '{patient_id}'. Available types: {', '.join(patient_data.keys())}."
    
    return f"EHR data for patient {patient_id} ({data_type}): {requested_data}"

@tool(
    args_schema=GuidelineSummarizerInput,
    return_direct=False
)
def guideline_summarizer(topic: str) -> str:
    """Generates a summary of new or existing medical guidelines for a given topic.
    This simulates the LLM's ability to autonomously create new tool-like content.
    """
    print(f"[DEBUG] Generating guideline summary for topic: '{topic}'")
    if "type 2 diabetes treatment" in topic.lower():
        return "[Newly Generated Guideline Summary]: Recent guidelines for Type 2 Diabetes treatment emphasize personalized care, early initiation of metformin, and consideration of SGLT2 inhibitors or GLP-1 receptor agonists for patients with cardiovascular or renal disease. Lifestyle modifications remain crucial."
    elif "hypertension management" in topic.lower():
        return "[Newly Generated Guideline Summary]: Updated hypertension guidelines suggest a target blood pressure of <130/80 mmHg for most adults. First-line agents include thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers. Lifestyle interventions are paramount."
    else:
        return f"[Newly Generated Guideline Summary]: A new guideline summary for '{topic}' would involve synthesizing information from various sources. For demonstration, let's assume a summary has been generated for this new topic."

# --- 3. Medical Assistant Class ---

class MedicalAssistant:
    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(openai_api_key=openai_api_key, temperature=0)

        self.tools = [
            medical_database_search,
            ehr_retrieval,
            guideline_summarizer,
        ]

        # Define the prompt for the ReAct agent
        template = '''
        Answer the following questions as best you can. You have access to the following tools:

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
        Thought:{agent_scratchpad}
        '''

        prompt = PromptTemplate.from_template(template)

        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    def run(self, query: str) -> str:
        """Runs the medical assistant with a given query."""
        print(f"\n--- User Query: {query} ---")
        try:
            result = self.agent_executor.invoke({"input": query})
            return result['output']
        except Exception as e:
            return f"An error occurred: {e}"

# --- 4. Main Execution Block ---

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables or .env file.")
        print("Please set it to run the LLM-powered agent. Using a placeholder for demonstration.")
        # For demonstration purposes without a key, you'd typically mock the LLM as well.
        # Here, we'll proceed, but the LLM calls will fail without a valid key.
        # openai_api_key = "YOUR_OPENAI_API_KEY_HERE" # Uncomment and replace for actual use

    assistant = MedicalAssistant(openai_api_key=openai_api_key)

    queries = [
        "What are the drug interactions for warfarin?",
        "Retrieve medications for patient_123.",
        "Summarize the latest guidelines for type 2 diabetes treatment.",
        "What are the symptoms of type 2 diabetes and what medications are typically prescribed?",
        "Get allergies for patient_456.",
        "Find information about a new guideline for pediatric asthma management and tell me what the key takeaways are."
    ]

    for q in queries:
        response = assistant.run(q)
        print(f"\n--- Assistant Response ---")
        print(response)
        print("="*50)
