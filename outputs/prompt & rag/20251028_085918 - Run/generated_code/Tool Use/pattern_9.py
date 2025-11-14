import os
from dotenv import load_dotenv
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- 1. LLM Agent (Controller) Setup ---
# Initialize OpenAI LLM
# Ensure you have OPENAI_API_KEY set in your environment variables or .env file
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 2. External Tools (Simulated) ---

# Medical Knowledge Base Search (using Chroma)
medical_knowledge_base_data = [
    "Hypertension: Also known as high blood pressure. Treatment often involves lifestyle changes (diet, exercise) and medications like ACE inhibitors, ARBs, diuretics, or beta-blockers. Regular monitoring is crucial.",
    "Type 2 Diabetes: A chronic condition affecting how your body processes blood sugar. Managed through diet, exercise, and sometimes medication (e.g., Metformin, insulin). Complications include heart disease, kidney disease, and nerve damage.",
    "Migraine: A type of headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Treatments include pain relievers and preventive medications.",
    "Common Cold: A viral infection of your nose and throat. Symptoms can include a runny nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, and a low-grade fever. Rest and hydration are key.",
    "Appendicitis: Inflammation of the appendix. Symptoms include sudden pain that begins on the right side of the lower abdomen, nausea, vomiting, and fever. Requires surgical removal of the appendix.",
    "Asthma: A condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out and shortness of breath. Managed with bronchodilators and anti-inflammatory medications.",
    "Pneumonia: An infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. Treatment depends on the cause (bacterial, viral, fungal).",
    "Influenza (Flu): A contagious respiratory illness caused by flu viruses. Symptoms are generally more severe than a common cold and can include fever, muscle aches, headache, fatigue, and cough. Antiviral drugs can be prescribed."
]

# Initialize embeddings for Chroma
embeddings = OpenAIEmbeddings()

# Create a persistent Chroma client (in-memory for this example)
chroma_db = Chroma.from_documents(
    documents=[{'page_content': text} for text in medical_knowledge_base_data], # Chroma expects Document objects or dicts
    embedding=embeddings,
    collection_name="medical_knowledge_base"
)

def search_medical_knowledge_base(query: str) -> str:
    """Searches the medical knowledge base for information related to a query.
    Use this tool to find evidence-based information about diseases, symptoms, treatments, or medical guidelines.
    """
    print(f"\n[DEBUG] Searching medical knowledge base for: '{query}'")
    results = chroma_db.similarity_search(query, k=3)
    if not results:
        return "No relevant information found in the medical knowledge base."
    return "\n-- Medical Knowledge Base Search Results --\n" + "\n--\n".join([doc.page_content for doc in results])

def get_ehr_data(patient_id: str, data_type: str) -> str:
    """Retrieves simulated Electronic Health Record (EHR) data for a given patient.
    Use this tool to get specific patient data like 'lab results', 'medication history', or 'allergies'.
    Example: get_ehr_data(patient_id='P123', data_type='lab results')
    """
    print(f"\n[DEBUG] Fetching EHR data for patient_id='{patient_id}', data_type='{data_type}'")
    simulated_ehr_data = {
        "P123": {
            "lab results": "Blood pressure: 140/90 mmHg, Glucose: 130 mg/dL, Cholesterol: LDL 150 mg/dL, HDL 35 mg/dL.",
            "medication history": "Lisinopril 10mg daily, Metformin 500mg twice daily.",
            "allergies": "Penicillin"
        },
        "P456": {
            "lab results": "Blood pressure: 120/80 mmHg, Glucose: 90 mg/dL.",
            "medication history": "None.",
            "allergies": "None"
        }
    }
    patient_data = simulated_ehr_data.get(patient_id)
    if patient_data:
        return f"\n-- EHR Data for Patient {patient_id} ({data_type}) --\n" + patient_data.get(data_type, "Data type not found.")
    return f"Patient ID {patient_id} not found in EHR system."

def analyze_imaging(image_id: str) -> str:
    """Simulates an API call to analyze a diagnostic imaging study (e.g., X-ray, MRI).
    Provide an image_id to get a simulated report. Example: analyze_imaging(image_id='XRAY-CHEST-001')
    """
    print(f"\n[DEBUG] Analyzing imaging for image_id: '{image_id}'")
    simulated_reports = {
        "XRAY-CHEST-001": "Chest X-ray report: Lungs clear, no acute cardiopulmonary abnormalities. Minor calcification noted in aortic knob.",
        "MRI-BRAIN-002": "Brain MRI report: No acute intracranial hemorrhage or mass effect. Age-related white matter changes noted."
    }
    report = simulated_reports.get(image_id)
    if report:
        return f"\n-- Diagnostic Imaging Report for {image_id} --\n" + report
    return f"No imaging report found for {image_id}."

def calculate_dosage(drug_name: str, patient_weight_kg: float, dosage_mg_per_kg: float) -> str:
    """Calculates a drug dosage based on patient weight and mg/kg dosage.
    Args:
        drug_name (str): The name of the drug.
        patient_weight_kg (float): The patient's weight in kilograms.
        dosage_mg_per_kg (float): The dosage in milligrams per kilogram.
    Example: calculate_dosage(drug_name='Amoxicillin', patient_weight_kg=70.0, dosage_mg_per_kg=15.0)
    """
    print(f"\n[DEBUG] Calculating dosage for {drug_name} for {patient_weight_kg}kg at {dosage_mg_per_kg} mg/kg")
    if patient_weight_kg <= 0 or dosage_mg_per_kg <= 0:
        return "Patient weight and dosage per kg must be positive values."
    total_dosage_mg = patient_weight_kg * dosage_mg_per_kg
    return f"\n-- Dosage Calculation --\nFor {drug_name}, a patient weighing {patient_weight_kg} kg at {dosage_mg_per_kg} mg/kg requires a total dosage of {total_dosage_mg:.2f} mg."

def check_drug_interactions(drugs: List[str]) -> str:
    """Simulates checking for potential drug interactions between a list of drugs.
    Example: check_drug_interactions(drugs=['Lisinopril', 'Metformin'])
    """
    print(f"\n[DEBUG] Checking drug interactions for: {', '.join(drugs)}")
    simulated_interactions = {
        frozenset(['Lisinopril', 'Metformin']): "Potential for increased risk of lactic acidosis with concurrent use, especially in patients with renal impairment. Monitor renal function.",
        frozenset(['Warfarin', 'Aspirin']): "Increased risk of bleeding. Use with caution and monitor INR.",
        frozenset(['Penicillin', 'Allopurinol']): "Increased risk of skin rash with concurrent use."
    }
    # Sort drugs to ensure consistent key for frozenset
    sorted_drugs = frozenset(sorted(drugs))
    interaction = simulated_interactions.get(sorted_drugs)
    if interaction:
        return f"\n-- Drug Interaction Alert --\nDrugs: {', '.join(drugs)}\nInteraction: {interaction}"
    return f"No significant interactions found for {', '.join(drugs)} in our simulated database."

# --- 3. Tool Definition (Langchain Tool objects) ---
tools = [
    Tool(
        name="MedicalKnowledgeBaseSearch",
        func=search_medical_knowledge_base,
        description=(
            "Searches the medical knowledge base for information related to a query. "
            "Use this tool to find evidence-based information about diseases, symptoms, treatments, or medical guidelines."
        ),
    ),
    Tool(
        name="EHRDataRetrieval",
        func=get_ehr_data,
        description=(
            "Retrieves simulated Electronic Health Record (EHR) data for a given patient. "
            "Input should be a patient ID and the specific data type (e.g., 'lab results', 'medication history', 'allergies'). "
            "Example: get_ehr_data(patient_id='P123', data_type='lab results')"
        ),
    ),
    Tool(
        name="DiagnosticImagingAnalysis",
        func=analyze_imaging,
        description=(
            "Simulates an API call to analyze a diagnostic imaging study (e.g., X-ray, MRI). "
            "Provide an image_id to get a simulated report. Example: analyze_imaging(image_id='XRAY-CHEST-001')"
        ),
    ),
    Tool(
        name="MedicalDosageCalculator",
        func=calculate_dosage,
        description=(
            "Calculates a drug dosage based on patient weight and mg/kg dosage. "
            "Input needs 'drug_name' (str), 'patient_weight_kg' (float), and 'dosage_mg_per_kg' (float). "
            "Example: calculate_dosage(drug_name='Amoxicillin', patient_weight_kg=70.0, dosage_mg_per_kg=15.0)"
        ),
    ),
    Tool(
        name="DrugInteractionChecker",
        func=check_drug_interactions,
        description=(
            "Simulates checking for potential drug interactions between a list of drugs. "
            "Input should be a list of drug names (List[str]). "
            "Example: check_drug_interactions(drugs=['Lisinopril', 'Metformin'])"
        ),
    ),
]

# --- 4. Agent Initialization ---
prompt = PromptTemplate.from_template(
    """You are a highly skilled AI-powered Clinical Decision Support System. "
    "Your goal is to assist medical professionals by providing accurate, evidence-based information, "
    "personalized recommendations, and safety alerts based on the available tools. "
    "Always use the provided tools to gather relevant information before formulating a response. "
    "Prioritize patient safety and evidence-based medicine. If you need patient-specific data, "
    "ask for the patient ID if it's not provided.

    You have access to the following tools:
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
    """
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 5. Main interaction loop (Conceptual UI) ---
def main():
    print("Welcome to the AI-Powered Clinical Decision Support System!")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nMedical Professional: ")
        if user_query.lower() == 'exit':
            break

        try:
            print("\n[System Processing...]\n")
            response = agent_executor.invoke({"input": user_query})
            print("\nAI Assistant:")
            print(response["output"])
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try rephrasing your query or check the input format.")

if __name__ == "__main__":
    main()
