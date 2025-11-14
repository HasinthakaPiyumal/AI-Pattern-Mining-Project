import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables from .env file
load_dotenv()

# --- 0. Configuration and LLM Initialization ---

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. Medical Knowledge Base (Simulated Tool) ---

# Simulate a medical knowledge base (in a real app, this would be a vector DB query)
medical_data = {
    "fever": "Fever is a temporary increase in your body temperature, often due to an illness. Treatment includes rest, fluids, and fever-reducing medication.",
    "headache": "Headache is a pain in any region of the head. Causes can vary from stress to more serious conditions. Treatment depends on the cause.",
    "diabetes type 2": "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). Management often includes diet, exercise, and medication.",
    "hypertension": "Hypertension (high blood pressure) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle changes and medication are common treatments.",
    "appendicitis": "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon on the lower right side of your abdomen. It typically causes pain in the lower right abdomen. Treatment usually involves surgery."
}

def get_medical_info(query: str) -> str:
    """Retrieves medical information based on a query.
    For example, get_medical_info("fever") will return details about fever.
    """
    print(f"\n[TOOL CALL] Searching medical knowledge base for: {query}")
    query_lower = query.lower()
    for key, value in medical_data.items():
        if query_lower in key or key in query_lower:
            return value
    return "No specific information found for that medical query in the knowledge base."

medical_knowledge_tool = Tool(
    name="MedicalKnowledgeBase",
    func=get_medical_info,
    description="Useful for retrieving detailed medical information about symptoms, diseases, or conditions."
)

# --- 2. Patient History and Context Management (Memory) ---

# Using ConversationBufferWindowMemory to store recent interactions
# In a real app, this would be linked to a patient's electronic health record (EHR)
patient_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5, # Stores the last 5 exchanges
    return_messages=True,
    input_key="input"
)

# Example patient data (for demonstration)
patient_ehr = {
    "patient_id": "P12345",
    "name": "John Doe",
    "age": 55,
    "allergies": ["penicillin"],
    "current_medications": ["Metformin", "Lisinopril"],
    "past_medical_history": ["Type 2 Diabetes", "Hypertension"]
}

def get_patient_summary(patient_id: str) -> str:
    """Retrieves a summary of the patient's electronic health record.
    Use this to get patient demographics, allergies, current medications, and past medical history.
    """
    print(f"\n[TOOL CALL] Retrieving patient summary for: {patient_id}")
    if patient_id == patient_ehr["patient_id"]:
        summary = f"Patient ID: {patient_ehr['patient_id']}\n"
        summary += f"Name: {patient_ehr['name']}, Age: {patient_ehr['age']}\n"
        summary += f"Allergies: {', '.join(patient_ehr['allergies']) if patient_ehr['allergies'] else 'None'}\n"
        summary += f"Current Medications: {', '.join(patient_ehr['current_medications']) if patient_ehr['current_medications'] else 'None'}\n"
        summary += f"Past Medical History: {', '.join(patient_ehr['past_medical_history']) if patient_ehr['past_medical_history'] else 'None'}"
        return summary
    return "Patient not found or invalid patient ID."

patient_summary_tool = Tool(
    name="PatientSummaryRetriever",
    func=get_patient_summary,
    description="Useful for retrieving a summary of the patient's medical history and current status. Input should be a patient ID."
)


# --- 3. Drug Interaction Checker (Simulated Tool) ---

def check_drug_interactions(medications: str) -> str:
    """Checks for potential drug-drug interactions between a comma-separated list of medications.
    E.g., check_drug_interactions("Metformin, Lisinopril, Ibuprofen")
    """
    print(f"\n[TOOL CALL] Checking drug interactions for: {medications}")
    med_list = [m.strip().lower() for m in medications.split(',')]
    interactions = []

    # Dummy interaction logic
    if "metformin" in med_list and "ibuprofen" in med_list:
        interactions.append("Metformin and Ibuprofen: Increased risk of kidney problems. Monitor renal function.")
    if "lisinopril" in med_list and "potassium supplements" in med_list:
        interactions.append("Lisinopril and Potassium Supplements: Increased risk of hyperkalemia. Avoid concurrent use or monitor closely.")
    if any(allergy.lower() in med_list for allergy in patient_ehr["allergies"]):
        allergies_triggered = [med for med in med_list if med in [a.lower() for a in patient_ehr["allergies"]]]
        interactions.append(f"WARNING: Patient has allergies to {', '.join(allergies_triggered)}.")

    if interactions:
        return "Potential Drug Interactions Found:\n" + "\n".join(interactions)
    return "No significant drug interactions or allergies detected for the provided medications."

drug_interaction_tool = Tool(
    name="DrugInteractionChecker",
    func=check_drug_interactions,
    description="Useful for checking potential drug-drug interactions. Input should be a comma-separated string of medication names. Also checks against patient allergies."
)

# --- 4. Differential Diagnosis Planner (Integrated into LLM Agent) ---
# This will be handled by the LLM's reasoning within the main agent using the available tools.
# The LLM will use MedicalKnowledgeBase and PatientSummaryRetriever to formulate diagnoses.

# --- 5. Main Orchestrator (LangChain Agent) ---

# Define the tools available to the agent
agent_tools = [medical_knowledge_tool, patient_summary_tool, drug_interaction_tool]

# Define the prompt for the agent
# It instructs the agent on its role and how to use the tools.
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI-powered diagnostic assistant for healthcare professionals. "
        "Your goal is to assist in diagnosis, treatment planning, and drug interaction checks. "
        "Always consider the patient's context and history when providing recommendations. "
        "Prioritize patient safety, especially regarding allergies and drug interactions. "
        "You have access to a MedicalKnowledgeBase, a PatientSummaryRetriever, and a DrugInteractionChecker."
        "When asked about a patient, first try to retrieve their summary using PatientSummaryRetriever with a patient ID if available. "
        "When providing diagnostic suggestions, explain your reasoning and suggest further steps or tests if appropriate. "
        "When checking drug interactions, ensure you consider all current and proposed medications, and also check against known patient allergies."
    )),
    MessagesPlaceholder(variable_name="chat_history"), # For conversational memory
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the ReAct agent
agent = create_react_agent(llm, agent_tools, agent_prompt)

# Create the AgentExecutor with memory
agent_executor = AgentExecutor(
    agent=agent,
    tools=agent_tools,
    verbose=True, # Set to True to see the thought process of the agent
    memory=patient_memory,
    handle_parsing_errors=True
)

# --- Example Usage ---

if __name__ == "__main__":
    print("\n--- AI Diagnostic Assistant Initialized ---")
    print("How can I assist you with patient diagnosis or information today?")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nHealthcare Professional: ")
        if user_input.lower() == 'exit':
            break

        try:
            # Include patient_id in the input if relevant, or let the LLM ask for it
            # For this example, we'll try to implicitly include it or let the LLM use the tool
            response = agent_executor.invoke({"input": user_input, "patient_id": patient_ehr["patient_id"]})
            print(f"\nAI Diagnostic Assistant: {response['output']}")
        except Exception as e:
            print(f"\nAI Diagnostic Assistant (Error): An error occurred: {e}")

    print("\n--- Session Ended ---")