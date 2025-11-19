import streamlit as st
import os
from loguru import logger
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.python.tool import PythonREPLTool

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
logger.add("file_diag_assistant.log", rotation="500 MB")

# --- 3. Specialized Tools ---

@tool
def get_medical_knowledge(query: str) -> str:
    """Accesses a comprehensive medical knowledge base to retrieve information on diseases, drugs, treatments, and diagnostic criteria.
    Input should be a clear medical query, e.g., 'symptoms of pneumonia' or 'dosage for amoxicillin'.
    """
    logger.info(f"Tool: get_medical_knowledge called with query: {query}")
    # Mocking an API call by returning predefined data
    if "pneumonia" in query.lower():
        return "Pneumonia symptoms include cough, fever, shortness of breath, and chest pain. Diagnosis involves chest X-ray and sputum culture."
    elif "diabetes" in query.lower():
        return "Diabetes Mellitus is a metabolic disease that causes high blood sugar. Treatment involves insulin, medication, and lifestyle changes."
    else:
        return f"Information for '{query}' not found in medical knowledge base. Please try a more specific query."

@tool
def analyze_medical_image(image_description: str) -> str:
    """Interprets medical images (e.g., X-rays, MRIs, CT scans) to provide preliminary findings.
    Input should be a description of the image and what to look for, e.g., 'X-ray of lungs, look for signs of consolidation' or 'MRI of brain, check for tumors'.
    """
    logger.info(f"Tool: analyze_medical_image called with description: {image_description}")
    # Mocking an external model/microservice
    if "x-ray" in image_description.lower() and "lungs" in image_description.lower():
        if "consolidation" in image_description.lower():
            return "Preliminary analysis of lung X-ray indicates signs of consolidation in the lower left lobe, consistent with pneumonia."
        else:
            return "Preliminary analysis of lung X-ray shows no obvious abnormalities."
    elif "mri" in image_description.lower() and "brain" in image_description.lower():
        if "tumor" in image_description.lower():
            return "Preliminary analysis of brain MRI shows a suspicious lesion in the frontal lobe, suggestive of a tumor. Further investigation recommended."
        else:
            return "Preliminary analysis of brain MRI shows no significant findings."
    else:
        return f"Cannot analyze image for '{image_description}'. Image type or area of interest not recognized by the mock image analysis tool."

@tool
def get_patient_ehr(patient_id: str) -> str:
    """Securely accesses and synthesizes a patient's electronic health records (EHR).
    Input should be a valid patient ID. Returns a summary of medical history, lab results, and current medications as a JSON string.
    """
    logger.info(f"Tool: get_patient_ehr called for patient ID: {patient_id}")
    # Mocking secure EHR access
    if patient_id == "P001":
        return str({
            "name": "Jane Doe",
            "age": 45,
            "conditions": ["Hypertension", "Type 2 Diabetes"],
            "medications": ["Lisinopril 10mg", "Metformin 500mg"],
            "lab_results": {"glucose": "180 mg/dL", "blood_pressure": "140/90 mmHg"},
            "symptoms_history": ["Recent cough", "fatigue"]
        })
    elif patient_id == "P002":
        return str({
            "name": "John Smith",
            "age": 62,
            "conditions": ["Asthma"],
            "medications": ["Albuterol inhaler"],
            "lab_results": {"spirometry": "FEV1 70%"},
            "symptoms_history": ["Wheezing", "shortness of breath"]
        })
    else:
        return f"Patient with ID '{patient_id}' not found in EHR."

@tool
def check_symptoms(symptoms: str) -> str:
    """Suggests potential diagnoses based on a list of reported symptoms.
    Input should be a comma-separated string of symptoms, e.g., 'fever, cough, headache'.
    """
    logger.info(f"Tool: check_symptoms called with symptoms: {symptoms}")
    symptoms_lower = symptoms.lower()
    if "fever" in symptoms_lower and "cough" in symptoms_lower and "shortness of breath" in symptoms_lower:
        return "Possible diagnoses include Pneumonia, Bronchitis, or COVID-19. Consider further tests."
    elif "headache" in symptoms_lower and "stiff neck" in symptoms_lower and "fever" in symptoms_lower:
        return "Possible diagnoses include Meningitis. Urgent medical attention is advised."
    elif "chest pain" in symptoms_lower and "left arm pain" in symptoms_lower:
        return "Possible diagnoses include Angina or Myocardial Infarction. Seek immediate emergency care."
    else:
        return "Based on the provided symptoms, no specific common diagnosis can be suggested by the symptom checker. Please provide more details or consult a healthcare professional."

@tool
def get_clinical_guidelines(condition: str) -> str:
    """Retrieves and applies evidence-based clinical practice guidelines for a specific medical condition.
    Input should be a clear medical condition, e.g., 'Type 2 Diabetes Management' or 'Hypertension Treatment'.
    """
    logger.info(f"Tool: get_clinical_guidelines called for condition: {condition}")
    # Mocking an API call by returning predefined data
    if "type 2 diabetes management" in condition.lower():
        return "Clinical guidelines for Type 2 Diabetes Management typically include metformin as first-line therapy, regular exercise, dietary changes, and monitoring of blood glucose levels. Consider GLP-1 receptor agonists or SGLT2 inhibitors for advanced cases."
    elif "hypertension treatment" in condition.lower():
        return "Clinical guidelines for Hypertension Treatment recommend lifestyle modifications (diet, exercise) and pharmacotherapy with ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics, depending on patient comorbidities."
    else:
        return f"Clinical guidelines for '{condition}' not found in the database."

# Code Interpreter Tool
python_repl_tool = PythonREPLTool()
python_repl_tool.name = "python_interpreter"
python_repl_tool.description = "A Python interpreter for executing Python code. Useful for mathematical calculations, data analysis, and simulations. Input should be valid Python code."
# Note: PythonREPLTool uses 'exec' internally, which can be a security risk in a production environment.
# For demonstration, it's acceptable, but be cautious in real applications.

# --- 2. Tool Integration Layer ---
tools = [
    get_medical_knowledge,
    analyze_medical_image,
    get_patient_ehr,
    check_symptoms,
    get_clinical_guidelines,
    python_repl_tool,
]

# --- 1. Core AI Agent (LLM Orchestrator) ---
# Ensure OPENAI_API_KEY environment variable is set
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI-powered diagnostic assistant for healthcare professionals. Your goal is to provide comprehensive diagnostic support and treatment recommendations by intelligently using the available tools. Be thorough, accurate, and always prioritize patient safety. If you need more information, ask clarifying questions."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 4. User Interface (Frontend) with Streamlit ---
st.set_page_config(page_title="AI Diagnostic Assistant", layout="wide")
st.title("👨‍⚕️ AI-Powered Diagnostic Assistant")

st.markdown("""
Welcome to the AI Diagnostic Assistant. This tool helps healthcare professionals by leveraging a powerful AI agent augmented with specialized medical tools.
Enter a patient's case details or a medical query below to get diagnostic insights and recommendations.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Enter your medical query or patient case details:")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and consulting medical tools..."):
            try:
                response = agent_executor.invoke({"input": user_query})
                st.markdown(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                logger.info(f"Assistant response: {response['output']}")
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                logger.error(f"Error during agent execution: {e}")

st.sidebar.header("How to Use:")
st.sidebar.markdown("""
1.  **Enter a query:** Describe the patient's symptoms, lab results, or ask a general medical question.
2.  **Example queries:**
    *   "Diagnose patient P001 who has recent cough and fatigue. They also have a history of hypertension."
    *   "What are the clinical guidelines for Type 2 Diabetes management?"
    *   "Analyze an X-ray of lungs looking for pneumonia."
    *   "Calculate the BMI for a patient weighing 70kg and 1.75m tall using the code interpreter."
""")