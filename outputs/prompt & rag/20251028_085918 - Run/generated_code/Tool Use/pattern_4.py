import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Placeholder Tool Definitions ---

@tool
def get_medical_knowledge(query: str) -> str:
    """
    Accesses a vast medical knowledge base to retrieve information about diseases,
    conditions, symptoms, drugs, and medical procedures.
    Input should be a specific medical query.
    """
    st.info(f"Accessing Medical Knowledge Base for: '{query}'")
    # In a real application, this would query a vector database (e.g., Chroma, Pinecone)
    # with embedded medical texts, or a structured medical ontology.
    # For now, it's a placeholder.
    if "diabetes" in query.lower():
        return "Retrieved information on Diabetes Mellitus: a chronic metabolic disease characterized by elevated blood glucose levels. Types include Type 1, Type 2, and Gestational. Management involves diet, exercise, medication, and insulin."
    return f"Simulated retrieval of medical knowledge for '{query}'. (Placeholder response)"

@tool
def process_ehr_data(patient_id: str, query: str) -> str:
    """
    Securely accesses and interprets a patient's Electronic Health Record (EHR) data.
    Can retrieve patient history, current medications, lab results, and previous diagnoses.
    Input requires a patient_id and a specific query about the patient's record.
    """
    st.info(f"Processing EHR data for Patient ID: '{patient_id}' with query: '{query}'")
    # This would involve secure API calls to an EHR system and NLP processing.
    if patient_id == "PAT001":
        if "medications" in query.lower():
            return "Patient PAT001 is currently on Metformin (for Type 2 Diabetes) and Lisinopril (for hypertension)."
        if "allergies" in query.lower():
            return "Patient PAT001 has an allergy to Penicillin."
        return f"Simulated EHR data for Patient ID '{patient_id}' related to '{query}'. (Placeholder response)"
    return f"Patient ID '{patient_id}' not found or query '{query}' is too general. (Placeholder response)"

@tool
def analyze_imaging(image_url: str, patient_id: str = None) -> str:
    """
    Integrates with an AI-powered diagnostic imaging analysis system to interpret
    medical images (e.g., X-rays, MRIs, CT scans).
    Input should be a URL or identifier for the image, and optionally a patient_id.
    """
    st.info(f"Analyzing imaging from URL: '{image_url}' for Patient ID: '{patient_id if patient_id else 'N/A'}'")
    # This would typically involve sending the image to a specialized AI model endpoint.
    if "chest-xray-PAT001" in image_url:
        return "Simulated Chest X-ray analysis for PAT001: Findings suggest mild cardiomegaly with clear lung fields. No acute infiltrates or effusions."
    return f"Simulated imaging analysis for '{image_url}'. (Placeholder response)"

@tool
def get_differential_diagnosis(symptoms: str, patient_history: str = "") -> str:
    """
    Suggests a list of probable diagnoses based on reported symptoms and patient history.
    Input should be a comma-separated string of symptoms and optionally a patient history summary.
    """
    st.info(f"Generating differential diagnosis for symptoms: '{symptoms}' and history: '{patient_history}'")
    # This could be a complex system, potentially leveraging a knowledge graph or a specialized LLM for diagnosis.
    if "fever, cough, fatigue" in symptoms.lower():
        return "Differential diagnoses: Common cold, Influenza, Bronchitis, Pneumonia, COVID-19."
    return f"Simulated differential diagnosis for symptoms: '{symptoms}'. (Placeholder response)"

@tool
def get_treatment_guidelines(diagnosis: str, patient_factors: str = "") -> str:
    """
    Retrieves evidence-based treatment guidelines and prescription recommendations
    for a given diagnosis, considering patient-specific factors.
    Input should be a diagnosis and optionally patient-specific factors (e.g., allergies, comorbidities).
    """
    st.info(f"Retrieving treatment guidelines for diagnosis: '{diagnosis}' with factors: '{patient_factors}'")
    # This would query a structured database of treatment protocols.
    if "type 2 diabetes" in diagnosis.lower():
        return "Treatment guidelines for Type 2 Diabetes: Lifestyle modifications (diet, exercise), first-line medication typically Metformin, consider SGLT2 inhibitors or GLP-1 receptor agonists based on cardiovascular/renal risk. Monitor HbA1c regularly."
    return f"Simulated treatment guidelines for '{diagnosis}'. (Placeholder response)"

# List of all tools
tools = [
    get_medical_knowledge,
    process_ehr_data,
    analyze_imaging,
    get_differential_diagnosis,
    get_treatment_guidelines
]

# --- LLM Setup ---
try:
    # Initialize the ChatOpenAI model. Ensure OPENAI_API_KEY is set in your environment variables.
    llm = ChatOpenAI(model="gpt-4", temperature=0) 
except Exception as e:
    st.error(f"Failed to initialize ChatOpenAI. Ensure OPENAI_API_KEY is set in your .env file. Error: {e}")
    st.stop() # Stop the app if LLM can't be initialized

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI-powered Clinical Diagnostic Assistant. Your goal is to assist healthcare professionals by providing accurate and comprehensive information for diagnosis and treatment planning. You have access to specialized medical tools. Use these tools effectively to answer questions and provide detailed insights. Always prioritize patient safety and evidence-based information. When asking for patient ID, ensure you mention it's for EHR access."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create an agent that uses the LLM and the defined tools
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Streamlit Application ---
st.set_page_config(page_title="Clinical Diagnostic Assistant", layout="wide")

st.title("👨‍⚕️ AI Clinical Diagnostic Assistant")
st.markdown("""
Welcome, healthcare professional! This assistant leverages a Large Language Model (LLM) augmented with specialized medical tools to aid in diagnosis and treatment planning. Ask questions about patient symptoms, medical conditions, or request information from patient records.
""")

# User input text area
user_query = st.text_area("Enter your medical query or patient case:", height=150, placeholder="e.g., 'Patient PAT001 presents with fever, cough, and fatigue. What are the possible differential diagnoses based on their EHR and general medical knowledge?'")

if st.button("Get Assistant's Response", type="primary"):
    if user_query:
        with st.spinner("Thinking and consulting medical tools..."):
            try:
                # Invoke the agent with the user's query
                response = agent_executor.invoke({"input": user_query})
                st.subheader("Assistant's Response:")
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Please ensure your OPENAI_API_KEY is correctly set in a `.env` file.")
    else:
        st.warning("Please enter a query to get a response.")

st.sidebar.header("How to Use")
st.sidebar.markdown("""
1.  **Enter your query:** Describe the patient's symptoms, a medical question, or a request for information.
2.  **Click \"Get Assistant's Response\":** The AI will analyze your query and use its specialized tools to provide an informed answer.
3.  **Example Queries:**
    *   \"What are the common treatments for Type 2 Diabetes?\"
    *   \"Patient PAT001 has a history of high blood pressure and recent lab results show elevated blood glucose. Can you provide their current medications from EHR?\"
    *   \"I have an image URL 'chest-xray-PAT001'. Can you analyze it?\"
""")

st.sidebar.header("Disclaimer")
st.sidebar.warning("""
This AI assistant is for informational and educational purposes only and should not be considered medical advice. Always consult with a qualified healthcare professional for diagnosis and treatment. The tools integrated are simulated for demonstration purposes.
""")