import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
import requests
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool


# --- Configuration (Placeholder values) ---
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
FASTAPI_PORT = 8000

# --- Simulated External Knowledge Sources ---
medical_database = {
    "ICD10": {
        "J11": "Influenza, virus not identified",
        "I10": "Essential (primary) hypertension",
        "E11": "Type 2 diabetes mellitus"
    },
    "Drugs": {
        "Paracetamol": {"class": "Analgesic", "usage": "Pain relief, fever reduction"},
        "Lisinopril": {"class": "ACE inhibitor", "usage": "Hypertension treatment"}
    }
}

medical_knowledge_graph = {
    "Influenza": {
        "symptoms": ["fever", "cough", "sore throat", "muscle aches"],
        "treatments": ["antivirals", "rest", "fluids"]
    },
    "Hypertension": {
        "risk_factors": ["obesity", "smoking", "high sodium intake"],
        "treatments": ["ACE inhibitors", "diuretics", "lifestyle changes"]
    }
}

def get_ehr_data_simulate(patient_id: str) -> Dict[str, Any]:
    logger.info(f"Simulating EHR data retrieval for patient_id: {patient_id}")
    if patient_id == "P123":
        return {
            "patient_id": "P123",
            "name": "John Doe",
            "age": 55,
            "conditions": ["Hypertension", "Type 2 Diabetes"],
            "medications": ["Lisinopril", "Metformin"],
            "allergies": ["Penicillin"],
            "last_visit_date": "2023-10-26"
        }
    return {}

# --- Langchain Tools ---
@tool
def medical_db_query(query_type: str, item_code: str = None, item_name: str = None) -> Dict[str, Any]:
    """Queries the structured medical database for ICD codes or drug information.
    Use query_type='ICD10' with item_code for ICD codes. Use query_type='Drugs' with item_name for drug info."""
    logger.info(f"Querying medical DB: type={query_type}, code={item_code}, name={item_name}")
    if query_type == "ICD10" and item_code:
        return {item_code: medical_database["ICD10"].get(item_code, "Not found")}
    elif query_type == "Drugs" and item_name:
        return {item_name: medical_database["Drugs"].get(item_name, "Not found")}
    return {"error": "Invalid query type or missing item identifier."}

@tool
def knowledge_graph_query(concept: str) -> Dict[str, Any]:
    """Queries the medical knowledge graph for related information like symptoms, treatments, or risk factors.
    Provide a medical concept like 'Influenza' or 'Hypertension'.""" 
    logger.info(f"Querying knowledge graph for: {concept}")
    return {concept: medical_knowledge_graph.get(concept, "Not found in KG")}

@tool
def ehr_data_retrieval(patient_id: str) -> Dict[str, Any]:
    """Retrieves real-time Electronic Health Record (EHR) data for a given patient_id.
    This tool is critical for personalized patient context."""
    logger.info(f"Retrieving EHR data for patient: {patient_id}")
    return get_ehr_data_simulate(patient_id)

# --- LLM and Agent Setup ---
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

tools = [
    medical_db_query,
    knowledge_graph_query,
    ehr_data_retrieval
]

# Define the prompt for the agent
prompt_template = PromptTemplate.from_template(
    """You are a highly skilled Medical Diagnosis Assistant. Your goal is to assist healthcare professionals by providing accurate diagnostic suggestions and treatment plans based on patient information and by leveraging external medical knowledge.

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

Patient Information: {patient_info}

Symptoms: {symptoms}

Question: Based on the provided patient information and symptoms, what is a likely diagnosis and a recommended treatment plan? Also, identify any potential drug interactions or contraindications given the patient's current medications and allergies.
Thought:{agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- FastAPI Backend ---
app = FastAPI()

class PatientData(BaseModel):
    patient_id: str
    symptoms: str
    patient_history: str = ""
    lab_results: str = ""

@app.post("/diagnose")
async def diagnose_patient(data: PatientData):
    logger.info(f"Received diagnosis request for patient: {data.patient_id}")
    try:
        patient_info_string = f"Patient ID: {data.patient_id}\nHistory: {data.patient_history}\nLab Results: {data.lab_results}"
        
        response = agent_executor.invoke({
            "patient_info": patient_info_string,
            "symptoms": data.symptoms
        })
        
        return {"diagnosis": response["output"]}
    except Exception as e:
        logger.error(f"Error during diagnosis: {e}")
        return {"error": str(e), "diagnosis": "Could not process diagnosis due to an internal error."}

# --- Streamlit Frontend ---
def streamlit_app():
    st.set_page_config(layout="wide")
    st.title("🧠 Medical Diagnosis Assistant")
    st.write("This AI assistant helps healthcare professionals by augmenting LLM capabilities with external medical knowledge.")

    st.header("Patient Information")
    patient_id = st.text_input("Patient ID", "P123")
    symptoms = st.text_area("Symptoms (e.g., fever, cough, fatigue)", "Patient reports persistent cough, mild fever, and muscle aches for 3 days.")
    patient_history = st.text_area("Patient History (e.g., pre-existing conditions, past surgeries)", "Patient has a history of hypertension and Type 2 diabetes.")
    lab_results = st.text_area("Lab Results (e.g., CBC, X-ray findings)", "CBC within normal limits. Chest X-ray shows mild bronchial inflammation.")

    if st.button("Get Diagnosis and Treatment Plan"):
        if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
            st.error("Please set your OPENAI_API_KEY.")
            st.stop()

        st.info("Querying AI assistant...")
        try:
            payload = {
                "patient_id": patient_id,
                "symptoms": symptoms,
                "patient_history": patient_history,
                "lab_results": lab_results
            }
            
            # Assuming FastAPI is running on localhost:8000
            fastapi_url = f"http://localhost:{FASTAPI_PORT}/diagnose"
            response = requests.post(fastapi_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.subheader("AI-Assisted Diagnosis and Treatment Plan:")
                st.write(result.get("diagnosis", "No diagnosis provided."))
            else:
                st.error(f"Error from backend: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to FastAPI backend. Please ensure it's running on port {FASTAPI_PORT}.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- Main Execution (Instructions for running) ---
if __name__ == "__main__":
    # To run this application:
    # 1. Save the file as main.py
    # 2. Install necessary libraries: pip install streamlit fastapi uvicorn requests loguru langchain-openai
    # 3. Set your OPENAI_API_KEY environment variable or replace "YOUR_OPENAI_API_KEY" above.
    
    # To run the FastAPI backend (in one terminal):
    # uvicorn main:app --host 0.0.0.0 --port 8000
    
    # To run the Streamlit frontend (in another terminal):
    # streamlit run main.py --server.port 8501
    
    # Note: For this single-file setup, you'll need two separate terminal commands.
    # The streamlit_app() function will be executed when you run 'streamlit run main.py'
    # The FastAPI 'app' object will be served when you run 'uvicorn main:app'
    streamlit_app()
