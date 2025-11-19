import os
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
import json

# LangChain specific imports
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# Streamlit for UI
import streamlit as st

# --- 0. Configuration and Initialization ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it.")
    st.stop()

logger.add("app.log", rotation="10 MB")
logger.info("Application started.")

# Placeholder for ChromaDB persistence
CHROMA_DB_DIR = "./chroma_db"

# --- 1. Pydantic Models for Constraint Satisfaction ---
class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication.")
    dosage: str = Field(..., description="Dosage instructions for the medication (e.g., '10mg daily').")
    frequency: str = Field(..., description="Frequency of medication intake (e.g., 'once a day', 'twice a day').")
    notes: Optional[str] = Field(None, description="Any additional notes or instructions for the medication.")

class LifestyleRecommendation(BaseModel):
    type: str = Field(..., description="Type of lifestyle recommendation (e.g., 'Diet', 'Exercise', 'Stress Management').")
    details: str = Field(..., description="Specific details of the recommendation (e.g., 'Low-sodium diet', '30 minutes walking daily').")

class TreatmentPlan(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    conditions: List[str] = Field(..., description="List of chronic conditions being addressed.")
    medications: List[Medication] = Field(..., description="List of prescribed medications.")
    lifestyle_recommendations: List[LifestyleRecommendation] = Field(..., description="List of lifestyle recommendations.")
    goals: List[str] = Field(..., description="Treatment goals for the patient.")
    notes: Optional[str] = Field(None, description="General notes or additional information for the plan.")

class PatientProfile(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    age: int = Field(..., gt=0, description="Patient's age.")
    gender: str = Field(..., description="Patient's gender (e.g., 'Male', 'Female', 'Other').")
    chronic_conditions: List[str] = Field(..., description="List of chronic conditions diagnosed.")
    allergies: List[str] = Field(default_factory=list, description="List of patient's allergies.")
    current_medications: List[str] = Field(default_factory=list, description="List of medications the patient is currently taking.")
    health_goals: List[str] = Field(default_factory=list, description="Patient's health goals.")
    recent_lab_results: Dict[str, Any] = Field(default_factory=dict, description="Recent lab results (e.g., {'Blood Glucose': '120 mg/dL'}).")

# --- 2. Knowledge Base and RAG Setup ---
def load_and_split_docs(file_path: str) -> List[Any]:
    if not os.path.exists(file_path):
        logger.warning(f"Medical guidelines file not found at {file_path}. Creating a placeholder.")
        with open(file_path, "w") as f:
            f.write("General guidelines for diabetes management: monitor blood sugar, regular exercise, healthy diet, consult doctor for medication. Hypertension management: reduce sodium, regular exercise, prescribed medication. Allergy management: avoid known allergens, antihistamines. Always consult a medical professional.")

    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    logger.info(f"Loaded and split {len(docs)} documents from {file_path}.")
    return docs

def get_vector_store(docs: List[Any], collection_name: str = "medical_guidelines_collection") -> Chroma:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        logger.info(f"Loading existing ChromaDB from {CHROMA_DB_DIR}")
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=collection_name)
    else:
        logger.info(f"Creating new ChromaDB at {CHROMA_DB_DIR}")
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_DIR, collection_name=collection_name)
        vectorstore.persist()
    return vectorstore

MEDICAL_GUIDELINES_FILE = "medical_guidelines.txt"
medical_docs = load_and_split_docs(MEDICAL_GUIDELINES_FILE)
vector_store = get_vector_store(medical_docs)
retriever = vector_store.as_retriever()

# --- 3. LLM Orchestration and Reasoning Engine ---
llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# Define tools for the agent
@tool
def retrieve_medical_info(query: str) -> str:
    """Searches the medical knowledge base for relevant information based on the query."""
    logger.info(f"Retrieving medical info for query: {query}")
    docs = retriever.get_relevant_documents(query)
    return "\n---\n".join([doc.page_content for doc in docs])

@tool
def validate_treatment_plan(plan_json: str) -> str:
    """Validates a treatment plan against predefined medical constraints using Pydantic models.
    Input should be a JSON string representing a TreatmentPlan model. Returns 'Valid' or an error message."""
    logger.info("Attempting to validate treatment plan.")
    try:
        plan_data = json.loads(plan_json)
        TreatmentPlan(**plan_data)
        logger.info("Treatment plan validated successfully.")
        return "Valid"
    except ValidationError as e:
        logger.error(f"Treatment plan validation failed: {e.json()}")
        return f"Validation Error: {e.json()}"
    except json.JSONDecodeError:
        logger.error("Invalid JSON format for treatment plan.")
        return "Validation Error: Invalid JSON format."

tools = [retrieve_medical_info, validate_treatment_plan]

# Prompt templates for the agent
PLANNING_SYSTEM_MESSAGE = PromptTemplate.from_template(
    "You are an AI assistant specialized in generating personalized treatment plans for chronic disease management. "
    "You have access to a medical knowledge base and a plan validation tool. "
    "Your goal is to create a comprehensive, safe, and effective treatment plan for the patient, considering their profile and medical context. "
    "Always validate the generated plan using the 'validate_treatment_plan' tool before finalizing it. "
    "Ensure the output is a JSON string conforming to the TreatmentPlan Pydantic model. "
    "If validation fails, explain why and attempt to correct the plan."
)

ADAPTATION_SYSTEM_MESSAGE = PromptTemplate.from_template(
    "You are an AI assistant specialized in adapting personalized treatment plans for chronic disease management. "
    "You have access to a medical knowledge base and a plan validation tool. "
    "Your goal is to adjust the current treatment plan based on new patient feedback, ensuring it remains comprehensive, safe, and effective. "
    "Always validate the adapted plan using the 'validate_treatment_plan' tool before finalizing it. "
    "Ensure the output is a JSON string conforming to the TreatmentPlan Pydantic model. "
    "If validation fails, explain why and attempt to correct the plan."
)

# Function to create an agent executor
def create_plan_agent(system_message_template: PromptTemplate, patient_profile: PatientProfile, medical_context: str) -> AgentExecutor:
    prompt = system_message_template.partial(patient_info=patient_profile.json(), medical_context=medical_context)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor

# --- 4. Core Logic Functions ---
def generate_initial_plan_llm(patient_profile: PatientProfile) -> Optional[TreatmentPlan]:
    logger.info(f"Generating initial plan for patient: {patient_profile.patient_id}")
    medical_context = retrieve_medical_info(f"Guidelines for {', '.join(patient_profile.chronic_conditions)} and patient's allergies: {', '.join(patient_profile.allergies)}")
    
    initial_prompt_content = f"Generate a personalized treatment plan for the following patient:\n" \
                             f"Patient Profile: {patient_profile.json()}\n" \
                             f"Relevant Medical Context: {medical_context}\n" \
                             f"Output the plan as a JSON string conforming to the TreatmentPlan Pydantic model."
    
    # Use a simpler LLM call for direct generation, then validate
    # For direct agent usage, the agent itself would handle validation
    # Here, we'll try to get the LLM to output JSON and then validate explicitly
    chat_history = [HumanMessage(content=initial_prompt_content)]
    response = llm.invoke(chat_history)
    generated_json = response.content
    
    try:
        plan_data = json.loads(generated_json)
        validation_result = validate_treatment_plan(generated_json)
        if validation_result == "Valid":
            logger.info("Initial plan generated and validated successfully.")
            return TreatmentPlan(**plan_data)
        else:
            logger.error(f"Initial plan validation failed: {validation_result}")
            st.error(f"Failed to generate a valid initial plan: {validation_result}")
            return None
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Error parsing or validating initial plan JSON: {e}")
        st.error(f"Error parsing or validating initial plan JSON: {e}\nRaw LLM output: {generated_json}")
        return None

def adapt_plan_llm(current_plan: TreatmentPlan, feedback: str) -> Optional[TreatmentPlan]:
    logger.info(f"Adapting plan for patient {current_plan.patient_id} with feedback: {feedback}")
    medical_context = retrieve_medical_info(f"Guidelines for {', '.join(current_plan.conditions)}")

    adaptation_prompt_content = f"Current Treatment Plan: {current_plan.json()}\n" \
                                f"Patient Feedback: {feedback}\n" \
                                f"Relevant Medical Context: {medical_context}\n" \
                                f"Based on the feedback, adapt the current treatment plan. Output the adapted plan as a JSON string conforming to the TreatmentPlan Pydantic model."

    chat_history = [HumanMessage(content=adaptation_prompt_content)]
    response = llm.invoke(chat_history)
    adapted_json = response.content

    try:
        plan_data = json.loads(adapted_json)
        validation_result = validate_treatment_plan(adapted_json)
        if validation_result == "Valid":
            logger.info("Adapted plan generated and validated successfully.")
            return TreatmentPlan(**plan_data)
        else:
            logger.error(f"Adapted plan validation failed: {validation_result}")
            st.error(f"Failed to generate a valid adapted plan: {validation_result}")
            return None
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Error parsing or validating adapted plan JSON: {e}")
        st.error(f"Error parsing or validating adapted plan JSON: {e}\nRaw LLM output: {adapted_json}")
        return None


# --- 5. Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI-powered Personalized Treatment Plan Generator")
st.title("💊 AI-powered Personalized Treatment Plan Generator")
st.markdown("This application assists healthcare professionals in generating and adapting personalized treatment plans for chronic disease management using LLMs.")

# Initialize session state for the treatment plan
if "treatment_plan" not in st.session_state:
    st.session_state.treatment_plan = None

with st.sidebar:
    st.header("Patient Profile Input")
    patient_id = st.text_input("Patient ID", value="patient_001")
    age = st.number_input("Age", min_value=1, max_value=120, value=55)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    chronic_conditions_input = st.text_area("Chronic Conditions (comma-separated)", "Diabetes Type 2, Hypertension")
    allergies_input = st.text_area("Allergies (comma-separated)", "Penicillin")
    current_medications_input = st.text_area("Current Medications (comma-separated)", "Metformin, Lisinopril")
    health_goals_input = st.text_area("Health Goals (comma-separated)", "Lower blood sugar, Reduce blood pressure, Increase physical activity")
    recent_lab_results_json = st.text_area("Recent Lab Results (JSON format)", "{\"Blood Glucose\": \"180 mg/dL\", \"Blood Pressure\": \"145/90 mmHg\"}")

    generate_button = st.button("Generate Initial Treatment Plan")

    patient_profile = None
    if generate_button:
        try:
            conditions = [c.strip() for c in chronic_conditions_input.split(",") if c.strip()]
            allergies = [a.strip() for a in allergies_input.split(",") if a.strip()]
            current_meds = [m.strip() for m in current_medications_input.split(",") if m.strip()]
            goals = [g.strip() for g in health_goals_input.split(",") if g.strip()]
            lab_results = json.loads(recent_lab_results_json) if recent_lab_results_json.strip() else {}

            patient_profile = PatientProfile(
                patient_id=patient_id,
                age=age,
                gender=gender,
                chronic_conditions=conditions,
                allergies=allergies,
                current_medications=current_meds,
                health_goals=goals,
                recent_lab_results=lab_results
            )
            logger.info(f"Patient profile created: {patient_profile.json()}")

            with st.spinner("Generating initial treatment plan..."):
                st.session_state.treatment_plan = generate_initial_plan_llm(patient_profile)
                if st.session_state.treatment_plan:
                    st.success("Initial treatment plan generated successfully!")
                else:
                    st.error("Could not generate a valid initial treatment plan. Check logs for details.")

        except ValidationError as e:
            st.error(f"Invalid Patient Profile Input: {e}")
            logger.error(f"Patient Profile Validation Error: {e}")
        except json.JSONDecodeError:
            st.error("Invalid JSON format for Recent Lab Results.")
        except Exception as e:
            st.error(f"An unexpected error occurred during patient profile creation or plan generation: {e}")
            logger.error(f"Unexpected error: {e}", exc_info=True)

col1, col2 = st.columns(2)

with col1:
    st.header("Current Treatment Plan")
    if st.session_state.treatment_plan:
        plan = st.session_state.treatment_plan
        st.json(plan.dict())
        
        st.subheader("Medications")
        for med in plan.medications:
            st.markdown(f"- **{med.name}**: {med.dosage}, {med.frequency} ({med.notes or 'No notes'}) ")
        
        st.subheader("Lifestyle Recommendations")
        for rec in plan.lifestyle_recommendations:
            st.markdown(f"- **{rec.type}**: {rec.details}")
        
        st.subheader("Goals")
        for goal in plan.goals:
            st.markdown(f"- {goal}")
        
        if plan.notes:
            st.subheader("Notes")
            st.write(plan.notes)

    else:
        st.info("No treatment plan generated yet. Please input patient details and click 'Generate Initial Treatment Plan'.")

with col2:
    st.header("Adapt Plan Based on Feedback")
    if st.session_state.treatment_plan:
        feedback_input = st.text_area("Simulate Patient Progress/Feedback", 
                                      "Patient's blood glucose is still high (220 mg/dL) despite current medication. Experiencing mild fatigue. Wants more exercise options.")
        adapt_button = st.button("Adapt Treatment Plan")

        if adapt_button:
            with st.spinner("Adapting treatment plan based on feedback..."):
                if st.session_state.treatment_plan:
                    adapted_plan = adapt_plan_llm(st.session_state.treatment_plan, feedback_input)
                    if adapted_plan:
                        st.session_state.treatment_plan = adapted_plan
                        st.success("Treatment plan adapted successfully!")
                    else:
                        st.error("Could not adapt the treatment plan. Check logs for details.")
                else:
                    st.warning("Please generate an initial plan first.")
    else:
        st.info("Generate an initial plan first to adapt it.")
