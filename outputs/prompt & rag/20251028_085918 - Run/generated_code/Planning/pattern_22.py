import streamlit as st
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

# LangChain/LangGraph/DSPy components (simplified)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
import operator
from typing import TypedDict, Annotated

# Mock LLM and DSPy for demonstration without actual API keys
class MockLLM:
    def invoke(self, messages):
        last_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        logger.info(f"MockLLM received: {last_user_message[:100]}...")

        if "plan a medication regimen" in last_user_message.lower():
            return HumanMessage(content="{\"medication_plan\": [{\"medication_name\": \"Metformin\", \"dosage\": \"500mg\", \"frequency\": \"twice daily\", \"time_of_day\": \"morning, evening\"}, {\"medication_name\": \"Lisinopril\", \"dosage\": \"10mg\", \"frequency\": \"once daily\", \"time_of_day\": \"morning\"}]}")
        elif "plan a diet" in last_user_message.lower():
            return HumanMessage(content="{\"diet_plan\": [{\"meal_type\": \"Breakfast\", \"food_items\": \"Oatmeal, Berries\", \"calories\": 300}, {\"meal_type\": \"Lunch\", \"food_items\": \"Grilled Chicken Salad\", \"calories\": 500}]}")
        elif "plan an exercise routine" in last_user_message.lower():
            return HumanMessage(content="{\"exercise_plan\": [{\"activity\": \"Walking\", \"duration_minutes\": 30, \"frequency\": \"daily\"}]}")
        elif "reflect on the plan" in last_user_message.lower():
            return HumanMessage(content="The initial plan seems reasonable, but ensure diet adheres to specific patient allergies. Consider adding a flexibility option.")
        elif "adapt the plan" in last_user_message.lower():
            return HumanMessage(content="{\"medication_plan\": [{\"medication_name\": \"Metformin\", \"dosage\": \"500mg\", \"frequency\": \"twice daily\", \"time_of_day\": \"morning, evening\"}], \"diet_plan\": [{\"meal_type\": \"Breakfast\", \"food_items\": \"Oatmeal, Berries (low sugar)\", \"calories\": 300}, {\"meal_type\": \"Lunch\", \"food_items\": \"Grilled Chicken Salad\", \"calories\": 500}], \"exercise_plan\": [{\"activity\": \"Walking\", \"duration_minutes\": 30, \"frequency\": \"daily\"}], \"notes\": \"Adjusted diet for lower sugar intake based on patient feedback.\"}")
        return HumanMessage(content=f"Mock LLM response for: {last_user_message}")

llm = MockLLM()

# DSPy-like components (minimal implementation for demonstration)
class PlanSignature(LCBaseModel):
    plan_type: str = Field(description="Type of plan to generate (medication, diet, exercise)")
    patient_context: str = Field(description="Relevant patient medical history, preferences, and current symptoms")
    medical_guidelines: str = Field(description="Relevant medical guidelines for the chronic disease")
    output_plan_json: str = Field(description="JSON string of the generated plan based on the plan_type")

class PredictPlan:
    def __init__(self, llm_model):
        self.llm = llm_model

    def __call__(self, plan_type: str, patient_context: str, medical_guidelines: str) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"You are an expert AI assistant for chronic disease management. Generate a detailed {plan_type} plan."),
            ("human", f"Plan a {plan_type} regimen for a patient with the following context: {patient_context}. Adhere to these medical guidelines: {medical_guidelines}. Output the plan as a JSON string for {plan_type}.")
        ])
        messages = prompt_template.format_messages(
            plan_type=plan_type,
            patient_context=patient_context,
            medical_guidelines=medical_guidelines
        )
        response_message = self.llm.invoke(messages)
        return response_message.content

predict_plan = PredictPlan(llm)

# Load environment variables
load_dotenv()

# --- Pydantic Models for Data Validation ---
class Medication(BaseModel):
    medication_name: str
    dosage: str
    frequency: str
    time_of_day: str

class MedicationPlan(BaseModel):
    medication_plan: List[Medication]

class DietItem(BaseModel):
    meal_type: str
    food_items: str
    calories: Optional[int]

class DietPlan(BaseModel):
    diet_plan: List[DietItem]

class ExerciseActivity(BaseModel):
    activity: str
    duration_minutes: int
    frequency: str

class ExercisePlan(BaseModel):
    exercise_plan: List[ExerciseActivity]

class PatientProfile(BaseModel):
    patient_id: str
    name: str
    age: int
    chronic_disease: str
    allergies: List[str] = []
    dietary_restrictions: List[str] = []
    physical_limitations: List[str] = []
    current_symptoms: List[str] = []
    medication_history: List[str] = []
    last_feedback: Optional[str] = None

class MedicalFact(BaseModel):
    topic: str
    content: str
    disease: Optional[str] = None

class OverallPlan(BaseModel):
    medication_plan: Optional[MedicationPlan] = None
    diet_plan: Optional[DietPlan] = None
    exercise_plan: Optional[ExercisePlan] = None
    notes: Optional[str] = None

# --- ChromaDB Setup ---
CHROMA_PATH = "./chroma_db"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    patients_collection = client.get_or_create_collection(
        name="patients",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    medical_collection = client.get_or_create_collection(
        name="medical_knowledge",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    st.error(f"Error initializing ChromaDB: {e}")
    st.stop()

# Add some initial data to ChromaDB (if not already present)
def add_initial_data():
    if patients_collection.count() == 0:
        logger.info("Adding initial patient data...")
        patient_data = PatientProfile(
            patient_id="patient_001",
            name="Alice Smith",
            age=55,
            chronic_disease="Type 2 Diabetes",
            allergies=["Penicillin"],
            dietary_restrictions=["low sugar", "gluten-free"],
            physical_limitations=["knee pain"],
            current_symptoms=["high blood sugar", "fatigue"],
            medication_history=["Metformin"]
        )
        patients_collection.add(
            documents=[patient_data.json()],
            metadatas=[{"patient_id": patient_data.patient_id, "type": "patient_profile"}],
            ids=[patient_data.patient_id]
        )
        logger.info("Initial patient data added.")

    if medical_collection.count() == 0:
        logger.info("Adding initial medical knowledge...")
        medical_facts = [
            MedicalFact(topic="Type 2 Diabetes Management", content="Regular exercise and a balanced low-sugar diet are crucial for managing Type 2 Diabetes.", disease="Type 2 Diabetes"),
            MedicalFact(topic="Metformin Guidelines", content="Metformin is a common medication for Type 2 Diabetes, typically taken with meals.", disease="Type 2 Diabetes"),
            MedicalFact(topic="Low Sugar Diet", content="Focus on whole grains, lean proteins, and plenty of vegetables. Avoid processed foods and sugary drinks.")
        ]
        docs = [fact.json() for fact in medical_facts]
        metas = [{"topic": fact.topic, "disease": fact.disease, "type": "medical_fact"} for fact in medical_facts]
        ids = [f"medical_fact_{i}" for i in range(len(medical_facts))]
        medical_collection.add(documents=docs, metadatas=metas, ids=ids)
        logger.info("Initial medical knowledge added.")

add_initial_data()

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    patient_id: str
    patient_profile: Optional[PatientProfile]
    medical_knowledge: List[str]
    user_input: str
    current_plan: Optional[OverallPlan]
    feedback: Optional[str]
    log: Annotated[List[str], operator.add]

# --- LangGraph Tools ---
@tool
def retrieve_patient_profile_tool(patient_id: str) -> str:
    try:
        results = patients_collection.get(ids=[patient_id], include=['documents'])
        if results['documents']:
            logger.info(f"Retrieved patient profile for {patient_id}")
            return results['documents'][0]
        logger.warning(f"Patient profile not found for {patient_id}")
        return "No patient profile found."
    except Exception as e:
        logger.error(f"Error retrieving patient profile: {e}")
        return f"Error: {e}"

@tool
def retrieve_medical_knowledge_tool(query: str, disease: Optional[str] = None) -> str:
    try:
        where_clause = {"disease": disease} if disease else None
        results = medical_collection.query(
            query_texts=[query],
            n_results=3,
            where=where_clause,
            include=['documents']
        )
        if results['documents']:
            logger.info(f"Retrieved medical knowledge for query: {query}")
            return "\n".join(results['documents'][0])
        logger.warning(f"No medical knowledge found for query: {query}")
        return "No medical knowledge found."
    except Exception as e:
        logger.error(f"Error retrieving medical knowledge: {e}")
        return f"Error: {e}"

@tool
def send_reminder_tool(patient_id: str, message: str) -> str:
    logger.info(f"Simulating sending reminder to {patient_id}: {message}")
    return f"Reminder sent to {patient_id}: {message}"

@tool
def schedule_appointment_tool(patient_id: str, appointment_details: str) -> str:
    logger.info(f"Simulating scheduling appointment for {patient_id}: {appointment_details}")
    return f"Appointment scheduled for {patient_id}: {appointment_details}"

# --- LangGraph Nodes ---
def fetch_patient_and_medical_data(state: AgentState) -> AgentState:
    patient_id = state["patient_id"]
    patient_profile_json = retrieve_patient_profile_tool.invoke({"patient_id": patient_id})
    patient_profile = None
    if patient_profile_json and patient_profile_json != "No patient profile found.":
        try:
            patient_profile = PatientProfile.parse_raw(patient_profile_json)
            logger.info(f"Parsed patient profile for {patient_id}")
            st.session_state.logs.append(f"Fetched patient profile for {patient_id}.")
        except ValidationError as e:
            logger.error(f"Validation error for patient profile: {e}")
            st.session_state.logs.append(f"Error validating patient profile: {e}")
            patient_profile = None
    
    medical_knowledge = []
    if patient_profile:
        disease_knowledge = retrieve_medical_knowledge_tool.invoke({"query": patient_profile.chronic_disease, "disease": patient_profile.chronic_disease})
        general_management_knowledge = retrieve_medical_knowledge_tool.invoke({"query": f"{patient_profile.chronic_disease} management"})
        if disease_knowledge and disease_knowledge != "No medical knowledge found.":
            medical_knowledge.append(disease_knowledge)
            st.session_state.logs.append(f"Fetched disease-specific knowledge for {patient_profile.chronic_disease}.")
        if general_management_knowledge and general_management_knowledge != "No medical knowledge found.":
            medical_knowledge.append(general_management_knowledge)
            st.session_state.logs.append(f"Fetched general management knowledge.")

    return {
        "patient_profile": patient_profile,
        "medical_knowledge": medical_knowledge,
        "log": [f"Fetched data for {patient_id}."]
    }

def generate_initial_plan(state: AgentState) -> AgentState:
    patient_profile = state["patient_profile"]
    medical_knowledge = state["medical_knowledge"]

    if not patient_profile or not medical_knowledge:
        st.session_state.logs.append("Error: Missing patient profile or medical knowledge for planning.")
        return {"log": ["Planning failed: missing data."]}

    patient_context_str = patient_profile.json()
    medical_guidelines_str = "\n".join(medical_knowledge)

    medication_plan_json = predict_plan(plan_type="medication", patient_context=patient_context_str, medical_guidelines=medical_guidelines_str)
    diet_plan_json = predict_plan(plan_type="diet", patient_context=patient_context_str, medical_guidelines=medical_guidelines_str)
    exercise_plan_json = predict_plan(plan_type="exercise", patient_context=patient_context_str, medical_guidelines=medical_guidelines_str)

    current_plan = OverallPlan()
    try:
        current_plan.medication_plan = MedicationPlan.parse_raw(medication_plan_json)
        st.session_state.logs.append("Generated medication plan.")
    except ValidationError as e:
        st.session_state.logs.append(f"Medication plan validation error: {e}")
        logger.error(f"Medication plan validation error: {e}")
    try:
        current_plan.diet_plan = DietPlan.parse_raw(diet_plan_json)
        st.session_state.logs.append("Generated diet plan.")
    except ValidationError as e:
        st.session_state.logs.append(f"Diet plan validation error: {e}")
        logger.error(f"Diet plan validation error: {e}")
    try:
        current_plan.exercise_plan = ExercisePlan.parse_raw(exercise_plan_json)
        st.session_state.logs.append("Generated exercise plan.")
    except ValidationError as e:
        st.session_state.logs.append(f"Exercise plan validation error: {e}")
        logger.error(f"Exercise plan validation error: {e}")

    return {"current_plan": current_plan, "log": ["Generated initial plan."]}

def check_constraints_and_reflect(state: AgentState) -> AgentState:
    patient_profile = state["patient_profile"]
    current_plan = state["current_plan"]
    log_messages = []
    reflection_needed = False
    reflection_notes = []

    if not patient_profile or not current_plan:
        log_messages.append("Error: Missing patient profile or plan for constraint checking/reflection.")
        return {"log": log_messages}

    # Constraint checking (Pydantic validation is already done during plan generation)
    # Additional custom logic for constraints
    if current_plan.diet_plan:
        for item in current_plan.diet_plan.diet_plan:
            if "low sugar" in patient_profile.dietary_restrictions and "sugar" in item.food_items.lower():
                log_messages.append(f"Constraint alert: Diet item '{item.food_items}' might violate low sugar restriction.")
                reflection_needed = True
                reflection_notes.append(f"Diet item '{item.food_items}' might violate low sugar restriction.")
            if "gluten-free" in patient_profile.dietary_restrictions and any(g in item.food_items.lower() for g in ["bread", "pasta", "wheat"]):
                log_messages.append(f"Constraint alert: Diet item '{item.food_items}' might violate gluten-free restriction.")
                reflection_needed = True
                reflection_notes.append(f"Diet item '{item.food_items}' might violate gluten-free restriction.")

    if current_plan.exercise_plan:
        for activity in current_plan.exercise_plan.exercise_plan:
            if "knee pain" in patient_profile.physical_limitations and activity.activity.lower() == "running":
                log_messages.append(f"Constraint alert: Exercise '{activity.activity}' might be unsuitable due to knee pain.")
                reflection_needed = True
                reflection_notes.append(f"Exercise '{activity.activity}' might be unsuitable due to knee pain.")

    # Introspective reasoning (mock LLM call)
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that critically reviews health plans. Identify potential issues, inconsistencies, or areas for improvement given patient context."),
        ("human", f"Review the following plan for patient {patient_profile.name} with {patient_profile.chronic_disease}. Patient Profile: {patient_profile.json()}. Current Plan: {current_plan.json()}. Any specific issues identified: {', '.join(reflection_notes) if reflection_notes else 'None'}. Provide a concise reflection.")
    ])
    reflection_message = llm.invoke(reflection_prompt.format_messages())
    log_messages.append(f"Introspection: {reflection_message.content}")

    if reflection_needed or "needs refinement" in reflection_message.content.lower(): # Simple check for reflection indicating refinement
        st.session_state.logs.append("Plan needs refinement based on constraints or introspection.")
        return {"log": log_messages, "reflection_needed": True}
    
    st.session_state.logs.append("Plan passed constraint checks and introspection.")
    return {"log": log_messages, "reflection_needed": False}

def adapt_plan_extrospectively(state: AgentState) -> AgentState:
    patient_profile = state["patient_profile"]
    current_plan = state["current_plan"]
    feedback = state["feedback"]
    log_messages = []

    if not patient_profile or not current_plan:
        log_messages.append("Error: Missing patient profile or plan for adaptation.")
        return {"log": log_messages}

    adaptation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an adaptive AI assistant for chronic disease management. Refine the existing plan based on new patient feedback or identified issues."),
        ("human", f"Patient Profile: {patient_profile.json()}. Current Plan: {current_plan.json()}. New Feedback/Issues: {feedback if feedback else 'None provided, refine based on previous introspection.'}. Generate an adapted overall plan in JSON format. Include a 'notes' field for changes.")
    ])
    adapted_plan_json = llm.invoke(adaptation_prompt.format_messages()).content
    
    try:
        adapted_plan = OverallPlan.parse_raw(adapted_plan_json)
        st.session_state.logs.append("Adapted plan based on feedback/introspection.")
        return {"current_plan": adapted_plan, "log": log_messages + ["Plan adapted."]}
    except ValidationError as e:
        st.session_state.logs.append(f"Adapted plan validation error: {e}")
        logger.error(f"Adapted plan validation error: {e}")
        return {"log": log_messages + ["Plan adaptation failed due to validation error."]}

def execute_actions(state: AgentState) -> AgentState:
    patient_profile = state["patient_profile"]
    current_plan = state["current_plan"]
    log_messages = []

    if not patient_profile or not current_plan:
        log_messages.append("Error: Missing patient profile or plan for execution.")
        return {"log": log_messages}

    if current_plan.medication_plan:
        for med in current_plan.medication_plan.medication_plan:
            reminder_msg = f"Time for {med.medication_name} {med.dosage} ({med.frequency}, {med.time_of_day})."
            send_reminder_tool.invoke({"patient_id": patient_profile.patient_id, "message": reminder_msg})
            st.session_state.logs.append(f"Sent medication reminder for {med.medication_name}.")

    if current_plan.diet_plan:
        diet_summary = ", ".join([f"{item.meal_type}: {item.food_items}" for item in current_plan.diet_plan.diet_plan])
        send_reminder_tool.invoke({"patient_id": patient_profile.patient_id, "message": f"Remember your diet today: {diet_summary}"})
        st.session_state.logs.append("Sent diet reminder.")

    if current_plan.exercise_plan:
        exercise_summary = ", ".join([f"{act.activity} for {act.duration_minutes} min ({act.frequency})" for act in current_plan.exercise_plan.exercise_plan])
        send_reminder_tool.invoke({"patient_id": patient_profile.patient_id, "message": f"Don't forget your exercise: {exercise_summary}"})
        st.session_state.logs.append("Sent exercise reminder.")

    schedule_appointment_tool.invoke({"patient_id": patient_profile.patient_id, "appointment_details": "Next check-up in 3 months"})
    st.session_state.logs.append("Scheduled follow-up appointment.")

    return {"log": log_messages + ["Actions executed."]}

# --- LangGraph Graph Definition ---
workflow = StateGraph(AgentState)

workflow.add_node("fetch_data", fetch_patient_and_medical_data)
workflow.add_node("generate_plan", generate_initial_plan)
workflow.add_node("check_and_reflect", check_constraints_and_reflect)
workflow.add_node("adapt_plan", adapt_plan_extrospectively)
workflow.add_node("execute_actions", execute_actions)

workflow.set_entry_point("fetch_data")

workflow.add_edge("fetch_data", "generate_plan")
workflow.add_edge("generate_plan", "check_and_reflect")

workflow.add_conditional_edges(
    "check_and_reflect",
    lambda state: "adapt_plan" if state.get("reflection_needed") else "execute_actions",
    {
        "adapt_plan": "adapt_plan",
        "execute_actions": "execute_actions",
    },
)
workflow.add_edge("adapt_plan", "execute_actions")
workflow.add_edge("execute_actions", END)

app_graph = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Chronic Disease Assistant")
st.title("AI-Powered Chronic Disease Management Assistant")

if "logs" not in st.session_state:
    st.session_state.logs = []
if "current_plan_display" not in st.session_state:
    st.session_state.current_plan_display = None

def display_plan(plan: OverallPlan):
    st.subheader("Generated Health Plan")
    if plan.medication_plan and plan.medication_plan.medication_plan:
        st.write("#### Medication Plan")
        meds_df = pd.DataFrame([m.dict() for m in plan.medication_plan.medication_plan])
        st.dataframe(meds_df)
    if plan.diet_plan and plan.diet_plan.diet_plan:
        st.write("#### Diet Plan")
        diet_df = pd.DataFrame([d.dict() for d in plan.diet_plan.diet_plan])
        st.dataframe(diet_df)
    if plan.exercise_plan and plan.exercise_plan.exercise_plan:
        st.write("#### Exercise Plan")
        ex_df = pd.DataFrame([e.dict() for e in plan.exercise_plan.exercise_plan])
        st.dataframe(ex_df)
    if plan.notes:
        st.write("#### Notes")
        st.info(plan.notes)

with st.sidebar:
    st.header("Patient Information")
    patient_id_input = st.text_input("Patient ID", value="patient_001")
    user_feedback_input = st.text_area("Patient Feedback/New Symptoms", help="e.g., 'Experienced mild stomach upset after Metformin', 'Found the exercise routine too strenuous.'")

    if st.button("Generate/Update Plan"):
        st.session_state.logs = []
        initial_state = {
            "patient_id": patient_id_input,
            "user_input": "Generate personalized health plan.",
            "feedback": user_feedback_input,
            "log": [],
        }
        
        st.session_state.logs.append("Starting plan generation/update...")
        try:
            # The LangGraph will automatically run through its nodes
            # We iterate to see intermediate states if needed, but for a simple flow, one run is often enough.
            # For this simplified example, we'll just run to completion.
            final_state = None
            for s in app_graph.stream(initial_state, config=RunnableConfig(recursion_limit=50)):
                final_state = s
                for key, value in s.items():
                    if key != '__end__':
                        logger.info(f"Intermediate state ({key}): {value}")
                        if "log" in value and value["log"]:
                            for msg in value["log"]:
                                if msg not in st.session_state.logs: # Avoid duplicate logs
                                    st.session_state.logs.append(msg)
            
            if final_state and "__end__" in final_state:
                final_data = final_state["__end__"]
                st.session_state.current_plan_display = final_data.get("current_plan")
                st.success("Plan generation/update complete!")
            else:
                st.error("Plan generation/update failed or did not complete.")

        except Exception as e:
            st.error(f"An error occurred during plan processing: {e}")
            st.session_state.logs.append(f"Error: {e}")
            logger.error(f"Error in graph execution: {e}")

    st.header("Application Logs")
    for log_entry in reversed(st.session_state.logs):
        st.text(log_entry)

# Main content area
if st.session_state.current_plan_display:
    display_plan(st.session_state.current_plan_display)
else:
    st.info("Enter patient ID and click 'Generate/Update Plan' to get started.")
