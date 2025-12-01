import os
import requests
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# --- 1. Structured Data Store (Pydantic Models & SQLAlchemy) ---
Base = declarative_base()

class UserProfileDB(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)
    medical_conditions = Column(String)
    allergies = Column(String)
    preferences = Column(String)

class MedicationDB(Base):
    __tablename__ = "medications"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    name = Column(String)
    dosage = Column(String)
    schedule = Column(String)
    last_refill = Column(DateTime)

class AppointmentDB(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    doctor = Column(String)
    date_time = Column(DateTime)
    reason = Column(String)

# Pydantic models for data validation and API interaction
class UserProfile(BaseModel):
    name: str
    age: int
    medical_conditions: str = "None"
    allergies: str = "None"
    preferences: str = "None"

class Medication(BaseModel):
    user_id: int
    name: str
    dosage: str
    schedule: str
    last_refill: Optional[datetime] = None

class Appointment(BaseModel):
    user_id: int
    doctor: str
    date_time: datetime
    reason: str

# SQLite Database Setup
DATABASE_URL = "sqlite:///health_assistant.db"
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- 2. Long-Term Health Record (ChromaDB & Sentence Transformers) ---
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

def add_health_record_to_vectordb(user_id: int, record: str):
    vector_db.add_texts(texts=[f"User {user_id}: {record}"], metadatas=[{"user_id": user_id}])
    vector_db.persist()

def retrieve_health_records_from_vectordb(user_id: int, query: str, k: int = 3) -> List[str]:
    docs = vector_db.similarity_search(f"User {user_id}: {query}", k=k)
    return [doc.page_content for doc in docs]

# --- 3. Tool Use Module (Langchain Tools) ---
@tool
def get_wearable_data(user_id: int, data_type: str, period: str = "daily") -> str:
    """Fetches simulated wearable data (e.g., steps, heart rate, sleep) for a given user and data type. 
    Args: user_id (int), data_type (str, e.g., 'steps', 'heart_rate', 'sleep'), period (str, e.g., 'daily', 'weekly')."""
    print(f"Simulating fetching {period} {data_type} data for user {user_id}...")
    # In a real app, this would call Apple Health/Fitbit APIs
    if data_type == "steps":
        return f"User {user_id} walked 8500 steps {period}."
    elif data_type == "heart_rate":
        return f"User {user_id} average heart rate was 72 bpm {period}."
    elif data_type == "sleep":
        return f"User {user_id} slept 7.5 hours {period}."
    return f"Could not retrieve {data_type} data."

@tool
def get_medical_info(query: str) -> str:
    """Retrieves simulated medical information for a given query (e.g., drug interactions, symptom descriptions).
    Args: query (str)."""
    print(f"Simulating medical information retrieval for: {query}...")
    # In a real app, this would call OpenFDA, MedlinePlus, etc.
    if "ibuprofen side effects" in query.lower():
        return "Common side effects of ibuprofen include stomach upset, heartburn, nausea, vomiting, dizziness, and headache."
    elif "diabetes symptoms" in query.lower():
        return "Symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision."
    return f"Simulated medical info for '{query}': Information relevant to '{query}' was found."

@tool
def schedule_appointment(user_id: int, doctor: str, date_time_str: str, reason: str) -> str:
    """Schedules a medical appointment for a user. 
    Args: user_id (int), doctor (str), date_time_str (str, e.g., '2024-12-25 10:00'), reason (str)."""
    print(f"Simulating scheduling appointment for user {user_id} with Dr. {doctor} on {date_time_str} for {reason}...")
    try:
        dt_obj = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")
        with SessionLocal() as db:
            new_appointment = AppointmentDB(user_id=user_id, doctor=doctor, date_time=dt_obj, reason=reason)
            db.add(new_appointment)
            db.commit()
            db.refresh(new_appointment)
        return f"Appointment with Dr. {doctor} successfully scheduled for {date_time_str} for {reason}."
    except ValueError:
        return "Invalid date/time format. Please use YYYY-MM-DD HH:MM."
    except Exception as e:
        return f"Failed to schedule appointment: {e}"

@tool
def refill_prescription(user_id: int, medication_name: str) -> str:
    """Simulates refilling a prescription for a user. 
    Args: user_id (int), medication_name (str)."""
    print(f"Simulating refilling prescription for user {user_id} - {medication_name}...")
    # In a real app, this would interact with a pharmacy API
    with SessionLocal() as db:
        med = db.query(MedicationDB).filter_by(user_id=user_id, name=medication_name).first()
        if med:
            med.last_refill = datetime.now()
            db.commit()
            return f"Prescription for {medication_name} refilled for user {user_id}."
        return f"Medication {medication_name} not found for user {user_id}."

@tool
def get_user_medications(user_id: int) -> str:
    """Retrieves all medications for a specific user from the database.
    Args: user_id (int)."""
    print(f"Retrieving medications for user {user_id}...")
    with SessionLocal() as db:
        medications = db.query(MedicationDB).filter_by(user_id=user_id).all()
        if medications:
            return "\n".join([f"- {m.name} ({m.dosage}, {m.schedule}) - Last Refill: {m.last_refill.strftime('%Y-%m-%d') if m.last_refill else 'N/A'}" for m in medications])
        return f"No medications found for user {user_id}."

@tool
def add_user_profile(name: str, age: int, medical_conditions: str = "None", allergies: str = "None", preferences: str = "None") -> str:
    """Adds a new user profile to the database. 
    Args: name (str), age (int), medical_conditions (str), allergies (str), preferences (str)."""
    print(f"Adding user profile for {name}...")
    try:
        user_profile = UserProfile(name=name, age=age, medical_conditions=medical_conditions, allergies=allergies, preferences=preferences)
        with SessionLocal() as db:
            new_user = UserProfileDB(
                name=user_profile.name,
                age=user_profile.age,
                medical_conditions=user_profile.medical_conditions,
                allergies=user_profile.allergies,
                preferences=user_profile.preferences
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
        return f"User {name} (ID: {new_user.id}) profile added successfully."
    except Exception as e:
        return f"Failed to add user profile: {e}"

@tool
def get_user_profile(user_id: int) -> str:
    """Retrieves a user profile by ID.
    Args: user_id (int)."""
    print(f"Retrieving user profile for ID: {user_id}...")
    with SessionLocal() as db:
        user = db.query(UserProfileDB).filter_by(id=user_id).first()
        if user:
            return f"Name: {user.name}, Age: {user.age}, Conditions: {user.medical_conditions}, Allergies: {user.allergies}, Preferences: {user.preferences}"
        return f"User with ID {user_id} not found."

# List of all tools available to the agent
tools = [
    get_wearable_data,
    get_medical_info,
    schedule_appointment,
    refill_prescription,
    get_user_medications,
    add_user_profile,
    get_user_profile,
]

# --- 4. Planning Module (LLM Core & Agent) ---

# Ensure OPENAI_API_KEY is set in your environment variables
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Conversational Memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5 # Keep last 5 turns of conversation
)

# Agent Prompt (ReAct-style)
agent_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are a helpful Personal Health Assistant AI. Your goal is to assist users with their health management tasks. \n"\
                  "You have access to various tools to gather information, schedule appointments, manage medications, and retrieve health records.\n"\
                  "Always try to be proactive and provide helpful, empathetic responses. \n"\
                  "If asked about a user's health history or current health, remember to first check the long-term health record or user profile.\n"\
                  "When scheduling or refilling, ensure you have all necessary details."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the ReAct agent
agent = create_react_agent(llm, tools, agent_prompt)

# Create the Agent Executor
health_assistant_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=conversational_memory,
    handle_parsing_errors=True,
)

# --- Main Interaction Loop ---
if __name__ == "__main__":
    print("Personalized Health Assistant Agent started. Type 'exit' to quit.")

    # Example: Add a user and some initial data
    print(health_assistant_agent.invoke({"input": "Add a new user profile named Alice, 30 years old, with no medical conditions, no allergies, and prefers morning workouts." }))
    print(health_assistant_agent.invoke({"input": "Add medication for user 1: name Aspirin, dosage 81mg, schedule daily, last refill 2024-01-15 09:00"}))
    add_health_record_to_vectordb(1, "Alice's initial checkup showed normal blood pressure and cholesterol. She aims to improve her cardiovascular health.")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            break

        response = health_assistant_agent.invoke({"input": user_input})
        print("\nAssistant:")
        print(response["output"])

        # Example of manually using vector DB for retrieval (agent would typically do this via prompt)
        if "health history" in user_input.lower() or "medical record" in user_input.lower():
            retrieved_records = retrieve_health_records_from_vectordb(1, user_input)
            if retrieved_records:
                print("Relevant health records found (from long-term memory):")
                for record in retrieved_records:
                    print(f"- {record}")

