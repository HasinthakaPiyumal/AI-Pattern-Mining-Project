import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- 1. Data Models (Pydantic) ---

class Medication(BaseModel):
    name: str = Field(description="Name of the medication")
    dosage: str = Field(description="Dosage of the medication (e.g., '10mg', 'once daily')")
    frequency: str = Field(description="How often the medication should be taken (e.g., 'morning', 'twice a day')")

class DietaryRestriction(BaseModel):
    type: str = Field(description="Type of dietary restriction (e.g., 'Gluten-Free', 'Low-Sodium')")
    details: str = Field(description="Specific details or examples for the restriction")

class ExerciseRoutine(BaseModel):
    type: str = Field(description="Type of exercise (e.g., 'Cardio', 'Strength Training')")
    duration_minutes: int = Field(description="Recommended duration in minutes")
    frequency: str = Field(description="How often the exercise should be performed (e.g., '3 times a week', 'daily')")

class Appointment(BaseModel):
    type: str = Field(description="Type of appointment (e.g., 'Follow-up', 'Specialist Consult')")
    date: str = Field(description="Suggested date for the appointment (e.g., 'YYYY-MM-DD')")
    time: str = Field(description="Suggested time for the appointment (e.g., 'HH:MM AM/PM')")
    notes: Optional[str] = Field(None, description="Any specific notes or instructions for the appointment")

class StructuredCarePlan(BaseModel):
    medications: List[Medication] = Field(description="List of recommended medications")
    dietary_restrictions: List[DietaryRestriction] = Field(description="List of dietary restrictions")
    exercise_routines: List[ExerciseRoutine] = Field(description="List of exercise routines")
    appointments: List[Appointment] = Field(description="List of follow-up appointments")

class PatientInput(BaseModel):
    patient_id: str = Field(description="Unique identifier for the patient")
    age: int = Field(description="Age of the patient")
    gender: str = Field(description="Gender of the patient")
    medical_conditions: List[str] = Field(description="List of diagnosed medical conditions")
    current_medications: List[str] = Field(description="List of medications the patient is currently taking")
    allergies: List[str] = Field(description="List of patient allergies")
    lifestyle_notes: str = Field(description="Additional notes about patient's lifestyle, habits, and preferences")

# --- 2. LLM Service & Post-processing Module ---

# Placeholder for OpenAI API Key. In a real application, use proper environment variable management.
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Mock ChatOpenAI for demonstration purposes. Replace with actual API key and model in production.
class MockChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt_template: str) -> str:
        # Simulate natural language care plan generation
        if "natural language care plan" in prompt_template.lower():
            return """Here is a personalized care plan:
            
            Medications:
            Take Metformin 500mg twice daily with meals.
            Continue with Lisinopril 10mg once daily in the morning.

            Dietary Recommendations:
            Follow a low-sodium diet, avoiding processed foods and excessive salt intake. Focus on fresh vegetables, fruits, and lean proteins.

            Exercise Plan:
            Engage in moderate-intensity cardio, such as brisk walking, for 30 minutes, 5 times a week.
            Include light strength training exercises (e.g., bodyweight) for 20 minutes, 2 times a week.

            Appointments:
            Schedule a follow-up with your primary care physician in 3 months. Date: 2024-10-26, Time: 10:00 AM.
            Consider a consultation with a registered dietitian for personalized meal planning.
            """
        # Simulate structured output generation
        elif "structured care plan as JSON" in prompt_template.lower():
            return StructuredCarePlan(
                medications=[
                    Medication(name="Metformin", dosage="500mg", frequency="twice daily"),
                    Medication(name="Lisinopril", dosage="10mg", frequency="once daily")
                ],
                dietary_restrictions=[
                    DietaryRestriction(type="Low-Sodium", details="Avoid processed foods and excessive salt. Focus on fresh vegetables, fruits, and lean proteins.")
                ],
                exercise_routines=[
                    ExerciseRoutine(type="Cardio (Brisk Walking)", duration_minutes=30, frequency="5 times a week"),
                    ExerciseRoutine(type="Strength Training (Bodyweight)", duration_minutes=20, frequency="2 times a week")
                ],
                appointments=[
                    Appointment(type="PCP Follow-up", date="2024-10-26", time="10:00 AM", notes="Schedule in 3 months."),
                    Appointment(type="Dietitian Consultation", date="TBD", time="TBD", notes="For personalized meal planning.")
                ]
            ).model_dump_json()
        return ""

llm_model = MockChatOpenAI()

# Prompt for initial natural language care plan generation
care_plan_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful healthcare assistant. Generate a personalized natural language care plan based on the patient's information."),
    ("human", "Patient ID: {patient_id}\nAge: {age}\nGender: {gender}\nMedical Conditions: {medical_conditions}\nCurrent Medications: {current_medications}\nAllergies: {allergies}\nLifestyle Notes: {lifestyle_notes}\n\nGenerate a comprehensive natural language care plan covering medications, dietary recommendations, exercise plan, and follow-up appointments.")
])

# Chain for natural language plan generation
nl_care_plan_chain = care_plan_prompt_template | llm_model

# Prompt for structured extraction (using pydantic's schema for guidance)
structured_extraction_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in parsing healthcare data. Extract the detailed components from the provided natural language care plan and format them precisely into the following JSON schema:\n{format_instructions}"),
    ("human", "Natural Language Care Plan: {natural_language_plan}")
])

# Chain for structured output generation using Pydantic schema
# For actual LLMs, you would typically use .with_structured_output(StructuredCarePlan) directly on the LLM
# However, since we are mocking, we will manually parse the mock JSON string.

# --- 3. API Layer (FastAPI) ---

app = FastAPI(
    title="Healthcare AI Assistant",
    description="AI assistant to generate and structure personalized patient care plans."
)

@app.post("/generate_care_plan", response_model=StructuredCarePlan)
async def generate_patient_care_plan(patient_data: PatientInput):
    try:
        # 1. Generate natural language care plan
        nl_plan = nl_care_plan_chain.invoke({
            "patient_id": patient_data.patient_id,
            "age": patient_data.age,
            "gender": patient_data.gender,
            "medical_conditions": ", ".join(patient_data.medical_conditions),
            "current_medications": ", ".join(patient_data.current_medications),
            "allergies": ", ".join(patient_data.allergies),
            "lifestyle_notes": patient_data.lifestyle_notes
        })

        # 2. Extract and structure information using a second LLM call (or robust parser)
        # In a real scenario with a capable LLM and Langchain's .with_structured_output,
        # the LLM would directly return a Pydantic object.
        # For this mock, we simulate that behavior by returning the Pydantic object's JSON.
        format_instructions = JsonOutputParser.get_format_instructions(StructuredCarePlan)
        
        # In a real setup, this would be `llm_model.with_structured_output(StructuredCarePlan).invoke(...)`
        # For our MockChatOpenAI, we have pre-programmed its behavior.
        structured_json_str = llm_model.invoke(
            structured_extraction_prompt_template.format(
                format_instructions=format_instructions,
                natural_language_plan=nl_plan
            )
        )
        
        structured_plan = StructuredCarePlan.model_validate_json(structured_json_str)
        
        return structured_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# To run this application:
# 1. Save the code as healthcare_ai_assistant.py
# 2. Install necessary libraries: pip install fastapi uvicorn pydantic langchain-openai langchain-core
# 3. Run from your terminal: uvicorn healthcare_ai_assistant:app --reload
# 4. Access the API documentation at http://127.0.0.1:8000/docs