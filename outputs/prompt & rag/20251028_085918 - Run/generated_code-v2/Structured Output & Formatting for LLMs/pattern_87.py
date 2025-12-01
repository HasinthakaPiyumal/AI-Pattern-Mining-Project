from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputFunctionsParser
from langchain.chains import create_structured_output_runnable
import os

# Pydantic Models for Structured Output
class Medication(BaseModel):
    drug: str = Field(..., description="Name of the medication.")
    dosage: str = Field(..., description="Dosage of the medication (e.g., '10mg', '2 pills').")
    frequency: str = Field(..., description="Frequency of medication intake (e.g., 'twice daily', 'every morning').")
    notes: Optional[str] = Field(None, description="Any specific notes or instructions for this medication.")

class Appointment(BaseModel):
    type: str = Field(..., description="Type of appointment (e.g., 'Cardiology Check-up', 'Physical Therapy').")
    date: str = Field(..., description="Date of the appointment (e.g., 'YYYY-MM-DD').")
    time: str = Field(..., description="Time of the appointment (e.g., 'HH:MM AM/PM').")
    location: Optional[str] = Field(None, description="Location of the appointment.")
    provider: Optional[str] = Field(None, description="Healthcare provider for the appointment.")

class DietaryRecommendation(BaseModel):
    item: str = Field(..., description="Specific dietary item or category (e.g., 'Low Sodium', 'High Protein Meal').")
    instructions: str = Field(..., description="Detailed instructions for the dietary recommendation.")

class ExerciseActivity(BaseModel):
    activity: str = Field(..., description="Type of exercise or activity (e.g., 'Walking', 'Stretching', 'Yoga').")
    duration_minutes: int = Field(..., description="Recommended duration in minutes.")
    frequency: str = Field(..., description="Frequency of the activity (e.g., 'daily', '3 times a week').")
    intensity: Optional[str] = Field(None, description="Suggested intensity level (e.g., 'light', 'moderate').")

class MonitoringInstruction(BaseModel):
    parameter: str = Field(..., description="Parameter to monitor (e.g., 'Blood Pressure', 'Blood Glucose', 'Weight').")
    frequency: str = Field(..., description="How often to monitor (e.g., 'daily', 'before meals').")
    target_range: Optional[str] = Field(None, description="Target range for the monitored parameter.")
    notes: Optional[str] = Field(None, description="Any specific notes or instructions for monitoring.")

class EducationalResource(BaseModel):
    title: str = Field(..., description="Title of the educational resource.")
    description: str = Field(..., description="Brief description of the resource.")
    link: Optional[str] = Field(None, description="URL or reference to the resource.")

class CarePlan(BaseModel):
    patient_id: Optional[str] = Field(None, description="Unique identifier for the patient.")
    plan_overview: str = Field(..., description="A general overview and summary of the patient's care plan.")
    medication_schedule: List[Medication] = Field(default_factory=list, description="List of prescribed medications and their schedule.")
    appointment_schedule: List[Appointment] = Field(default_factory=list, description="List of scheduled appointments.")
    dietary_recommendations: List[DietaryRecommendation] = Field(default_factory=list, description="List of dietary advice.")
    exercise_activity_plan: List[ExerciseActivity] = Field(default_factory=list, description="List of recommended exercises and activities.")
    monitoring_instructions: List[MonitoringInstruction] = Field(default_factory=list, description="List of instructions for self-monitoring.")
    educational_resources: List[EducationalResource] = Field(default_factory=list, description="List of helpful educational materials.")
    emergency_contact_info: Optional[str] = Field(None, description="Information on who to contact in an emergency.")

# FastAPI Application
app = FastAPI(
    title="Personalized Patient Care Plan Generator",
    description="API to generate structured patient care plans using LLMs."
)

# LLM Setup
# Ensure OPENAI_API_KEY environment variable is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Prompt Template
# The prompt clearly instructs the LLM to output a JSON object conforming to the CarePlan schema.
CARE_PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant tasked with generating comprehensive and personalized patient care plans based on provided patient information. The output MUST be a JSON object that strictly adheres to the provided schema."),
    ("human", "Generate a detailed care plan for the following patient: {patient_summary}\n\nEnsure the output is a JSON object with the following structure:\n{format_instructions}\n\nBegin the JSON object now:")
])

# LLM Chain for Structured Output
# create_structured_output_runnable uses Pydantic schema to guide LLM and parse output
care_plan_generator_chain = create_structured_output_runnable(CarePlan, llm, CARE_PLAN_PROMPT)

# API Endpoint
@app.post("/generate-care-plan", response_model=CarePlan)
async def generate_patient_care_plan(patient_summary_text: str):
    try:
        # Invoke the LLM chain to generate the structured care plan
        care_plan = await care_plan_generator_chain.ainvoke({"patient_summary": patient_summary_text})
        return care_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate care plan: {str(e)}")

# To run this application:
# 1. Save the code as 'patient_care_plan_generator.py'
# 2. Set your OpenAI API key as an environment variable: export OPENAI_API_KEY='your_api_key'
# 3. Install necessary libraries: pip install fastapi uvicorn pydantic langchain-openai
# 4. Run the API server: uvicorn patient_care_plan_generator:app --reload
# 5. Access the API documentation at http://127.0.0.1:8000/docs