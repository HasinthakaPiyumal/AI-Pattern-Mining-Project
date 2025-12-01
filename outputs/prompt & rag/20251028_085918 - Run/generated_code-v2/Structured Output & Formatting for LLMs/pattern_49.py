from pydantic import BaseModel, Field
from typing import List, Optional

class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication.")
    dosage: str = Field(..., description="Dosage instructions (e.g., '10mg', '2 units').")
    frequency: str = Field(..., description="How often the medication should be taken (e.g., 'daily', 'twice a day').")
    notes: Optional[str] = Field(None, description="Any additional notes for this medication.")

class DietaryRestriction(BaseModel):
    type: str = Field(..., description="Type of dietary restriction (e.g., 'low-sodium', 'gluten-free', 'diabetic-friendly').")
    details: str = Field(..., description="Specific details or examples for the dietary restriction.")

class ExerciseRoutine(BaseModel):
    type: str = Field(..., description="Type of exercise (e.g., 'walking', 'swimming', 'strength training').")
    duration_minutes: int = Field(..., description="Duration of the exercise in minutes.")
    frequency: str = Field(..., description="How often the exercise should be performed (e.g., '3 times a week', 'daily').")
    notes: Optional[str] = Field(None, description="Any additional notes for this exercise routine.")

class Appointment(BaseModel):
    date: str = Field(..., description="Date of the appointment (e.g., 'YYYY-MM-DD').")
    time: str = Field(..., description="Time of the appointment (e.g., 'HH:MM AM/PM').")
    specialty: str = Field(..., description="Specialty of the doctor or clinic (e.g., 'Cardiology', 'Endocrinology').")
    notes: Optional[str] = Field(None, description="Any additional notes for this appointment.")

class TreatmentPlan(BaseModel):
    patient_name: str = Field(..., description="Full name of the patient.")
    chronic_disease: str = Field(..., description="The chronic disease being managed.")
    plan_duration_weeks: int = Field(..., description="Duration of the treatment plan in weeks.")
    medications: List[Medication] = Field(default_factory=list, description="List of medications in the plan.")
    dietary_restrictions: List[DietaryRestriction] = Field(default_factory=list, description="List of dietary restrictions.")
    exercise_routines: List[ExerciseRoutine] = Field(default_factory=list, description="List of exercise routines.")
    appointments: List[Appointment] = Field(default_factory=list, description="List of scheduled appointments.")
    general_recommendations: str = Field(..., description="General recommendations or lifestyle advice.")