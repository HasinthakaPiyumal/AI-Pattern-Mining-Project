from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, time

class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dosage: str = Field(..., description="Dosage instructions (e.g., '10mg daily')")
    frequency: str = Field(..., description="How often the medication should be taken (e.g., 'once a day', 'every 8 hours')")
    notes: Optional[str] = Field(None, description="Any additional notes for this medication")

class DietRecommendation(BaseModel):
    type: str = Field(..., description="Type of diet recommendation (e.g., 'Low-sodium', 'Diabetic-friendly', 'High-fiber')")
    details: str = Field(..., description="Specific details of the diet recommendation")

class ExerciseRoutine(BaseModel):
    activity: str = Field(..., description="Type of physical activity (e.g., 'Walking', 'Yoga', 'Strength Training')")
    duration_minutes: int = Field(..., description="Recommended duration in minutes")
    frequency: str = Field(..., description="How often the exercise should be performed (e.g., '3 times a week', 'daily')")

class Appointment(BaseModel):
    date: date = Field(..., description="Date of the appointment")
    time: time = Field(..., description="Time of the appointment")
    with_doctor: str = Field(..., description="Name or specialty of the healthcare professional")
    purpose: str = Field(..., description="Purpose of the appointment (e.g., 'Follow-up', 'Blood work')")

class MonitoringTask(BaseModel):
    task_name: str = Field(..., description="Name of the monitoring task (e.g., 'Blood Pressure Check', 'Blood Glucose Level')")
    frequency: str = Field(..., description="How often the task should be performed (e.g., 'daily', 'weekly', 'before meals')")
    target_range: Optional[str] = Field(None, description="Optional target range for the monitored value")

class CarePlan(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient")
    summary: str = Field(..., description="Overall summary of the personalized care plan")
    medications: List[Medication] = Field(default_factory=list, description="List of prescribed medications")
    diet_recommendations: List[DietRecommendation] = Field(default_factory=list, description="List of diet recommendations")
    exercise_routines: List[ExerciseRoutine] = Field(default_factory=list, description="List of exercise routines")
    appointments: List[Appointment] = Field(default_factory=list, description="List of scheduled appointments")
    monitoring_tasks: List[MonitoringTask] = Field(default_factory=list, description="List of monitoring tasks")
