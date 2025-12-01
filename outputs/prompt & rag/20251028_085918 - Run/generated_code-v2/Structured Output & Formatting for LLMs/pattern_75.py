from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json

# 1. Pydantic Models for Structured Output Layer
class Medication(BaseModel):
    name: str
    dosage: str
    frequency: str

class DietaryRecommendation(BaseModel):
    type: str
    description: str
    restrictions: Optional[str] = None

class ExercisePlan(BaseModel):
    activity: str
    duration: str
    frequency: str
    intensity: Optional[str] = None

class Appointment(BaseModel):
    date: str
    type: str
    doctor: str
    notes: Optional[str] = None

class TreatmentPlan(BaseModel):
    patient_id: str
    disease: str
    medications: List[Medication]
    dietary_recommendations: List[DietaryRecommendation]
    exercise_plans: List[ExercisePlan]
    appointments: List[Appointment]
    overall_notes: Optional[str] = None

# Input model for the API endpoint
class PatientInput(BaseModel):
    patient_id: str
    disease: str

# FastAPI Application
app = FastAPI()

# Simulate LLM core and parser
def generate_and_parse_treatment_plan(patient_id: str, disease: str) -> TreatmentPlan:
    # Simulate LLM generating a structured JSON output directly
    # In a real scenario, this would be an API call to an LLM like GPT-4 or Gemini Pro
    # prompted to output in JSON format.
    mock_llm_output_json = f"""
    {{
      "patient_id": "{patient_id}",
      "disease": "{disease}",
      "medications": [
        {{
          "name": "Metformin",
          "dosage": "500mg",
          "frequency": "Twice daily with meals"
        }},
        {{
          "name": "Insulin Glargine",
          "dosage": "10 units",
          "frequency": "Once daily at bedtime"
        }}
      ],
      "dietary_recommendations": [
        {{
          "type": "Low Carb",
          "description": "Focus on lean proteins, non-starchy vegetables, and healthy fats. Limit refined sugars and processed foods.",
          "restrictions": "Avoid sugary drinks, white bread, pasta, and potatoes."
        }},
        {{
          "type": "Portion Control",
          "description": "Be mindful of portion sizes to manage carbohydrate intake."
        }}
      ],
      "exercise_plans": [
        {{
          "activity": "Brisk Walking",
          "duration": "30 minutes",
          "frequency": "5 times per week",
          "intensity": "Moderate"
        }},
        {{
          "activity": "Strength Training",
          "duration": "20 minutes",
          "frequency": "3 times per week",
          "intensity": "Light to Moderate"
        }}
      ],
      "appointments": [
        {{
          "date": "2024-07-15",
          "type": "Endocrinologist Follow-up",
          "doctor": "Dr. Smith",
          "notes": "Review blood sugar logs and adjust medication."
        }},
        {{
          "date": "2024-08-01",
          "type": "Dietitian Consultation",
          "doctor": "Ms. Johnson",
          "notes": "Develop a personalized meal plan."
        }}
      ],
      "overall_notes": "Patient should regularly monitor blood glucose levels and report any unusual symptoms."
    }}
    """

    # Plan Parser / Extractor (LLM-based structured extraction handled by Pydantic)
    # The LLM is assumed to generate valid JSON directly, which Pydantic can parse.
    return TreatmentPlan.model_validate_json(mock_llm_output_json)

@app.post("/generate_plan", response_model=TreatmentPlan)
async def generate_plan(patient_input: PatientInput):
    # Orchestrate LLM interaction and parsing
    treatment_plan = generate_and_parse_treatment_plan(
        patient_input.patient_id,
        patient_input.disease
    )
    return treatment_plan
