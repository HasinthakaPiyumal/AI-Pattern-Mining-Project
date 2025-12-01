import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# 1. Define Pydantic models for structured output
class Medication(BaseModel):
    name: str
    dosage: str
    frequency: str
    notes: str = Field(default="")

class Test(BaseModel):
    name: str
    details: str
    frequency: str = Field(default="")

class Recommendation(BaseModel):
    type: str  # e.g., "lifestyle", "dietary", "follow-up"
    description: str

class TreatmentPlan(BaseModel):
    patient_id: str
    diagnosis: str
    chief_complaint: str
    medications: List[Medication] = Field(default_factory=list)
    tests: List[Test] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    follow_up_date: str
    plan_notes: str = Field(default="")

# 2. Simulate LLM generating a natural language treatment plan
def generate_natural_language_plan(patient_info: Dict[str, Any]) -> str:
    patient_name = patient_info.get("name", "John Doe")
    patient_id = patient_info.get("id", "P12345")
    diagnosis = patient_info.get("diagnosis", "Type 2 Diabetes")
    symptoms = patient_info.get("symptoms", "Fatigue, increased thirst, frequent urination")
    history = patient_info.get("history", "Family history of diabetes, sedentary lifestyle")

    plan_text = f"""
Patient: {patient_name} (ID: {patient_id})
Diagnosis: {diagnosis}
Chief Complaint: Patient reports {symptoms} for the past few weeks.
Medical History: {history}.

Treatment Plan:

1. Medications:
   - Metformin 500mg, twice daily with meals. Notes: Start low and titrate up as tolerated.
   - Atorvastatin 20mg, once daily at bedtime.

2. Diagnostic Tests:
   - HbA1c test, repeat in 3 months. Details: To monitor blood sugar control.
   - Lipid Panel, repeat in 3 months. Details: To assess cholesterol levels.

3. Recommendations:
   - Lifestyle: Engage in moderate-intensity exercise for at least 30 minutes, 5 days a week.
   - Dietary: Follow a balanced diet, limiting processed sugars and saturated fats. Consider consultation with a dietitian.
   - Follow-up: Schedule a follow-up appointment in 3 months to review test results and treatment efficacy.

Additional Notes: Educate patient on signs and symptoms of hypoglycemia and hyperglycemia. Provide resources for diabetes management.
"""
    return plan_text

# 3. Simulate parsing natural language into structured JSON using an LLM (mocked)
def parse_natural_language_to_structured_json(natural_language_plan: str, patient_id: str) -> TreatmentPlan:
    # In a real application, an LLM would be prompted to extract entities
    # and format them into JSON based on the TreatmentPlan schema.
    # For this simulation, we'll hardcode a structured output based on the example above.
    # A library like `instructor` with `openai` or `langchain` with a structured output parser
    # would be used here.
    
    # Mocked structured output (assuming the LLM successfully extracts and formats)
    mock_structured_data = {
        "patient_id": patient_id,
        "diagnosis": "Type 2 Diabetes",
        "chief_complaint": "Patient reports fatigue, increased thirst, frequent urination for the past few weeks.",
        "medications": [
            {"name": "Metformin", "dosage": "500mg", "frequency": "twice daily", "notes": "Start low and titrate up as tolerated."},
            {"name": "Atorvastatin", "dosage": "20mg", "frequency": "once daily", "notes": ""}
        ],
        "tests": [
            {"name": "HbA1c test", "details": "To monitor blood sugar control.", "frequency": "repeat in 3 months"},
            {"name": "Lipid Panel", "details": "To assess cholesterol levels.", "frequency": "repeat in 3 months"}
        ],
        "recommendations": [
            {"type": "Lifestyle", "description": "Engage in moderate-intensity exercise for at least 30 minutes, 5 days a week."},
            {"type": "Dietary", "description": "Follow a balanced diet, limiting processed sugars and saturated fats. Consider consultation with a dietitian."},
            {"type": "Follow-up", "description": "Schedule a follow-up appointment in 3 months to review test results and treatment efficacy."}
        ],
        "follow_up_date": "3 months from now",
        "plan_notes": "Educate patient on signs and symptoms of hypoglycemia and hyperglycemia. Provide resources for diabetes management."
    }
    
    return TreatmentPlan(**mock_structured_data)

# 4. Simulate EHR integration
def integrate_with_ehr(treatment_plan: TreatmentPlan) -> None:
    print("\n--- Integrating with EHR System ---")
    print("Structured Treatment Plan for Patient ID:", treatment_plan.patient_id)
    print(json.dumps(treatment_plan.dict(), indent=2))
    print("--- EHR Integration Complete ---")

if __name__ == "__main__":
    patient_data = {
        "name": "Alice Smith",
        "id": "P67890",
        "diagnosis": "Hypertension",
        "symptoms": "occasional headaches, dizziness",
        "history": "Smoker, high-stress job"
    }

    # Step 1: LLM generates natural language plan
    print("Generating natural language treatment plan...")
    nl_plan = generate_natural_language_plan(patient_data)
    print("\n--- Natural Language Plan ---")
    print(nl_plan)
    print("---------------------------\n")

    # Step 2: Parse natural language plan into structured JSON
    print("Parsing natural language plan into structured JSON...")
    structured_plan = parse_natural_language_to_structured_json(nl_plan, patient_data["id"])
    
    # Step 3: Integrate with EHR
    integrate_with_ehr(structured_plan)

    print("\nApplication workflow completed.")
