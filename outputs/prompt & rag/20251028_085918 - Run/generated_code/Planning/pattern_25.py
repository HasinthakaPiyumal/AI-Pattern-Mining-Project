import json
from typing import List, Dict
import gradio as gr
from pydantic import BaseModel, Field

class PatientProfile(BaseModel):
    name: str
    age: int
    gender: str
    chronic_diseases: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    recent_vitals: Dict[str, float] = Field(default_factory=dict)

class TreatmentStep(BaseModel):
    description: str
    action: str
    target_metric: str
    duration_days: int

class TreatmentPlan(BaseModel):
    patient_name: str
    generated_date: str
    steps: List[TreatmentStep]
    notes: List[str] = Field(default_factory=list)

class KnowledgeBase:
    def __init__(self):
        self._drug_interactions = {
            ("Metformin", "Insulin"): "Monitor blood sugar closely due to increased hypoglycemia risk.",
            ("Warfarin", "Aspirin"): "Increased bleeding risk.",
            ("Lisinopril", "Ibuprofen"): "Reduced efficacy of lisinopril and potential kidney issues."
        }
        self._clinical_guidelines = {
            "Diabetes": {
                "initial_management": "Lifestyle modification (diet, exercise). First-line medication: Metformin.",
                "target_hba1c": "<7% for most adults."
            },
            "Hypertension": {
                "initial_management": "Lifestyle changes. First-line medications: Thiazide diuretics, ACE inhibitors, ARBs, Calcium Channel Blockers.",
                "target_bp": "<130/80 mmHg."
            }
        }

    def check_drug_interaction(self, med1: str, med2: str) -> str or None:
        pair1 = tuple(sorted((med1, med2)))
        return self._drug_interactions.get(pair1)

    def get_disease_guidelines(self, disease: str) -> Dict or None:
        return self._clinical_guidelines.get(disease)

class LLMService:
    def __init__(self):
        pass # In a real application, this would initialize an actual LLM client (e.g., OpenAI, Anthropic)

    def generate_plan_draft(self, prompt: str) -> str:
        # Simulate LLM response for demonstration purposes
        if "Diabetes" in prompt and "Hypertension" in prompt:
            return json.dumps({
                "steps": [
                    {"description": "Initiate Metformin and ACE Inhibitor.", "action": "Prescription", "target_metric": "Blood Sugar, Blood Pressure", "duration_days": 90},
                    {"description": "Recommend personalized diet and exercise plan.", "action": "Consultation", "target_metric": "Weight, Activity Level", "duration_days": 90},
                    {"description": "Schedule follow-up in 3 months.", "action": "Appointment", "target_metric": "Compliance", "duration_days": 90}
                ],
                "notes": ["Consider potential interaction between Metformin and other medications."]
            })
        elif "Diabetes" in prompt:
            return json.dumps({
                "steps": [
                    {"description": "Start Metformin 500mg daily.", "action": "Prescription", "target_metric": "Blood Sugar", "duration_days": 30},
                    {"description": "Educate on diabetic diet and exercise.", "action": "Education", "target_metric": "HbA1c", "duration_days": 30},
                    {"description": "Monitor blood glucose daily.", "action": "Monitoring", "target_metric": "Blood Sugar", "duration_days": 30}
                ],
                "notes": ["Initial plan for Type 2 Diabetes management."]
            })
        elif "Hypertension" in prompt:
            return json.dumps({
                "steps": [
                    {"description": "Prescribe Lisinopril 10mg daily.", "action": "Prescription", "target_metric": "Blood Pressure", "duration_days": 30},
                    {"description": "Advise on low-sodium diet and regular physical activity.", "action": "Education", "target_metric": "Blood Pressure", "duration_days": 30},
                    {"description": "Follow up in 1 month to assess blood pressure.", "action": "Appointment", "target_metric": "Blood Pressure", "duration_days": 30}
                ],
                "notes": ["Initial plan for Hypertension management."]
            })
        return json.dumps({
            "steps": [
                {"description": "General health assessment.", "action": "Observation", "target_metric": "Overall Health", "duration_days": 7}
            ],
            "notes": ["No specific chronic disease plan generated."]
        })

class TreatmentPlanGenerator:
    def __init__(self):
        self._knowledge_base = KnowledgeBase()
        self._llm_service = LLMService()

    def generate_initial_plan(self, patient_profile: PatientProfile) -> Dict:
        prompt = f"Generate a comprehensive treatment plan for a patient named {patient_profile.name}, {patient_profile.age} years old, with chronic diseases: {', '.join(patient_profile.chronic_diseases)}. Current medications: {', '.join(patient_profile.current_medications)}. Allergies: {', '.join(patient_profile.allergies)}. Recent vitals: {patient_profile.recent_vitals}. Focus on managing these conditions and providing actionable steps."
        raw_plan_string = self._llm_service.generate_plan_draft(prompt)
        return json.loads(raw_plan_string)

    def parse_llm_plan(self, patient_name: str, raw_plan_data: Dict) -> TreatmentPlan:
        steps = [TreatmentStep(**step_data) for step_data in raw_plan_data.get("steps", [])]
        import datetime
        today_date = datetime.date.today().strftime("%Y-%m-%d")
        return TreatmentPlan(patient_name=patient_name, generated_date=today_date, steps=steps, notes=raw_plan_data.get("notes", []))

    def optimize_and_validate_plan(self, patient_profile: PatientProfile, initial_plan: TreatmentPlan) -> TreatmentPlan:
        optimized_plan = initial_plan.copy(deep=True)
        validation_notes = []

        # Check drug interactions
        for i, med1 in enumerate(patient_profile.current_medications):
            for j, med2 in enumerate(patient_profile.current_medications):
                if i < j:
                    interaction = self._knowledge_base.check_drug_interaction(med1, med2)
                    if interaction:
                        validation_notes.append(f"Warning: Drug interaction detected between {med1} and {med2}. Recommendation: {interaction}")
        
        # Check for allergies with proposed actions/medications (simplified)
        for step in optimized_plan.steps:
            for allergy in patient_profile.allergies:
                if allergy.lower() in step.description.lower() or allergy.lower() in step.action.lower():
                    validation_notes.append(f"Warning: Proposed action/medication '{step.description}' might conflict with patient allergy: {allergy}.")
        
        # Check against clinical guidelines (simplified for core disease)
        for disease in patient_profile.chronic_diseases:
            guidelines = self._knowledge_base.get_disease_guidelines(disease)
            if guidelines:
                validation_notes.append(f"Consider clinical guidelines for {disease}: {guidelines.get('initial_management', '')}. Target HbA1c: {guidelines.get('target_hba1c', 'N/A')}, Target BP: {guidelines.get('target_bp', 'N/A')}.")

        optimized_plan.notes.extend(validation_notes)
        return optimized_plan

    def generate_full_plan(self, patient_profile: PatientProfile) -> TreatmentPlan:
        raw_plan_data = self.generate_initial_plan(patient_profile)
        initial_plan = self.parse_llm_plan(patient_profile.name, raw_plan_data)
        final_plan = self.optimize_and_validate_plan(patient_profile, initial_plan)
        return final_plan

plan_generator = TreatmentPlanGenerator()

def generate_plan_ui(
    name: str,
    age: int,
    gender: str,
    chronic_diseases_str: str,
    allergies_str: str,
    current_medications_str: str,
    recent_vitals_str: str
) -> str:
    try:
        chronic_diseases = [d.strip() for d in chronic_diseases_str.split(',') if d.strip()]
        allergies = [a.strip() for a in allergies_str.split(',') if a.strip()]
        current_medications = [m.strip() for m in current_medications_str.split(',') if m.strip()]
        recent_vitals = json.loads(recent_vitals_str) if recent_vitals_str else {}

        patient_profile = PatientProfile(
            name=name,
            age=age,
            gender=gender,
            chronic_diseases=chronic_diseases,
            allergies=allergies,
            current_medications=current_medications,
            recent_vitals=recent_vitals
        )
        
        treatment_plan = plan_generator.generate_full_plan(patient_profile)
        return treatment_plan.json(indent=2)
    except Exception as e:
        return f"Error generating plan: {e}"

iface = gr.Interface(
    fn=generate_plan_ui,
    inputs=[
        gr.Textbox(label="Patient Name"),
        gr.Number(label="Age", value=45),
        gr.Dropdown(choices=["Male", "Female", "Other"], label="Gender"),
        gr.Textbox(label="Chronic Diseases (comma-separated)", value="Diabetes, Hypertension"),
        gr.Textbox(label="Allergies (comma-separated)", value="Penicillin"),
        gr.Textbox(label="Current Medications (comma-separated)", value="Metformin, Lisinopril"),
        gr.Textbox(label="Recent Vitals (JSON format)", value=json.dumps({"blood_pressure_systolic": 145, "blood_pressure_diastolic": 90, "hba1c": 8.2}))
    ],
    outputs=gr.Json(label="Generated Treatment Plan"),
    title="AI-Powered Personalized Treatment Plan Generator",
    description="Enter patient details to generate an adaptive treatment plan for chronic disease management."
)

iface.launch()