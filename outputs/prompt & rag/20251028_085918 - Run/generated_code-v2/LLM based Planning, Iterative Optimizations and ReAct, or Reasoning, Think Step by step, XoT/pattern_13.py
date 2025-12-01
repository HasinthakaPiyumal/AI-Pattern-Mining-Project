import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import openai

# Configuration for OpenAI API
# You should set OPENAI_API_KEY as an environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class PatientData(BaseModel):
    symptoms: List[str]
    history: List[str]
    lab_results: Dict[str, Any]

class DiagnosticStep(BaseModel):
    step_name: str
    description: str

class ClinicalDetail(BaseModel):
    step_name: str
    differential_diagnoses: List[str]
    clinical_indicators: List[str]
    test_recommendations: List[str]

# Mock LLM calls - in a real application, these would interact with actual LLM APIs
# For demonstration, we'll simulate responses
def call_llm(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    # In a real scenario, this would be an API call to an LLM provider
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message['content']

    # Mock responses for demonstration
    if "propose diagnostic steps" in prompt.lower():
        return """
        [
            {"step_name": "Initial Assessment", "description": "Review patient chief complaint and vital signs."},
            {"step_name": "Symptom Analysis", "description": "Deep dive into symptom characteristics and onset."},
            {"step_name": "Physical Examination", "description": "Conduct relevant physical examination."},
            {"step_name": "Laboratory Tests", "description": "Order initial blood work and imaging."},
            {"step_name": "Specialist Consultation", "description": "Recommend consultation with relevant specialist."}
        ]
        """
    elif "generate detailed clinical information for Initial Assessment" in prompt.lower():
        return """
        {
            "step_name": "Initial Assessment",
            "differential_diagnoses": ["Common Cold", "Influenza", "Allergic Rhinitis"],
            "clinical_indicators": ["Fever", "Cough", "Sore Throat"],
            "test_recommendations": ["Rapid Strep Test", "Flu Swab"]
        }
        """
    elif "generate detailed clinical information for Symptom Analysis" in prompt.lower():
        return """
        {
            "step_name": "Symptom Analysis",
            "differential_diagnoses": ["Bronchitis", "Pneumonia", "Asthma Exacerbation"],
            "clinical_indicators": ["Dyspnea", "Wheezing", "Productive Cough"],
            "test_recommendations": ["Chest X-ray", "Spirometry"]
        }
        """
    elif "generate detailed clinical information for Physical Examination" in prompt.lower():
        return """
        {
            "step_name": "Physical Examination",
            "differential_diagnoses": ["Appendicitis", "Diverticulitis", "Kidney Stones"],
            "clinical_indicators": ["Abdominal Tenderness", "Rebound Tenderness", "Flank Pain"],
            "test_recommendations": ["Abdominal Ultrasound", "CT Abdomen"]
        }
        """
    elif "generate detailed clinical information for Laboratory Tests" in prompt.lower():
        return """
        {
            "step_name": "Laboratory Tests",
            "differential_diagnoses": ["Anemia", "Infection", "Electrolyte Imbalance"],
            "clinical_indicators": ["Low Hemoglobin", "High WBC", "Abnormal Sodium"],
            "test_recommendations": ["CBC", "CMP", "Urinalysis"]
        }
        """
    elif "generate detailed clinical information for Specialist Consultation" in prompt.lower():
        return """
        {
            "step_name": "Specialist Consultation",
            "differential_diagnoses": ["Autoimmune Disease", "Rare Genetic Condition", "Complex Neurological Disorder"],
            "clinical_indicators": ["Persistent Unexplained Symptoms", "Multiple Organ Involvement", "Failed Standard Treatments"],
            "test_recommendations": ["Referral to Rheumatology", "Neurology Consult", "Genetic Counseling"]
        }
        """
    return "[]" # Default empty list for other cases


# LLM 1: Diagnostic Path Planner
def diagnostic_path_planner(patient_data: PatientData) -> List[DiagnosticStep]:
    prompt = f"Given the patient's symptoms: {', '.join(patient_data.symptoms)}, history: {', '.join(patient_data.history)}, and lab results: {patient_data.lab_results}, propose diagnostic steps as a JSON list of objects with 'step_name' and 'description' keys. Each step should be a logical progression in a diagnostic process."
    response = call_llm(prompt, model="gpt-4") # Use a more capable model for planning
    try:
        steps_raw = eval(response) # Using eval for simplicity, safer parsing methods recommended for production
        return [DiagnosticStep(**step) for step in steps_raw]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing diagnostic steps: {e}")

# LLM 2: Clinical Detail Generator
def clinical_detail_generator(step: DiagnosticStep) -> ClinicalDetail:
    prompt = f"Given the diagnostic step '{step.step_name}' with description '{step.description}', generate detailed clinical information including differential diagnoses, relevant clinical indicators, and specific diagnostic test recommendations. Provide this as a JSON object with 'step_name', 'differential_diagnoses' (list of strings), 'clinical_indicators' (list of strings), and 'test_recommendations' (list of strings) keys."
    response = call_llm(prompt, model="gpt-3.5-turbo") # Can use a faster model for generation
    try:
        details_raw = eval(response) # Using eval for simplicity, safer parsing methods recommended for production
        return ClinicalDetail(**details_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing clinical details for step '{step.step_name}': {e}")

@app.post("/diagnose")
async def diagnose_patient(patient_data: PatientData):
    """
    Endpoint to diagnose a patient using the two-LLM framework.
    """
    try:
        # Step 1: LLM1 proposes diagnostic steps
        diagnostic_steps = diagnostic_path_planner(patient_data)

        # Step 2: LLM2 generates clinical details for each step
        detailed_diagnoses = []
        for step in diagnostic_steps:
            details = clinical_detail_generator(step)
            detailed_diagnoses.append(details)

        return {"diagnostic_process": detailed_diagnoses}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")