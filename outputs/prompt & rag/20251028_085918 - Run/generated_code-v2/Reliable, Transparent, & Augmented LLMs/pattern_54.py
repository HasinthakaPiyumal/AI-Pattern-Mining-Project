from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import re

# 1. Data Models (`data_models.py`)
class LabResult(BaseModel):
    test_name: str
    value: Any
    unit: Optional[str] = None
    reference_range: Optional[str] = None

class ImagingFinding(BaseModel):
    modality: str
    description: str
    impression: Optional[str] = None

class Medication(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None

class PatientHistory(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    past_medical_history: List[str]
    current_medications: List[Medication]
    allergies: List[str]

class StandardizedPatientData(BaseModel):
    patient_id: str
    history: PatientHistory
    lab_results: List[LabResult]
    imaging_findings: List[ImagingFinding]
    drug_interactions: List[str]

class DiagnosticReport(BaseModel):
    patient_id: str
    potential_diagnoses: List[str]
    explanation: str
    suggested_tests: List[str]
    treatment_recommendations: List[str]

# 2. External Tool Simulators (`medical_tools.py`)
def simulate_lab_results(patient_id: str) -> Dict[str, Any]:
    if patient_id == "P001":
        return {
            "glucose": {"value": 125, "unit": "mg/dL", "reference": "70-100"},
            "hba1c": {"value": 7.2, "unit": "%", "reference": "<5.7"},
            "cholesterol": {"value": 220, "unit": "mg/dL", "reference": "<200"},
            "tsh": {"value": 3.5, "unit": "uIU/mL", "reference": "0.4-4.0"}
        }
    return {}

def simulate_imaging_report(patient_id: str) -> str:
    if patient_id == "P001":
        return "CT scan of the abdomen showed fatty liver changes and a small renal cyst. No acute abnormalities."  
    return "No imaging reports found."

def simulate_ehr_data(patient_id: str) -> Dict[str, Any]:
    if patient_id == "P001":
        return {
            "patient_id": "P001",
            "age": 55,
            "gender": "Male",
            "chief_complaint": "Fatigue and increased thirst for 3 months.",
            "past_medical_history": ["Hypertension", "Hyperlipidemia"],
            "current_medications": [
                {"name": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
                {"name": "Atorvastatin", "dosage": "20mg", "frequency": "daily"}
            ],
            "allergies": ["Penicillin"]
        }
    return {}

def simulate_drug_interaction_check(medications_list: List[Medication]) -> List[str]:
    interactions = []
    med_names = {med.name.lower() for med in medications_list}

    if "lisinopril" in med_names and "potassium supplements" in med_names: # Example interaction
        interactions.append("Lisinopril and potassium supplements can increase risk of hyperkalemia.")
    
    # Simulate a generic interaction based on common patterns
    if len(med_names) > 1 and len(interactions) == 0: # If multiple meds but no specific interaction found
        interactions.append("No major drug interactions identified among the listed medications, but caution is advised with polypharmacy.")
    elif len(med_names) == 1:
        interactions.append("Single medication, no interactions to check.")

    return interactions

# 3. Information Extraction & Standardization Layer (`information_processor.py`)
class InformationProcessor:
    def process_lab_results(self, raw_lab_data: Dict[str, Any]) -> List[LabResult]:
        lab_results = []
        for test_name, data in raw_lab_data.items():
            lab_results.append(LabResult(
                test_name=test_name,
                value=data.get("value"),
                unit=data.get("unit"),
                reference_range=data.get("reference")
            ))
        return lab_results

    def process_imaging_report(self, raw_imaging_report: str) -> List[ImagingFinding]:
        findings = []
        if raw_imaging_report and raw_imaging_report != "No imaging reports found.":
            # Simple parsing for demonstration
            modality_match = re.search(r"(CT scan|MRI|X-ray)", raw_imaging_report, re.IGNORECASE)
            modality = modality_match.group(0) if modality_match else "Unknown"
            description = raw_imaging_report
            impression_match = re.search(r"impression: (.*?)(?=\.)", raw_imaging_report, re.IGNORECASE)
            impression = impression_match.group(1).strip() if impression_match else None
            findings.append(ImagingFinding(
                modality=modality,
                description=description,
                impression=impression
            ))
        return findings

    def process_ehr_data(self, raw_ehr_data: Dict[str, Any]) -> PatientHistory:
        medications = [Medication(**m) for m in raw_ehr_data.get("current_medications", [])]
        return PatientHistory(
            patient_id=raw_ehr_data.get("patient_id"),
            age=raw_ehr_data.get("age"),
            gender=raw_ehr_data.get("gender"),
            chief_complaint=raw_ehr_data.get("chief_complaint"),
            past_medical_history=raw_ehr_data.get("past_medical_history", []),
            current_medications=medications,
            allergies=raw_ehr_data.get("allergies", [])
        )
    
    def standardize_patient_data(self, patient_id: str) -> StandardizedPatientData:
        raw_lab_data = simulate_lab_results(patient_id)
        raw_imaging_report = simulate_imaging_report(patient_id)
        raw_ehr_data = simulate_ehr_data(patient_id)

        lab_results = self.process_lab_results(raw_lab_data)
        imaging_findings = self.process_imaging_report(raw_imaging_report)
        patient_history = self.process_ehr_data(raw_ehr_data)
        drug_interactions = simulate_drug_interaction_check(patient_history.current_medications)

        return StandardizedPatientData(
            patient_id=patient_id,
            history=patient_history,
            lab_results=lab_results,
            imaging_findings=imaging_findings,
            drug_interactions=drug_interactions
        )

# 4. LLM-based Synthesis & Reasoning (`llm_synthesizer.py`)
class LLMSynthesizer:
    def synthesize_diagnosis(self, patient_data: StandardizedPatientData) -> DiagnosticReport:
        # In a real application, this would involve an actual LLM API call.
        # The prompt would be constructed using patient_data.
        # For this simulation, we'll generate a plausible report based on the provided P001 data.

        # Simulate LLM reasoning for P001 based on known data:
        # Glucose 125 (high), HbA1c 7.2 (diabetic range), Cholesterol 220 (high)
        # Chief complaint: Fatigue, increased thirst (classic diabetes symptoms)
        # Imaging: Fatty liver (common with metabolic syndrome/diabetes)
        # Past Medical History: Hypertension, Hyperlipidemia (risk factors for diabetes)

        potential_diagnoses = ["Type 2 Diabetes Mellitus", "Metabolic Syndrome"]
        explanation = (
            f"Patient {patient_data.patient_id} presents with fatigue and increased thirst. "
            "Lab results indicate elevated fasting glucose (125 mg/dL) and HbA1c (7.2%), consistent with Type 2 Diabetes Mellitus. "
            "Cholesterol is also elevated (220 mg/dL). "
            "Past medical history includes hypertension and hyperlipidemia, further supporting metabolic dysregulation. "
            "Imaging shows fatty liver, which is frequently associated with insulin resistance and Type 2 Diabetes. "
            "Simulated drug interaction check found no critical interactions among current medications (Lisinopril, Atorvastatin) based on provided data, but overall polypharmacy should be monitored."
        )
        suggested_tests = ["Fasting Lipid Panel", "Urinalysis for Microalbumin", "Ophthalmology consult", "Foot exam"]
        treatment_recommendations = [
            "Dietary and lifestyle modifications (e.g., reduced sugar intake, regular exercise)",
            "Initiate Metformin (if no contraindications)",
            "Continue current medications (Lisinopril, Atorvastatin) as prescribed and monitor",
            "Regular follow-up with endocrinologist/PCP"
        ]
        
        # If patient_id is not P001, provide a generic response
        if patient_data.patient_id != "P001":
             potential_diagnoses = ["Further investigation needed"]
             explanation = "Insufficient data to provide a detailed diagnosis. Please provide more patient information."
             suggested_tests = ["Comprehensive metabolic panel", "Full physical examination"]
             treatment_recommendations = ["Consult with a specialist"]

        return DiagnosticReport(
            patient_id=patient_data.patient_id,
            potential_diagnoses=potential_diagnoses,
            explanation=explanation,
            suggested_tests=suggested_tests,
            treatment_recommendations=treatment_recommendations
        )

# 5. Contextual Response Generation & User Interface (`app.py`)
class MedicalDiagnosticAssistant:
    def __init__(self):
        self.information_processor = InformationProcessor()
        self.llm_synthesizer = LLMSynthesizer()
        self.feedback_data = [] # Conceptual feedback storage

    def get_patient_diagnosis(self, patient_id: str) -> Optional[DiagnosticReport]:
        print(f"Retrieving and standardizing data for patient {patient_id}...")
        standardized_data = self.information_processor.standardize_patient_data(patient_id)
        
        if not standardized_data.history.patient_id: # Basic check if EHR data was found
            print(f"Error: No EHR data found for patient ID {patient_id}. Cannot proceed with diagnosis.")
            return None

        print("Synthesizing diagnostic report with LLM...")
        diagnostic_report = self.llm_synthesizer.synthesize_diagnosis(standardized_data)
        return diagnostic_report

    def display_report(self, report: DiagnosticReport):
        print("\n--- Medical Diagnostic Report ---")
        print(f"Patient ID: {report.patient_id}")
        print("\nPotential Diagnoses:")
        for diagnosis in report.potential_diagnoses:
            print(f"  - {diagnosis}")
        print("\nExplanation:")
        print(f"  {report.explanation}")
        print("\nSuggested Further Tests:")
        for test in report.suggested_tests:
            print(f"  - {test}")
        print("\nTreatment Recommendations:")
        for recommendation in report.treatment_recommendations:
            print(f"  - {recommendation}")
        print("-----------------------------------")

    def collect_feedback(self, patient_id: str, report: DiagnosticReport, feedback_text: str):
        self.feedback_data.append({
            "patient_id": patient_id,
            "report_summary": report.potential_diagnoses,
            "feedback": feedback_text,
            "timestamp": "(current_time)" # In a real app, use datetime
        })
        print("Feedback collected. Thank you!")

    def run_cli(self):
        print("Welcome to the Medical Diagnostic Assistant CLI!")
        while True:
            patient_id = input("\nEnter Patient ID (e.g., P001) or 'exit' to quit: ").strip()
            if patient_id.lower() == 'exit':
                break
            
            report = self.get_patient_diagnosis(patient_id)
            if report:
                self.display_report(report)
                
                feedback_needed = input("\nWould you like to provide feedback on this report? (yes/no): ").strip().lower()
                if feedback_needed == 'yes':
                    feedback_text = input("Please enter your feedback: ").strip()
                    self.collect_feedback(patient_id, report, feedback_text)
            print("\n")

if __name__ == "__main__":
    app = MedicalDiagnosticAssistant()
    app.run_cli()