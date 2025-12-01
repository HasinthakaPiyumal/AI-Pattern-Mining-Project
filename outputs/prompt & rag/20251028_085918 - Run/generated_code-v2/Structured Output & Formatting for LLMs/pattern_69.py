from pydantic import BaseModel, Field
import json
import re
from typing import List, Optional, Dict, Any

class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication.")
    dosage: str = Field(..., description="Dosage of the medication (e.g., '10mg', '2 pills').")
    frequency: str = Field(..., description="Frequency of administration (e.g., 'daily', 'twice a day').")
    duration: Optional[str] = Field(None, description="Optional duration for the medication (e.g., '7 days', 'until finished').")

class LifestyleChange(BaseModel):
    description: str = Field(..., description="Description of the lifestyle change (e.g., 'Increase daily water intake', 'Engage in light exercise').")

class FollowUp(BaseModel):
    type: str = Field(..., description="Type of follow-up (e.g., 'Doctor visit', 'Lab test', 'Imaging').")
    schedule: str = Field(..., description="Schedule for the follow-up (e.g., 'in 2 weeks', 'after 3 months', 'as needed').")

class MedicalTreatmentPlan(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    diagnosis: str = Field(..., description="The diagnosed condition.")
    medications: List[Medication] = Field([], description="List of prescribed medications.")
    lifestyle_changes: List[LifestyleChange] = Field([], description="List of recommended lifestyle changes.")
    follow_ups: List[FollowUp] = Field([], description="List of scheduled follow-up appointments/tests.")
    notes: Optional[str] = Field(None, description="Any additional notes or general advice.")
    side_effects_warnings: Optional[str] = Field(None, description="General warnings about potential side effects.")

def generate_treatment_plan_llm_mock(patient_data: Dict[str, Any]) -> str:
    patient_id = patient_data.get("id", "UNKNOWN_PATIENT")
    diagnosis = patient_data.get("diagnosis", "unspecified condition")
    symptoms = ", ".join(patient_data.get("symptoms", []))

    return f"""
Patient ID: {patient_id}
Diagnosis: {diagnosis}

Based on the patient's symptoms ({symptoms}) and diagnosis, here is a proposed treatment plan:

Medications:
1. Take Acetaminophen 500mg daily for 7 days.
2. Take Amoxicillin 250mg twice a day until finished (approx. 10 days).

Lifestyle Changes:
*   Increase daily water intake to 2-3 liters.
*   Get at least 30 minutes of light exercise most days of the week.
*   Ensure 7-8 hours of sleep per night.

Follow-up:
*   Schedule a doctor visit in 2 weeks to reassess symptoms.
*   Consider a blood test after 3 months.

Notes: Monitor for any adverse reactions to medications. If symptoms worsen, contact your doctor immediately.
Side Effects: Drowsiness may occur with Acetaminophen. Gastrointestinal upset possible with Amoxicillin.
"""

def extract_and_structure_plan(natural_language_plan: str, patient_id: str, diagnosis: str) -> MedicalTreatmentPlan:
    medications: List[Medication] = []
    lifestyle_changes: List[LifestyleChange] = []
    follow_ups: List[FollowUp] = []
    notes: Optional[str] = None
    side_effects_warnings: Optional[str] = None

    med_pattern = re.compile(r"\d+\.\s*Take\s+([A-Za-z0-9\s]+?)\s+([0-9\sA-Za-z]+?)\s+(daily|twice a day|three times a day|as needed|every\s+\d+\s+hours|until finished)\s*(?:\((approx\.\s*\d+\s+days|for\s+\d+\s+days)\))?\.")
    lifestyle_pattern = re.compile(r"\*\s*(.+)")
    follow_up_pattern = re.compile(r"\*\s*(Schedule a|Consider a|Get a)\s*([A-Za-z\s]+?)\s*(in\s+\d+\s+weeks|after\s+\d+\s+months|as needed|to reassess symptoms|to check progress)\s*\.?")
    notes_pattern = re.compile(r"Notes:\s*(.+)")
    side_effects_pattern = re.compile(r"Side Effects:\s*(.+)")

    sections = natural_language_plan.split('\n\n')
    
    for section in sections:
        if "Medications:" in section:
            for line in section.split('\n'):
                match = med_pattern.search(line)
                if match:
                    name, dosage, frequency, duration_group = match.groups()
                    medications.append(Medication(
                        name=name.strip(),
                        dosage=dosage.strip(),
                        frequency=frequency.strip(),
                        duration=duration_group.strip() if duration_group else None
                    ))
        elif "Lifestyle Changes:" in section:
            for line in section.split('\n'):
                match = lifestyle_pattern.search(line)
                if match:
                    lifestyle_changes.append(LifestyleChange(description=match.group(1).strip()))
        elif "Follow-up:" in section:
            for line in section.split('\n'):
                match = follow_up_pattern.search(line)
                if match:
                    follow_ups.append(FollowUp(type=match.group(2).strip(), schedule=match.group(3).strip()))
        
        notes_match = notes_pattern.search(section)
        if notes_match: 
            notes = notes_match.group(1).strip()
        
        side_effects_match = side_effects_pattern.search(section)
        if side_effects_match:
            side_effects_warnings = side_effects_match.group(1).strip()

    return MedicalTreatmentPlan(
        patient_id=patient_id,
        diagnosis=diagnosis,
        medications=medications,
        lifestyle_changes=lifestyle_changes,
        follow_ups=follow_ups,
        notes=notes,
        side_effects_warnings=side_effects_warnings
    )

def evaluate_plan(structured_plan: MedicalTreatmentPlan) -> Dict[str, Any]:
    evaluation_results = {
        "status": "Passed",
        "warnings": [],
        "suggestions": []
    }

    if not structured_plan.medications:
        evaluation_results["status"] = "Failed"
        evaluation_results["warnings"].append("No medications prescribed. Is this intentional?")
    
    if not structured_plan.follow_ups:
        evaluation_results["warnings"].append("No follow-up scheduled. Ensure patient monitoring.")

    med_names = {m.name.lower() for m in structured_plan.medications}
    if "acetaminophen" in med_names and "amoxicillin" in med_names:
         evaluation_results["warnings"].append("Check for potential interactions between Acetaminophen and Amoxicillin (mock check).")

    return evaluation_results

def main():
    patient_data = {
        "id": "P001",
        "symptoms": ["fever", "sore throat", "fatigue"],
        "history": ["mild allergies"],
        "diagnosis": "Viral Infection"
    }

    print("\n--- Step 1: LLM Generates Natural Language Plan ---")
    natural_plan = generate_treatment_plan_llm_mock(patient_data)
    print(natural_plan)

    print("\n--- Step 2: Extracting and Structuring Plan ---")
    structured_plan = extract_and_structure_plan(
        natural_plan,
        patient_id=patient_data["id"],
        diagnosis=patient_data["diagnosis"]
    )
    print("Structured Plan (JSON output):\n")
    print(json.dumps(structured_plan.model_dump(), indent=4))

    print("\n--- Step 3: Evaluating Structured Plan ---")
    evaluation = evaluate_plan(structured_plan)
    print(json.dumps(evaluation, indent=4))

if __name__ == "__main__":
    main()