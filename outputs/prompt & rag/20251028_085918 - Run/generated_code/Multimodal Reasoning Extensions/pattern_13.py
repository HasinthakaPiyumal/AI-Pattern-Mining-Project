"""Utility functions for loading patient data."""

def load_patient_data(patient_id: str) -> dict:
    """
    Simulates loading multimodal patient data based on a patient ID.
    In a real application, this would fetch data from a database or storage system.
    """
    print(f"[DataLoader] Loading data for patient: {patient_id}")
    # Mock data for demonstration purposes
    if patient_id == "P001":
        return {
            "patient_id": patient_id,
            "imaging_paths": {
                "xray_chest": "data/P001_chest_xray.png",
                "mri_head": "data/P001_head_mri.png"
            },
            "lab_reports": [
                "CBC: WBC 12.5 (High), HGB 13.0, PLT 250.",
                "CRP: 15 mg/L (Elevated).",
                "Blood Culture: Negative."
            ],
            "patient_history": "Patient P001, 55 y/o male, presented with fever, cough, and shortness of breath for 3 days. History of asthma and hypertension.",
            "symptoms": "Fever, cough, shortness of breath, fatigue."
        }
    elif patient_id == "P002":
        return {
            "patient_id": patient_id,
            "imaging_paths": {
                "xray_chest": "data/P002_chest_xray.png",
                # No MRI for this patient
            },
            "lab_reports": [
                "CBC: WBC 8.2, HGB 14.5, PLT 280.",
                "Electrolytes: All within normal limits."
            ],
            "patient_history": "Patient P002, 30 y/o female, routine check-up. No significant medical history.",
            "symptoms": "None."
        }
    else:
        return {"patient_id": patient_id, "error": "Patient data not found"}
