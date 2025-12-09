from pydantic import BaseModel
from typing import List, Dict, Any

class Abnormality(BaseModel):
    id: str
    type: str
    bbox: List[int]
    characteristics: Dict[str, Any] = {}

class PatientEHRData(BaseModel):
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    symptoms: List[str]
    lab_results: Dict[str, Any]
    medications: List[str]

class DifferentialDiagnosis(BaseModel):
    diagnosis: str
    likelihood: float
    justification: str

class MedicalDiagnosisAssistant:
    def __init__(self):
        pass

    def _preprocess_and_detect_abnormalities(self, image_path: str) -> List[Abnormality]:
        print(f"Processing image: {image_path} for abnormality detection.")
        dummy_abnormalities = [
            Abnormality(id="lesion_001", type="lesion", bbox=[100, 150, 200, 250]),
            Abnormality(id="tumor_001", type="tumor", bbox=[300, 350, 450, 500])
        ]
        return dummy_abnormalities

    def _characterize_abnormalities(self, abnormalities: List[Abnormality], image_path: str) -> List[Abnormality]:
        print(f"Characterizing abnormalities from image: {image_path}.")
        for abnormality in abnormalities:
            if abnormality.type == "lesion":
                abnormality.characteristics = {"size_mm": 15, "shape": "oval", "density": "low"}
            elif abnormality.type == "tumor":
                abnormality.characteristics = {"size_mm": 30, "shape": "irregular", "density": "high", "malignancy_score": 0.85}
        return abnormalities

    def _extract_ehr_data(self, ehr_text: str) -> PatientEHRData:
        print("Extracting data from EHR text.")
        dummy_ehr_data = PatientEHRData(
            patient_id="P12345",
            age=62,
            gender="Male",
            medical_history=["Hypertension", "Diabetes Type 2"],
            symptoms=["cough", "fatigue", "weight loss"],
            lab_results={"CRP": "elevated", "WBC": "normal"},
            medications=["Lisinopril", "Metformin"]
        )
        return dummy_ehr_data

    def _correlate_and_diagnose(
        self,
        characterized_abnormalities: List[Abnormality],
        patient_ehr_data: PatientEHRData
    ) -> List[DifferentialDiagnosis]:
        print("Correlating image findings with EHR data to generate differential diagnoses.")
        diagnoses = []
        for abnormality in characterized_abnormalities:
            if abnormality.type == "tumor" and "weight loss" in patient_ehr_data.symptoms:
                diagnoses.append(DifferentialDiagnosis(
                    diagnosis="Lung Carcinoma (suspected)",
                    likelihood=0.9,
                    justification=f"Irregular tumor of {abnormality.characteristics.get('size_mm', 'N/A')}mm with high density, combined with patient history of weight loss and cough."
                ))
            elif abnormality.type == "lesion" and patient_ehr_data.age > 60:
                diagnoses.append(DifferentialDiagnosis(
                    diagnosis="Benign Lung Nodule",
                    likelihood=0.7,
                    justification=f"Small oval lesion of {abnormality.characteristics.get('size_mm', 'N/A')}mm, common in older patients."
                ))
        return diagnoses

    def _generate_report(
        self,
        characterized_abnormalities: List[Abnormality],
        patient_ehr_data: PatientEHRData,
        differential_diagnoses: List[DifferentialDiagnosis]
    ) -> str:
        print("Generating diagnostic report.")
        report_parts = [
            f"DIAGNOSTIC REPORT FOR PATIENT ID: {patient_ehr_data.patient_id}",
            f"Age: {patient_ehr_data.age}, Gender: {patient_ehr_data.gender}",
            "\n--- Clinical Context ---",
            f"Medical History: {', '.join(patient_ehr_data.medical_history)}",
            f"Symptoms: {', '.join(patient_ehr_data.symptoms)}",
            f"Lab Results: {patient_ehr_data.lab_results}",
            f"Medications: {', '.join(patient_ehr_data.medications)}",
            "\n--- Imaging Findings ---"
        ]

        for abn in characterized_abnormalities:
            report_parts.append(f"  - Abnormality {abn.id} ({abn.type}):")
            report_parts.append(f"    Bounding Box: {abn.bbox}")
            for char_key, char_val in abn.characteristics.items():
                report_parts.append(f"    {char_key.replace('_', ' ').title()}: {char_val}")

        report_parts.append("\n--- Differential Diagnoses ---")
        if not differential_diagnoses:
            report_parts.append("  No specific differential diagnoses generated based on current data.")
        else:
            for diag in differential_diagnoses:
                report_parts.append(f"  - Diagnosis: {diag.diagnosis}")
                report_parts.append(f"    Likelihood: {diag.likelihood:.2f}")
                report_parts.append(f"    Justification: {diag.justification}")
        report_parts.append("\n--- End of Report ---")
        return "\n".join(report_parts)

    def diagnose(self, image_path: str, ehr_text: str) -> str:
        print("Starting medical diagnosis process (DDCoT pattern)...")

        detected_abnormalities = self._preprocess_and_detect_abnormalities(image_path)
        print(f"Detected {len(detected_abnormalities)} abnormalities.")

        characterized_abnormalities = self._characterize_abnormalities(detected_abnormalities, image_path)
        print("Abnormalities characterized.")

        patient_ehr_data = self._extract_ehr_data(ehr_text)
        print("EHR data extracted.")

        differential_diagnoses = self._correlate_and_diagnose(
            characterized_abnormalities, patient_ehr_data
        )
        print(f"Generated {len(differential_diagnoses)} differential diagnoses.")

        final_report = self._generate_report(
            characterized_abnormalities, patient_ehr_data, differential_diagnoses
        )
        print("Diagnostic report generated.")
        print("Diagnosis process completed.")
        return final_report