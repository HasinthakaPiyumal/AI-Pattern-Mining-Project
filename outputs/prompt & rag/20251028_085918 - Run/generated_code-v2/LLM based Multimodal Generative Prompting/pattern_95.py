import os
from PIL import Image
import random

# --- 1. Data Ingestion Layer ---
class PatientDataLoader:
    def load_image(self, image_path):
        try:
            return Image.open(image_path)
        except FileNotFoundError:
            return None # Simulate image not found

    def load_text_history(self, text_content):
        return text_content

    def load_lab_report(self, lab_data):
        return lab_data

# --- 2. Modality-Specific Analysis Modules (Duties) ---

class ImageAnalysisModule:
    def __init__(self, model=None):
        self.model = model # Placeholder for a real DL model (e.g., PyTorch model)

    def analyze(self, image):
        if image is None:
            return {"image_analysis_status": "No image provided"}
        
        # Simulate image analysis: detect anomalies based on a simple heuristic or a dummy model output
        findings = {"image_anomalies": [], "potential_issues": []}
        
        # In a real scenario, this would use self.model to process the image
        if random.random() > 0.6: # Simulate finding an anomaly
            findings["image_anomalies"].append("Nodule detected in upper right lung field")
            findings["potential_issues"].append("Respiratory concern")
        if random.random() > 0.8:
            findings["image_anomalies"].append("Evidence of bone fracture in distal radius")
            findings["potential_issues"].append("Orthopedic concern")
            
        findings["image_analysis_summary"] = "Image analysis completed."
        return findings

class TextAnalysisModule:
    def __init__(self, nlp_model=None):
        self.nlp_model = nlp_model # Placeholder for an NLP model (e.g., transformers pipeline)

    def analyze(self, text_history):
        # Simulate text analysis: extract keywords or identify symptoms
        findings = {"symptoms": [], "medical_history_notes": []}
        
        text_history_lower = text_history.lower()
        if "fever" in text_history_lower or "high temperature" in text_history_lower:
            findings["symptoms"].append("Fever")
        if "cough" in text_history_lower or "chest pain" in text_history_lower:
            findings["symptoms"].append("Cough/Chest Pain")
        if "diabetes" in text_history_lower:
            findings["medical_history_notes"].append("History of Diabetes")
        if "hypertension" in text_history_lower:
            findings["medical_history_notes"].append("History of Hypertension")
            
        findings["text_analysis_summary"] = "Text history interpreted."
        return findings

class LabReportAnalysisModule:
    def analyze(self, lab_data):
        findings = {"critical_values": [], "abnormal_results": [], "lab_summary": ""}

        if not lab_data:
            findings["lab_summary"] = "No lab data provided."
            return findings

        for test, result_info in lab_data.items():
            value = result_info.get("value")
            unit = result_info.get("unit")
            ref_range = result_info.get("reference_range")

            if value is None or ref_range is None:
                continue
            
            if isinstance(ref_range, tuple) and (value < ref_range[0] or value > ref_range[1]):
                findings["abnormal_results"].append(f"{test}: {value} {unit} (Ref: {ref_range[0]}-{ref_range[1]}) - Abnormal")
                if test in ["White Blood Cell Count", "C-Reactive Protein"] and (value > ref_range[1] * 1.5):
                     findings["critical_values"].append(f"{test} critically high")
            elif isinstance(ref_range, str) and result_info.get("status") == "abnormal":
                findings["abnormal_results"].append(f"{test}: {value} {unit} - Abnormal")

        findings["lab_summary"] = "Lab report analysis completed."
        return findings

# --- 3. Orchestrator (DDCoT Core) ---
class MedicalDiagnosisOrchestrator:
    def __init__(self):
        self.data_loader = PatientDataLoader()
        self.image_analyzer = ImageAnalysisModule()
        self.text_analyzer = TextAnalysisModule()
        self.lab_analyzer = LabReportAnalysisModule()

    def perform_diagnosis(self, image_path, text_history, lab_report_data):
        print("\n--- Starting Multimodal Medical Diagnosis (DDCoT) ---")
        full_findings = {}

        # Step 1: Load Raw Data
        print("1. Loading patient data...")
        image = self.data_loader.load_image(image_path)
        text = self.data_loader.load_text_history(text_history)
        lab_data = self.data_loader.load_lab_report(lab_report_data)
        
        # Step 2: Perform Modality-Specific Analysis (Duty Distinct Chain of Thought)
        print("2. Analyzing medical images...")
        image_findings = self.image_analyzer.analyze(image)
        full_findings["image_analysis"] = image_findings
        print(f"   Image Findings: {image_findings}")

        print("3. Interpreting textual medical history...")
        text_findings = self.text_analyzer.analyze(text)
        full_findings["text_analysis"] = text_findings
        print(f"   Text Findings: {text_findings}")

        print("4. Evaluating lab results...")
        lab_findings = self.lab_analyzer.analyze(lab_data)
        full_findings["lab_analysis"] = lab_findings
        print(f"   Lab Findings: {lab_findings}")

        # Step 3: Synthesize Findings and Propose Differential Diagnosis
        print("\n5. Synthesizing all findings for differential diagnosis...")
        differential_diagnosis = []
        suggested_actions = []

        # Combine insights from image, text, and labs
        # Example logic for synthesis (highly simplified)
        if "Respiratory concern" in image_findings.get("potential_issues", []) and "Cough/Chest Pain" in text_findings.get("symptoms", []):
            differential_diagnosis.append("Possible Pneumonia or Bronchitis")
            suggested_actions.append("Recommend further lung imaging (e.g., HRCT) and sputum culture.")

        if "History of Diabetes" in text_findings.get("medical_history_notes", []) and any("critically high" in val for val in lab_findings.get("critical_values", [])):
            differential_diagnosis.append("Diabetic Ketoacidosis (DKA) or other metabolic crisis")
            suggested_actions.append("Immediate blood glucose and electrolyte panel, consider insulin therapy.")
            
        if any("fracture" in anomaly for anomaly in image_findings.get("image_anomalies", [])):
            differential_diagnosis.append("Bone Fracture")
            suggested_actions.append("Consult Orthopedics for immobilization/treatment.")

        if not differential_diagnosis:
            differential_diagnosis.append("No clear differential diagnosis based on current data. Further investigation needed.")
            suggested_actions.append("Monitor patient, re-evaluate symptoms, consider more specialized tests.")

        print("\n--- Diagnosis Summary ---")
        print(f"Full Consolidated Findings: {full_findings}")
        print(f"\nDifferential Diagnosis: {'; '.join(differential_diagnosis)}")
        print(f"Suggested Investigative/Treatment Actions: {'; '.join(suggested_actions)}")
        print("--- End Diagnosis ---")
        
        return {
            "full_findings": full_findings,
            "differential_diagnosis": differential_diagnosis,
            "suggested_actions": suggested_actions
        }

# --- Main Execution for Demonstration ---
if __name__ == "__main__":
    # Create a dummy image file for testing
    dummy_image_path = "dummy_xray.png"
    try:
        Image.new("RGB", (100, 100), color = "red").save(dummy_image_path)
    except Exception as e:
        print(f"Could not create dummy image (may not have Pillow installed or write permission): {e}")
        dummy_image_path = None

    # Sample Patient Data 1 (Respiratory concern)
    patient_data_1 = {
        "image_path": dummy_image_path,
        "text_history": "Patient presents with persistent cough for 3 days, mild fever, and shortness of breath. No significant medical history.",
        "lab_report": {
            "White Blood Cell Count": {"value": 15.2, "unit": "x10^9/L", "reference_range": (4.0, 10.0)},
            "C-Reactive Protein": {"value": 85, "unit": "mg/L", "reference_range": (0.0, 5.0)},
            "Hemoglobin": {"value": 14.5, "unit": "g/dL", "reference_range": (13.0, 17.0)},
        }
    }

    # Sample Patient Data 2 (Diabetes related metabolic issue)
    patient_data_2 = {
        "image_path": None, # No image for this case
        "text_history": "Patient is a known diabetic, complaining of increased thirst and frequent urination. Has felt weak for the past 24 hours.",
        "lab_report": {
            "Blood Glucose": {"value": 450, "unit": "mg/dL", "reference_range": (70, 120)},
            "Sodium": {"value": 132, "unit": "mmol/L", "reference_range": (135, 145)},
            "Potassium": {"value": 3.0, "unit": "mmol/L", "reference_range": (3.5, 5.0)},
        }
    }

    # Initialize and run the orchestrator
    orchestrator = MedicalDiagnosisOrchestrator()

    print("\n===== DIAGNOSIS CASE 1 =====")
    orchestrator.perform_diagnosis(**patient_data_1)

    print("\n===== DIAGNOSIS CASE 2 =====")
    orchestrator.perform_diagnosis(**patient_data_2)

    # Clean up dummy image
    if dummy_image_path and os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)
