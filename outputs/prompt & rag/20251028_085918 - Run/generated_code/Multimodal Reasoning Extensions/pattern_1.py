import json
from PIL import Image
import io
import base64

# Placeholder for transformers and other heavy libraries. 
# In a real application, these would be loaded and used properly.
# For demonstration, we'll simulate their behavior.

class SymptomAnalyzer:
    def __init__(self):
        # In a real application, you would load an LLM here, e.g.,
        # from transformers import pipeline
        # self.llm_pipeline = pipeline("text-generation", model="distilbert-base-uncased")
        pass

    def analyze(self, symptoms_text: str) -> dict:
        """Simulates LLM-based symptom analysis."""
        print(f"[SymptomAnalyzer] Analyzing symptoms: {symptoms_text[:50]}...")
        # Simulate LLM output for symptom extraction
        extracted_symptoms = {
            "primary_complaint": "abdominal pain" if "abdominal pain" in symptoms_text.lower() else "headache",
            "severity": "severe",
            "duration": "3 days",
            "history": "patient reports similar episodes in the past"
        }
        print(f"[SymptomAnalyzer] Extracted: {extracted_symptoms}")
        return extracted_symptoms

class MedicalImageInterpreter:
    def __init__(self):
        # In a real application, you would load a vision model here, e.g.,
        # from transformers import BlipProcessor, BlipForConditionalGeneration
        # self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        pass

    def interpret(self, image_data_base64: str) -> dict:
        """Simulates image interpretation (e.g., captioning or anomaly detection)."""
        print("[MedicalImageInterpreter] Interpreting medical image...")
        try:
            # Decode base64 image data for simulated processing
            image_bytes = base64.b64decode(image_data_base64)
            image = Image.open(io.BytesIO(image_bytes))
            # In a real scenario, pass 'image' to your vision model
            # For demonstration, we'll return a fixed finding
            findings = {
                "image_type": "X-ray",
                "identified_anomalies": "Possible pneumonia in right lung base",
                "confidence": 0.85
            }
            print(f"[MedicalImageInterpreter] Findings: {findings}")
            return findings
        except Exception as e:
            print(f"[MedicalImageInterpreter] Error processing image: {e}")
            return {"error": str(e)}

class LabResultEvaluator:
    def __init__(self):
        # Optionally, load an LLM for more nuanced interpretation here
        pass

    def evaluate(self, lab_results: dict) -> dict:
        """Evaluates lab test results, flagging abnormalities and providing preliminary interpretations."""
        print("[LabResultEvaluator] Evaluating lab results...")
        abnormal_results = {}
        interpretations = []

        for test_name, result in lab_results.items():
            value = result.get("value")
            unit = result.get("unit")
            reference_range = result.get("reference_range") # e.g., "5.0-10.0"

            if value is None or reference_range is None:
                interpretations.append(f"Skipping {test_name}: incomplete data.")
                continue

            try:
                min_ref, max_ref = map(float, reference_range.split('-'))
                if not (min_ref <= value <= max_ref):
                    abnormal_results[test_name] = {
                        "value": value,
                        "unit": unit,
                        "reference_range": reference_range,
                        "status": "abnormal"
                    }
                    interpretations.append(
                        f"{test_name} is {value}{unit}, which is outside the normal range ({reference_range}{unit})."
                    )
                else:
                    interpretations.append(
                        f"{test_name} is {value}{unit}, which is within the normal range ({reference_range}{unit})."
                    )
            except ValueError:
                interpretations.append(f"Could not parse reference range for {test_name}: {reference_range}")

        # Simulate LLM adding more context to abnormal results
        if abnormal_results:
            llm_interpretation = "Considering the abnormal findings, further investigation into potential inflammatory markers or infection is recommended."
            interpretations.append(llm_interpretation)

        final_evaluation = {
            "abnormal_findings": abnormal_results,
            "interpretations": interpretations
        }
        print(f"[LabResultEvaluator] Evaluation: {final_evaluation}")
        return final_evaluation

class FindingsSynthesizer:
    def __init__(self):
        # Load a powerful LLM for synthesis, e.g., using OpenAI API or a large open-source model
        # self.llm_synthesizer = ...
        pass

    def synthesize(self, symptom_analysis: dict, image_findings: dict, lab_evaluation: dict) -> dict:
        """Synthesizes findings from all duties into a comprehensive diagnostic report."""
        print("[FindingsSynthesizer] Synthesizing all findings...")

        # Construct a comprehensive prompt for the LLM based on all inputs
        prompt_parts = [
            "Based on the following patient data, provide a differential diagnosis, reasoning, and recommend further steps:",
            "--- Symptoms ---",
            f"Primary complaint: {symptom_analysis.get('primary_complaint')}",
            f"Severity: {symptom_analysis.get('severity')}",
            f"Duration: {symptom_analysis.get('duration')}",
            f"History: {symptom_analysis.get('history')}",
            "--- Image Findings ---",
            f"Image Type: {image_findings.get('image_type')}",
            f"Identified Anomalies: {image_findings.get('identified_anomalies')}",
            f"Confidence: {image_findings.get('confidence')}",
            "--- Lab Results ---",
            "Abnormal Findings: " + json.dumps(lab_evaluation.get('abnormal_findings', {})),
            "Lab Interpretations: " + "\n".join(lab_evaluation.get('interpretations', []))
        ]
        full_prompt = "\n".join(prompt_parts)
        
        # Simulate LLM generating a diagnosis based on the prompt
        differential_diagnosis = [
            "1. Pneumonia (supported by image findings and potentially elevated white blood cell count)",
            "2. Bronchitis (less likely given image findings, but possible)",
            "3. Gastroesophageal Reflux Disease (GERD) if abdominal pain is primary, but image/labs point elsewhere"
        ]
        reasoning = (
            "The patient presents with abdominal pain (symptom) and the X-ray shows possible pneumonia in the right lung base. "
            "Lab results might indicate inflammation, further supporting an infectious process. "
            "The combination strongly points towards a respiratory infection."
        )
        recommendations = [
            "1. Further chest imaging (e.g., CT scan) for detailed assessment.",
            "2. Sputum culture to identify causative organism.",
            "3. Start empirical antibiotic therapy."
        ]

        report = {
            "differential_diagnosis": differential_diagnosis,
            "reasoning": reasoning,
            "recommendations": recommendations,
            "full_prompt_for_llm": full_prompt # Include for debugging/audit
        }
        print(f"[FindingsSynthesizer] Report: {report}")
        return report

class MedicalDiagnosisAssistant:
    def __init__(self):
        self.symptom_analyzer = SymptomAnalyzer()
        self.image_interpreter = MedicalImageInterpreter()
        self.lab_evaluator = LabResultEvaluator()
        self.findings_synthesizer = FindingsSynthesizer()

    def diagnose(self, symptoms_text: str, medical_image_data_base64: str, lab_results: dict) -> dict:
        print("\n--- Starting Medical Diagnosis Assistant ---")
        
        # Duty 1: Symptom Analysis
        symptom_analysis = self.symptom_analyzer.analyze(symptoms_text)

        # Duty 2: Medical Image Interpretation
        image_findings = self.image_interpreter.interpret(medical_image_data_base64)

        # Duty 3: Lab Result Evaluation
        lab_evaluation = self.lab_evaluator.evaluate(lab_results)

        # Duty 4: Findings Synthesis
        final_report = self.findings_synthesizer.synthesize(
            symptom_analysis, image_findings, lab_evaluation
        )

        print("\n--- Diagnosis Complete ---")
        return {
            "symptom_analysis": symptom_analysis,
            "image_findings": image_findings,
            "lab_evaluation": lab_evaluation,
            "final_report": final_report
        }

# --- Example Usage ---
if __name__ == "__main__":
    # Sample Inputs
    sample_symptoms = (
        "The patient is a 45-year-old male presenting with severe abdominal pain for the past 3 days. "
        "He also reports a persistent cough and mild fever. No relevant past medical history."
    )

    # A tiny, valid base64 encoded PNG image (1x1 transparent pixel)
    # In a real app, this would be a full medical image.
    sample_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    sample_lab_results = {
        "White Blood Cell Count": {"value": 15.2, "unit": "x10^9/L", "reference_range": "4.0-10.0"},
        "C-Reactive Protein": {"value": 55.0, "unit": "mg/L", "reference_range": "<5.0"},
        "Hemoglobin": {"value": 14.5, "unit": "g/dL", "reference_range": "13.5-17.5"},
        "Glucose": {"value": 95, "unit": "mg/dL", "reference_range": "70-100"}
    }

    assistant = MedicalDiagnosisAssistant()
    diagnosis_output = assistant.diagnose(
        symptoms_text=sample_symptoms,
        medical_image_data_base64=sample_image_base64,
        lab_results=sample_lab_results
    )

    print("\n*** Final Diagnostic Report ***")
    print(json.dumps(diagnosis_output['final_report'], indent=4))

    # Another example with different symptoms to show flexibility
    print("\n--- Another Patient Scenario ---")
    sample_symptoms_2 = (
        "Patient reports persistent headache for a week, accompanied by nausea. "
        "No fever or visual disturbances. Has a history of migraines."
    )
    # For simplicity, using same image and lab results to show symptom analyzer changes
    diagnosis_output_2 = assistant.diagnose(
        symptoms_text=sample_symptoms_2,
        medical_image_data_base64=sample_image_base64,
        lab_results=sample_lab_results
    )
    print("\n*** Final Diagnostic Report (Scenario 2) ***")
    print(json.dumps(diagnosis_output_2['final_report'], indent=4))
