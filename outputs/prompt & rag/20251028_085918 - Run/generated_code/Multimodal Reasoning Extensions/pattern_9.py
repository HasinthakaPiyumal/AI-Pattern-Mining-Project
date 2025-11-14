import os
from PIL import Image
from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Mock LLM for demonstration purposes ---
class MockLLM:
    """A mock LLM class to simulate responses for demonstration."""
    def invoke(self, prompt: str) -> str:
        if "differential diagnoses" in prompt.lower():
            return "Possible diagnoses: pneumonia, bronchitis, asthma. Justification: Based on cough, shortness of breath, and chest X-ray findings."
        elif "likelihood" in prompt.lower():
            return "Pneumonia: High (80%), Bronchitis: Medium (15%), Asthma: Low (5%)."
        elif "recommendations" in prompt.lower():
            return "Recommended next steps: sputum culture, antibiotics for pneumonia, follow-up in 3 days."
        return f"Mock LLM response for: {prompt[:50]}..."

# --- Core Components --- 
class DataIngestion:
    """Handles loading and preprocessing of various patient data types."""
    def __init__(self):
        pass

    def load_patient_data(self, patient_id: str, image_paths: List[str], text_data: str, lab_results: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[DataIngestion] Loading data for Patient ID: {patient_id}")
        # Simulate basic validation for image paths
        for path in image_paths:
            if not os.path.exists(path):
                print(f"Warning: Image file not found at {path}")

        return {
            "patient_id": patient_id,
            "image_paths": image_paths,
            "text_data": text_data,
            "lab_results": lab_results
        }

class ImageAnalyzer:
    """Analyzes medical images for abnormalities."""
    def __init__(self):
        pass

    def analyze_images(self, image_paths: List[str]) -> str:
        print("[ImageAnalyzer] Analyzing images...")
        findings = []
        for i, path in enumerate(image_paths):
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    findings.append(f"Image {i+1} ({os.path.basename(path)}) - Dimensions: {width}x{height}. Simulated finding: Minor opacity in lower left lobe.")
            except FileNotFoundError:
                findings.append(f"Image {i+1} ({os.path.basename(path)}) - File not found, cannot analyze.")
            except Exception as e:
                findings.append(f"Image {i+1} ({os.path.basename(path)}) - Error analyzing: {e}")
        return "\n".join(findings)

class TextualDataExtractor:
    """Extracts key symptoms, conditions, and relevant history from unstructured text."""
    def __init__(self):
        pass

    def extract_data(self, text_data: str) -> Dict[str, Any]:
        print("[TextualDataExtractor] Extracting data from text...")
        # Simple keyword extraction for demonstration
        symptoms = []
        history = []

        if "cough" in text_data.lower():
            symptoms.append("cough")
        if "fever" in text_data.lower():
            symptoms.append("fever")
        if "shortness of breath" in text_data.lower():
            symptoms.append("shortness of breath")
        if "smoking history" in text_data.lower():
            history.append("smoking history")
        if "diabetes" in text_data.lower():
            history.append("diabetes")

        return {
            "extracted_symptoms": ", ".join(symptoms) if symptoms else "None",
            "extracted_history": ", ".join(history) if history else "None",
            "summary": text_data[:100] + "..." if len(text_data) > 100 else text_data # Simplified summary
        }

class LabResultCorrelator:
    """Correlates lab results with known medical conditions and findings."""
    def __init__(self):
        pass

    def correlate_results(self, lab_results: Dict[str, Any]) -> str:
        print("[LabResultCorrelator] Correlating lab results...")
        insights = []
        if "CRP" in lab_results and lab_results["CRP"] > 5:
            insights.append(f"Elevated C-Reactive Protein (CRP: {lab_results['CRP']}) indicates inflammation.")
        if "WBC" in lab_results and lab_results["WBC"] > 11000:
            insights.append(f"Elevated White Blood Cell count (WBC: {lab_results['WBC']}) suggests infection.")
        if "Glucose" in lab_results and lab_results["Glucose"] > 125:
            insights.append(f"High Glucose level (Glucose: {lab_results['Glucose']}) might indicate diabetes or pre-diabetes.")
        
        return "\n".join(insights) if insights else "No significant lab abnormalities found."

class MultimodalSynthesizer:
    """Synthesizes findings from all preceding modules to propose differential diagnoses."""
    def __init__(self, llm_client: Any):
        self.llm = llm_client
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a medical diagnostic AI. Synthesize the provided multimodal findings to propose a list of differential diagnoses with brief justifications."),
                ("user", "Image findings:\n{image_findings}\n\nTextual findings:\n{text_findings}\n\nLab insights:\n{lab_insights}\n\nBased on these, what are the top differential diagnoses and why?")
            ]
        )
        self.chain = {"image_findings": RunnablePassthrough(), "text_findings": RunnablePassthrough(), "lab_insights": RunnablePassthrough()} | self.prompt | StrOutputParser()

    def synthesize_findings(self, image_findings: str, text_findings: Dict[str, Any], lab_insights: str) -> str:
        print("[MultimodalSynthesizer] Synthesizing multimodal findings...")
        text_summary = f"Symptoms: {text_findings['extracted_symptoms']}. History: {text_findings['extracted_history']}. Summary: {text_findings['summary']}"
        return self.chain.invoke({"image_findings": image_findings, "text_findings": text_summary, "lab_insights": lab_insights})

class DiagnosisEvaluator:
    """Evaluates the likelihood of each differential diagnosis based on all gathered evidence."""
    def __init__(self, llm_client: Any):
        self.llm = llm_client
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a medical AI. Evaluate the provided differential diagnoses against all available patient findings and assign a likelihood score (e.g., High, Medium, Low) or a percentage."),
                ("user", "Differential diagnoses:\n{differential_diagnoses}\n\nAll patient findings:\n{all_findings}\n\nEvaluate the likelihood of each diagnosis:")
            ]
        )
        self.chain = {"differential_diagnoses": RunnablePassthrough(), "all_findings": RunnablePassthrough()} | self.prompt | StrOutputParser()

    def evaluate_diagnoses(self, differential_diagnoses: str, all_findings: str) -> str:
        print("[DiagnosisEvaluator] Evaluating differential diagnoses...")
        return self.chain.invoke({"differential_diagnoses": differential_diagnoses, "all_findings": all_findings})

class RecommendationEngine:
    """Proposes further diagnostic tests or treatment plans."""
    def __init__(self, llm_client: Any):
        self.llm = llm_client
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a medical AI. Based on the most likely diagnoses, propose further diagnostic tests and potential treatment plans."),
                ("user", "Likely diagnoses with likelihoods:\n{likely_diagnoses}\n\nWhat are the recommended next diagnostic steps and potential treatment considerations?")
            ]
        )
        self.chain = {"likely_diagnoses": RunnablePassthrough()} | self.prompt | StrOutputParser()

    def propose_recommendations(self, likely_diagnoses: str) -> str:
        print("[RecommendationEngine] Proposing recommendations...")
        return self.chain.invoke({"likely_diagnoses": likely_diagnoses})


class DDCoTMedicalAssistant:
    """Orchestrates the entire diagnostic workflow following the DDCoT pattern."""
    def __init__(self, llm_client: Any):
        self.data_ingestion = DataIngestion()
        self.image_analyzer = ImageAnalyzer()
        self.textual_data_extractor = TextualDataExtractor()
        self.lab_result_correlator = LabResultCorrelator()
        self.multimodal_synthesizer = MultimodalSynthesizer(llm_client)
        self.diagnosis_evaluator = DiagnosisEvaluator(llm_client)
        self.recommendation_engine = RecommendationEngine(llm_client)

    def diagnose_patient(self, patient_id: str, image_paths: List[str], text_data: str, lab_results: Dict[str, Any]) -> Dict[str, str]:
        print("\n--- Starting DDCoT Medical Diagnosis ---")

        # Step 1: Data Ingestion
        patient_data = self.data_ingestion.load_patient_data(patient_id, image_paths, text_data, lab_results)
        print("Data Ingested.\n")

        # Step 2: Visual Reasoning (Image Analysis)
        image_findings = self.image_analyzer.analyze_images(patient_data["image_paths"])
        print("Image Analysis Complete.\n")

        # Step 3: Linguistic Reasoning (Textual Data Extraction)
        text_findings = self.textual_data_extractor.extract_data(patient_data["text_data"])
        print("Textual Data Extraction Complete.\n")

        # Step 4: Data Integration & Reasoning (Lab Result Correlation)
        lab_insights = self.lab_result_correlator.correlate_results(patient_data["lab_results"])
        print("Lab Result Correlation Complete.\n")

        # Combine all findings for synthesis and evaluation steps
        all_findings_str = (
            f"Image Findings: {image_findings}\n"
            f"Textual Findings: Symptoms: {text_findings['extracted_symptoms']}, History: {text_findings['extracted_history']}, Summary: {text_findings['summary']}\n"
            f"Lab Insights: {lab_insights}"
        )

        # Step 5: Multimodal Synthesis (Propose Differential Diagnoses)
        differential_diagnoses = self.multimodal_synthesizer.synthesize_findings(image_findings, text_findings, lab_insights)
        print("Differential Diagnoses Proposed.\n")

        # Step 6: Probabilistic Reasoning (Evaluate Diagnoses)
        likely_diagnoses = self.diagnosis_evaluator.evaluate_diagnoses(differential_diagnoses, all_findings_str)
        print("Diagnoses Evaluated.\n")

        # Step 7: Actionable Recommendation (Propose Next Steps)
        recommendations = self.recommendation_engine.propose_recommendations(likely_diagnoses)
        print("Recommendations Proposed.\n")

        final_report = {
            "patient_id": patient_id,
            "image_findings": image_findings,
            "text_findings": text_findings,
            "lab_insights": lab_insights,
            "differential_diagnoses": differential_diagnoses,
            "likely_diagnoses": likely_diagnoses,
            "recommendations": recommendations
        }

        print("--- DDCoT Medical Diagnosis Complete ---")
        return final_report

if __name__ == "__main__":
    # Create a dummy image file for demonstration
    dummy_image_path = "dummy_xray.png"
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (600, 400), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10,10), "Simulated X-Ray Image", fill=(255,255,0))
        img.save(dummy_image_path)
        print(f"Created dummy image: {dummy_image_path}")
    except ImportError:
        print("Pillow not installed, cannot create dummy image. Please install with 'pip install Pillow'.")
        dummy_image_path = None # Set to None if Pillow isn't available

    # Initialize the LLM client (using the MockLLM)
    mock_llm = MockLLM()

    # Initialize the DDCoT Medical Assistant
    assistant = DDCoTMedicalAssistant(llm_client=mock_llm)

    # Sample Patient Data
    patient_data = {
        "patient_id": "P1001",
        "image_paths": [dummy_image_path] if dummy_image_path else [],
        "text_data": "Patient presents with persistent cough, mild fever, and shortness of breath for 3 days. Has a history of smoking for 10 years.",
        "lab_results": {"CRP": 8.5, "WBC": 13500, "Glucose": 95}
    }

    # Run the diagnostic process
    report = assistant.diagnose_patient(**patient_data)

    print("\n--- Final Diagnostic Report --- ")
    for key, value in report.items():
        print(f"{key.replace('_', ' ').title()}:\n{value}\n")

    # Clean up dummy image
    if dummy_image_path and os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)
        print(f"Cleaned up dummy image: {dummy_image_path}")
