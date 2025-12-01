import random
from typing import List, Dict, Union, Any

class MedicalLLM:
    """Simulated Large Language Model for medical diagnosis with confidence estimation."""

    def __init__(self):
        self.knowledge_base = {
            "fever and rash": [("Measles", 0.90), ("Rubella", 0.75), ("Drug Reaction", 0.50)],
            "severe headache and stiff neck": [("Meningitis", 0.95), ("Migraine", 0.60), ("Tension Headache", 0.30)],
            "fatigue and joint pain": [("Lupus", 0.88), ("Rheumatoid Arthritis", 0.80), ("Chronic Fatigue Syndrome", 0.65)],
            "chest pain and shortness of breath": [("Heart Attack", 0.98), ("Anxiety Attack", 0.70), ("Pneumonia", 0.60)],
            "abdominal pain and nausea": [("Appendicitis", 0.92), ("Gastritis", 0.78), ("Food Poisoning", 0.65)],
        }

    def _simulate_confidence(self, diagnosis: str, base_confidence: float) -> float:
        """Simulates a slight variation in confidence for demonstration."""
        variation = random.uniform(-0.05, 0.05)
        return round(max(0.01, min(0.99, base_confidence + variation)), 2)

    def diagnose_with_confidence(self, symptoms: str) -> List[Dict[str, Union[str, float]]]:
        """Generates potential diagnoses with self-rated confidence scores."""
        symptoms_lower = symptoms.lower().strip()
        
        if symptoms_lower in self.knowledge_base:
            simulated_diagnoses_data = self.knowledge_base[symptoms_lower]
            results = []
            for diag, conf in simulated_diagnoses_data:
                results.append({
                    "diagnosis": diag,
                    "confidence": self._simulate_confidence(diag, conf)
                })
            return sorted(results, key=lambda x: x["confidence"], reverse=True)
        else:
            # Default or random diagnoses for unknown symptoms
            random_diagnoses = [
                ("Common Cold", 0.40),
                ("Flu", 0.55),
                ("Stress-related symptoms", 0.35)
            ]
            results = []
            for diag, conf in random_diagnoses:
                 results.append({
                    "diagnosis": diag,
                    "confidence": self._simulate_confidence(diag, conf)
                })
            return sorted(results, key=lambda x: x["confidence"], reverse=True)

class MedicalDiagnosticAssistant:
    """Orchestrates diagnosis and provides recommendations based on LLM confidence."""

    def __init__(self, llm: MedicalLLM, high_confidence_threshold: float = 0.85, moderate_confidence_threshold: float = 0.60):
        if not (0 <= high_confidence_threshold <= 1 and 0 <= moderate_confidence_threshold <= 1):
            raise ValueError("Confidence thresholds must be between 0 and 1.")
        if moderate_confidence_threshold >= high_confidence_threshold:
            raise ValueError("Moderate confidence threshold must be less than high confidence threshold.")

        self.llm = llm
        self.high_confidence_threshold = high_confidence_threshold
        self.moderate_confidence_threshold = moderate_confidence_threshold

    def get_diagnostic_recommendation(self, symptoms: str) -> Dict[str, Any]:
        """Provides a diagnostic recommendation and actionable advice based on LLM confidence."""
        llm_diagnoses = self.llm.diagnose_with_confidence(symptoms)

        if not llm_diagnoses:
            return {
                "recommendation": "No clear diagnosis",
                "advice": "Consider broader investigations or specialist consultation.",
                "diagnoses": []
            }
        
        # Take the top diagnosis for primary recommendation logic
        top_diagnosis = llm_diagnoses[0]
        diagnosis_name = top_diagnosis["diagnosis"]
        confidence_score = top_diagnosis["confidence"]

        recommendation = ""
        advice = ""

        if confidence_score >= self.high_confidence_threshold:
            recommendation = "High Confidence Diagnosis"
            advice = f"The AI has high confidence in '{diagnosis_name}'. Consider confirming with standard diagnostic procedures. No immediate need for extensive review unless other factors suggest otherwise."
        elif confidence_score >= self.moderate_confidence_threshold:
            recommendation = "Moderate Confidence Diagnosis"
            advice = f"The AI suggests '{diagnosis_name}' with moderate confidence. Further tests or a second opinion are highly recommended to confirm or rule out this diagnosis."
        else:
            recommendation = "Low Confidence Diagnosis / Flag for Review"
            advice = f"The AI's confidence in '{diagnosis_name}' is low. This case should be flagged for immediate human review, requiring comprehensive medical evaluation and potentially specialist consultation. Consider a wider differential diagnosis."

        return {
            "recommendation": recommendation,
            "advice": advice,
            "diagnoses": llm_diagnoses,
            "top_diagnosis_confidence": confidence_score
        }

if __name__ == "__main__":
    # --- Demonstration of the Medical Diagnostic Confidence Assistant ---

    # 1. Initialize the simulated LLM
    medical_llm = MedicalLLM()

    # 2. Initialize the Diagnostic Assistant with custom thresholds (optional)
    assistant = MedicalDiagnosticAssistant(medical_llm, high_confidence_threshold=0.88, moderate_confidence_threshold=0.65)

    print("\n--- Case 1: High Confidence Scenario ---")
    symptoms1 = "severe headache and stiff neck"
    result1 = assistant.get_diagnostic_recommendation(symptoms1)
    print(f"Symptoms: {symptoms1}")
    print(f"Recommendation: {result1['recommendation']}")
    print(f"Top Diagnosis: {result1['diagnoses'][0]['diagnosis']} (Confidence: {result1['top_diagnosis_confidence']:.2f})")
    print(f"Advice: {result1['advice']}")
    print("All Diagnoses:")
    for d in result1['diagnoses']:
        print(f"  - {d['diagnosis']}: {d['confidence']:.2f}")

    print("\n--- Case 2: Moderate Confidence Scenario ---")
    symptoms2 = "fatigue and joint pain"
    result2 = assistant.get_diagnostic_recommendation(symptoms2)
    print(f"Symptoms: {symptoms2}")
    print(f"Recommendation: {result2['recommendation']}")
    print(f"Top Diagnosis: {result2['diagnoses'][0]['diagnosis']} (Confidence: {result2['top_diagnosis_confidence']:.2f})")
    print(f"Advice: {result2['advice']}")
    print("All Diagnoses:")
    for d in result2['diagnoses']:
        print(f"  - {d['diagnosis']}: {d['confidence']:.2f}")

    print("\n--- Case 3: Low Confidence / Review Scenario ---")
    symptoms3 = "persistent cough and mild fever"
    result3 = assistant.get_diagnostic_recommendation(symptoms3)
    print(f"Symptoms: {symptoms3}")
    print(f"Recommendation: {result3['recommendation']}")
    if result3['diagnoses']:
        print(f"Top (Low Confidence) Diagnosis: {result3['diagnoses'][0]['diagnosis']} (Confidence: {result3['top_diagnosis_confidence']:.2f})")
    else:
        print("No specific diagnoses provided by LLM.")
    print(f"Advice: {result3['advice']}")
    if result3['diagnoses']:
        print("All Diagnoses:")
        for d in result3['diagnoses']:
            print(f"  - {d['diagnosis']}: {d['confidence']:.2f}")

    print("\n--- Case 4: Another Moderate Confidence Scenario ---")
    symptoms4 = "abdominal pain and nausea"
    result4 = assistant.get_diagnostic_recommendation(symptoms4)
    print(f"Symptoms: {symptoms4}")
    print(f"Recommendation: {result4['recommendation']}")
    print(f"Top Diagnosis: {result4['diagnoses'][0]['diagnosis']} (Confidence: {result4['top_diagnosis_confidence']:.2f})")
    print(f"Advice: {result4['advice']}")
    print("All Diagnoses:")
    for d in result4['diagnoses']:
        print(f"  - {d['diagnosis']}: {d['confidence']:.2f}")