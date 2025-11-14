import random

class MedicalLLM:
    """Simulates a Large Language Model for medical diagnosis with XAI features."""

    def __init__(self):
        # In a real application, this would load a finetuned LLM model
        # For this simulation, we'll use predefined responses.
        pass

    def get_diagnosis(self, patient_data: dict) -> dict:
        """
        Generates a diagnosis, reasoning, and confidence score based on patient data.
        Demonstrates explicit reasoning paths and confidence estimations.
        """
        symptoms = patient_data.get("symptoms", "")
        medical_history = patient_data.get("medical_history", "")
        lab_results = patient_data.get("lab_results", "")
        imaging_analysis = patient_data.get("imaging_analysis", "")

        # Simple rule-based simulation for demonstration purposes
        if "fever" in symptoms.lower() and "cough" in symptoms.lower():
            diagnosis = "Common Cold or Flu"
            reasoning = (
                "Based on reported symptoms of fever and cough, common respiratory infections "
                "like the common cold or flu are highly suspected. Further testing (e.g., flu test) "
                "may be required for definitive diagnosis."
            )
            confidence = random.uniform(0.75, 0.95) # High confidence
        elif "chest pain" in symptoms.lower() and "shortness of breath" in symptoms.lower() and "cardiac markers elevated" in lab_results.lower():
            diagnosis = "Myocardial Infarction (Heart Attack)"
            reasoning = (
                "The combination of chest pain, shortness of breath, and elevated cardiac markers "
                "strongly indicates a myocardial infarction. Immediate medical attention and "
                "further cardiac investigations are critical."
            )
            confidence = random.uniform(0.90, 0.99) # Very high confidence
        elif "abdominal pain" in symptoms.lower() and "nausea" in symptoms.lower() and "appendicitis" in imaging_analysis.lower():
            diagnosis = "Acute Appendicitis"
            reasoning = (
                "Patient presents with abdominal pain and nausea, supported by imaging analysis "
                "suggesting appendicitis. Surgical consultation is recommended."
            )
            confidence = random.uniform(0.90, 0.98) # Very high confidence
        else:
            diagnosis = "Undetermined / Requires More Information"
            reasoning = (
                "The provided information is insufficient for a confident diagnosis. "
                "Please provide more detailed symptoms, complete medical history, and additional diagnostic test results."
            )
            confidence = random.uniform(0.40, 0.60) # Lower confidence, potential abstention candidate

        # Simulate controlled abstention for very low confidence cases
        if confidence < 0.50:
            diagnosis = "Abstain: Insufficient Data"
            reasoning += " The system is abstaining from a definitive diagnosis due to low confidence and lack of comprehensive data."
            confidence = 0.0 # Indicate abstention

        return {
            "diagnosis": diagnosis,
            "reasoning": reasoning,
            "confidence": round(confidence, 2)
        }