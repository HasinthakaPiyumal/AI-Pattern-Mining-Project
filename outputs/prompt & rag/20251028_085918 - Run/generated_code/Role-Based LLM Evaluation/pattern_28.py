import json

class MedicalDiagnosisModel:
    def __init__(self, name="Primary AI Diagnoser"):
        self.name = name

    def generate_diagnosis(self, patient_data: dict) -> dict:
        """Simulates an initial AI diagnosis based on patient data."""
        print(f"[{self.name}] Generating initial diagnosis for patient: {patient_data['name']}")
        # In a real system, this would involve complex ML models
        # For this example, we'll create a simple, potentially flawed diagnosis
        diagnosis = {
            "patient_name": patient_data["name"],
            "symptoms": patient_data.get("symptoms", []),
            "initial_diagnosis": "Possible general malaise",
            "suggested_tests": [],
            "treatment_plan": "Monitor symptoms",
            "confidence_score": 0.6,
            "patient_data": patient_data # Include for ethicist check
        }

        if "fatigue" in patient_data.get("symptoms", []) and "sore throat" in patient_data.get("symptoms", []):
            diagnosis["initial_diagnosis"] = "Possible common cold"
            diagnosis["suggested_tests"].append("general physical exam")
            diagnosis["treatment_plan"] = "Rest, hydration, symptomatic relief"
            diagnosis["confidence_score"] = 0.7
        
        if "severe_headache" in patient_data.get("symptoms", []) or "blurred vision" in patient_data.get("symptoms", []):
            diagnosis["initial_diagnosis"] = "Migraine or other neurological concern"
            diagnosis["suggested_tests"].append("neurological exam")
            diagnosis["treatment_plan"] = "Pain relief, consider specialist referral if persistent"
            diagnosis["confidence_score"] = 0.8
        
        if "chest_pain" in patient_data.get("symptoms", []) and "shortness of breath" in patient_data.get("symptoms", []):
            diagnosis["initial_diagnosis"] = "Possible cardiac event or severe respiratory issue"
            diagnosis["suggested_tests"].extend(["ECG", "blood tests (cardiac markers)", "chest X-ray"])
            diagnosis["treatment_plan"] = "Immediate medical attention, monitor vitals"
            diagnosis["confidence_score"] = 0.95

        return diagnosis

class PersonaAgent:
    def __init__(self, name: str, role_description: str):
        self.name = name
        self.role_description = role_description

    def evaluate(self, diagnosis: dict) -> dict:
        """Evaluates the given diagnosis from the agent's specific perspective."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")

class GeneralPractitionerAgent(PersonaAgent):
    def __init__(self):
        super().__init__("General Practitioner", "Focuses on common conditions, overall patient health, initial treatment plans, and general medical guidelines.")

    def evaluate(self, diagnosis: dict) -> dict:
        feedback = {
            "agent": self.name,
            "perspective": self.role_description,
            "comments": [],
            "suggestions": []
        }
        print(f"[{self.name}] Evaluating diagnosis for {diagnosis['patient_name']}...")

        if diagnosis["confidence_score"] < 0.75:
            feedback["comments"].append("Confidence score is moderate; consider further common diagnostic steps.")
            feedback["suggestions"].append("Ensure all common and easily treatable causes are ruled out.")
        
        if not diagnosis.get("treatment_plan") or diagnosis["treatment_plan"] == "Monitor symptoms":
            feedback["comments"].append("Treatment plan is too vague or missing actionable steps.")
            feedback["suggestions"].append("Propose a more specific preliminary treatment plan.")
        
        if "common cold" in diagnosis["initial_diagnosis"].lower():
            if "fever" not in diagnosis.get("symptoms", []):
                feedback["comments"].append("Diagnosis of common cold seems plausible given symptoms.")
            else:
                feedback["comments"].append("Fever alongside common cold symptoms might indicate flu or other infection.")
                feedback["suggestions"].append("Suggest advising a flu test or further investigation if symptoms worsen.")

        return feedback

class MedicalSpecialistAgent(PersonaAgent):
    def __init__(self, specialty: str):
        super().__init__(f"{specialty} Specialist", f"Provides in-depth analysis related to {specialty}, considering complex cases and advanced diagnostics.")
        self.specialty = specialty

    def evaluate(self, diagnosis: dict) -> dict:
        feedback = {
            "agent": self.name,
            "perspective": self.role_description,
            "comments": [],
            "suggestions": []
        }
        print(f"[{self.name}] Evaluating diagnosis for {diagnosis['patient_name']}...")

        if self.specialty == "Cardiologist":
            if "chest_pain" in diagnosis.get("symptoms", []):
                feedback["comments"].append("Chest pain and shortness of breath strongly suggest a cardiac workup is critical.")
                if "ECG" not in diagnosis.get("suggested_tests", []):
                    feedback["suggestions"].append("Strongly recommend an immediate ECG and cardiac enzyme markers.")
                if "cardiac event" not in diagnosis["initial_diagnosis"].lower():
                    feedback["comments"].append("The initial diagnosis might be underestimating cardiac risk.")
                    feedback["suggestions"].append("Advise immediate referral for cardiology consultation.")
            else:
                feedback["comments"].append(f"No primary {self.specialty} concerns from initial symptoms, but maintain vigilance.")
        
        if self.specialty == "Neurologist":
            if "severe_headache" in diagnosis.get("symptoms", []) or "blurred vision" in diagnosis.get("symptoms", []):
                feedback["comments"].append("Severe headache with vision changes warrants thorough neurological assessment.")
                if "neurological exam" not in diagnosis.get("suggested_tests", []):
                    feedback["suggestions"].append("Recommend a detailed neurological examination and potentially imaging (MRI/CT scan of the brain).")
                if "Migraine" not in diagnosis["initial_diagnosis"].lower() and "neurological concern" not in diagnosis["initial_diagnosis"].lower():
                    feedback["comments"].append("Consider other neurological causes beyond common headaches.")
            else:
                feedback["comments"].append(f"No primary {self.specialty} concerns from initial symptoms.")

        return feedback

class PatientAdvocateAgent(PersonaAgent):
    def __init__(self):
        super().__init__("Patient Advocate", "Evaluates the diagnosis from the patient's perspective, focusing on clarity, empathy, potential impact on quality of life, and patient rights.")

    def evaluate(self, diagnosis: dict) -> dict:
        feedback = {
            "agent": self.name,
            "perspective": self.role_description,
            "comments": [],
            "suggestions": []
        }
        print(f"[{self.name}] Evaluating diagnosis for {diagnosis['patient_name']}...")

        if diagnosis["confidence_score"] < 0.7:
            feedback["comments"].append("The diagnosis seems uncertain, which might cause significant patient anxiety.")
            feedback["suggestions"].append("Suggest explaining any diagnostic uncertainty clearly and transparently to the patient, outlining next steps.")
        
        if not diagnosis.get("treatment_plan") or diagnosis["treatment_plan"] == "Monitor symptoms":
            feedback["comments"].append("A vague or missing treatment plan leaves the patient without clear direction or hope.")
            feedback["suggestions"].append("Ensure a clear, understandable treatment plan is provided, including self-care advice and expected timelines.")
        
        # Check if potential impact on daily life is considered (simplified)
        if "chest_pain" in diagnosis.get("symptoms", []) or "severe_headache" in diagnosis.get("symptoms", []):
            feedback["comments"].append("These symptoms could significantly impact the patient's quality of life and ability to perform daily tasks.")
            feedback["suggestions"].append("Advise discussing the potential impact of the condition on daily activities and exploring support systems or accommodations.")

        return feedback

class EthicistAgent(PersonaAgent):
    def __init__(self):
        super().__init__("Ethicist", "Reviews the diagnosis for ethical implications, potential biases, fairness, and adherence to medical ethics principles (autonomy, beneficence, non-maleficence, justice).")

    def evaluate(self, diagnosis: dict) -> dict:
        feedback = {
            "agent": self.name,
            "perspective": self.role_description,
            "comments": [],
            "suggestions": []
        }
        print(f"[{self.name}] Evaluating diagnosis for {diagnosis['patient_name']}...")

        # Simulate checking for potential bias (simplified example)
        # In a real scenario, this would involve analyzing data for demographic biases
        if diagnosis["confidence_score"] < 0.6: # Arbitrary low score as a trigger for ethical review
            feedback["comments"].append("A low confidence score might indicate insufficient data or potential for diagnostic bias leading to misdiagnosis.")
            feedback["suggestions"].append("Recommend further data collection, expert human review, or a second AI opinion to ensure diagnostic fairness and accuracy.")

        if "aggressive treatment" in diagnosis.get("treatment_plan", "").lower() and diagnosis["confidence_score"] < 0.8:
            feedback["comments"].append("Proposing aggressive treatment with only moderate diagnostic confidence raises ethical questions regarding beneficence and non-maleficence (doing good vs. doing harm).")
            feedback["suggestions"].append("Advise a thorough risk-benefit analysis, exploring less invasive alternatives, and ensuring fully informed patient consent before proceeding with aggressive treatments.")
        
        # Check for privacy concerns (simplified, assuming patient_data is directly in diagnosis)
        # In a real system, this would involve checking data access logs and anonymization policies.
        if diagnosis.get("patient_data") and len(diagnosis["patient_data"]) > 5: # Arbitrary check for 'too much' direct patient data in diagnosis
            feedback["comments"].append("Review data included in the diagnosis for privacy implications; ensure only strictly necessary information is used.")
            feedback["suggestions"].append("Emphasize strict data minimization principles and secure handling of sensitive patient information in line with GDPR/HIPAA.")

        return feedback

class DiagnosisEvaluationSystem:
    def __init__(self, primary_diagnoser: MedicalDiagnosisModel, evaluator_agents: list[PersonaAgent]):
        self.primary_diagnoser = primary_diagnoser
        self.evaluator_agents = evaluator_agents

    def evaluate_diagnosis(self, patient_data: dict) -> dict:
        """Orchestrates the persona-driven evaluation of a medical diagnosis."""
        print("\n--- Starting Initial AI Diagnosis ---")
        initial_diagnosis = self.primary_diagnoser.generate_diagnosis(patient_data)
        print("\n--- Initial Diagnosis Output ---")
        print(initial_diagnosis)

        print("\n--- Starting Persona-Driven Evaluation ---")
        all_feedback = []
        for agent in self.evaluator_agents:
            feedback = agent.evaluate(initial_diagnosis)
            all_feedback.append(feedback)
        
        print("\n--- Compiling Final Assessment ---")
        refined_assessment = {
            "patient_name": patient_data["name"],
            "initial_diagnosis": initial_diagnosis,
            "evaluator_feedback": all_feedback,
            "final_recommendations": []
        }

        # Simple aggregation of unique suggestions
        for feedback in all_feedback:
            for suggestion in feedback["suggestions"]:
                if suggestion not in refined_assessment["final_recommendations"]:
                    refined_assessment["final_recommendations"].append(suggestion)

        if not refined_assessment["final_recommendations"]:
            refined_assessment["final_recommendations"].append("Initial diagnosis appears robust based on current evaluations.")

        print("\n--- Final Refined Assessment ---")
        print(json.dumps(refined_assessment, indent=2))
        return refined_assessment

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize the primary AI diagnosis model (simulated)
    primary_ai = MedicalDiagnosisModel()

    # 2. Initialize the persona-driven evaluator agents
    evaluators = [
        GeneralPractitionerAgent(),
        MedicalSpecialistAgent(specialty="Cardiologist"),
        MedicalSpecialistAgent(specialty="Neurologist"),
        PatientAdvocateAgent(),
        EthicistAgent()
    ]

    # 3. Initialize the evaluation system
    evaluation_system = DiagnosisEvaluationSystem(primary_ai, evaluators)

    # 4. Prepare patient data scenarios
    patient_data_1 = {
        "name": "Alice Smith",
        "age": 45,
        "symptoms": ["fatigue", "sore throat", "runny nose"],
        "medical_history": "none"
    }

    patient_data_2 = {
        "name": "Bob Johnson",
        "age": 60,
        "symptoms": ["sudden severe chest pain", "shortness of breath", "left arm tingling"],
        "medical_history": "hypertension, high cholesterol"
    }
    
    patient_data_3 = {
        "name": "Carol White",
        "age": 30,
        "symptoms": ["persistent severe headache", "blurred vision", "sensitivity to light"],
        "medical_history": "diagnosed migraines for 10 years"
    }

    print("=========================================")
    print("Scenario 1: Common Cold-like Symptoms")
    evaluation_system.evaluate_diagnosis(patient_data_1)
    print("\n\n=========================================")
    print("Scenario 2: Potential Cardiac Event")
    evaluation_system.evaluate_diagnosis(patient_data_2)
    print("\n\n=========================================")
    print("Scenario 3: Severe Headache")
    evaluation_system.evaluate_diagnosis(patient_data_3)
    print("=========================================")