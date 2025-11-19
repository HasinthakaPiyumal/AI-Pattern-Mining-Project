
class MedicalPersona:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

class LLMAgent:
    def __init__(self, persona):
        self.persona = persona

    def evaluate(self, patient_case, initial_ai_plan):
        # Simulate LLM reasoning based on persona's system prompt
        # In a real system, this would involve an actual LLM API call
        evaluation_report = {
            "persona": self.persona.name,
            "perspective": self.persona.system_prompt,
            "evaluation": f"As a {self.persona.name}, my assessment of the patient case (ID: {patient_case['patient_id']}) and the proposed plan: '{initial_ai_plan['diagnosis']} - {initial_ai_plan['treatment_plan']}' is as follows:\n\n" \
                          f"Based on my expertise in {self.persona.name.lower()} and considering the details provided, I would focus on... [Simulated detailed critique from {self.persona.name}'s viewpoint, e.g., identifying risks, suggesting alternatives, or questioning ethical implications]."
        }
        return evaluation_report

class DiagnosisReviewSystem:
    def __init__(self, personas):
        self.agents = [LLMAgent(persona) for persona in personas]

    def conduct_review(self, patient_case, initial_ai_plan):
        individual_evaluations = []
        for agent in self.agents:
            report = agent.evaluate(patient_case, initial_ai_plan)
            individual_evaluations.append(report)

        return self._synthesize_reviews(individual_evaluations)

    def _synthesize_reviews(self, evaluations):
        comprehensive_review = """Comprehensive Medical Diagnosis Review
========================================

Individual Persona Evaluations:
-------------------------------
"""
        for eval_report in evaluations:
            comprehensive_review += f"Persona: {eval_report['persona']}\n"
            comprehensive_review += f"Perspective: {eval_report['perspective']}\n"
            comprehensive_review += f"Evaluation: {eval_report['evaluation']}\n\n"

        # Simulate synthesis logic - in a real scenario, another LLM or rule-based system
        # would process these individual evaluations to find commonalities, conflicts, etc.
        comprehensive_review += """Overall Synthesis and Recommendations:
----------------------------------------
Based on the diverse perspectives presented by the General Practitioner, Cardiologist, Pharmacist, and Medical Ethicist, the following key points emerge:

- **Agreement Points**: [Simulated points of consensus among personas]
- **Conflicting Views/Areas for Further Discussion**: [Simulated areas where personas might disagree or offer different priorities]
- **Identified Risks/Ethical Considerations**: [Simulated risks or ethical concerns raised by specific personas]
- **Recommended Adjustments/Next Steps**: [Simulated recommendations for modifying the initial plan based on the reviews]

This multi-faceted review provides a more robust assessment than a singular viewpoint, highlighting critical aspects for enhanced patient care.
"""
        return comprehensive_review

if __name__ == "__main__":
    # 1. Initial AI Diagnosis & Treatment Plan (Simulated)
    patient_case = {
        "patient_id": "P001",
        "age": 65,
        "gender": "Male",
        "symptoms": "Chest pain, shortness of breath, fatigue",
        "medical_history": "Hypertension, mild diabetes"
    }

    initial_ai_plan = {
        "diagnosis": "Suspected Coronary Artery Disease (CAD) with potential angina.",
        "treatment_plan": "Prescribe nitroglycerin, statins, and recommend lifestyle changes. Schedule for a stress test and follow-up in 2 weeks."
    }

    print("\n--- Initial AI Diagnosis and Treatment Plan ---")
    print(f"Patient ID: {patient_case['patient_id']}")
    print(f"Diagnosis: {initial_ai_plan['diagnosis']}")
    print(f"Treatment Plan: {initial_ai_plan['treatment_plan']}")
    print("-----------------------------------------------\n")

    # 2. Persona Definitions
    gp_persona = MedicalPersona(
        name="General Practitioner",
        system_prompt="You are a General Practitioner. Evaluate the diagnosis and treatment plan from a holistic patient care perspective, considering general health, comorbidities, and initial management."
    )

    cardio_persona = MedicalPersona(
        name="Cardiologist",
        system_prompt="You are a Cardiologist. Critically assess the diagnosis and treatment plan for cardiovascular accuracy, appropriateness of tests, and efficacy for heart-related conditions. Focus on cardiac risks and interventions."
    )

    pharma_persona = MedicalPersona(
        name="Pharmacist",
        system_prompt="You are a Pharmacist. Review the prescribed medications for drug interactions, appropriate dosages, patient adherence, and potential side effects based on the patient's medical history."
    )

    ethicist_persona = MedicalPersona(
        name="Medical Ethicist",
        system_prompt="You are a Medical Ethicist. Evaluate the treatment plan for ethical considerations, patient autonomy, informed consent, potential biases, and fairness in resource allocation."
    )

    # 3. Initialize Review System
    review_system = DiagnosisReviewSystem([gp_persona, cardio_persona, pharma_persona, ethicist_persona])

    # 4. Conduct Review
    comprehensive_review_output = review_system.conduct_review(patient_case, initial_ai_plan)

    # 5. Output & Reporting
    print(comprehensive_review_output)
