from medical_prompt_engine import MedicalPromptEngine
from knowledge_consolidator import KnowledgeConsolidator
from utility_module import UtilityModule
from llm_simulator import LLM_Simulator

class MedicalDiagnosisAssistant:
    def __init__(self):
        self.prompt_engine = MedicalPromptEngine()
        self.knowledge_consolidator = KnowledgeConsolidator()
        self.utility_module = UtilityModule()
        self.llm_simulator = LLM_Simulator()
        self.dialog_history = [] # Stores past patient interactions

    def diagnose_patient(self, symptoms: str, patient_id: str):
        print(f"\n--- Diagnosing Patient {patient_id} ---")
        print(f"Symptoms provided: {symptoms}")

        # 1. Update dialog history with current symptoms
        self.dialog_history.append(f"Patient {patient_id} reported: {symptoms}")

        # 2. Get external evidence
        evidence = self.knowledge_consolidator.get_evidence(symptoms)
        print(f"Consolidated Medical Evidence: {evidence[:70]}...")

        # 3. Get automated feedback
        feedback = self.utility_module.get_feedback(symptoms, self.dialog_history)
        print(f"Automated Feedback: {feedback}")

        # 4. Define task instructions
        task_instructions = "You are a highly experienced medical diagnostic AI. Your goal is to provide a differential diagnosis and potential treatment suggestions based on the provided patient information, medical history, and external evidence. Be precise and consider all contextual information."

        # 5. Construct the dynamic prompt
        prompt = self.prompt_engine.construct_prompt(
            task_instructions=task_instructions,
            user_query=symptoms,
            dialog_history=self.dialog_history,
            evidence=evidence,
            feedback=feedback
        )
        print(f"\n--- Generated Prompt (first 200 chars): ---\n{prompt[:200]}...")

        # 6. Get diagnosis from LLM Simulator
        llm_response = self.llm_simulator.generate_response(prompt)
        print(f"\n--- LLM Diagnosis for Patient {patient_id} ---")
        print(llm_response)
        print("--------------------------------------")
        return llm_response

if __name__ == "__main__":
    assistant = MedicalDiagnosisAssistant()

    # Example 1: Patient with flu-like symptoms
    assistant.diagnose_patient("fever, cough, body aches, sore throat", "P001")

    # Example 2: Patient with abdominal pain (dialog history will be accumulated)
    assistant.diagnose_patient("severe lower right abdominal pain, nausea, loss of appetite", "P002")

    # Example 3: Patient with headache and stiff neck
    assistant.diagnose_patient("sudden severe headache, stiff neck, sensitivity to light", "P003")