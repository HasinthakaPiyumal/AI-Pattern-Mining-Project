class SimulatedLLM:
    def __init__(self):
        pass

    def _generate_mock_cot(self, prompt_type, diagnosis=None, patient_data=None, correct_diagnosis=None):
        if prompt_type == "initial":
            return f"Initial CoT for patient with symptoms {patient_data['symptoms']}: Based on these, a possible diagnosis is {diagnosis}."
        elif prompt_type == "incorrect_reasoning":
            return f"Incorrect reasoning CoT: The LLM initially thought {diagnosis} because of X, Y, Z, but it was actually {correct_diagnosis}. This was due to misinterpreting A, B."
        elif prompt_type == "correct_reasoning":
            return f"Correct reasoning CoT: Given the correct diagnosis is {correct_diagnosis}, the reasoning is based on P, Q, R, which aligns with symptoms {patient_data['symptoms']}."
        return ""

    def diagnose(self, patient_data):
        symptoms_str = ", ".join(patient_data['symptoms'])
        if "fatigue" in patient_data['symptoms'] and "joint pain" in patient_data['symptoms']:
            diagnosis = "Lupus"
        elif "fever" in patient_data['symptoms'] and "rash" in patient_data['symptoms']:
            diagnosis = "Measles"
        else:
            diagnosis = "Common Cold"
        cot = self._generate_mock_cot("initial", diagnosis=diagnosis, patient_data=patient_data)
        return diagnosis, cot

    def generate_cot_explanation(self, diagnosis, patient_data):
        return self._generate_mock_cot("initial", diagnosis=diagnosis, patient_data=patient_data)

    def generate_corrective_cot(self, incorrect_diagnosis, correct_diagnosis, patient_data):
        incorrect_cot = self._generate_mock_cot("incorrect_reasoning", diagnosis=incorrect_diagnosis, patient_data=patient_data, correct_diagnosis=correct_diagnosis)
        correct_cot = self._generate_mock_cot("correct_reasoning", diagnosis=incorrect_diagnosis, patient_data=patient_data, correct_diagnosis=correct_diagnosis)
        return incorrect_cot, correct_cot

class MedicalExpertFeedback:
    def __init__(self):
        self.ground_truths = {
            "patient_001": {"diagnosis": "Lupus", "is_rare": True},
            "patient_002": {"diagnosis": "Measles", "is_rare": False},
            "patient_003": {"diagnosis": "Rare Autoimmune Disease X", "is_rare": True},
            "patient_004": {"diagnosis": "Common Cold", "is_rare": False},
            "patient_005": {"diagnosis": "Lupus", "is_rare": True},
        }

    def review_diagnosis(self, patient_id, llm_diagnosis):
        if patient_id in self.ground_truths:
            correct_diagnosis_info = self.ground_truths[patient_id]
            is_correct = (llm_diagnosis == correct_diagnosis_info["diagnosis"])
            return is_correct, correct_diagnosis_info["diagnosis"]
        return False, "Unknown/No Ground Truth"

class AutoDiCoTOrchestrator:
    def __init__(self, llm_model, expert_feedback_module):
        self.llm = llm_model
        self.expert_feedback = expert_feedback_module
        self.exemplars = []

    def process_patient_case(self, patient_id, patient_data):
        print(f"\nProcessing Patient {patient_id}:")
        initial_diagnosis, initial_cot = self.llm.diagnose(patient_data)
        print(f"  LLM Initial Diagnosis: {initial_diagnosis}")
        print(f"  LLM Initial CoT: {initial_cot}")

        is_correct, correct_diagnosis = self.expert_feedback.review_diagnosis(patient_id, initial_diagnosis)

        if not is_correct:
            print(f"  Diagnosis incorrect. Correct diagnosis: {correct_diagnosis}")
            incorrect_cot, correct_cot_from_feedback = self.llm.generate_corrective_cot(initial_diagnosis, correct_diagnosis, patient_data)
            print(f"  LLM Incorrect Reasoning CoT: {incorrect_cot}")
            print(f"  LLM Correct Reasoning CoT: {correct_cot_from_feedback}")
            self.exemplars.append({
                "patient_id": patient_id,
                "patient_data": patient_data,
                "llm_initial_diagnosis": initial_diagnosis,
                "llm_initial_cot": initial_cot,
                "is_correct": False,
                "correct_diagnosis": correct_diagnosis,
                "incorrect_reasoning_cot": incorrect_cot,
                "correct_reasoning_cot": correct_cot_from_feedback
            })
        else:
            print(f"  Diagnosis correct: {initial_diagnosis}")
            self.exemplars.append({
                "patient_id": patient_id,
                "patient_data": patient_data,
                "llm_initial_diagnosis": initial_diagnosis,
                "llm_initial_cot": initial_cot,
                "is_correct": True,
                "correct_diagnosis": initial_diagnosis
            })
        print(f"  Current exemplars count: {len(self.exemplars)}")

# Main execution
if __name__ == "__main__":
    llm = SimulatedLLM()
    expert_feedback = MedicalExpertFeedback()
    orchestrator = AutoDiCoTOrchestrator(llm, expert_feedback)

    # Simulate patient cases
    patient_cases = {
        "patient_001": {"symptoms": ["fatigue", "joint pain", "skin rash"], "medical_history": ["chronic illness"]},
        "patient_002": {"symptoms": ["fever", "cough", "rash"], "medical_history": []},
        "patient_003": {"symptoms": ["unexplained weight loss", "muscle weakness", "difficulty swallowing"], "medical_history": ["autoimmune conditions in family"]},
        "patient_004": {"symptoms": ["sore throat", "runny nose"], "medical_history": []},
        "patient_005": {"symptoms": ["fatigue", "joint pain", "hair loss"], "medical_history": ["family history of lupus"]}
    }

    for patient_id, data in patient_cases.items():
        orchestrator.process_patient_case(patient_id, data)

    print("\n--- Stored Exemplars ---")
    for i, exemplar in enumerate(orchestrator.exemplars):
        print(f"Exemplar {i+1}:")
        print(f"  Patient ID: {exemplar['patient_id']}")
        print(f"  Initial LLM Diagnosis: {exemplar['llm_initial_diagnosis']}")
        print(f"  Correct: {exemplar['is_correct']}")
        if not exemplar['is_correct']:
            print(f"  Correct Diagnosis: {exemplar['correct_diagnosis']}")
            print(f"  Incorrect Reasoning CoT: {exemplar['incorrect_reasoning_cot']}")
            print(f"  Correct Reasoning CoT: {exemplar['correct_reasoning_cot']}")
        print("-------------------------")
