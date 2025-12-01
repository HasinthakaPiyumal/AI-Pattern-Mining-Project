import collections

class LLMSimulator:
    def generate_hypotheses(self, current_patient_state):
        symptoms = current_patient_state.get("symptoms", [])
        history = current_patient_state.get("history", "")
        
        if "fever" in symptoms and "cough" in symptoms:
            return ["Common Cold", "Flu", "Pneumonia"]
        elif "headache" in symptoms and "stiff neck" in symptoms:
            return ["Meningitis", "Severe Migraine"]
        elif "chest pain" in symptoms and "shortness of breath" in symptoms:
            return ["Heart Attack", "Anxiety Attack", "Pneumonia"]
        elif "fatigue" in symptoms and "weight loss" in symptoms and "diabetes" in history:
            return ["Uncontrolled Diabetes", "Thyroid Dysfunction", "Cancer"]
        else:
            return ["General Illness", "Stress-related Symptoms"]

    def evaluate_hypothesis(self, hypothesis, patient_data):
        symptoms = patient_data.get("symptoms", [])
        history = patient_data.get("history", "")
        lab_results = patient_data.get("lab_results", {})

        score = 0
        treatment_plan = "" 
        further_questions = []

        if hypothesis == "Common Cold":
            if "runny nose" in symptoms and "sore throat" in symptoms:
                score += 0.8
            treatment_plan = "Rest, fluids, over-the-counter medication."
        elif hypothesis == "Flu":
            if "body aches" in symptoms and "fatigue" in symptoms and "fever" in symptoms:
                score += 0.9
            treatment_plan = "Antiviral medication, rest, fluids."
            further_questions.append("Have you had a flu shot recently?")
        elif hypothesis == "Pneumonia":
            if "difficulty breathing" in symptoms and "chest pain" in symptoms and lab_results.get("x-ray") == "infiltrates":
                score += 0.95
            treatment_plan = "Antibiotics, oxygen therapy."
            further_questions.append("Do you have a productive cough?")
        elif hypothesis == "Heart Attack":
            if "chest pain" in symptoms and "left arm pain" in symptoms and lab_results.get("ecg") == "abnormal":
                score += 0.99
            treatment_plan = "Emergency medical attention, aspirin, nitrates."
            further_questions.append("Do you have a history of heart disease?")
        elif hypothesis == "Uncontrolled Diabetes":
            if "increased thirst" in symptoms and "frequent urination" in symptoms and lab_results.get("blood_glucose") > 200:
                score += 0.9
            treatment_plan = "Insulin adjustment, dietary changes, exercise."
            further_questions.append("How regularly do you monitor your blood sugar?")
        
        return {"score": min(score, 1.0), "treatment": treatment_plan, "further_questions": further_questions}

class ThoughtNode:
    def __init__(self, hypothesis, patient_state, evaluation_score, parent=None):
        self.hypothesis = hypothesis
        self.patient_state = patient_state
        self.evaluation_score = evaluation_score
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

class DiagnosticAssistant:
    def __init__(self, llm_simulator, max_depth=3, beam_width=2, confidence_threshold=0.95):
        self.llm_simulator = llm_simulator
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.confidence_threshold = confidence_threshold

    def diagnose(self, initial_patient_data):
        root_node = ThoughtNode(hypothesis="Initial Assessment", patient_state=initial_patient_data, evaluation_score=0.0)
        current_frontier = collections.deque([(root_node, 0)])
        best_diagnosis = None
        highest_confidence = 0.0

        while current_frontier:
            current_node, current_depth = current_frontier.popleft()

            if current_node.evaluation_score >= self.confidence_threshold:
                if current_node.evaluation_score > highest_confidence:
                    highest_confidence = current_node.evaluation_score
                    best_diagnosis = current_node
                continue

            if current_depth >= self.max_depth:
                continue

            hypotheses = self.llm_simulator.generate_hypotheses(current_node.patient_state)
            evaluated_hypotheses = []

            for hyp in hypotheses:
                evaluation_result = self.llm_simulator.evaluate_hypothesis(hyp, current_node.patient_state)
                evaluated_hypotheses.append({"hypothesis": hyp, "score": evaluation_result["score"], "treatment": evaluation_result["treatment"]})
            
            # Sort and select top hypotheses (beam search)
            evaluated_hypotheses.sort(key=lambda x: x["score"], reverse=True)
            selected_hypotheses = evaluated_hypotheses[:self.beam_width]

            for selected_hyp in selected_hypotheses:
                child_node = ThoughtNode(hypothesis=selected_hyp["hypothesis"], 
                                         patient_state=current_node.patient_state, 
                                         evaluation_score=selected_hyp["score"], 
                                         parent=current_node)
                current_node.add_child(child_node)
                current_frontier.append((child_node, current_depth + 1))

                if child_node.evaluation_score > highest_confidence:
                    highest_confidence = child_node.evaluation_score
                    best_diagnosis = child_node
        
        if best_diagnosis:  
            # Backtrack to reconstruct the path, though for final diagnosis, the best node is enough.
            final_diagnosis_info = self.llm_simulator.evaluate_hypothesis(best_diagnosis.hypothesis, best_diagnosis.patient_state)
            return {
                "diagnosis": best_diagnosis.hypothesis,
                "confidence": best_diagnosis.evaluation_score,
                "treatment_plan": final_diagnosis_info["treatment"],
                "path_taken": self._reconstruct_path(best_diagnosis)
            }
        else:
            return {"diagnosis": "Undetermined", "confidence": 0.0, "treatment_plan": "Further investigation needed.", "path_taken": []}

    def _reconstruct_path(self, node):
        path = []
        current = node
        while current:
            path.append(current.hypothesis)
            current = current.parent
        return path[::-1]


if __name__ == "__main__":
    llm_sim = LLMSimulator()
    assistant = DiagnosticAssistant(llm_sim)

    # Example 1: Simple Cold
    patient_data_1 = {"symptoms": ["fever", "cough", "runny nose", "sore throat"], "history": ""
    }
    print("\n--- Diagnosing Patient 1 (Simple Cold) ---")
    result_1 = assistant.diagnose(patient_data_1)
    print(f"Diagnosis: {result_1['diagnosis']}")
    print(f"Confidence: {result_1['confidence']:.2f}")
    print(f"Treatment Plan: {result_1['treatment_plan']}")
    print(f"Path Taken: {result_1['path_taken']}")

    # Example 2: Potential Heart Attack
    patient_data_2 = {"symptoms": ["chest pain", "shortness of breath", "left arm pain"], 
                      "history": "family history of heart disease",
                      "lab_results": {"ecg": "abnormal"}
    }
    print("\n--- Diagnosing Patient 2 (Potential Heart Attack) ---")
    result_2 = assistant.diagnose(patient_data_2)
    print(f"Diagnosis: {result_2['diagnosis']}")
    print(f"Confidence: {result_2['confidence']:.2f}")
    print(f"Treatment Plan: {result_2['treatment_plan']}")
    print(f"Path Taken: {result_2['path_taken']}")

    # Example 3: Complex case with diabetes history
    patient_data_3 = {"symptoms": ["fatigue", "weight loss", "increased thirst"], 
                      "history": "diabetes",
                      "lab_results": {"blood_glucose": 250}
    }
    print("\n--- Diagnosing Patient 3 (Complex Diabetes) ---")
    result_3 = assistant.diagnose(patient_data_3)
    print(f"Diagnosis: {result_3['diagnosis']}")
    print(f"Confidence: {result_3['confidence']:.2f}")
    print(f"Treatment Plan: {result_3['treatment_plan']}")
    print(f"Path Taken: {result_3['path_taken']}")

    # Example 4: Undetermined case
    patient_data_4 = {"symptoms": ["mild rash", "itchiness"], "history": "no significant history"}
    print("\n--- Diagnosing Patient 4 (Undetermined) ---")
    result_4 = assistant.diagnose(patient_data_4)
    print(f"Diagnosis: {result_4['diagnosis']}")
    print(f"Confidence: {result_4['confidence']:.2f}")
    print(f"Treatment Plan: {result_4['treatment_plan']}")
    print(f"Path Taken: {result_4['path_taken']}")
