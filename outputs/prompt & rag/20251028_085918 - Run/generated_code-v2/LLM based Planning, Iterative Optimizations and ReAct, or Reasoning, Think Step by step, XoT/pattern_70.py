import random

class ReasoningStep:
    def __init__(self, step_id: int, thought: str, justification: str, score: float = 0.0):
        self.step_id = step_id
        self.thought = thought
        self.justification = justification
        self.score = score

    def __repr__(self):
        return f"Step {self.step_id}: {self.thought} (Score: {self.score:.2f})\n  Justification: {self.justification}"

class ReasoningPath:
    def __init__(self, path_id: int, steps: list[ReasoningStep]):
        self.path_id = path_id
        self.steps = steps
        self.average_score = self._calculate_average_score()

    def _calculate_average_score(self):
        if not self.steps:
            return 0.0
        return sum(step.score for step in self.steps) / len(self.steps)

    def __repr__(self):
        steps_repr = "\n".join(str(step) for step in self.steps)
        return f"Path {self.path_id} (Avg Score: {self.average_score:.2f}):\n{steps_repr}"

class DiagnosticHypothesis:
    def __init__(self, hypothesis_id: int, diagnosis: str, reasoning_paths: list[ReasoningPath]):
        self.hypothesis_id = hypothesis_id
        self.diagnosis = diagnosis
        self.reasoning_paths = reasoning_paths
        self.overall_score = self._calculate_overall_score()

    def _calculate_overall_score(self):
        if not self.reasoning_paths:
            return 0.0
        return sum(path.average_score for path in self.reasoning_paths) / len(self.reasoning_paths)

    def __repr__(self):
        return f"Hypothesis {self.hypothesis_id}: {self.diagnosis} (Overall Score: {self.overall_score:.2f})"

class MedicalDiagnosticAssistant:
    def __init__(self, medical_knowledge: dict):
        self.medical_knowledge = medical_knowledge

    def _generate_hypotheses(self, symptoms: list[str]) -> list[str]:
        """Simulates generating multiple diagnostic hypotheses based on symptoms."""
        # In a real system, an LLM would generate these based on input.
        # Here, we simulate by picking relevant conditions from knowledge or general ones.
        possible_conditions = []
        for symptom in symptoms:
            if "fever" in symptom:
                possible_conditions.extend(["Influenza", "Common Cold", "Bacterial Infection"])
            if "cough" in symptom:
                possible_conditions.extend(["Bronchitis", "Allergies", "Pneumonia"])
            if "headache" in symptom:
                possible_conditions.extend(["Migraine", "Tension Headache", "Sinusitis"])
            if "fatigue" in symptom:
                possible_conditions.extend(["Chronic Fatigue Syndrome", "Anemia", "Hypothyroidism"])
        
        # Remove duplicates and ensure a diverse set, but keep it plausible
        hypotheses = list(set(possible_conditions))
        if not hypotheses and symptoms:
            hypotheses = ["General Viral Illness", "Stress-related Symptoms"]
        elif not hypotheses:
            hypotheses = ["Unknown Condition"]
        
        # Add some slight variations/alternatives for DiVeRSe pattern
        if "Influenza" in hypotheses and "Viral Infection" not in hypotheses:
            hypotheses.append("Viral Infection")
        if "Bacterial Infection" in hypotheses and "Localized Infection" not in hypotheses:
            hypotheses.append("Localized Infection")

        return list(set(hypotheses)) # Ensure uniqueness after variations

    def _generate_reasoning_path(self, hypothesis: str, symptoms: list[str]) -> list[ReasoningStep]:
        """Simulates generating a reasoning path for a given hypothesis and symptoms.
        Each path is a sequence of steps, like a Chain-of-Thought.
        """
        steps = []
        step_id_counter = 1

        # Step 1: Initial assessment based on primary symptom
        primary_symptom = symptoms[0] if symptoms else "patient reports general discomfort"
        thought1 = f"Considering {hypothesis} based on {primary_symptom}."
        justification1 = f"Initial screening suggests {hypothesis} as a possibility given the patient's primary complaint of '{primary_symptom}'."
        steps.append(ReasoningStep(step_id_counter, thought1, justification1))
        step_id_counter += 1

        # Step 2: Corroborate with other symptoms/patient history
        other_symptoms = ", ".join(symptoms[1:]) if len(symptoms) > 1 else "no other specific symptoms mentioned"
        thought2 = f"Evaluating {hypothesis} against additional symptoms: {other_symptoms}."
        justification2 = f"The presence of {other_symptoms} further supports/contradicts {hypothesis} based on typical disease presentation. For example, if '{hypothesis}' often presents with fever, and fever is present, this strengthens the case."
        steps.append(ReasoningStep(step_id_counter, thought2, justification2))
        step_id_counter += 1

        # Step 3: Consider lab results/medical history (simulated)
        simulated_lab_result = "elevated white blood cell count" if "Bacterial Infection" in hypothesis else "normal blood count"
        thought3 = f"Integrating simulated lab results ({simulated_lab_result}) and medical history for {hypothesis}."
        justification3 = f"A {simulated_lab_result} would be a significant indicator for/against {hypothesis}. Patient's past medical history (e.g., allergies, chronic conditions) would also be crucial."
        steps.append(ReasoningStep(step_id_counter, thought3, justification3))
        step_id_counter += 1

        # Step 4: Differential diagnosis consideration (simulated)
        alternative_diagnosis = random.choice([h for h in self.medical_knowledge.keys() if h != hypothesis]) if self.medical_knowledge else "another common illness"
        thought4 = f"Ruling out/considering {alternative_diagnosis} as a differential diagnosis for {hypothesis}."
        justification4 = f"It is important to differentiate {hypothesis} from conditions like {alternative_diagnosis} that share similar symptoms but require different treatments. Key distinguishing factors are being considered."
        steps.append(ReasoningStep(step_id_counter, thought4, justification4))
        step_id_counter += 1

        # Step 5: Final conclusion for this path
        thought5 = f"Concluding this reasoning path for {hypothesis}."
        justification5 = f"Based on the synthesis of all previous steps, this path either confirms {hypothesis} or finds it less likely."
        steps.append(ReasoningStep(step_id_counter, thought5, justification5))

        return steps

    def _score_reasoning_step(self, step: ReasoningStep, symptoms: list[str]) -> float:
        """Scores an individual reasoning step based on simulated medical evidence and logic.
        In a real system, this would involve querying a medical knowledge base or an evaluation LLM.
        """
        score = 0.5 # Base score

        # Simulate scoring based on keywords and simple logic
        justification_lower = step.justification.lower()
        thought_lower = step.thought.lower()

        # Positive indicators
        if "supports" in justification_lower or "strengthens" in justification_lower or "confirms" in justification_lower:
            score += random.uniform(0.1, 0.3)
        if "indicator for" in justification_lower:
            score += random.uniform(0.1, 0.2)

        # Negative indicators
        if "contradicts" in justification_lower or "less likely" in justification_lower or "rule out" in thought_lower:
            score -= random.uniform(00.05, 0.15)
        
        # Alignment with symptoms (very basic simulation)
        for symptom in symptoms:
            if symptom.lower() in justification_lower or symptom.lower() in thought_lower:
                score += 0.05 # Small boost for symptom relevance

        # Consistency check (simulated - if a step is too contradictory, lower score)
        if "contradicts" in justification_lower and "supports" in justification_lower:
            score -= 0.2 # Significant penalty for self-contradiction

        # Ensure score is within a reasonable range
        return max(0.1, min(1.0, score + random.uniform(-0.1, 0.1))) # Add some randomness for diversity

    def diagnose(self, patient_symptoms: list[str], num_paths_per_hypothesis: int = 3) -> dict:
        """Applies the DiVeRSe pattern to provide a diagnosis."""
        print(f"\n--- Diagnosing Patient with Symptoms: {', '.join(patient_symptoms)} ---")

        # 1. Generate multiple diagnostic hypotheses (prompts)
        initial_hypotheses_str = self._generate_hypotheses(patient_symptoms)
        print(f"Generated Initial Hypotheses: {', '.join(initial_hypotheses_str)}")

        all_diagnostic_hypotheses: list[DiagnosticHypothesis] = []

        for i, hypothesis_str in enumerate(initial_hypotheses_str):
            reasoning_paths_for_hypothesis: list[ReasoningPath] = []
            print(f"\n  -- Processing Hypothesis: {hypothesis_str} --")

            # 2. For each, perform SelfConsistency, generating multiple reasoning paths
            for path_idx in range(num_paths_per_hypothesis):
                print(f"    Generating Reasoning Path {path_idx + 1} for {hypothesis_str}...")
                raw_steps = self._generate_reasoning_path(hypothesis_str, patient_symptoms)
                
                # 3. Score reasoning paths based on each step
                scored_steps = []
                for step in raw_steps:
                    step.score = self._score_reasoning_step(step, patient_symptoms)
                    scored_steps.append(step)
                
                reasoning_paths_for_hypothesis.append(ReasoningPath(path_idx + 1, scored_steps))
            
            current_hypothesis = DiagnosticHypothesis(i + 1, hypothesis_str, reasoning_paths_for_hypothesis)
            all_diagnostic_hypotheses.append(current_hypothesis)
            print(f"  {current_hypothesis}")
            for path in reasoning_paths_for_hypothesis:
                print(f"    Path {path.path_id}: Avg Score {path.average_score:.2f}")

        # 4. Select a final response based on overall scores
        if not all_diagnostic_hypotheses:
            return {"final_diagnosis": "No diagnosis could be determined.", "confidence_score": 0.0, "details": []}

        # Sort hypotheses by overall score in descending order
        all_diagnostic_hypotheses.sort(key=lambda h: h.overall_score, reverse=True)
        best_hypothesis = all_diagnostic_hypotheses[0]

        # Find the best reasoning path for the best hypothesis
        best_hypothesis.reasoning_paths.sort(key=lambda p: p.average_score, reverse=True)
        best_path = best_hypothesis.reasoning_paths[0]

        print(f"\n--- Final DiVeRSe Diagnosis ---")
        print(f"Most Probable Diagnosis: {best_hypothesis.diagnosis}")
        print(f"Confidence Score (Aggregated DiVeRSe Score): {best_hypothesis.overall_score:.2f}")
        print(f"Highest Scoring Path (for this diagnosis):\n{best_path}")

        return {
            "final_diagnosis": best_hypothesis.diagnosis,
            "confidence_score": best_hypothesis.overall_score,
            "best_reasoning_path": {
                "diagnosis": best_hypothesis.diagnosis,
                "path_score": best_path.average_score,
                "steps": [{
                    "step_id": step.step_id,
                    "thought": step.thought,
                    "justification": step.justification,
                    "score": step.score
                } for step in best_path.steps]
            },
            "all_hypotheses_scores": [{
                "diagnosis": h.diagnosis,
                "overall_score": h.overall_score,
                "top_path_score": h.reasoning_paths[0].average_score if h.reasoning_paths else 0.0
            } for h in all_diagnostic_hypotheses]
        }

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate a basic medical knowledge base (keywords for scoring)
    medical_knowledge_base = {
        "Influenza": {"symptoms": ["fever", "cough", "fatigue", "body aches"], "treatment": "antivirals, rest"},
        "Common Cold": {"symptoms": ["cough", "sore throat", "runny nose"], "treatment": "rest, fluids"},
        "Bacterial Infection": {"symptoms": ["high fever", "localized pain", "elevated white blood cell count"], "treatment": "antibiotics"},
        "Bronchitis": {"symptoms": ["cough", "mucus", "chest discomfort"], "treatment": "rest, bronchodilators"},
        "Migraine": {"symptoms": ["severe headache", "nausea", "light sensitivity"], "treatment": "pain relievers"},
        "Viral Infection": {"symptoms": ["fever", "fatigue", "aches"], "treatment": "supportive care"},
        "Localized Infection": {"symptoms": ["pain", "swelling", "redness"], "treatment": "antibiotics, drainage"},
    }

    assistant = MedicalDiagnosticAssistant(medical_knowledge=medical_knowledge_base)

    patient_a_symptoms = ["fever", "cough", "fatigue"]
    diagnosis_a = assistant.diagnose(patient_a_symptoms)
    print(f"\n--- Detailed Result for Patient A ---")
    print(f"Final Diagnosis: {diagnosis_a['final_diagnosis']} (Confidence: {diagnosis_a['confidence_score']:.2f})")
    # print(json.dumps(diagnosis_a, indent=2)) # Uncomment to see full JSON output (requires import json)

    print("\n" + "="*80 + "\n")

    patient_b_symptoms = ["severe headache", "nausea"]
    diagnosis_b = assistant.diagnose(patient_b_symptoms, num_paths_per_hypothesis=4)
    print(f"\n--- Detailed Result for Patient B ---")
    print(f"Final Diagnosis: {diagnosis_b['final_diagnosis']} (Confidence: {diagnosis_b['confidence_score']:.2f})")

    print("\n" + "="*80 + "\n")

    patient_c_symptoms = ["localized pain", "swelling"]
    diagnosis_c = assistant.diagnose(patient_c_symptoms)
    print(f"\n--- Detailed Result for Patient C ---")
    print(f"Final Diagnosis: {diagnosis_c['final_diagnosis']} (Confidence: {diagnosis_c['confidence_score']:.2f})")

    print("\n" + "="*80 + "\n")

    patient_d_symptoms = [] # No specific symptoms
    diagnosis_d = assistant.diagnose(patient_d_symptoms)
    print(f"\n--- Detailed Result for Patient D ---")
    print(f"Final Diagnosis: {diagnosis_d['final_diagnosis']} (Confidence: {diagnosis_d['confidence_score']:.2f})")
