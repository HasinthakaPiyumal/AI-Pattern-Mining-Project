import random

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = {
            "fever": {"is_symptom_of": ["flu", "malaria", "pneumonia"]},
            "headache": {"is_symptom_of": ["flu", "migraine", "tension_headache"]},
            "cough": {"is_symptom_of": ["flu", "pneumonia", "bronchitis"]},
            "fatigue": {"is_symptom_of": ["flu", "anemia", "chronic_fatigue_syndrome"]},
            "flu": {"has_symptom": ["fever", "headache", "cough", "fatigue"], "treatable_by": ["antivirals", "rest"]},
            "malaria": {"has_symptom": ["fever", "chills", "sweating"], "treatable_by": ["antimalarials"]},
            "pneumonia": {"has_symptom": ["fever", "cough", "shortness_of_breath"], "treatable_by": ["antibiotics"]},
            "migraine": {"has_symptom": ["headache", "nausea", "light_sensitivity"], "treatable_by": ["triptans", "pain_relievers"]},
            "diabetes_type2": {"has_symptom": ["frequent_urination", "increased_thirst", "fatigue"], "complication": ["heart_disease", "kidney_disease"], "treatable_by": ["metformin", "lifestyle_changes"]},
            "metformin": {"treats": ["diabetes_type2"], "has_side_effect": ["nausea", "diarrhea"]},
            "antivirals": {"treats": ["flu"]},
            "antibiotics": {"treats": ["pneumonia", "bronchitis"]}
        }

    def get_neighbors(self, entity):
        return self.graph.get(entity, {})

class LLMAgent:
    def __init__(self, kg):
        self.kg = kg

    def extract_initial_entities(self, patient_data):
        # Simulate LLM extracting entities from patient data
        possible_entities = ["fever", "headache", "cough", "fatigue", "frequent_urination", "increased_thirst", "diabetes_type2"]
        extracted = [e for e in possible_entities if e.replace('_', ' ') in patient_data.lower()]
        if not extracted:
            # Fallback for more general cases or if specific entities aren't hardcoded
            if "fever" in patient_data.lower() or "high temperature" in patient_data.lower():
                extracted.append("fever")
            if "headache" in patient_data.lower() or "head pain" in patient_data.lower():
                extracted.append("headache")
        return extracted if extracted else ["unknown_symptom"]

    def evaluate_and_prune_paths(self, query, current_paths, candidate_paths, top_n=3):
        # Simulate LLM evaluating relevance and pruning paths
        # In a real scenario, this would involve complex reasoning based on the query
        # For simulation, we'll prioritize paths that connect to more distinct entities or are shorter.
        scored_paths = []
        for path in candidate_paths:
            path_length = len(path)
            unique_entities = set()
            for step in path:
                unique_entities.add(step[0])
                unique_entities.add(step[2])
            score = len(unique_entities) - path_length # More unique entities, shorter path is better
            scored_paths.append((score, path))

        scored_paths.sort(key=lambda x: x[0], reverse=True)
        return [p for s, p in scored_paths[:top_n]]

    def reason_and_answer(self, query, reasoning_paths, max_depth_reached):
        # Simulate LLM reasoning and generating an answer
        if not reasoning_paths and max_depth_reached:
            return "Based on the available information and knowledge graph exploration, I cannot confidently provide a diagnosis or treatment recommendation. Further medical evaluation is advised.", True

        if not reasoning_paths:
            return None, False # No sufficient paths yet

        diagnoses = set()
        treatments = set()
        path_explanations = []

        for path in reasoning_paths:
            explanation = "Patient symptoms suggest: "
            current_entity = None
            for i, (e1, rel, e2) in enumerate(path):
                if i == 0:
                    explanation += f"{e1}"
                    current_entity = e2
                else:
                    explanation += f" --({rel})--> {e2}"
                    current_entity = e2

                if rel == "is_symptom_of":
                    diagnoses.add(e2)
                elif rel == "treatable_by":
                    treatments.add(e2)

            path_explanations.append(explanation)

        if diagnoses or treatments:
            diagnosis_str = ", ".join(diagnoses) if diagnoses else "No specific diagnosis inferred"
            treatment_str = ", ".join(treatments) if treatments else "No specific treatments recommended"
            final_answer = f"Possible Diagnoses: {diagnosis_str}\nRecommended Treatments: {treatment_str}\n\nReasoning Paths:\n" + "\n".join(path_explanations)
            return final_answer, True
        else:
            return None, False

class ToGAlgorithmicFramework:
    def __init__(self, llm_agent, kg, max_depth=3, beam_width=3):
        self.llm = llm_agent
        self.kg = kg
        self.max_depth = max_depth
        self.beam_width = beam_width

    def run(self, patient_data):
        print(f"\n--- Starting ToG for Patient Data: {patient_data} ---")
        query = patient_data

        # Phase 1: Initialization
        initial_entities = self.llm.extract_initial_entities(patient_data)
        if not initial_entities:
            return self.llm.reason_and_answer(query, [], True)[0]

        # Initialize beam with single-entity paths for initial entities
        current_beam = [[(e, "starts_with", e)] for e in initial_entities if e != "unknown_symptom"]
        if not current_beam:
             return self.llm.reason_and_answer(query, [], True)[0]

        all_reasoning_paths = []
        max_depth_reached = False

        for depth in range(self.max_depth):
            print(f"\n--- Exploration Phase (Depth {depth + 1}/{self.max_depth}) ---")
            if not current_beam:
                print("Beam is empty, stopping exploration.")
                break

            new_candidate_paths = []
            for path in current_beam:
                last_entity = path[-1][2] # Get the last entity in the path
                neighbors = self.kg.get_neighbors(last_entity)
                
                for relation, connected_entities in neighbors.items():
                    for next_entity in connected_entities:
                        new_path = path + [(last_entity, relation, next_entity)]
                        new_candidate_paths.append(new_path)

            if not new_candidate_paths:
                print("No new paths found for exploration.")
                break

            # Phase 2b: Prune
            current_beam = self.llm.evaluate_and_prune_paths(query, current_beam, new_candidate_paths, self.beam_width)
            all_reasoning_paths.extend(current_beam)
            
            # Phase 3: Reasoning (Interim Check)
            answer, sufficient = self.llm.reason_and_answer(query, current_beam, False)
            if sufficient and answer is not None:
                print("Sufficient reasoning paths found. Generating answer.")
                return answer

        max_depth_reached = True
        print("\n--- Max depth reached or no further relevant paths. Final Reasoning Phase ---")
        final_answer, _ = self.llm.reason_and_answer(query, all_reasoning_paths, max_depth_reached)
        return final_answer

# Main execution
if __name__ == "__main__":
    medical_kg = MedicalKnowledgeGraph()
    llm_mock_agent = LLMAgent(medical_kg)
    tog_framework = ToGAlgorithmicFramework(llm_mock_agent, medical_kg, max_depth=3, beam_width=3)

    patient_cases = [
        "Patient has high fever, persistent cough, and fatigue.",
        "Patient reports severe headache and nausea, sensitive to light.",
        "Patient experiences frequent urination and increased thirst, and has fatigue.",
        "Patient has chills and sweating along with a fever.",
        "Patient feels generally unwell but no specific strong symptoms."
    ]

    for case in patient_cases:
        result = tog_framework.run(case)
        print(f"\n--- FINAL RESULT FOR CASE: {case} ---")
        print(result)
        print("="*80)
