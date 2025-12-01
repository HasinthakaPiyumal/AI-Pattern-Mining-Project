class KnowledgeGraph:
    def __init__(self):
        self.facts = [] # Stores (subject, predicate, object) triples

    def add_fact(self, subject, predicate, obj):
        self.facts.append((subject, predicate, obj))

    def get_facts(self, subject=None, predicate=None, obj=None):
        results = []
        for fact in self.facts:
            match = True
            if subject is not None and fact[0] != subject:
                match = False
            if predicate is not None and fact[1] != predicate:
                match = False
            if obj is not None and fact[2] != obj:
                match = False
            if match:
                results.append(fact)
        return results

    def update_fact(self, old_subject, old_predicate, old_obj, new_obj):
        for i, fact in enumerate(self.facts):
            if fact == (old_subject, old_predicate, old_obj):
                self.facts[i] = (old_subject, old_predicate, new_obj)
                return True
        return False

    def delete_fact(self, subject, predicate, obj):
        if (subject, predicate, obj) in self.facts:
            self.facts.remove((subject, predicate, obj))
            return True
        return False

    def query_paths(self, start_entity, target_relation, max_depth=3):
        # Simple BFS to find paths from start_entity to anything related by target_relation
        paths = []
        queue = [(start_entity, [], 0)] # (current_entity, current_path_triples, current_depth)
        visited = set()

        while queue:
            current_entity, current_path, current_depth = queue.pop(0)

            if (current_entity, current_depth) in visited:
                continue
            visited.add((current_entity, current_depth))

            if current_depth > max_depth:
                continue
            
            # Check for direct relation to target_relation
            for s, p, o in self.facts:
                if s == current_entity and p == target_relation:
                    paths.append(current_path + [(s, p, o)])

            # Explore neighbors
            for s, p, o in self.facts:
                if s == current_entity and o != current_entity: # Connect to object of a triple
                    new_path = current_path + [(s, p, o)]
                    queue.append((o, new_path, current_depth + 1))
                elif o == current_entity and s != current_entity: # Connect to subject of a triple
                    new_path = current_path + [(s, p, o)]
                    queue.append((s, new_path, current_depth + 1))
        
        # Filter for paths that actually lead to the target_relation and have a relevant entity
        filtered_paths = []
        for path in paths:
            # Ensure the last triple is the target relation and the path isn't empty
            if path and path[-1][1] == target_relation: # and path[-1][0] == start_entity and path[-1][1] == target_relation:
                # Further filtering to make sure the target relation's subject is part of the path flow
                # This simple query_paths primarily finds paths originating from start_entity
                # A more sophisticated graph query engine would be needed for complex paths.
                filtered_paths.append(path)
        
        return filtered_paths


class LLMReasoningCore:
    def __init__(self, kg):
        self.kg = kg
        # Simulate a basic symptom-to-disease mapping for demonstration
        self.disease_symptoms = {
            "Influenza": {"fever", "cough", "sore throat"},
            "Common Cold": {"runny nose", "sore throat", "sneezing"},
            "Pneumonia": {"cough", "chest pain", "shortness of breath", "fever"},
            "Migraine": {"headache", "nausea", "sensitivity to light"},
            "Diabetes": {"frequent urination", "increased thirst", "fatigue"},
        }

    def analyze_symptoms(self, symptoms_text):
        extracted_symptoms = set()
        text_lower = symptoms_text.lower()
        for disease, symptoms_set in self.disease_symptoms.items():
            for symptom in symptoms_set:
                if symptom in text_lower:
                    extracted_symptoms.add(symptom)
        return list(extracted_symptoms)

    def generate_reasoning_path(self, symptom_entities, potential_diagnosis):
        all_paths = []
        for symptom in symptom_entities:
            # Find paths from symptom to the potential diagnosis
            paths_from_symptom = self.kg.query_paths(symptom, "causes") # Assuming 'causes' is a relation to a disease
            for path in paths_from_symptom:
                # Check if the potential_diagnosis is present in the path's objects
                if any(triple[2] == potential_diagnosis for triple in path):
                    all_paths.append(path)
        return all_paths

    def diagnose(self, symptoms_text):
        extracted_symptoms = self.analyze_symptoms(symptoms_text)
        if not extracted_symptoms:
            return "No specific symptoms recognized.", []

        potential_diagnoses = {}
        for disease, symptoms_set in self.disease_symptoms.items():
            match_count = len(extracted_symptoms.intersection(symptoms_set))
            if match_count > 0:
                potential_diagnoses[disease] = match_count
        
        if not potential_diagnoses:
            return "Cannot determine a diagnosis based on the provided symptoms and current knowledge.", []

        # Select the diagnosis with most matching symptoms for simplicity
        best_diagnosis = max(potential_diagnoses, key=potential_diagnoses.get)
        
        reasoning_paths = self.generate_reasoning_path(extracted_symptoms, best_diagnosis)

        explanation = [
            f"Based on the symptoms: {', '.join(extracted_symptoms)}, "
            f"the most likely diagnosis is {best_diagnosis}."
        ]
        if reasoning_paths:
            explanation.append("Reasoning Path(s):")
            for i, path in enumerate(reasoning_paths):
                path_str = f"Path {i+1}: "
                for s, p, o in path:
                    path_str += f"({s} {p} {o}) -> "
                explanation.append(path_str.rstrip(" -> "))
        else:
            explanation.append("No explicit reasoning path found in the Knowledge Graph for this diagnosis.")

        return "\n".join(explanation), reasoning_paths


# --- Demonstration --- 
if __name__ == "__main__":
    # Initialize Knowledge Graph
    medical_kg = KnowledgeGraph()

    # Populate with some medical facts (entity-relation-entity triples)
    medical_kg.add_fact("fever", "is_symptom_of", "Influenza")
    medical_kg.add_fact("cough", "is_symptom_of", "Influenza")
    medical_kg.add_fact("sore throat", "is_symptom_of", "Influenza")
    medical_kg.add_fact("Influenza", "causes", "fatigue")
    medical_kg.add_fact("Influenza", "requires", "antiviral medication")

    medical_kg.add_fact("runny nose", "is_symptom_of", "Common Cold")
    medical_kg.add_fact("sore throat", "is_symptom_of", "Common Cold")
    medical_kg.add_fact("sneezing", "is_symptom_of", "Common Cold")
    medical_kg.add_fact("Common Cold", "causes", "discomfort")

    medical_kg.add_fact("cough", "is_symptom_of", "Pneumonia")
    medical_kg.add_fact("chest pain", "is_symptom_of", "Pneumonia")
    medical_kg.add_fact("shortness of breath", "is_symptom_of", "Pneumonia")
    medical_kg.add_fact("Pneumonia", "requires", "antibiotics")
    medical_kg.add_fact("Pneumonia", "is_serious_condition", "True")

    medical_kg.add_fact("headache", "is_symptom_of", "Migraine")
    medical_kg.add_fact("nausea", "is_symptom_of", "Migraine")
    medical_kg.add_fact("sensitivity to light", "is_symptom_of", "Migraine")

    medical_kg.add_fact("frequent urination", "is_symptom_of", "Diabetes")
    medical_kg.add_fact("increased thirst", "is_symptom_of", "Diabetes")
    medical_kg.add_fact("fatigue", "is_symptom_of", "Diabetes")
    medical_kg.add_fact("Diabetes", "requires", "insulin therapy")


    # Initialize LLM Reasoning Core
    diagnostic_assistant = LLMReasoningCore(medical_kg)

    print("--- Medical Diagnostic Assistant ---")

    # Scenario 1: Basic Diagnosis
    patient_symptoms = "I have a fever and a terrible cough, and my throat is sore."
    print(f"\nPatient symptoms: '{patient_symptoms}'")
    diagnosis_output, reasoning = diagnostic_assistant.diagnose(patient_symptoms)
    print(diagnosis_output)

    # Scenario 2: Another Diagnosis
    patient_symptoms_2 = "My nose is runny, and I keep sneezing. My throat also hurts a bit."
    print(f"\nPatient symptoms: '{patient_symptoms_2}'")
    diagnosis_output_2, reasoning_2 = diagnostic_assistant.diagnose(patient_symptoms_2)
    print(diagnosis_output_2)

    # Scenario 3: Diagnosis requiring correction/update
    print("\n--- Demonstrating Correction ---")
    print("Initial facts about Pneumonia requiring:")
    print(medical_kg.get_facts(subject="Pneumonia", predicate="requires"))

    # Let's say we find out 'Pneumonia' requires 'hospitalization' in some severe cases
    # We will 'correct' an existing fact or add a new one if it's a nuance
    # For this demo, let's update an existing 'requires' relation if it was wrong or incomplete.
    print("\nDoctor observes that some Pneumonia cases also require oxygen therapy, not just antibiotics.")
    # To simulate correction, we'll modify the 'requires' fact directly.
    # In a real system, this would be more nuanced, possibly adding a new fact or a more specific one.
    
    # Let's add a new fact to demonstrate infusion, as updating 'antibiotics' might be too strong.
    medical_kg.add_fact("Pneumonia", "requires", "oxygen therapy")
    medical_kg.add_fact("Pneumonia", "can_be_severe", "True")

    print("Updated facts about Pneumonia requiring/severity:")
    print(medical_kg.get_facts(subject="Pneumonia", predicate="requires"))
    print(medical_kg.get_facts(subject="Pneumonia", predicate="can_be_severe"))

    # Now, if we were to query for Pneumonia reasoning again, the new fact would be available.
    # This simple query_paths doesn't explicitly show 'requires' for diagnosis, but it could be extended.
    print("\n--- After knowledge correction/infusion, re-diagnosing (simulated) ---")
    patient_symptoms_3 = "I have a persistent cough, chest pain, and difficulty breathing. Also a fever."
    print(f"\nPatient symptoms: '{patient_symptoms_3}'")
    diagnosis_output_3, reasoning_3 = diagnostic_assistant.diagnose(patient_symptoms_3)
    print(diagnosis_output_3)
    
    # Example of a doctor identifying an erroneous path and correcting it
    print("\n--- Doctor correcting a perceived erroneous fact in KG ---")
    print("Suppose 'Influenza causes fatigue' is found to be too general or sometimes incorrect.")
    print("Current fact: ", medical_kg.get_facts("Influenza", "causes", "fatigue"))
    if medical_kg.update_fact("Influenza", "causes", "fatigue", "mild fatigue"):
        print("Fact updated: Influenza now causes mild fatigue.")
    else:
        print("Fact not found for update.")

    print("Updated fact: ", medical_kg.get_facts("Influenza", "causes", "mild fatigue"))

    print("\n--- End of Demonstration ---")
