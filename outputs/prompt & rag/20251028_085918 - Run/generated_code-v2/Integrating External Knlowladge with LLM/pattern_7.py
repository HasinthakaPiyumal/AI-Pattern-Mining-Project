class MedicalKnowledgeGraph:
    """A simplified in-memory representation of a Medical Knowledge Graph."""

    def __init__(self):
        self.entities = {}
        self.relations = []
        self._next_entity_id = 1

    def add_entity(self, entity_type, name, attributes=None):
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        self.entities[entity_id] = {
            "id": entity_id,
            "type": entity_type,
            "name": name,
            "attributes": attributes if attributes is not None else {}
        }
        return entity_id

    def add_relation(self, source_id, relation_type, target_id, attributes=None):
        if source_id not in self.entities or target_id not in self.entities:
            raise ValueError("Source or target entity not found in KG.")
        self.relations.append({
            "source": source_id,
            "type": relation_type,
            "target": target_id,
            "attributes": attributes if attributes is not None else {}
        })

    def get_paths(self, start_entity_name, max_hops=2):
        """
        Retrieves simple paths from a starting entity name up to max_hops.
        This is a highly simplified path retrieval for demonstration.
        """
        start_entities = [e_id for e_id, entity in self.entities.items() if entity["name"] == start_entity_name]
        if not start_entities:
            return []

        found_paths = []
        queue = [(e_id, [self.entities[e_id]], 0) for e_id in start_entities] # (current_entity_id, current_path, current_hops)
        visited = set(start_entities)

        while queue:
            current_id, current_path, hops = queue.pop(0)

            if hops >= max_hops:
                continue

            for relation in self.relations:
                if relation["source"] == current_id:
                    target_id = relation["target"]
                    if target_id not in visited:
                        new_path = current_path + [relation, self.entities[target_id]]
                        found_paths.append(new_path)
                        visited.add(target_id)
                        queue.append((target_id, new_path, hops + 1))
                elif relation["target"] == current_id: # Also consider inverse relations for simplicity of path finding
                    source_id = relation["source"]
                    if source_id not in visited:
                        new_path = [self.entities[source_id], relation] + current_path
                        found_paths.append(new_path)
                        visited.add(source_id)
                        queue.append((source_id, new_path, hops + 1))
        return found_paths

class KG_Tuned_LLM_Simulator:
    """
    Simulates an LLM fine-tuned for KG interaction.
    In a real scenario, this would involve a transformer model (e.g., Llama, BERT-based) 
    fine-tuned on instruction-following datasets derived from KG data.
    """
    def __init__(self, kg: MedicalKnowledgeGraph):
        self.kg = kg
        # In a real system, a pre-trained LLM would be loaded here
        # and its weights would reflect the instruction tuning.

    def _simulate_planning(self, symptoms: list[str]) -> list[tuple]:
        """
        Simulates the 'Planning Optimization' task.
        Generates a hypothetical sequence of KG queries/operations based on symptoms.
        """
        print(f"[LLM Simulator] Generating KG plan for symptoms: {symptoms}")
        # This would be a complex LLM inference, here we use simple heuristics.
        plan = []
        if "fever" in symptoms or "cough" in symptoms:
            plan.append(("Symptom", "associated_with", "Disease"))
            plan.append(("Disease", "treated_by", "Treatment"))
        if "chest pain" in symptoms:
            plan.append(("Symptom", "indicates", "Condition"))
            plan.append(("Condition", "requires", "DiagnosticTest"))
        
        # Add a general relation to explore related conditions
        plan.append(("Disease", "has_complication", "Complication"))

        print(f"[LLM Simulator] Generated plan: {plan}")
        return plan

    def _simulate_reasoning(self, kg_retrieved_data: list, patient_history: str) -> dict:
        """
        Simulates the 'Retrieval-Reasoning Optimization' task.
        Reasons over retrieved KG data and patient history to suggest diagnosis.
        """
        print("[LLM Simulator] Performing reasoning on retrieved KG data and patient history.")
        
        diagnosis_likelihood = {}
        evidence_snippets = []
        suggested_actions = []

        # Simple heuristic reasoning based on retrieved paths
        for path in kg_retrieved_data:
            path_description = " -> ".join([str(item["name"]) if isinstance(item, dict) and "name" in item else str(item["type"]) for item in path if isinstance(item, dict) and ("name" in item or "type" in item)])
            evidence_snippets.append(f"Path: {path_description}")
            
            # Example: Identify potential diseases from paths
            for i, item in enumerate(path):
                if isinstance(item, dict) and item.get("type") == "Disease":
                    disease_name = item.get("name")
                    diagnosis_likelihood[disease_name] = diagnosis_likelihood.get(disease_name, 0) + 1
                    
                if isinstance(item, dict) and item.get("type") == "Treatment":
                    treatment_name = item.get("name")
                    suggested_actions.append(f"Consider treatment: {treatment_name}")

        final_diagnosis = "Uncertain Diagnosis"
        if diagnosis_likelihood:
            final_diagnosis = max(diagnosis_likelihood, key=diagnosis_likelihood.get)
            evidence_snippets.append(f"Most likely disease based on KG paths: {final_diagnosis}")

        if "elderly" in patient_history.lower() and "fever" in diagnosis_likelihood:
            suggested_actions.append("Monitor for complications given patient's age.")

        return {
            "diagnosis": final_diagnosis,
            "evidence": "\n".join(evidence_snippets),
            "suggestions": "\n".join(list(set(suggested_actions))) # Remove duplicates
        }

# --- Instruction Tuning Conceptual Explanation ---
# In a real-world scenario, the 'KG_Tuned_LLM_Simulator' would represent an actual LLM
# that has undergone 'Instruction Tuning'. This involves:
# 
# 1.  **Data Curation**: Creating a dataset of (instruction, KG_grounding, response) triplets.
#     *   **Planning Optimization**: Instructions like "Given symptoms X, what is a likely diagnostic path in the KG?"
#         Responses would be structured KG relation paths (e.g., "Symptom -> associated_with -> Disease -> treated_by -> Drug").
#     *   **Retrieval-Reasoning Optimization**: Instructions like "Given retrieved KG facts A, B, C, and patient history D, what is the diagnosis and supporting evidence?"
#         Responses would be natural language diagnoses with explicit citations to the provided KG facts.
#     This dataset would be automatically or semi-automatically generated by traversing the KG and synthesizing questions/answers.
# 
# 2.  **Fine-tuning**: Using this dataset to fine-tune a pre-trained LLM (e.g., using Hugging Face Transformers library with PEFT/LoRA).
#     The LLM learns to follow these KG-specific instructions, generating structured plans or grounded reasoning.
#     Libraries like 'trl' (Transformer Reinforcement Learning) or 'accelerate' can assist in this process.
# 
# 3.  **Deployment**: The fine-tuned LLM is then deployed, and its inference capabilities are what are simulated in the `_simulate_planning`
#     and `_simulate_reasoning` methods above.


def run_diagnostic_assistant():
    print("\n--- Medical Diagnostic Assistant (KG-Grounding Demo) ---")

    # 1. Initialize and populate a Medical Knowledge Graph
    medical_kg = MedicalKnowledgeGraph()

    # Add some example medical entities
    symptom_fever = medical_kg.add_entity("Symptom", "fever")
    symptom_cough = medical_kg.add_entity("Symptom", "cough")
    symptom_chest_pain = medical_kg.add_entity("Symptom", "chest pain")
    symptom_fatigue = medical_kg.add_entity("Symptom", "fatigue")

    disease_flu = medical_kg.add_entity("Disease", "Influenza")
    disease_pneumonia = medical_kg.add_entity("Disease", "Pneumonia")
    disease_bronchitis = medical_kg.add_entity("Disease", "Bronchitis")
    disease_heart_attack = medical_kg.add_entity("Disease", "Myocardial Infarction")

    drug_oseltamivir = medical_kg.add_entity("Treatment", "Oseltamivir")
    drug_antibiotics = medical_kg.add_entity("Treatment", "Antibiotics")
    drug_bronchodilators = medical_kg.add_entity("Treatment", "Bronchodilators")

    test_xray = medical_kg.add_entity("DiagnosticTest", "Chest X-ray")
    test_ecg = medical_kg.add_entity("DiagnosticTest", "ECG")

    # Add relations
    medical_kg.add_relation(symptom_fever, "associated_with", disease_flu, {"strength": 0.8})
    medical_kg.add_relation(symptom_cough, "associated_with", disease_flu, {"strength": 0.7})
    medical_kg.add_relation(symptom_fever, "associated_with", disease_pneumonia, {"strength": 0.9})
    medical_kg.add_relation(symptom_cough, "associated_with", disease_pneumonia, {"strength": 0.85})
    medical_kg.add_relation(symptom_fatigue, "associated_with", disease_pneumonia, {"strength": 0.6})
    medical_kg.add_relation(symptom_cough, "associated_with", disease_bronchitis, {"strength": 0.95})
    medical_kg.add_relation(symptom_chest_pain, "indicates", disease_heart_attack, {"strength": 0.9})

    medical_kg.add_relation(disease_flu, "treated_by", drug_oseltamivir)
    medical_kg.add_relation(disease_pneumonia, "treated_by", drug_antibiotics)
    medical_kg.add_relation(disease_bronchitis, "treated_by", drug_bronchodilators)

    medical_kg.add_relation(disease_pneumonia, "requires", test_xray)
    medical_kg.add_relation(disease_heart_attack, "requires", test_ecg)

    medical_kg.add_relation(disease_flu, "has_complication", disease_pneumonia, {"risk": "moderate"})

    # 2. Initialize the KG-Tuned LLM Simulator
    llm_assistant = KG_Tuned_LLM_Simulator(medical_kg)

    while True:
        user_symptoms_input = input("\nEnter patient symptoms (comma-separated, e.g., fever, cough): ").strip().lower()
        if not user_symptoms_input:
            print("Please enter symptoms.")
            continue
        
        patient_history_input = input("Enter relevant patient history (e.g., 'elderly, smoker'): ").strip().lower()

        input_symptoms = [s.strip() for s in user_symptoms_input.split(',') if s.strip()]

        print("\n--- Processing Patient Data ---")
        
        # Step 1: LLM generates a KG-grounded plan
        kg_plan = llm_assistant._simulate_planning(input_symptoms)
        
        # Step 2: Use the KG to retrieve data based on the plan
        # This part simplifies the plan execution. In reality, the LLM's plan
        # would guide a series of specific KG queries.
        retrieved_kg_data = []
        for symptom_name in input_symptoms:
            paths = medical_kg.get_paths(symptom_name, max_hops=2)
            if paths:
                retrieved_kg_data.extend(paths)
                print(f"[KG] Retrieved {len(paths)} paths for symptom '{symptom_name}'.")

        if not retrieved_kg_data:
            print("[KG] No relevant paths found for the given symptoms in the Knowledge Graph.")
            print("Diagnosis: Unclear. No KG grounding found.")
        else:
            # Step 3: LLM performs reasoning on retrieved KG data
            diagnosis_result = llm_assistant._simulate_reasoning(retrieved_kg_data, patient_history_input)

            print("\n--- Diagnostic Result ---")
            print(f"Diagnosis: {diagnosis_result['diagnosis']}")
            print(f"\nEvidence from KG:\n{diagnosis_result['evidence']}")
            print(f"\nSuggestions:\n{diagnosis_result['suggestions']}")

        another_query = input("\nProcess another patient? (yes/no): ").strip().lower()
        if another_query != 'yes':
            break

if __name__ == "__main__":
    run_diagnostic_assistant()