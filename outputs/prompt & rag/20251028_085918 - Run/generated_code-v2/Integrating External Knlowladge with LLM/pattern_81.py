class KGComponent:
    def __init__(self):
        self.medical_knowledge = self._load_medical_knowledge()

    def _load_medical_knowledge(self):
        # Simplified in-memory knowledge graph for demonstration
        # Format: {entity: {relation: [related_entity, ...]}}
        return {
            "Fever": {"indicates": ["Influenza", "Pneumonia", "Malaria"]},
            "Cough": {"indicates": ["Influenza", "Pneumonia", "Bronchitis"]},
            "Fatigue": {"indicates": ["Influenza", "Anemia", "Chronic Fatigue Syndrome"]},
            "Influenza": {"causes": ["Fever", "Cough", "Fatigue"], "treatable_by": ["Oseltamivir"]},
            "Pneumonia": {"causes": ["Fever", "Cough", "Shortness of Breath"], "treatable_by": ["Antibiotics"]},
            "Malaria": {"causes": ["Fever", "Chills", "Sweating"], "treatable_by": ["Artemisinin-based combination therapies"]},
            "Anemia": {"causes": ["Fatigue", "Pale Skin"], "treatable_by": ["Iron Supplements"]},
            "Oseltamivir": {"treats": ["Influenza"]},
            "Antibiotics": {"treats": ["Pneumonia", "Bacterial Infections"]},
            "Iron Supplements": {"treats": ["Anemia"]}
        }

    def find_reasoning_paths(self, symptoms, history=None, test_results=None):
        relevant_triples = set()
        patient_conditions = set(symptoms)

        if history:
            patient_conditions.update(history)
        if test_results:
            patient_conditions.update(test_results)

        for condition in patient_conditions:
            if condition in self.medical_knowledge:
                for relation, targets in self.medical_knowledge[condition].items():
                    for target in targets:
                        relevant_triples.add((condition, relation, target))
                # Also find paths where this condition is a target
                for entity, relations_map in self.medical_knowledge.items():
                    for relation, targets in relations_map.items():
                        if condition in targets:
                            relevant_triples.add((entity, relation, condition))

        return list(relevant_triples)


class PromptGenerator:
    def generate_llm_prompt(self, patient_query, kg_triples):
        prompt_parts = []
        prompt_parts.append(f"Patient Information: {patient_query}")
        prompt_parts.append("Relevant Medical Knowledge (Entity-Relation-Entity triples):")
        
        if not kg_triples:
            prompt_parts.append("  No specific relevant medical knowledge found.")
        else:
            for entity1, relation, entity2 in kg_triples:
                prompt_parts.append(f"  {entity1} {relation} {entity2};")

        prompt_parts.append("\nBased on the patient information and the provided medical knowledge, provide a differential diagnosis, an explanation for each diagnosis, and potential treatment recommendations. Prioritize the most likely diagnoses.\nDifferential Diagnosis:")
        return "\n".join(prompt_parts)


class LLMIntegrator:
    def get_llm_response(self, prompt):
        # Mock LLM response for demonstration purposes
        # In a real application, this would involve an API call to a large language model
        if "Fever" in prompt and "Cough" in prompt and "Fatigue" in prompt and "Influenza causes Fever" in prompt:
            return (
                "Differential Diagnosis:\n"
                "1. Influenza: Highly likely given the common symptoms (fever, cough, fatigue) and direct knowledge graph link. \n   Treatment: Rest, fluids, Oseltamivir (if within 48 hours of symptom onset).\
                \n2. Pneumonia: Possible, as it also presents with fever and cough. The KG suggests it causes fever and cough. Requires further investigation. \n   Treatment: Antibiotics, oxygen therapy if severe.\
                \n3. Common Cold: Less severe, but shares some symptoms. \n   Treatment: Symptomatic relief (e.g., pain relievers, decongestants)."
            )
        elif "Fatigue" in prompt and "Pale Skin" in prompt and "Anemia causes Fatigue" in prompt:
             return (
                "Differential Diagnosis:\n"
                "1. Anemia: Highly likely given fatigue and pale skin, directly supported by KG. \n   Treatment: Iron supplements, dietary changes.\
                \n2. Chronic Fatigue Syndrome: Possible, but less direct KG support for these specific symptoms alone. \n   Treatment: Symptomatic management, lifestyle adjustments.\
                \n3. Hypothyroidism: Could cause fatigue, but pale skin is less typical without other symptoms. \n   Treatment: Thyroid hormone replacement."
             )
        else:
            return (
                "Differential Diagnosis:\n"
                "1. Undetermined: More information needed to provide a specific diagnosis. \n   Treatment: Symptomatic relief, further diagnostic tests."
            )


class MedicalDiagnosticAssistant:
    def __init__(self):
        self.kg_component = KGComponent()
        self.prompt_generator = PromptGenerator()
        self.llm_integrator = LLMIntegrator()

    def diagnose_patient(self, symptoms, history=None, test_results=None):
        patient_query = f"Symptoms: {', '.join(symptoms)}. "
        if history: patient_query += f"History: {', '.join(history)}. "
        if test_results: patient_query += f"Test Results: {', '.join(test_results)}."

        kg_triples = self.kg_component.find_reasoning_paths(symptoms, history, test_results)
        llm_prompt = self.prompt_generator.generate_llm_prompt(patient_query, kg_triples)
        llm_response = self.llm_integrator.get_llm_response(llm_prompt)
        return llm_response


if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    print("--- Scenario 1: Influenza-like symptoms ---")
    symptoms1 = ["Fever", "Cough", "Fatigue"]
    diagnosis1 = assistant.diagnose_patient(symptoms1)
    print(f"Patient Input: {', '.join(symptoms1)}")
    print("\nLLM Output:")
    print(diagnosis1)

    print("\n--- Scenario 2: Anemia-like symptoms ---")
    symptoms2 = ["Fatigue", "Pale Skin"]
    diagnosis2 = assistant.diagnose_patient(symptoms2)
    print(f"Patient Input: {', '.join(symptoms2)}")
    print("\nLLM Output:")
    print(diagnosis2)

    print("\n--- Scenario 3: Less specific symptoms ---")
    symptoms3 = ["Headache"]
    diagnosis3 = assistant.diagnose_patient(symptoms3)
    print(f"Patient Input: {', '.join(symptoms3)}")
    print("\nLLM Output:")
    print(diagnosis3)

    print("\n--- Scenario 4: Pneumonia-related symptoms ---")
    symptoms4 = ["Fever", "Cough", "Shortness of Breath"]
    diagnosis4 = assistant.diagnose_patient(symptoms4)
    print(f"Patient Input: {', '.join(symptoms4)}")
    print("\nLLM Output:")
    print(diagnosis4)

    print("\n--- Scenario 5: Symptoms with history ---")
    symptoms5 = ["Fever", "Cough"]
    history5 = ["recently traveled"]
    diagnosis5 = assistant.diagnose_patient(symptoms5, history=history5)
    print(f"Patient Input: Symptoms: {', '.join(symptoms5)}, History: {', '.join(history5)}")
    print("\nLLM Output:")
    print(diagnosis5)