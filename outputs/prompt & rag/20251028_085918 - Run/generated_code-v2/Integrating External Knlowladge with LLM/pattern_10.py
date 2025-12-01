import networkx as nx

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self._build_kg()

    def _build_kg(self):
        # Add diseases
        self.graph.add_node("Influenza", type="disease")
        self.graph.add_node("Pneumonia", type="disease")
        self.graph.add_node("Common Cold", type="disease")
        self.graph.add_node("Appendicitis", type="disease")
        self.graph.add_node("Migraine", type="disease")
        self.graph.add_node("Strep Throat", type="disease")

        # Add symptoms
        self.graph.add_node("Fever", type="symptom")
        self.graph.add_node("Cough", type="symptom")
        self.graph.add_node("Sore Throat", type="symptom")
        self.graph.add_node("Runny Nose", type="symptom")
        self.graph.add_node("Headache", type="symptom")
        self.graph.add_node("Fatigue", type="symptom")
        self.graph.add_node("Body Aches", type="symptom")
        self.graph.add_node("Nausea", type="symptom")
        self.graph.add_node("Abdominal Pain", type="symptom")
        self.graph.add_node("Loss of Appetite", type="symptom")
        self.graph.add_node("Chills", type="symptom")
        self.graph.add_node("Shortness of Breath", type="symptom")
        self.graph.add_node("Swollen Lymph Nodes", type="symptom")
        self.graph.add_node("Muscle Pain", type="symptom")

        # Add tests
        self.graph.add_node("Flu Test", type="test")
        self.graph.add_node("Chest X-ray", type="test")
        self.graph.add_node("CBC Blood Test", type="test")
        self.graph.add_node("CT Scan Abdomen", type="test")
        self.graph.add_node("Throat Swab", type="test")

        # Add treatments
        self.graph.add_node("Antivirals", type="treatment")
        self.graph.add_node("Antibiotics", type="treatment")
        self.graph.add_node("Pain Relievers", type="treatment")
        self.graph.add_node("Rest and Fluids", type="treatment")
        self.graph.add_node("Surgery", type="treatment")

        # Add relationships
        # Influenza
        self.graph.add_edge("Influenza", "Fever", relation="has_symptom")
        self.graph.add_edge("Influenza", "Cough", relation="has_symptom")
        self.graph.add_edge("Influenza", "Sore Throat", relation="has_symptom")
        self.graph.add_edge("Influenza", "Body Aches", relation="has_symptom")
        self.graph.add_edge("Influenza", "Fatigue", relation="has_symptom")
        self.graph.add_edge("Influenza", "Chills", relation="has_symptom")
        self.graph.add_edge("Influenza", "Flu Test", relation="diagnosed_by")
        self.graph.add_edge("Influenza", "Antivirals", relation="treated_by")
        self.graph.add_edge("Influenza", "Rest and Fluids", relation="treated_by")

        # Pneumonia
        self.graph.add_edge("Pneumonia", "Fever", relation="has_symptom")
        self.graph.add_edge("Pneumonia", "Cough", relation="has_symptom")
        self.graph.add_edge("Pneumonia", "Shortness of Breath", relation="has_symptom")
        self.graph.add_edge("Pneumonia", "Chills", relation="has_symptom")
        self.graph.add_edge("Pneumonia", "Chest X-ray", relation="diagnosed_by")
        self.graph.add_edge("Pneumonia", "Antibiotics", relation="treated_by")

        # Common Cold
        self.graph.add_edge("Common Cold", "Runny Nose", relation="has_symptom")
        self.graph.add_edge("Common Cold", "Sore Throat", relation="has_symptom")
        self.graph.add_edge("Common Cold", "Cough", relation="has_symptom")
        self.graph.add_edge("Common Cold", "Headache", relation="has_symptom")
        self.graph.add_edge("Common Cold", "Rest and Fluids", relation="treated_by")
        self.graph.add_edge("Common Cold", "Pain Relievers", relation="treated_by")

        # Appendicitis
        self.graph.add_edge("Appendicitis", "Abdominal Pain", relation="has_symptom")
        self.graph.add_edge("Appendicitis", "Nausea", relation="has_symptom")
        self.graph.add_edge("Appendicitis", "Loss of Appetite", relation="has_symptom")
        self.graph.add_edge("Appendicitis", "Fever", relation="has_symptom")
        self.graph.add_edge("Appendicitis", "CT Scan Abdomen", relation="diagnosed_by")
        self.graph.add_edge("Appendicitis", "Surgery", relation="treated_by")

        # Migraine
        self.graph.add_edge("Migraine", "Headache", relation="has_symptom")
        self.graph.add_edge("Migraine", "Nausea", relation="has_symptom")
        self.graph.add_edge("Migraine", "Fatigue", relation="has_symptom")
        self.graph.add_edge("Migraine", "Pain Relievers", relation="treated_by")
        self.graph.add_edge("Migraine", "Rest and Fluids", relation="treated_by")

        # Strep Throat
        self.graph.add_edge("Strep Throat", "Sore Throat", relation="has_symptom")
        self.graph.add_edge("Strep Throat", "Fever", relation="has_symptom")
        self.graph.add_edge("Strep Throat", "Swollen Lymph Nodes", relation="has_symptom")
        self.graph.add_edge("Strep Throat", "Body Aches", relation="has_symptom")
        self.graph.add_edge("Strep Throat", "Throat Swab", relation="diagnosed_by")
        self.graph.add_edge("Strep Throat", "Antibiotics", relation="treated_by")


class KGInteractionTools:
    def __init__(self, kg: MedicalKnowledgeGraph):
        self.kg = kg.graph

    def get_related_entities(self, entity_name: str, relation_type: str):
        if entity_name not in self.kg:
            return []
        related = []
        for neighbor in self.kg.neighbors(entity_name):
            if self.kg.get_edge_data(entity_name, neighbor).get("relation") == relation_type:
                related.append(neighbor)
        return related

    def get_entities_by_type(self, entity_type: str):
        return [node for node, data in self.kg.nodes(data=True) if data.get("type") == entity_type]

    def get_info(self, entity_name: str):
        if entity_name in self.kg:
            return self.kg.nodes[entity_name]
        return None


class LLMAgent:
    def __init__(self, kg_tools: KGInteractionTools):
        self.kg_tools = kg_tools

    def _simulate_llm_reasoning(self, query: str, context: list):
        # This method simulates the LLM's natural language processing and reasoning
        # In a real application, this would involve calling a true LLM (e.g., OpenAI, Gemini)
        # For this prototype, we'll use simple keyword matching and pre-defined logic.

        if "diagnose" in query.lower() and "symptoms" in query.lower():
            return self._diagnose_logic(context)
        elif "suggest tests" in query.lower():
            return self._suggest_tests_logic(context)
        elif "suggest treatments" in query.lower():
            return self._suggest_treatments_logic(context)
        return "I am not sure how to respond to that based on the current context."

    def _diagnose_logic(self, context: list):
        patient_symptoms = context.get("patient_symptoms", [])
        if not patient_symptoms:
            return "Please provide symptoms for diagnosis."

        potential_diagnoses = {}
        for symptom in patient_symptoms:
            related_diseases = self.kg_tools.get_related_entities(symptom, "has_symptom")
            for disease in related_diseases:
                if disease not in potential_diagnoses:
                    potential_diagnoses[disease] = {"matching_symptoms": [], "missing_symptoms": []}
                potential_diagnoses[disease]["matching_symptoms"].append(symptom)

        # Refine diagnoses based on all patient symptoms
        final_diagnoses = []
        for disease, data in potential_diagnoses.items():
            disease_symptoms = self.kg_tools.get_related_entities(disease, "has_symptom")
            current_matching = set(data["matching_symptoms"])
            current_missing = set(disease_symptoms) - current_matching

            # Update based on overall patient symptoms, not just initial trigger
            for patient_symptom in patient_symptoms:
                if patient_symptom in disease_symptoms and patient_symptom not in current_matching:
                    current_matching.add(patient_symptom)
                elif patient_symptom not in disease_symptoms and patient_symptom not in current_missing:
                     # A symptom is present in patient but not in disease's known symptoms
                     # This can be used to penalize or filter out unlikely diseases
                    pass # For simplicity, we just look for matches and missing known symptoms

            data["matching_symptoms"] = list(current_matching)
            data["missing_symptoms"] = list(set(disease_symptoms) - current_matching)

            if len(data["matching_symptoms"]) > 0: # Only consider diseases with at least one matching symptom
                final_diagnoses.append((disease, data))

        # Sort by number of matching symptoms (desc) and then by number of missing symptoms (asc)
        final_diagnoses.sort(key=lambda x: (len(x[1]["matching_symptoms"]), -len(x[1]["missing_symptoms"])), reverse=True)

        if not final_diagnoses:
            return {
                "diagnosis": "No clear diagnosis based on provided symptoms.",
                "justification": "The symptoms provided do not strongly align with any known conditions in the medical knowledge graph.",
                "suggested_tests": [],
                "suggested_treatments": []
            }

        best_diagnosis = final_diagnoses[0]
        disease_name = best_diagnosis[0]
        justification_parts = [
            f"Based on the provided symptoms, the most likely diagnosis is {disease_name}."
        ]
        justification_parts.append(f"Matching symptoms: {', '.join(best_diagnosis[1]['matching_symptoms'])}.")
        if best_diagnosis[1]['missing_symptoms']:
            justification_parts.append(f"Note that typical symptoms for {disease_name} also include: {', '.join(best_diagnosis[1]['missing_symptoms'])}. Further investigation may be needed.")

        suggested_tests = self.kg_tools.get_related_entities(disease_name, "diagnosed_by")
        suggested_treatments = self.kg_tools.get_related_entities(disease_name, "treated_by")

        return {
            "diagnosis": disease_name,
            "differential_diagnoses": [d[0] for d in final_diagnoses[:3]], # Top 3
            "justification": " ".join(justification_parts),
            "suggested_tests": suggested_tests,
            "suggested_treatments": suggested_treatments
        }

    def _suggest_tests_logic(self, context: dict):
        disease = context.get("disease")
        if not disease:
            return "Please specify a disease to suggest tests."
        tests = self.kg_tools.get_related_entities(disease, "diagnosed_by")
        return f"For {disease}, common tests include: {', '.join(tests) if tests else 'No specific tests found in KG.'}"

    def _suggest_treatments_logic(self, context: dict):
        disease = context.get("disease")
        if not disease:
            return "Please specify a disease to suggest treatments."
        treatments = self.kg_tools.get_related_entities(disease, "treated_by")
        return f"For {disease}, common treatments include: {', '.join(treatments) if treatments else 'No specific treatments found in KG.'}"

    def process_patient_case(self, patient_symptoms: list):
        print(f"\nLLM Agent received symptoms: {', '.join(patient_symptoms)}")

        # Step 1: LLM Agent queries KG for initial related diseases based on symptoms
        print("Agent: Initial exploration of KG for diseases related to symptoms...")
        initial_context = {"patient_symptoms": patient_symptoms}
        diagnostic_result = self._simulate_llm_reasoning("diagnose with symptoms", initial_context)

        return diagnostic_result


def main():
    print("Initializing Medical Diagnostic Assistant...")
    kg = MedicalKnowledgeGraph()
    kg_tools = KGInteractionTools(kg)
    llm_agent = LLMAgent(kg_tools)
    print("Assistant ready. Enter patient symptoms (comma-separated) or 'exit' to quit.")

    while True:
        user_input = input("\nEnter symptoms: ").strip()
        if user_input.lower() == 'exit':
            break

        symptoms = [s.strip().title() for s in user_input.split(',') if s.strip()]
        if not symptoms:
            print("Please enter at least one symptom.")
            continue

        # Ensure symptoms are known in the KG for a more focused demo
        known_symptoms = kg_tools.get_entities_by_type("symptom")
        valid_symptoms = [s for s in symptoms if s in known_symptoms]
        if not valid_symptoms:
            print("None of the entered symptoms are recognized. Known symptoms include:")
            print(f"Known: {', '.join(known_symptoms)}")
            continue
        elif len(valid_symptoms) < len(symptoms):
            print(f"Note: Some entered symptoms were not recognized in the KG: {', '.join(set(symptoms) - set(valid_symptoms))}. Proceeding with recognized symptoms.")

        result = llm_agent.process_patient_case(valid_symptoms)

        print("\n--- Diagnostic Report ---")
        print(f"Primary Diagnosis: {result['diagnosis']}")
        if result.get('differential_diagnoses'):
            print(f"Differential Diagnoses (Top 3): {', '.join(result['differential_diagnoses'])}")
        print(f"Justification: {result['justification']}")
        print(f"Suggested Tests: {', '.join(result['suggested_tests']) if result['suggested_tests'] else 'N/A'}")
        print(f"Suggested Treatments: {', '.join(result['suggested_treatments']) if result['suggested_treatments'] else 'N/A'}")

if __name__ == "__main__":
    main()