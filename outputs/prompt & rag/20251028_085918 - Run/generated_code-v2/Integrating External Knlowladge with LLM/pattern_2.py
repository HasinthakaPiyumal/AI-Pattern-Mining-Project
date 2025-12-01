import networkx as nx

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_sample_kg()

    def _build_sample_kg(self):
        # Nodes: Medical entities
        diseases = ["Flu", "Common Cold", "Pneumonia", "Malaria", "Appendicitis"]
        symptoms = ["Fever", "Cough", "Sore Throat", "Headache", "Body Aches", "Fatigue", "Chills", "Shortness of Breath", "Nausea", "Abdominal Pain", "Diarrhea"]
        tests = ["Flu Test", "Chest X-ray", "Blood Test", "CT Scan"]
        treatments = ["Paracetamol", "Antibiotics", "Rest", "Fluids"]

        self.graph.add_nodes_from(diseases, type="disease")
        self.graph.add_nodes_from(symptoms, type="symptom")
        self.graph.add_nodes_from(tests, type="test")
        self.graph.add_nodes_from(treatments, type="treatment")

        # Edges: Relationships
        # Flu relations
        self.graph.add_edge("Flu", "Fever", relation="has_symptom")
        self.graph.add_edge("Flu", "Cough", relation="has_symptom")
        self.graph.add_edge("Flu", "Body Aches", relation="has_symptom")
        self.graph.add_edge("Flu", "Fatigue", relation="has_symptom")
        self.graph.add_edge("Fever", "Flu", relation="is_symptom_of")
        self.graph.add_edge("Cough", "Flu", relation="is_symptom_of")
        self.graph.add_edge("Flu Test", "Flu", relation="diagnoses")
        self.graph.add_edge("Flu", "Paracetamol", relation="treatable_by")

        # Common Cold relations
        self.graph.add_edge("Common Cold", "Cough", relation="has_symptom")
        self.graph.add_edge("Common Cold", "Sore Throat", relation="has_symptom")
        self.graph.add_edge("Common Cold", "Headache", relation="has_symptom")
        self.graph.add_edge("Sore Throat", "Common Cold", relation="is_symptom_of")
        self.graph.add_edge("Common Cold", "Rest", relation="treatable_by")

        # Pneumonia relations
        self.graph.add_edge("Pneumonia", "Fever", relation="has_symptom")
        self.graph.add_edge("Pneumonia", "Cough", relation="has_symptom")
        self.graph.add_edge("Pneumonia", "Shortness of Breath", relation="has_symptom")
        self.graph.add_edge("Chest X-ray", "Pneumonia", relation="diagnoses")
        self.graph.add_edge("Pneumonia", "Antibiotics", relation="treatable_by")

        # Appendicitis relations
        self.graph.add_edge("Appendicitis", "Abdominal Pain", relation="has_symptom")
        self.graph.add_edge("Appendicitis", "Nausea", relation="has_symptom")
        self.graph.add_edge("CT Scan", "Appendicitis", relation="diagnoses")

        # General relations
        self.graph.add_edge("Paracetamol", "Fever", relation="alleviates")
        self.graph.add_edge("Antibiotics", "Pneumonia", relation="treats")

class PlanningModule:
    def generate_plan(self, patient_symptoms: list) -> list:
        plan = []
        if "Fever" in patient_symptoms or "Cough" in patient_symptoms:
            plan.extend(["is_symptom_of", "diagnoses", "treatable_by"])
        if "Abdominal Pain" in patient_symptoms:
            plan.extend(["is_symptom_of", "diagnoses"])
        return list(set(plan))

class RetrievalModule:
    def __init__(self, kg: MedicalKnowledgeGraph):
        self.kg = kg

    def retrieve_reasoning_paths(self, plan: list, query_entities: list) -> list:
        retrieved_paths = []
        for entity in query_entities:
            if entity not in self.kg.graph:
                continue

            for path in nx.all_simple_paths(self.kg.graph, source=entity, cutoff=3):
                current_path_relations = []
                valid_path = True
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge_data = self.kg.graph.get_edge_data(u, v)
                    if edge_data and "relation" in edge_data:
                        current_path_relations.append(edge_data["relation"])
                    else:
                        valid_path = False
                        break
                
                if valid_path and any(rel in plan for rel in current_path_relations):
                    path_with_relations = []
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        edge_data = self.kg.graph.get_edge_data(u, v)
                        path_with_relations.append((u, edge_data["relation"], v))
                    retrieved_paths.append(path_with_relations)
        return retrieved_paths

class ReasoningModule:
    def conduct_reasoning(self, patient_data: dict, retrieved_paths: list) -> (str, str):
        diagnosis = "Uncertain Diagnosis"
        explanation_parts = []

        symptoms_present = patient_data.get("symptoms", [])

        potential_diseases = {}

        for path in retrieved_paths:
            for i in range(len(path) - 1):
                u, relation, v = path[i]

                if relation == "is_symptom_of" and u in symptoms_present:
                    if self.kg.graph.nodes[v]["type"] == "disease":
                        potential_diseases[v] = potential_diseases.get(v, 0) + 1
                        explanation_parts.append(f"Patient symptom '{u}' is a symptom of '{v}'.")
                elif relation == "diagnoses":
                    if self.kg.graph.nodes[v]["type"] == "disease":
                        potential_diseases[v] = potential_diseases.get(v, 0) + 2 # Stronger evidence
                        explanation_parts.append(f"A test associated with '{v}' could be relevant.")

        if potential_diseases:
            # Simple scoring for diagnosis
            most_likely_disease = max(potential_diseases, key=potential_diseases.get)
            diagnosis = f"Most likely: {most_likely_disease}"
            
            explanation_parts.append(f"Based on the medical knowledge graph, '{most_likely_disease}' is indicated by the presence of matching symptoms.")

            # Add treatment suggestions if available
            for u, _, v in self.kg.graph.edges(most_likely_disease, data="relation"):
                if _ == "treatable_by":
                    explanation_parts.append(f"Treatment consideration: '{v}'.")

        if not explanation_parts:
            explanation = "No specific reasoning paths found for the provided symptoms in the knowledge graph."
        else:
            explanation = "\n".join(explanation_parts)

        return diagnosis, explanation

class ClinicalDiagnosticAssistant:
    def __init__(self):
        self.kg = MedicalKnowledgeGraph()
        self.planning_module = PlanningModule()
        self.retrieval_module = RetrievalModule(self.kg)
        self.reasoning_module = ReasoningModule()

    def diagnose(self, patient_data: dict):
        symptoms = patient_data.get("symptoms", [])

        plan = self.planning_module.generate_plan(symptoms)
        print(f"Generated Plan: {plan}")

        retrieved_paths = self.retrieval_module.retrieve_reasoning_paths(plan, symptoms)
        print(f"Retrieved Paths: {retrieved_paths}")

        diagnosis, explanation = self.reasoning_module.conduct_reasoning(patient_data, retrieved_paths)

        return diagnosis, explanation

if __name__ == "__main__":
    assistant = ClinicalDiagnosticAssistant()

    patient_case_1 = {"symptoms": ["Fever", "Cough", "Body Aches"]}
    diag_1, expl_1 = assistant.diagnose(patient_case_1)
    print("\n--- Patient Case 1 ---")
    print(f"Diagnosis: {diag_1}")
    print(f"Explanation:\n{expl_1}")

    patient_case_2 = {"symptoms": ["Abdominal Pain", "Nausea"]}
    diag_2, expl_2 = assistant.diagnose(patient_case_2)
    print("\n--- Patient Case 2 ---")
    print(f"Diagnosis: {diag_2}")
    print(f"Explanation:\n{expl_2}")

    patient_case_3 = {"symptoms": ["Sore Throat", "Headache"]}
    diag_3, expl_3 = assistant.diagnose(patient_case_3)
    print("\n--- Patient Case 3 ---")
    print(f"Diagnosis: {diag_3}")
    print(f"Explanation:\n{expl_3}")

    patient_case_4 = {"symptoms": ["Fatigue", "Chills"]}
    diag_4, expl_4 = assistant.diagnose(patient_case_4)
    print("\n--- Patient Case 4 ---")
    print(f"Diagnosis: {diag_4}")
    print(f"Explanation:\n{expl_4}")
