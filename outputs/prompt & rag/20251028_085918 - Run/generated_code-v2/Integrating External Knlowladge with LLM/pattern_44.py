
import networkx as nx

class MedicalKnowledgeGraph:
    """Simulates a medical knowledge graph for demonstration purposes."""
    def __init__(self):
        self.graph = nx.DiGraph()
        self._populate_graph()

    def _populate_graph(self):
        # Entities: Patients, Symptoms, Diseases, Drugs
        self.graph.add_nodes_from([
            "John Doe", "Jane Smith", "Fever", "Cough", "Headache",
            "Flu", "Common Cold", "Migraine", "Tylenol", "Amoxicillin",
            "COVID-19", "Shortness of Breath"
        ])

        # Relations
        self.graph.add_edge("John Doe", "Fever", relation="has_symptom")
        self.graph.add_edge("John Doe", "Cough", relation="has_symptom")
        self.graph.add_edge("Jane Smith", "Headache", relation="has_symptom")
        self.graph.add_edge("Jane Smith", "Shortness of Breath", relation="has_symptom")

        self.graph.add_edge("Fever", "Flu", relation="associated_with")
        self.graph.add_edge("Cough", "Flu", relation="associated_with")
        self.graph.add_edge("Fever", "Common Cold", relation="associated_with")
        self.graph.add_edge("Cough", "Common Cold", relation="associated_with")
        self.graph.add_edge("Headache", "Migraine", relation="associated_with")
        self.graph.add_edge("Shortness of Breath", "COVID-19", relation="associated_with")

        self.graph.add_edge("Flu", "Tylenol", relation="treated_by")
        self.graph.add_edge("Common Cold", "Tylenol", relation="treated_by")
        self.graph.add_edge("Migraine", "Tylenol", relation="treated_by")

        # Example of drug interaction (simplified)
        self.graph.add_edge("Tylenol", "Alcohol", relation="interacts_with")

    def get_neighbors(self, entity, relation=None):
        """Get entities connected to a given entity via a specific relation."""
        results = []
        for u, v, data in self.graph.edges(data=True):
            if u == entity and (relation is None or data.get("relation") == relation):
                results.append((data.get("relation"), v))
        return results

    def find_paths(self, start_entity, end_entity_type, path_plan):
        """Find reasoning paths (instances) based on a path plan.
        path_plan is a list of (relation, target_entity_type).
        """
        # This is a simplified path finder. A real implementation would be more robust.
        current_paths = [[start_entity]]
        final_reasoning_paths = []

        for i, (relation, target_type) in enumerate(path_plan):
            next_paths = []
            for path in current_paths:
                last_entity = path[-1]
                neighbors = self.get_neighbors(last_entity, relation)
                for rel, neighbor in neighbors:
                    # In a real KG, we'd check if 'neighbor' is of 'target_type'
                    # For this simple demo, we just add the neighbor.
                    new_path = path + [(rel, neighbor)]
                    if i == len(path_plan) - 1: # If it's the last step in the plan
                        final_reasoning_paths.append(new_path)
                    else:
                        next_paths.append(new_path)
            current_paths = next_paths
            if not current_paths and i < len(path_plan) -1: # No paths found for intermediate step
                break
        return final_reasoning_paths


class LLMService:
    """Mocks an LLM service for generating plans and reasoning."""
    def generate_plan(self, query, patient_info):
        print(f"LLM: Generating plan for query: '{query}' with patient: {patient_info}")
        # In a real scenario, the LLM would analyze the query and KG schema
        # to output structured relation paths. Here, we hardcode for a demo.
        if "diagnoses" in query and "Fever" in patient_info and "Cough" in patient_info:
            return [
                ("has_symptom", "Symptom"),
                ("associated_with", "Disease"),
            ]
        elif "treatment" in query and "Migraine" in patient_info:
             return [
                ("has_symptom", "Headache"), # Assuming headache is a proxy for Migraine symptom here
                ("associated_with", "Migraine"),
                ("treated_by", "Drug"),
            ]
        return [] # Default empty plan

    def reason_and_answer(self, query, retrieved_paths):
        print(f"LLM: Reasoning based on query: '{query}' and paths: {retrieved_paths}")
        # The LLM would synthesize the retrieved information into an answer and explanation.
        if "diagnoses" in query and retrieved_paths:
            diagnosis = "Unknown"
            evidence = []
            for path in retrieved_paths:
                if len(path) >= 3 and path[-2][0] == "associated_with":
                    diagnosis = path[-1][1] # Extract the disease
                    evidence.append(f"{path[0]} has {path[1][1]} which is associated with {diagnosis}")
            return {
                "answer": f"Potential diagnosis: {diagnosis}.",
                "explanation": f"Based on the medical knowledge graph, we found evidence: {' '.join(evidence)}"
            }
        elif "treatment" in query and retrieved_paths:
            treatment = "Unknown"
            for path in retrieved_paths:
                if len(path) >= 3 and path[-2][0] == "treated_by":
                    treatment = path[-1][1] # Extract the drug
            return {
                "answer": f"Recommended treatment: {treatment}.",
                "explanation": f"Treatment derived from KG paths linking symptoms to disease and then to treatment."
            }

        return {"answer": "Could not provide a definitive answer.", "explanation": "No relevant reasoning paths found."}


class ClinicalDecisionSupportSystem:
    """Integrates LLM and KG for clinical decision support using RoG framework."""
    def __init__(self):
        self.kg = MedicalKnowledgeGraph()
        self.llm = LLMService()

    def run_query(self, patient_name, patient_symptoms, query):
        print(f"\n--- Processing Query for {patient_name} ---")
        patient_info = {"name": patient_name, "symptoms": patient_symptoms}

        # 1. Planning Module
        print("\n[1. Planning Module]")
        # LLM generates a plan (sequence of relation paths) grounded by KG structure
        # For this demo, the LLM will 'know' general patterns like symptom->disease->treatment
        plan = self.llm.generate_plan(query, patient_symptoms)
        if not plan:
            print("No relevant plan could be generated by the LLM.")
            return self.llm.reason_and_answer(query, []) # Fallback for no plan
        print(f"Generated Plan: {plan}")

        # 2. Retrieval Module
        print("\n[2. Retrieval Module]")
        # Use the plan to retrieve specific reasoning paths from the KG
        # We need a starting point for retrieval, which is the patient.
        # This step is simplified to find paths starting from patient's symptoms.
        all_reasoning_paths = []
        for symptom in patient_symptoms:
            # For each symptom, try to find paths as per the plan
            # This path_plan in find_paths should be relative to the current entity (symptom)
            # We'll adjust the plan for the find_paths to start from a symptom
            # Assuming plan expects start -> (rel, type) -> (rel, type)..
            # We will search for (symptom) -> (relation, target_type) -> ...

            # The current mock `find_paths` starts with an entity and takes a plan like `[(relation, target_type)]`.
            # Let's adapt the plan for the `find_paths` from the symptom itself.
            adapted_plan_for_symptom = []
            # Skip the first "has_symptom" if the plan already starts that way
            if plan and plan[0][0] == "has_symptom":
                adapted_plan_for_symptom = plan[1:]
            else:
                adapted_plan_for_symptom = plan # Use the whole plan if it doesn't start with has_symptom

            retrieved_paths_for_symptom = self.kg.find_paths(symptom, None, adapted_plan_for_symptom)
            # Prefix the patient and symptom to these paths for full context
            if retrieved_paths_for_symptom:
                for path in retrieved_paths_for_symptom:
                    all_reasoning_paths.append([patient_name, ("has_symptom", symptom)] + path)
            
        if not all_reasoning_paths:
            print("No valid reasoning paths retrieved from the KG.")
        else:
            print(f"Retrieved Reasoning Paths (simplified): {all_reasoning_paths}")

        # 3. Reasoning Module
        print("\n[3. Reasoning Module]")
        # LLM synthesizes retrieved paths into an answer and explanation
        result = self.llm.reason_and_answer(query, all_reasoning_paths)
        print("\n--- Result ---")
        print(f"Answer: {result['answer']}")
        print(f"Explanation: {result['explanation']}")
        return result


if __name__ == "__main__":
    cdss = ClinicalDecisionSupportSystem()

    # Example 1: Diagnosis query
    cdss.run_query(
        patient_name="John Doe",
        patient_symptoms=["Fever", "Cough"],
        query="What are the potential diagnoses for John Doe given his symptoms?"
    )

    # Example 2: Treatment query
    cdss.run_query(
        patient_name="Jane Smith",
        patient_symptoms=["Headache"],
        query="What is a recommended treatment for Jane Smith's condition?"
    )

    # Example 3: Query with no matching plan
    cdss.run_query(
        patient_name="John Doe",
        patient_symptoms=["Rash"],
        query="What causes this rash?"
    )

    # Example 4: Diagnosis query for a different disease
    cdss.run_query(
        patient_name="Jane Smith",
        patient_symptoms=["Shortness of Breath"],
        query="What are the potential diagnoses for Jane Smith given her symptoms?"
    )

