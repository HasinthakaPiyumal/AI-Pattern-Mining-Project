import networkx as nx

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_types = set()
        self.relation_types = set()

    def add_entity(self, entity_id, entity_type, attributes=None):
        if attributes is None:
            attributes = {}
        self.graph.add_node(entity_id, type=entity_type, **attributes)
        self.entity_types.add(entity_type)

    def add_relation(self, source_id, target_id, relation_type, attributes=None):
        if attributes is None:
            attributes = {}
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            self.graph.add_edge(source_id, target_id, type=relation_type, **attributes)
            self.relation_types.add(relation_type)

    def get_neighbors(self, entity_id, relation_type=None):
        neighbors = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            if relation_type is None or edge_data.get("type") == relation_type:
                neighbors.append((neighbor, edge_data.get("type")))
        return neighbors

    def get_schema(self):
        return {"entity_types": list(self.entity_types), "relation_types": list(self.relation_types)}

class PlanningModule:
    def __init__(self, kg_schema):
        self.kg_schema = kg_schema

    def _simulate_llm_plan(self, patient_query):
        # A very simplified LLM simulation for generating relation paths
        if "fever" in patient_query.lower() and "cough" in patient_query.lower():
            return ["symptom -> causes -> disease -> treated_by -> drug",
                    "symptom -> indicates -> condition -> managed_by -> therapy"]
        elif "headache" in patient_query.lower():
            return ["symptom -> causes -> disease -> prescribed_for -> drug"]
        return ["symptom -> causes -> disease"]

    def generate_relation_paths(self, patient_query):
        # In a real system, this would involve complex LLM prompting and parsing
        proposed_paths = self._simulate_llm_plan(patient_query)
        validated_paths = []
        for path_str in proposed_paths:
            parts = path_str.split(" -> ")
            # Basic validation: check if relation types exist in schema
            is_valid = True
            for i in range(1, len(parts), 2):
                if parts[i] not in self.kg_schema["relation_types"]:
                    is_valid = False
                    break
            if is_valid:
                validated_paths.append(path_str)
        return validated_paths

class RetrievalModule:
    def __init__(self, kg):
        self.kg = kg

    def _find_path_instances(self, start_entity, relation_path_str):
        path_parts = relation_path_str.split(" -> ")
        if not path_parts or len(path_parts) < 2:
            return []

        # Initialize paths with the starting entity
        current_paths = [[(start_entity, self.kg.graph.nodes[start_entity].get("type"))]]
        all_found_paths = []

        # Traverse the graph based on the relation path
        for i in range(1, len(path_parts), 2):
            relation_type = path_parts[i]
            next_entity_type_hint = path_parts[i + 1] if i + 1 < len(path_parts) else None
            
            new_paths = []
            for path in current_paths:
                last_node_id = path[-1][0]
                
                for neighbor_id, edge_type in self.kg.get_neighbors(last_node_id):
                    if edge_type == relation_type:
                        if next_entity_type_hint and self.kg.graph.nodes[neighbor_id].get("type") != next_entity_type_hint:
                            continue # Skip if entity type hint doesn't match
                        new_path = path + [(relation_type, neighbor_id, self.kg.graph.nodes[neighbor_id].get("type"))]
                        new_paths.append(new_path)
            current_paths = new_paths

        # Filter for complete paths matching the full length of the relation path segments
        expected_length = 1 + (len(path_parts) // 2) * 2  # entities + relations
        for path in current_paths:
            formatted_path = []
            for j, item in enumerate(path):
                if j == 0: # First node
                    formatted_path.append(item[0]) # Entity ID
                elif j % 2 == 1: # Relation
                    formatted_path.append(item[0]) # Relation type
                else: # Node after relation
                    formatted_path.append(item[1]) # Entity ID
            all_found_paths.append(formatted_path)

        return all_found_paths

    def retrieve_reasoning_paths(self, proposed_relation_paths, patient_symptoms):
        retrieved_paths = {}
        for path_str in proposed_relation_paths:
            for symptom in patient_symptoms: # Try to ground the path with patient's symptoms
                if self.kg.graph.has_node(symptom):
                    instances = self._find_path_instances(symptom, path_str)
                    if instances:
                        if path_str not in retrieved_paths:
                            retrieved_paths[path_str] = []
                        retrieved_paths[path_str].extend(instances)
        return retrieved_paths

class ReasoningModule:
    def __init__(self):
        pass

    def _simulate_llm_reasoning(self, patient_query, reasoning_paths):
        # A very simplified LLM simulation for reasoning and answer generation
        diagnosis = "Uncertain Diagnosis"
        treatment = "No specific treatment recommended."
        explanation = f"Based on your query: \"{patient_query}\"\n\nRetrieved Reasoning Paths:\n"

        if not reasoning_paths:
            explanation += "No relevant knowledge graph paths were found to support a diagnosis or treatment."
            return diagnosis, treatment, explanation

        # Simple aggregation logic for demonstration
        found_diseases = set()
        found_drugs = set()
        for path_type, paths in reasoning_paths.items():
            explanation += f"- Path Type: {path_type}\n"
            for path in paths:
                explanation += f"  - Path Instance: {" -> ".join(map(str, path))}\n"
                for i, item in enumerate(path):
                    if isinstance(item, str) and self.kg.graph.has_node(item):
                        node_type = self.kg.graph.nodes[item].get("type")
                        if node_type == "disease":
                            found_diseases.add(item)
                        elif node_type == "drug":
                            found_drugs.add(item)

        if found_diseases:
            diagnosis = f"Possible diseases: {", ".join(found_diseases)}"
        if found_drugs:
            treatment = f"Potential treatments (consult a physician): {", ".join(found_drugs)}"

        explanation += f"\nSummary:\nDiagnosis: {diagnosis}\nTreatment: {treatment}"
        return diagnosis, treatment, explanation

    def generate_diagnosis_and_treatment(self, patient_query, retrieved_reasoning_paths):
        self.kg = MedicalKnowledgeGraph() # Placeholder for graph access in reasoning; should ideally be passed or a global singleton if not instatiated this way.
        return self._simulate_llm_reasoning(patient_query, retrieved_reasoning_paths)

class ClinicalDecisionSupportSystem:
    def __init__(self, kg):
        self.kg = kg
        self.planning_module = PlanningModule(self.kg.get_schema())
        self.retrieval_module = RetrievalModule(self.kg)
        self.reasoning_module = ReasoningModule()

    def diagnose_and_recommend(self, patient_query, patient_symptoms):
        print("\n--- Planning Module: Generating Relation Paths ---")
        proposed_relation_paths = self.planning_module.generate_relation_paths(patient_query)
        print(f"Proposed Relation Paths: {proposed_relation_paths}")

        print("\n--- Retrieval Module: Retrieving Reasoning Paths ---")
        concrete_reasoning_paths = self.retrieval_module.retrieve_reasoning_paths(proposed_relation_paths, patient_symptoms)
        print(f"Concrete Reasoning Paths: {concrete_reasoning_paths}")

        print("\n--- Reasoning Module: Generating Diagnosis and Treatment ---")
        diagnosis, treatment, explanation = self.reasoning_module.generate_diagnosis_and_treatment(patient_query, concrete_reasoning_paths)
        print("\n--- Final Recommendation ---")
        print(f"Diagnosis: {diagnosis}")
        print(f"Treatment: {treatment}")
        print(f"\nExplanation:\n{explanation}")
        return diagnosis, treatment, explanation


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize Medical Knowledge Graph
    medical_kg = MedicalKnowledgeGraph()

    # Add entities
    medical_kg.add_entity("Fever", "symptom")
    medical_kg.add_entity("Cough", "symptom")
    medical_kg.add_entity("Headache", "symptom")
    medical_kg.add_entity("Sore Throat", "symptom")
    medical_kg.add_entity("Fatigue", "symptom")

    medical_kg.add_entity("Influenza", "disease", {"severity": "moderate"})
    medical_kg.add_entity("Common Cold", "disease", {"severity": "mild"})
    medical_kg.add_entity("Migraine", "disease", {"severity": "high"})
    medical_kg.add_entity("Strep Throat", "disease", {"severity": "moderate"})

    medical_kg.add_entity("Paracetamol", "drug", {"class": "pain reliever"})
    medical_kg.add_entity("Ibuprofen", "drug", {"class": "NSAID"})
    medical_kg.add_entity("Amoxicillin", "drug", {"class": "antibiotic"})
    medical_kg.add_entity("Rest", "therapy")
    medical_kg.add_entity("Fluids", "therapy")

    # Add relations
    medical_kg.add_relation("Fever", "Influenza", "causes")
    medical_kg.add_relation("Cough", "Influenza", "causes")
    medical_kg.add_relation("Fatigue", "Influenza", "causes")
    medical_kg.add_relation("Influenza", "Paracetamol", "treated_by")
    medical_kg.add_relation("Influenza", "Rest", "treated_by")
    medical_kg.add_relation("Influenza", "Fluids", "treated_by")

    medical_kg.add_relation("Fever", "Common Cold", "causes")
    medical_kg.add_relation("Cough", "Common Cold", "causes")
    medical_kg.add_relation("Common Cold", "Rest", "treated_by")
    medical_kg.add_relation("Common Cold", "Fluids", "treated_by")

    medical_kg.add_relation("Headache", "Migraine", "causes")
    medical_kg.add_relation("Migraine", "Ibuprofen", "prescribed_for")

    medical_kg.add_relation("Sore Throat", "Strep Throat", "indicates")
    medical_kg.add_relation("Strep Throat", "Amoxicillin", "managed_by")
    medical_kg.add_relation("Strep Throat", "Rest", "managed_by")

    # 2. Initialize the CDS System
    cds_system = ClinicalDecisionSupportSystem(medical_kg)

    # 3. Patient Query 1
    print("\n=============== Patient Query 1 ===============")
    patient_query_1 = "I have a fever and a cough. I feel very tired."
    patient_symptoms_1 = ["Fever", "Cough", "Fatigue"]
    cds_system.diagnose_and_recommend(patient_query_1, patient_symptoms_1)

    # 4. Patient Query 2
    print("\n=============== Patient Query 2 ===============")
    patient_query_2 = "I have a bad headache."
    patient_symptoms_2 = ["Headache"]
    cds_system.diagnose_and_recommend(patient_query_2, patient_symptoms_2)

    # 5. Patient Query 3 (no direct match for treatment path)
    print("\n=============== Patient Query 3 ===============")
    patient_query_3 = "I have a sore throat."
    patient_symptoms_3 = ["Sore Throat"]
    cds_system.diagnose_and_recommend(patient_query_3, patient_symptoms_3)

    # 6. Patient Query 4 (unsupported/new symptom)
    print("\n=============== Patient Query 4 ===============")
    patient_query_4 = "I have knee pain and a rash."
    patient_symptoms_4 = ["Knee Pain", "Rash"]
    cds_system.diagnose_and_recommend(patient_query_4, patient_symptoms_4)
