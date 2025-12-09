
import networkx as nx
import random

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id, entity_type, attributes=None):
        if attributes is None:
            attributes = {}
        self.graph.add_node(entity_id, type=entity_type, **attributes)

    def add_relation(self, source_id, target_id, relation_type, attributes=None):
        if attributes is None:
            attributes = {}
        self.graph.add_edge(source_id, target_id, type=relation_type, **attributes)

    def get_neighbors(self, entity_id, relation_type=None):
        neighbors = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            if relation_type is None or edge_data.get('type') == relation_type:
                neighbors.append((neighbor, edge_data.get('type')))
        return neighbors

    def find_paths(self, start_node, end_node=None, max_length=3, relation_paths_only=False):
        all_paths = []
        for path in nx.all_simple_paths(self.graph, source=start_node, target=end_node, cutoff=max_length) if end_node else nx.all_simple_paths(self.graph, source=start_node, cutoff=max_length):
            if relation_paths_only:
                relation_path = []
                for i in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    relation_path.append(edge_data.get('type'))
                all_paths.append(relation_path)
            else:
                all_paths.append(path)
        return all_paths

class PlanningModule:
    def __init__(self, kg):
        self.kg = kg
        # Simulate LLM for planning - in a real system, this would be an actual LLM call
        self.planning_prompts = {
            "fever and cough": ["symptom_of", "causes", "treats"],
            "chest pain and shortness of breath": ["symptom_of", "associated_with", "risk_factor_for", "treats"],
            "rash and joint pain": ["symptom_of", "diagnosed_by", "contraindicated_with"]
        }

    def generate_relation_paths(self, patient_symptoms):
        print(f"[Planning Module] Patient symptoms: {patient_symptoms}")
        # LLM simulation: generate relation paths (plans) grounded by KG
        # In a real scenario, an LLM would query the KG for possible relations
        # and construct a sequence of relations that might lead to a diagnosis/treatment.
        if patient_symptoms in self.planning_prompts:
            # For demonstration, we use pre-defined paths based on symptoms
            return self.planning_prompts[patient_symptoms]
        else:
            # Fallback to a generic path if symptoms are not specifically mapped
            return ["symptom_of", "causes", "treats"]

class RetrievalModule:
    def __init__(self, kg):
        self.kg = kg

    def retrieve_reasoning_paths(self, start_entity, relation_path_plan, max_retrievals=5):
        print(f"[Retrieval Module] Retrieving paths for plan: {relation_path_plan} starting from {start_entity}")
        retrieved_paths = []
        # In a real system, this would involve sophisticated graph traversal based on the plan
        # For demonstration, we'll do a simplified path search that matches relation types.

        current_paths = [[start_entity]]

        for relation_type in relation_path_plan:
            next_paths = []
            for path in current_paths:
                last_node = path[-1]
                for neighbor, edge_type in self.kg.get_neighbors(last_node):
                    if edge_type == relation_type:
                        new_path = list(path) # Create a copy to avoid modifying original path
                        new_path.append(neighbor)
                        next_paths.append(new_path)
            current_paths = next_paths

        # Filter for complete paths matching the length of the plan and get actual node sequences
        final_reasoning_paths = []
        for path in current_paths:
            if len(path) == len(relation_path_plan) + 1: # +1 because path includes start node
                final_reasoning_paths.append(path)
        
        # Limit the number of retrieved paths for demonstration
        return random.sample(final_reasoning_paths, min(len(final_reasoning_paths), max_retrievals))

class ReasoningModule:
    def __init__(self):
        # Simulate LLM for reasoning
        pass

    def conduct_reasoning(self, patient_context, retrieved_reasoning_paths):
        print(f"[Reasoning Module] Conducting reasoning for patient context: {patient_context}")
        print(f"[Reasoning Module] Based on retrieved paths: {retrieved_reasoning_paths}")

        if not retrieved_reasoning_paths:
            return "No specific reasoning paths found. Further investigation needed.", "No specific explanation available without clear reasoning paths."

        # LLM simulation: process retrieved paths to generate diagnosis/recommendation and explanation
        # In a real system, an LLM would synthesize information from the paths
        # and patient context to formulate a coherent answer.
        
        diagnosis_candidates = set()
        treatment_candidates = set()
        explanations = []

        for path in retrieved_reasoning_paths:
            path_explanation = "Path: "
            for i in range(len(path) - 1):
                source_node = path[i]
                target_node = path[i+1]
                edge_data = self.kg.graph.get_edge_data(source_node, target_node)
                relation_type = edge_data.get('type')
                
                path_explanation += f"({source_node}) -[{relation_type}]-> ({target_node}) "

                # Simple logic to extract potential diagnoses/treatments
                if relation_type == "symptom_of":
                    diagnosis_candidates.add(target_node)
                elif relation_type == "causes":
                    diagnosis_candidates.add(target_node)
                elif relation_type == "treats":
                    treatment_candidates.add(target_node)
            explanations.append(path_explanation.strip())
        
        # Formulate a simulated answer
        diagnosis = ", ".join(list(diagnosis_candidates)) if diagnosis_candidates else "Undetermined"
        treatment = ", ".join(list(treatment_candidates)) if treatment_candidates else "No specific treatment recommended based on paths."

        answer = f"Based on the medical knowledge graph and patient context: Diagnosis candidate(s): {diagnosis}. Potential treatment(s): {treatment}."
        full_explanation = "Reasoning Paths:\n" + "\n".join(explanations)
        
        return answer, full_explanation

class InstructionTuningSimulator:
    def __init__(self):
        pass

    def optimize_modules(self, planning_module, retrieval_module, reasoning_module):
        print("[Instruction Tuning Simulator] Optimizing planning and retrieval-reasoning modules...")
        # In a real scenario, this would involve fine-tuning LLMs with specific instruction sets
        # and feedback loops based on performance on planning and retrieval-reasoning tasks.
        # For this simulation, we acknowledge the step but don't implement complex logic.
        print("[Instruction Tuning Simulator] Optimization complete (simulated).")

class ICDSS_RoGFramework:
    def __init__(self):
        self.kg = MedicalKnowledgeGraph()
        self._initialize_kg()
        self.planning_module = PlanningModule(self.kg)
        self.retrieval_module = RetrievalModule(self.kg)
        self.reasoning_module = ReasoningModule()
        self.instruction_tuning_simulator = InstructionTuningSimulator()

    def _initialize_kg(self):
        # Diseases
        self.kg.add_entity("Influenza", "Disease", {"symptoms_common": ["Fever", "Cough", "Sore Throat"]})
        self.kg.add_entity("Pneumonia", "Disease", {"symptoms_common": ["Cough", "Shortness of Breath", "Chest Pain"]})
        self.kg.add_entity("Hypertension", "Disease", {"symptoms_common": ["Headache", "Dizziness"]})
        self.kg.add_entity("COVID-19", "Disease", {"symptoms_common": ["Fever", "Cough", "Fatigue", "Loss of Taste"]})
        self.kg.add_entity("Rheumatoid_Arthritis", "Disease", {"symptoms_common": ["Joint Pain", "Swelling", "Stiffness"]})
        self.kg.add_entity("Measles", "Disease", {"symptoms_common": ["Fever", "Rash", "Cough"]})

        # Symptoms
        self.kg.add_entity("Fever", "Symptom")
        self.kg.add_entity("Cough", "Symptom")
        self.kg.add_entity("Sore Throat", "Symptom")
        self.kg.add_entity("Shortness of Breath", "Symptom")
        self.kg.add_entity("Chest Pain", "Symptom")
        self.kg.add_entity("Headache", "Symptom")
        self.kg.add_entity("Dizziness", "Symptom")
        self.kg.add_entity("Fatigue", "Symptom")
        self.kg.add_entity("Loss of Taste", "Symptom")
        self.kg.add_entity("Joint Pain", "Symptom")
        self.kg.add_entity("Swelling", "Symptom")
        self.kg.add_entity("Stiffness", "Symptom")
        self.kg.add_entity("Rash", "Symptom")

        # Treatments/Drugs
        self.kg.add_entity("Paracetamol", "Drug")
        self.kg.add_entity("Antibiotics", "Drug")
        self.kg.add_entity("Ventilator_Support", "Treatment")
        self.kg.add_entity("Antivirals", "Drug")
        self.kg.add_entity("Immunosuppressants", "Drug")
        self.kg.add_entity("Vaccine", "Preventative")

        # Relations
        self.kg.add_relation("Fever", "Influenza", "symptom_of")
        self.kg.add_relation("Cough", "Influenza", "symptom_of")
        self.kg.add_relation("Sore Throat", "Influenza", "symptom_of")
        self.kg.add_relation("Influenza", "Paracetamol", "treatable_by")

        self.kg.add_relation("Cough", "Pneumonia", "symptom_of")
        self.kg.add_relation("Shortness of Breath", "Pneumonia", "symptom_of")
        self.kg.add_relation("Chest Pain", "Pneumonia", "symptom_of")
        self.kg.add_relation("Pneumonia", "Antibiotics", "treatable_by")

        self.kg.add_relation("Fever", "COVID-19", "symptom_of")
        self.kg.add_relation("Cough", "COVID-19", "symptom_of")
        self.kg.add_relation("Fatigue", "COVID-19", "symptom_of")
        self.kg.add_relation("Loss of Taste", "COVID-19", "symptom_of")
        self.kg.add_relation("COVID-19", "Antivirals", "treatable_by")
        self.kg.add_relation("COVID-19", "Ventilator_Support", "requires_in_severe_cases")
        self.kg.add_relation("COVID-19", "Vaccine", "preventable_by")

        self.kg.add_relation("Joint Pain", "Rheumatoid_Arthritis", "symptom_of")
        self.kg.add_relation("Swelling", "Rheumatoid_Arthritis", "symptom_of")
        self.kg.add_relation("Stiffness", "Rheumatoid_Arthritis", "symptom_of")
        self.kg.add_relation("Rheumatoid_Arthritis", "Immunosuppressants", "treatable_by")

        self.kg.add_relation("Fever", "Measles", "symptom_of")
        self.kg.add_relation("Rash", "Measles", "symptom_of")
        self.kg.add_relation("Cough", "Measles", "symptom_of")

        self.kg.add_relation("Influenza", "Pneumonia", "can_lead_to") # Complication
        self.kg.add_relation("Measles", "Pneumonia", "can_lead_to") # Complication


    def process_medical_query(self, patient_symptoms):
        print("\n--- Starting ICDSS Process ---")

        # 1. Planning Module
        relation_path_plan = self.planning_module.generate_relation_paths(patient_symptoms)
        print(f"Generated Relation Path Plan: {relation_path_plan}")

        # 2. Retrieval Module
        # For simplicity, we'll assume the starting point for retrieval is one of the patient's symptoms
        # In a real system, the LLM might identify the primary symptom/entity to start the search.
        start_entity_for_retrieval = patient_symptoms.split(' and ')[0] # Take first symptom as starting point
        if not self.kg.graph.has_node(start_entity_for_retrieval):
            print(f"Warning: Starting entity '{start_entity_for_retrieval}' not found in KG. Attempting to find a related symptom in KG.")
            found_start_entity = False
            for symptom in patient_symptoms.split(' and '):
                if self.kg.graph.has_node(symptom):
                    start_entity_for_retrieval = symptom
                    found_start_entity = True
                    break
            if not found_start_entity:
                print("Error: No valid starting symptom found in KG for retrieval.")
                return "Could not process query due to unknown symptoms.", ""


        retrieved_reasoning_paths = self.retrieval_module.retrieve_reasoning_paths(start_entity_for_retrieval, relation_path_plan)
        print(f"Retrieved Reasoning Paths ({len(retrieved_reasoning_paths)}): {retrieved_reasoning_paths}")

        # 3. Reasoning Module
        answer, explanation = self.reasoning_module.conduct_reasoning(patient_symptoms, retrieved_reasoning_paths)
        print(f"\nDiagnosis/Recommendation: {answer}")
        print(f"Explanation:\n{explanation}")

        print("--- ICDSS Process Complete ---")
        return answer, explanation

# --- Main Execution --- #
if __name__ == "__main__":
    icdss = ICDSS_RoGFramework()
    icdss.instruction_tuning_simulator.optimize_modules(icdss.planning_module, icdss.retrieval_module, icdss.reasoning_module)

    print("\n--- Testing with a Complex Medical Query (Fever and Cough) ---")
    query1_answer, query1_explanation = icdss.process_medical_query("Fever and Cough")

    print("\n--- Testing with another query (Chest Pain and Shortness of Breath) ---")
    query2_answer, query2_explanation = icdss.process_medical_query("Chest Pain and Shortness of Breath")

    print("\n--- Testing with another query (Rash and Joint Pain) ---")
    query3_answer, query3_explanation = icdss.process_medical_query("Rash and Joint Pain")

    print("\n--- Testing with an unknown query ---")
    query4_answer, query4_explanation = icdss.process_medical_query("headache and blurry vision")

