import networkx as nx
from transformers import pipeline

class KGManager:
    def __init__(self):
        self.kg = nx.DiGraph()

    def add_entity(self, entity_id, entity_type, attributes=None):
        if attributes is None:
            attributes = {}
        self.kg.add_node(entity_id, type=entity_type, **attributes)

    def add_relation(self, source_id, target_id, relation_type, attributes=None):
        if attributes is None:
            attributes = {}
        self.kg.add_edge(source_id, target_id, type=relation_type, **attributes)

    def get_neighbors(self, entity_id, relation_type=None):
        neighbors = []
        if entity_id in self.kg:
            for neighbor in self.kg.neighbors(entity_id):
                edge_data = self.kg.get_edge_data(entity_id, neighbor)
                if relation_type is None or edge_data.get('type') == relation_type:
                    neighbors.append((neighbor, edge_data.get('type')))
        return neighbors

    def find_paths(self, source_id, target_type, max_length=3):
        paths = []
        for path in nx.all_simple_paths(self.kg, source_id, target=None, cutoff=max_length):
            if self.kg.nodes[path[-1]].get('type') == target_type:
                paths.append(path)
        return paths

    def get_node_attributes(self, entity_id):
        return self.kg.nodes.get(entity_id, {})

class LLMNavigator:
    def __init__(self, kg_manager):
        self.kg_manager = kg_manager
        self.llm_pipeline = self._load_simulated_llm()

    def _load_simulated_llm(self):
        def simulated_llm(prompt_text):
            if "generate a plan" in prompt_text.lower():
                if "cancer" in prompt_text.lower() and "trial" in prompt_text.lower():
                    return "PLAN: Find all 'Trial' entities related to 'Cancer' through 'targets_disease' or 'studies_disease' relations."
                return "PLAN: Generic search based on keywords."
            elif "reason on path" in prompt_text.lower():
                if "patient" in prompt_text.lower() and "disease" in prompt_text.lower() and "trial" in prompt_text.lower():
                    return "REASONING: Based on the path provided, Patient {patient_id} with {disease} might be suitable for {trial_id} due to its focus on {disease}. Further analysis of inclusion criteria is recommended."
                return "REASONING: Unable to provide detailed reasoning for this path."
            return "LLM Response: I am a simulated LLM."
        return simulated_llm

    def generate_plan(self, patient_profile):
        prompt = f"Given a patient with profile: {patient_profile}, generate a plan to find suitable clinical trials in the Knowledge Graph."
        llm_response = self.llm_pipeline(prompt)
        plan = llm_response.replace("PLAN: ", "")
        return plan

    def execute_plan(self, plan_str):
        if "Find all 'Trial' entities related to 'Cancer'" in plan_str:
            cancer_trials = []
            for node_id, data in self.kg_manager.kg.nodes(data=True):
                if data.get('type') == 'Disease' and 'Cancer' in node_id:
                    for neighbor, rel_type in self.kg_manager.get_neighbors(node_id):
                        if rel_type in ['targets_disease', 'studies_disease'] and self.kg_manager.get_node_attributes(neighbor).get('type') == 'Trial':
                            cancer_trials.append(neighbor)
            return list(set(cancer_trials))
        return []

    def reason_on_path(self, patient_id, disease, trial_id, kg_path):
        prompt = f"Reason on the following KG path for Patient {patient_id} with Disease {disease} considering Trial {trial_id}: {kg_path}"
        llm_response = self.llm_pipeline(prompt)
        reasoning = llm_response.replace("REASONING: ", "")
        reasoning = reasoning.replace("{patient_id}", patient_id)
        reasoning = reasoning.replace("{disease}", disease)
        reasoning = reasoning.replace("{trial_id}", trial_id)
        return reasoning

if __name__ == "__main__":
    kg_manager = KGManager()

    kg_manager.add_entity("Patient_001", "Patient", {"age": 65, "gender": "male"})
    kg_manager.add_entity("Disease_Cancer", "Disease", {"name": "Cancer"})
    kg_manager.add_entity("Symptom_Fatigue", "Symptom", {"name": "Fatigue"})
    kg_manager.add_entity("Drug_ChemoA", "Treatment", {"name": "Chemotherapy A"})
    kg_manager.add_entity("Trial_CT001", "Trial", {"name": "Cancer Study Phase 2", "status": "Recruiting", "criteria": "Adults with advanced cancer"})
    kg_manager.add_entity("Trial_CT002", "Trial", {"name": "Fatigue Reduction Trial", "status": "Open"})
    kg_manager.add_entity("Disease_Diabetes", "Disease", {"name": "Diabetes"})

    kg_manager.add_relation("Patient_001", "Disease_Cancer", "has_disease")
    kg_manager.add_relation("Patient_001", "Symptom_Fatigue", "has_symptom")
    kg_manager.add_relation("Disease_Cancer", "Symptom_Fatigue", "manifests_as")
    kg_manager.add_relation("Trial_CT001", "Disease_Cancer", "targets_disease")
    kg_manager.add_relation("Trial_CT001", "Drug_ChemoA", "uses_treatment")
    kg_manager.add_relation("Trial_CT002", "Symptom_Fatigue", "targets_symptom")
    kg_manager.add_relation("Disease_Diabetes", "Trial_CT001", "excluded_from")

    llm_navigator = LLMNavigator(kg_manager)

    patient_profile_1 = {"id": "Patient_001", "diseases": ["Cancer"], "symptoms": ["Fatigue"]}
    patient_profile_2 = {"id": "Patient_002", "diseases": ["Diabetes"], "symptoms": ["Thirst"]}

    print("\n--- Scenario 1: Patient with Cancer ---")
    print(f"Patient Profile: {patient_profile_1}")
    
    # Planning Optimization Simulation
    plan = llm_navigator.generate_plan(patient_profile_1)
    print(f"LLM Generated Plan: {plan}")

    # Execute plan on KG
    potential_trials = llm_navigator.execute_plan(plan)
    print(f"Potential Trials (from KG based on plan): {potential_trials}")

    if potential_trials:
        for trial_id in potential_trials:
            # Simulate a KG path found
            simulated_kg_path = [
                "Patient_001", "has_disease", "Disease_Cancer",
                "Trial_CT001", "targets_disease", "Disease_Cancer",
                "Trial_CT001", "status", "Recruiting"
            ]
            # Retrieval-Reasoning Optimization Simulation
            reasoning = llm_navigator.reason_on_path(patient_profile_1['id'], "Cancer", trial_id, simulated_kg_path)
            print(f"LLM Reasoning for {trial_id}: {reasoning}")
    else:
        print("No relevant trials found based on the plan.")

    print("\n--- Scenario 2: Patient with Diabetes (no direct cancer trial match) ---")
    print(f"Patient Profile: {patient_profile_2}")

    plan_2 = llm_navigator.generate_plan(patient_profile_2)
    print(f"LLM Generated Plan: {plan_2}")

    potential_trials_2 = llm_navigator.execute_plan(plan_2)
    print(f"Potential Trials (from KG based on plan): {potential_trials_2}")

    if not potential_trials_2:
        print("No specific trials found for Diabetes in this simplified KG for cancer-related plans.")
        simulated_kg_path_no_match = [
            "Patient_002", "has_disease", "Disease_Diabetes",
            "Disease_Diabetes", "excluded_from", "Trial_CT001"
        ]
        reasoning_no_match = llm_navigator.reason_on_path(patient_profile_2['id'], "Diabetes", "N/A", simulated_kg_path_no_match)
        print(f"LLM Reasoning for no match: {reasoning_no_match}")
