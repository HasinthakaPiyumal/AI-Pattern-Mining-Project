import networkx as nx
import random

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def load_kg_from_dict(self, data):
        for entity, connections in data.items():
            self.graph.add_node(entity)
            for relation, targets in connections.items():
                for target in targets:
                    self.graph.add_node(target)
                    self.graph.add_edge(entity, target, type=relation)

    def get_outgoing_relations(self, entity):
        if entity not in self.graph:
            return []
        return list(set(self.graph[entity][neighbor]["type"] for neighbor in self.graph[entity]))

    def get_connected_entities(self, entity, relation_type):
        if entity not in self.graph:
            return []
        connected = []
        for neighbor in self.graph[entity]:
            if self.graph[entity][neighbor]["type"] == relation_type:
                connected.append(neighbor)
        return connected

class LLMSimulationModule:
    def __init__(self):
        pass

    def prune_relations(self, patient_query, available_relations):
        # Simple keyword matching simulation for LLM pruning
        relevant_relations = []
        query_keywords = set(patient_query.lower().split())

        for relation in available_relations:
            if any(keyword in relation.lower() for keyword in query_keywords) or \
               any(keyword in patient_query.lower() for keyword in relation.lower()):
                relevant_relations.append(relation)
        
        # If no relations match keywords, return a default set or all relations
        if not relevant_relations and available_relations:
            return available_relations # Fallback to all if no direct match
        
        return relevant_relations

class ToGRCoreModule:
    def __init__(self, kg, llm_simulator, max_iterations=5, max_entities_per_step=3):
        self.kg = kg
        self.llm_simulator = llm_simulator
        self.max_iterations = max_iterations
        self.max_entities_per_step = max_entities_per_step

    def diagnose(self, initial_symptoms, patient_history_query):
        active_entities = set(initial_symptoms)
        discovered_relation_chains = []
        
        for _ in range(self.max_iterations):
            if not active_entities:
                break

            next_candidate_entities = set()
            relations_to_explore = set()

            for entity in active_entities:
                relations_to_explore.update(self.kg.get_outgoing_relations(entity))
            
            # LLM-based relation pruning
            pruned_relations = self.llm_simulator.prune_relations(patient_history_query, list(relations_to_explore))
            
            if not pruned_relations:
                break

            current_iteration_chains = []

            for entity in active_entities:
                for relation in pruned_relations:
                    connected_entities = self.kg.get_connected_entities(entity, relation)
                    if connected_entities:
                        for next_entity in connected_entities:
                            current_iteration_chains.append((entity, relation, next_entity))
                            next_candidate_entities.add(next_entity)
            
            # Random sampling (RandomPrune) for entity pruning
            pruned_next_entities = list(next_candidate_entities)
            random.shuffle(pruned_next_entities)
            active_entities = set(pruned_next_entities[:self.max_entities_per_step])
            
            if current_iteration_chains:
                discovered_relation_chains.extend(current_iteration_chains)

        return list(set([entity for chain in discovered_relation_chains for entity in chain if entity not in initial_symptoms])), discovered_relation_chains


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize Knowledge Graph
    kg_data = {
        "fever": {"indicates": ["flu", "common cold", "malaria"]},
        "cough": {"indicates": ["flu", "common cold", "bronchitis"]},
        "headache": {"indicates": ["flu", "migraine"]},
        "flu": {"treated_by": ["antivirals", "rest"], "has_symptom": ["fever", "cough", "headache"]},
        "common cold": {"treated_by": ["rest", "pain relievers"], "has_symptom": ["fever", "cough"]},
        "malaria": {"treated_by": ["antimalarials"], "has_symptom": ["fever"]},
        "migraine": {"treated_by": ["triptans"], "has_symptom": ["headache"]},
        "bronchitis": {"treated_by": ["antibiotics", "rest"], "has_symptom": ["cough"]},
        "antivirals": {"drug_class": ["antiviral_medication"]},
        "antimalarials": {"drug_class": ["antimalarial_medication"]},
        "antibiotics": {"drug_class": ["antibiotic_medication"]},
        "triptans": {"drug_class": ["migraine_medication"]},
        "rest": {"type": ["supportive_care"]},
        "pain relievers": {"type": ["symptomatic_treatment"]}
    }

    kg = KnowledgeGraph()
    kg.load_kg_from_dict(kg_data)

    # 2. Initialize LLM Simulation Module
    llm_simulator = LLMSimulationModule()

    # 3. Initialize ToGR Core Module
    # max_entities_per_step determines the 'random pruning' strength
    tog_reasoner = ToGRCoreModule(kg, llm_simulator, max_iterations=3, max_entities_per_step=2)

    # 4. Run Diagnosis
    initial_symptoms = ["fever", "cough"]
    patient_query = "Patient has severe cough and a persistent fever. Also feels very tired. No recent travel."

    suggested_entities, relation_chains = tog_reasoner.diagnose(initial_symptoms, patient_query)

    print(f"Initial Symptoms: {initial_symptoms}")
    print(f"Patient Query: {patient_query}")
    print("\n--- Suggested Related Entities (Diseases/Treatments) ---")
    print(list(set(suggested_entities))) # Use set to remove duplicates

    print("\n--- Discovered Relation Chains ---")
    for chain in relation_chains:
        print(f"{chain[0]} --[{chain[1]}]--> {chain[2]}")

    print("\n--- Second Scenario ---")
    initial_symptoms_2 = ["headache"]
    patient_query_2 = "Patient has a severe headache, throbbing pain, sensitive to light."
    suggested_entities_2, relation_chains_2 = tog_reasoner.diagnose(initial_symptoms_2, patient_query_2)

    print(f"Initial Symptoms: {initial_symptoms_2}")
    print(f"Patient Query: {patient_query_2}")
    print("\n--- Suggested Related Entities (Diseases/Treatments) ---")
    print(list(set(suggested_entities_2))) # Use set to remove duplicates

    print("\n--- Discovered Relation Chains ---")
    for chain in relation_chains_2:
        print(f"{chain[0]} --[{chain[1]}]--> {chain[2]}")