import networkx as nx
import random
import streamlit as st

class LLMWrapper:
    def __init__(self):
        pass

    def prune_relations(self, current_path_relations, discovered_relations):
        pruned = []
        for rel_type, target_entity in discovered_relations:
            if "causes" in rel_type or "leads_to" in rel_type or "is_symptom_of" in rel_type or "is_complication_of" in rel_type or "can_lead_to" in rel_type or "can_cause" in rel_type:
                pruned.append((rel_type, target_entity))
            elif "associated_with" in rel_type:
                if random.random() < 0.7:
                    pruned.append((rel_type, target_entity))
        return pruned

def initialize_medical_kg():
    kg = nx.DiGraph()
    entities = [
        "Fever", "Cough", "Headache", "Fatigue", "Sore Throat", "Muscle Ache",
        "Rash", "Shortness of Breath", "Chest Pain", "Nausea", "Vomiting",
        "Diarrhea", "Abdominal Pain", "Joint Pain", "Swelling", "Weight Loss",
        "Influenza", "Common Cold", "Pneumonia", "COVID-19", "Bronchitis",
        "Strep Throat", "Allergies", "Arthritis", "Food Poisoning", "Appendicitis",
        "Migraine", "Asthma", "Heart Attack", "Gastritis", "Dengue Fever",
        "Viral Infection", "Bacterial Infection", "Inflammation", "Autoimmune Disease",
        "Surgery"
    ]
    kg.add_nodes_from(entities)

    relations = [
        ("Fever", "causes", "Fatigue"),
        ("Fever", "associated_with", "Headache"),
        ("Cough", "causes", "Sore Throat"),
        ("Cough", "associated_with", "Chest Pain"),
        ("Headache", "is_symptom_of", "Migraine"),
        ("Fatigue", "is_symptom_of", "Influenza"),
        ("Sore Throat", "is_symptom_of", "Strep Throat"),
        ("Muscle Ache", "is_symptom_of", "Influenza"),
        ("Rash", "is_symptom_of", "Dengue Fever"),
        ("Shortness of Breath", "is_symptom_of", "Pneumonia"),
        ("Chest Pain", "is_symptom_of", "Heart Attack"),
        ("Nausea", "is_symptom_of", "Food Poisoning"),
        ("Vomiting", "is_symptom_of", "Food Poisoning"),
        ("Diarrhea", "is_symptom_of", "Food Poisoning"),
        ("Abdominal Pain", "is_symptom_of", "Appendicitis"),
        ("Joint Pain", "is_symptom_of", "Arthritis"),

        ("Influenza", "causes", "Fever"),
        ("Influenza", "causes", "Cough"),
        ("Influenza", "causes", "Fatigue"),
        ("Influenza", "causes", "Muscle Ache"),

        ("Common Cold", "causes", "Cough"),
        ("Common Cold", "causes", "Sore Throat"),

        ("Pneumonia", "causes", "Fever"),
        ("Pneumonia", "causes", "Shortness of Breath"),
        ("Pneumonia", "is_complication_of", "Bronchitis"),

        ("COVID-19", "causes", "Fever"),
        ("COVID-19", "causes", "Cough"),
        ("COVID-19", "causes", "Shortness of Breath"),
        ("COVID-19", "can_lead_to", "Pneumonia"),

        ("Bronchitis", "causes", "Cough"),
        ("Bronchitis", "associated_with", "Chest Pain"),

        ("Strep Throat", "causes", "Sore Throat"),
        ("Strep Throat", "causes", "Fever"),

        ("Allergies", "causes", "Cough"),
        ("Allergies", "causes", "Sore Throat"),

        ("Arthritis", "causes", "Joint Pain"),
        ("Arthritis", "causes", "Inflammation"),

        ("Food Poisoning", "causes", "Nausea"),
        ("Food Poisoning", "causes", "Vomiting"),
        ("Food Poisoning", "causes", "Diarrhea"),
        ("Food Poisoning", "causes", "Abdominal Pain"),

        ("Appendicitis", "causes", "Abdominal Pain"),
        ("Appendicitis", "requires", "Surgery"),

        ("Migraine", "causes", "Headache"),
        ("Migraine", "associated_with", "Nausea"),

        ("Asthma", "causes", "Shortness of Breath"),
        ("Asthma", "aggravated_by", "Allergies"),

        ("Heart Attack", "causes", "Chest Pain"),
        ("Heart Attack", "causes", "Shortness of Breath"),

        ("Gastritis", "causes", "Abdominal Pain"),
        ("Gastritis", "causes", "Nausea"),

        ("Dengue Fever", "causes", "Fever"),
        ("Dengue Fever", "causes", "Rash"),
        ("Dengue Fever", "causes", "Muscle Ache"),

        ("Viral Infection", "causes", "Fever"),
        ("Viral Infection", "causes", "Fatigue"),
        ("Bacterial Infection", "causes", "Fever"),
        ("Inflammation", "is_part_of", "Autoimmune Disease"),
        ("Autoimmune Disease", "can_cause", "Joint Pain"),
        ("Autoimmune Disease", "can_cause", "Fatigue"),
    ]

    for s, r, t in relations:
        kg.add_edge(s, t, relation=r)
    return kg

def get_outgoing_relations(kg, entity):
    if entity not in kg:
        return []
    outgoing = []
    for successor in kg.successors(entity):
        relation_type = kg[entity][successor]['relation']
        outgoing.append((relation_type, successor))
    return outgoing

class ToGREngine:
    def __init__(self, kg, llm_wrapper, max_depth=3, num_entity_samples=2, max_paths_to_keep=5):
        self.kg = kg
        self.llm_wrapper = llm_wrapper
        self.max_depth = max_depth
        self.num_entity_samples = num_entity_samples
        self.max_paths_to_keep = max_paths_to_keep

    def RelationSearch(self, current_entity):
        return get_outgoing_relations(self.kg, current_entity)

    def LLM_based_RelationPruning(self, current_path_relations, discovered_relations):
        return self.llm_wrapper.prune_relations(current_path_relations, discovered_relations)

    def RandomPrune_EntityPruning(self, candidate_entities, num_to_keep):
        if len(candidate_entities) <= num_to_keep:
            return candidate_entities
        return random.sample(candidate_entities, num_to_keep)

    def explore(self, initial_symptoms_str):
        initial_symptoms_list = [s.strip() for s in initial_symptoms_str.split(',') if s.strip()]
        
        active_paths = [] # Stores (current_entity, full_path_list)
        for symptom in initial_symptoms_list:
            if symptom in self.kg.nodes:
                active_paths.append((symptom, [(None, "initial_symptom", symptom)]))
        
        final_paths = []

        for _ in range(self.max_depth):
            new_active_paths = []
            for current_entity, current_path in active_paths:
                if not current_entity:
                    continue

                discovered_relations_targets = self.RelationSearch(current_entity)
                
                current_path_relations_only = [p[1] for p in current_path]
                pruned_relations_targets = self.LLM_based_RelationPruning(
                    current_path_relations_only,
                    discovered_relations_targets
                )
                
                if not pruned_relations_targets:
                    final_paths.append(current_path)
                    continue

                candidate_entities_for_sampling = list(set([target for _, target in pruned_relations_targets]))
                
                next_entities_after_random_prune = self.RandomPrune_EntityPruning(
                    candidate_entities_for_sampling,
                    self.num_entity_samples
                )

                for rel_type, target_entity in pruned_relations_targets:
                    if target_entity in next_entities_after_random_prune:
                        new_path = current_path + [(current_entity, rel_type, target_entity)]
                        new_active_paths.append((target_entity, new_path))
            
            if len(new_active_paths) > self.max_paths_to_keep:
                active_paths = random.sample(new_active_paths, self.max_paths_to_keep)
            else:
                active_paths = new_active_paths
            
            if not active_paths:
                break

        final_paths.extend([path for _, path in active_paths])

        unique_paths = []
        seen_path_tuples = set()
        for path in final_paths:
            path_tuple = tuple(path)
            if path_tuple not in seen_path_tuples:
                unique_paths.append(path)
                seen_path_tuples.add(path_tuple)

        return unique_paths

# Streamlit User Interface
st.title("Relation-Based Differential Diagnosis Assistant")
st.write("Enter symptoms (comma-separated) to get potential diagnostic paths.")

user_symptoms = st.text_input("Symptoms:", "Fever, Cough")

if st.button("Diagnose"):
    if not user_symptoms:
        st.warning("Please enter at least one symptom.")
    else:
        kg = initialize_medical_kg()
        llm_wrapper = LLMWrapper()
        tog_engine = ToGREngine(kg, llm_wrapper, max_depth=4, num_entity_samples=2, max_paths_to_keep=10)
        
        st.info("Exploring potential diagnostic paths...")
        diagnostic_paths = tog_engine.explore(user_symptoms)
        
        if diagnostic_paths:
            st.subheader("Potential Diagnostic Paths:")
            for i, path in enumerate(diagnostic_paths):
                st.write(f"**Path {i+1}:**")
                path_str = []
                for j, (source, relation, target) in enumerate(path):
                    if source is None:
                        path_str.append(f"{target} (initial symptom)")
                    else:
                        path_str.append(f" -[{relation}]-> {target}")
                st.markdown("".join(path_str))
        else:
            st.write("No diagnostic paths found for the given symptoms.")