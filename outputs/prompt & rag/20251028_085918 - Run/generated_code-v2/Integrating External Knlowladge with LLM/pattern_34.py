
import streamlit as st
import networkx as nx
import random

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._load_sample_data()

    def _load_sample_data(self):
        # Add entities (nodes)
        self.graph.add_nodes_from([
            "Fever", "Headache", "Cough", "Fatigue", # Symptoms
            "Influenza", "Common Cold", "Migraine", "Pneumonia", # Diseases
            "Rest", "Fluids", "Pain Relievers", "Antibiotics", "Antivirals" # Treatments/Medications
        ])

        # Add relations (edges)
        self.graph.add_edges_from([
            ("Fever", "indicates", "Influenza", {"type": "indicates"}),
            ("Fever", "indicates", "Common Cold", {"type": "indicates"}),
            ("Fever", "indicates", "Pneumonia", {"type": "indicates"}),
            ("Headache", "indicates", "Influenza", {"type": "indicates"}),
            ("Headache", "indicates", "Migraine", {"type": "indicates"}),
            ("Cough", "indicates", "Common Cold", {"type": "indicates"}),
            ("Cough", "indicates", "Influenza", {"type": "indicates"}),
            ("Cough", "indicates", "Pneumonia", {"type": "indicates"}),
            ("Fatigue", "indicates", "Influenza", {"type": "indicates"}),
            ("Fatigue", "indicates", "Common Cold", {"type": "indicates"}),

            ("Influenza", "treated_by", "Rest", {"type": "treated_by"}),
            ("Influenza", "treated_by", "Fluids", {"type": "treated_by"}),
            ("Influenza", "treated_by", "Pain Relievers", {"type": "treated_by"}),
            ("Influenza", "treated_by", "Antivirals", {"type": "treated_by"}),

            ("Common Cold", "treated_by", "Rest", {"type": "treated_by"}),
            ("Common Cold", "treated_by", "Fluids", {"type": "treated_by"}),
            ("Common Cold", "treated_by", "Pain Relievers", {"type": "treated_by"}),

            ("Migraine", "treated_by", "Pain Relievers", {"type": "treated_by"}),
            ("Migraine", "treated_by", "Rest", {"type": "treated_by"}),

            ("Pneumonia", "treated_by", "Antibiotics", {"type": "treated_by"}),
            ("Pneumonia", "treated_by", "Rest", {"type": "treated_by"}),
            ("Pneumonia", "complication_of", "Influenza", {"type": "complication_of"}), # Example of a different relation type
        ])

    def get_outgoing_relations(self, entity):
        """Returns a list of (relation_type, target_entity) tuples for a given entity."""
        if entity not in self.graph:
            return []
        relations = []
        for u, v, data in self.graph.out_edges(entity, data=True):
            relations.append((data['type'], v))
        return relations


class LLMRelationPruner:
    def llm_prune_relations(self, current_path, available_relations, initial_symptoms):
        """
        Mock LLM-based relation pruning. In a real scenario, an LLM would analyze the
        current path, available relations, and symptoms to select the most relevant ones.
        For this mock, it prioritizes 'indicates' relations for initial symptoms
        and 'treated_by' for diseases, otherwise keeps all.
        """
        pruned_relations = []
        current_entity = current_path[-1] if current_path else None

        # Simple heuristic for pruning based on relation type and path context
        if not current_path and initial_symptoms: # If starting, prioritize 'indicates'
            for rel_type, target_entity in available_relations:
                if rel_type == "indicates":
                    pruned_relations.append((rel_type, target_entity))
            if not pruned_relations: # If no 'indicates', take all
                pruned_relations = list(available_relations)
        elif any(isinstance(node, str) and node in MedicalKnowledgeGraph().graph.nodes and "Disease" in node for node in current_path): # If a disease is in path, prioritize 'treated_by'
             for rel_type, target_entity in available_relations:
                if rel_type == "treated_by":
                    pruned_relations.append((rel_type, target_entity))
             if not pruned_relations: # If no 'treated_by', take all
                pruned_relations = list(available_relations)
        else: # Default: keep all for simplicity in mock
            pruned_relations = list(available_relations)

        # Further refine based on initial symptoms for relevance
        symptom_keywords = [s.lower() for s in initial_symptoms]
        filtered_by_symptoms = []
        for rel_type, target_entity in pruned_relations:
            # Simple check: if target entity is a symptom or directly related to a symptom
            # This is a very basic heuristic. A real LLM would be much more sophisticated.
            if any(s_kw in target_entity.lower() for s_kw in symptom_keywords) or \
               (current_entity in initial_symptoms and rel_type == "indicates") or \
               (current_path and len(current_path) == 1 and current_path[0] in initial_symptoms and rel_type == "indicates") or \
               (len(current_path) > 1 and rel_type == "treated_by") : # For diseases, any treatment is relevant
                filtered_by_symptoms.append((rel_type, target_entity))
        
        if not filtered_by_symptoms and pruned_relations: # Fallback if symptom filtering yields nothing
             return pruned_relations
        elif filtered_by_symptoms:
            return filtered_by_symptoms
        else:
            return []


def random_prune_entities(entities, num_to_keep):
    """Randomly samples entities to keep."""
    if len(entities) <= num_to_keep:
        return entities
    return random.sample(entities, num_to_keep)


def diagnose_with_togr(symptoms_input, kg, llm_pruner, max_depth=4, num_paths_per_step=3, num_entities_to_sample=2):
    """
    Performs Relation-Based Reasoning (ToGR) for medical diagnosis.
    Prioritizes relation chains, uses LLM for relation pruning, and random sampling for entity pruning.
    """
    initial_symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]
    if not initial_symptoms:
        return []

    # Initialize paths: Each path is a list of (entity, relation_type, entity, ...)
    # Start with initial symptoms as the first entities in paths
    active_paths = []
    for symptom in initial_symptoms:
        if symptom in kg.graph.nodes:
            active_paths.append([symptom])
    
    if not active_paths:
        return []

    final_paths = []

    for depth in range(max_depth):
        new_active_paths = []
        for current_path in active_paths:
            current_entity = current_path[-1]

            # Step b: Relation Search
            available_relations_targets = kg.get_outgoing_relations(current_entity)
            
            # Group by relation type to apply LLM pruning on relations themselves
            # available_relations will be a list of (relation_type, target_entity)
            
            # Step c: LLM-based Relation Pruning
            # The LLM pruner needs to suggest which *relation types* are promising next steps
            # For the mock, we pass all (relation_type, target_entity) pairs, and it filters.
            pruned_relations_targets = llm_pruner.llm_prune_relations(
                current_path=current_path,
                available_relations=available_relations_targets,
                initial_symptoms=initial_symptoms
            )

            # Group pruned relations by type for easier processing
            relations_by_type = {}
            for rel_type, target_entity in pruned_relations_targets:
                if rel_type not in relations_by_type:
                    relations_by_type[rel_type] = []
                relations_by_type[rel_type].append(target_entity)
            
            # Step d & e: Path Extension and Random Entity Pruning
            for rel_type, possible_next_entities in relations_by_type.items():
                # Apply random entity pruning for each relation type
                sampled_next_entities = random_prune_entities(possible_next_entities, num_entities_to_sample)
                
                for next_entity in sampled_next_entities:
                    new_path = current_path + [rel_type, next_entity]
                    new_active_paths.append(new_path)
        
        # Step f: Path Maintenance - keep a manageable number of paths
        if not new_active_paths:
            break # No new paths to explore
        
        # Sort paths (e.g., by length or some heuristic) and take the top N
        # For simplicity, we just shuffle and take `num_paths_per_step` unique paths
        random.shuffle(new_active_paths)
        active_paths = list(set(tuple(p) for p in new_active_paths))[:num_paths_per_step]
        
        # Add current active paths to final if they are 