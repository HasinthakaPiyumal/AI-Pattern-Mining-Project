import networkx as nx
import random

class MedicalKG:
    def __init__(self, data=None):
        self.graph = nx.DiGraph()
        if data:
            for entity_id, entity_data in data.get("entities", {}).items():
                self.add_entity(entity_id, entity_data["type"], entity_data.get("attributes"))
            for relation in data.get("relations", []):
                self.add_relation(relation["source"], relation["type"], relation["target"], relation.get("attributes"))

    def add_entity(self, entity_id, entity_type, attributes=None):
        if attributes is None:
            attributes = {}
        attributes["type"] = entity_type
        self.graph.add_node(entity_id, **attributes)

    def add_relation(self, source_id, relation_type, target_id, attributes=None):
        if attributes is None:
            attributes = {}
        self.graph.add_edge(source_id, target_id, relation_type=relation_type, **attributes)

    def get_neighbors(self, entity_id, relation_type=None):
        neighbors = []
        if entity_id not in self.graph:
            return neighbors
        for neighbor_id in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor_id)
            if relation_type is None or edge_data.get("relation_type") == relation_type:
                neighbors.append({
                    "entity": neighbor_id,
                    "type": self.graph.nodes[neighbor_id].get("type"),
                    "relation": edge_data.get("relation_type"),
                    "attributes": self.graph.nodes[neighbor_id],
                    "relation_attributes": edge_data
                })
        return neighbors

    def get_entities_by_type(self, entity_type):
        return [node for node, data in self.graph.nodes(data=True) if data.get("type") == entity_type]

    def search_by_keyword(self, keyword, entity_type=None):
        found_entities = []
        for node_id, data in self.graph.nodes(data=True):
            if keyword.lower() in str(node_id).lower():
                if entity_type is None or data.get("type") == entity_type:
                    found_entities.append(node_id)
            for attr_key, attr_val in data.items():
                if isinstance(attr_val, str) and keyword.lower() in attr_val.lower():
                    if entity_type is None or data.get("type") == entity_type:
                        found_entities.append(node_id)
                        break
        return list(set(found_entities))


class LLMSimulator:
    def __init__(self, kg: MedicalKG):
        self.kg = kg

    def search_kg(self, current_path_context, query_keywords):
        candidate_extensions = []

        # Simulate LLM extracting keywords from query and patient context
        all_relevant_keywords = set(query_keywords)
        for entity in current_path_context:
            if isinstance(entity, dict) and "entity" in entity: # Path contains dicts of entities
                all_relevant_keywords.add(entity["entity"].lower())
            elif isinstance(entity, str): # Path contains just entity names
                all_relevant_keywords.add(entity.lower())

        # Strategy 1: Explore neighbors of the last entity in the current path
        if current_path_context:
            last_entity = current_path_context[-1]
            entity_id_to_explore = last_entity["entity"] if isinstance(last_entity, dict) else last_entity
            neighbors = self.kg.get_neighbors(entity_id_to_explore)
            for neighbor in neighbors:
                score_contribution = 0.5 # Base score
                # Simulate LLM evaluating relevance
                for kw in all_relevant_keywords:
                    if kw in neighbor["entity"].lower() or \
                       (neighbor["type"] and kw in neighbor["type"].lower()) or \
                       (neighbor["relation"] and kw in neighbor["relation"].lower()):
                        score_contribution += 0.3
                candidate_extensions.append((neighbor, neighbor["relation"], score_contribution))

        # Strategy 2: Search for new entities directly related to query keywords (especially for initial or short paths)
        if not current_path_context or len(current_path_context) < 3: # Bias towards direct search for short paths
            for keyword in query_keywords:
                found_by_keyword = self.kg.search_by_keyword(keyword)
                for entity_id in found_by_keyword:
                    # Avoid adding entities already in the immediate path context for simplicity
                    is_already_in_path = False
                    if current_path_context:
                        for path_item in current_path_context:
                            item_id = path_item["entity"] if isinstance(path_item, dict) else path_item
                            if item_id == entity_id:
                                is_already_in_path = True
                                break
                    if not is_already_in_path:
                        score_contribution = 0.7 # Higher base score for direct keyword match
                        entity_data = self.kg.graph.nodes[entity_id]
                        candidate_extensions.append((
                            {"entity": entity_id, "type": entity_data.get("type"), "attributes": entity_data},
                            "keyword_match",
                            score_contribution
                        ))

        return candidate_extensions

    def prune_candidates(self, candidates, question, patient_profile, beam_width):
        # candidates: list of (current_path, new_step_entity_dict, new_step_relation_type, score_contribution)
        # where current_path is a list of (entity_dict, relation_type) tuples
        scored_paths = []

        for path_data in candidates:
            current_path = path_data[0]
            new_step_entity = path_data[1]
            new_step_relation = path_data[2]
            step_score_contribution = path_data[3]

            # Calculate total path score
            current_path_score = 0
            if current_path and current_path[0] and isinstance(current_path[0], tuple): # If path is (entity, relation)
                current_path_score = sum(item[0]["score"] for item in current_path if "score" in item[0])
            elif current_path and current_path[0] and isinstance(current_path[0], dict): # If path is just entity dicts
                 current_path_score = sum(item["score"] for item in current_path if "score" in item)

            # Simulate LLM-based scoring based on semantic relevance
            # This is a heuristic for demonstration purposes
            total_score = current_path_score + step_score_contribution

            # Additional scoring based on question and patient profile (simulated)
            question_lower = question.lower()
            if new_step_entity["type"] == "Disease" and any(kw in question_lower for kw in [new_step_entity["entity"].lower()] + list(patient_profile.values())):
                total_score += 0.4
            if new_step_entity["type"] == "Symptom" and new_step_entity["entity"].lower() in question_lower:
                total_score += 0.2
            if new_step_entity["type"] == "Test" and "test" in question_lower:
                total_score += 0.1

            # Incorporate patient profile (e.g., age, pre-existing conditions)
            if patient_profile.get("age") and new_step_entity.get("min_age") and patient_profile["age"] < new_step_entity["min_age"]:
                total_score -= 0.5 # Penalize if disease is for older age group

            # Create the new extended path, including the score contribution for the new entity
            new_step_with_score = new_step_entity.copy()
            new_step_with_score["score"] = step_score_contribution # Score for this specific step
            extended_path = current_path + [(new_step_with_score, new_step_relation)]

            scored_paths.append((total_score, extended_path))

        # Sort by score in descending order and select top N (beam_width)
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        return scored_paths[:beam_width]

def run_llm_guided_beam_search(
    kg: MedicalKG,
    llm_simulator: LLMSimulator,
    initial_symptoms: list,
    question: str,
    patient_profile: dict,
    beam_width: int = 3,
    max_iterations: int = 5
):
    # Initial beam: Start with paths based on initial symptoms
    initial_paths = []
    for symptom in initial_symptoms:
        # Directly add symptoms as initial path entities. Score them based on direct relevance.
        symptom_entity_data = {"entity": symptom, "type": "Symptom", "attributes": {"name": symptom}, "score": 1.0}
        initial_paths.append((1.0, [symptom_entity_data])) # (score, path)

    current_beam = initial_paths

    print(f"--- Starting Beam Search with Initial Symptoms: {initial_symptoms} ---")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        all_candidate_paths_for_next_beam = []

        for path_score, current_path_entities in current_beam:
            # current_path_entities is a list of (entity_dict, relation_type) for the path
            # For search_kg, we need a list of entity dicts (or just entity names)
            current_path_context_for_llm = []
            for item in current_path_entities:
                if isinstance(item, tuple):
                    current_path_context_for_llm.append(item[0]) # Just the entity dict
                else:
                    current_path_context_for_llm.append(item) # Just the entity dict from initial beam

            # LLM searches for candidates to extend the current path
            query_keywords = initial_symptoms + question.lower().split()
            new_steps_from_search = llm_simulator.search_kg(current_path_context_for_llm, query_keywords)

            for new_entity_data, new_relation_type, step_score_contribution in new_steps_from_search:
                # Path data for prune_candidates: (current_path, new_step_entity_dict, new_step_relation_type, score_contribution)
                # current_path_entities is already in the right format for prune_candidates's first element
                all_candidate_paths_for_next_beam.append(
                    (current_path_entities, new_entity_data, new_relation_type, step_score_contribution)
                )

        if not all_candidate_paths_for_next_beam:
            print("No new candidates found. Ending search.")
            break

        # LLM prunes and ranks candidates to form the next beam
        current_beam = llm_simulator.prune_candidates(
            all_candidate_paths_for_next_beam, question, patient_profile, beam_width
        )

        print(f"Top paths after iteration {iteration + 1}:")
        for score, path in current_beam:
            path_str = " -> ".join([f"({e['entity']} [{e['type']}])" if isinstance(e, dict) else f"({e[0]['entity']} [{e[0]['type']}] by {e[1]})" for e in path])
            print(f"  Score: {score:.2f}, Path: {path_str}")

    print("\n--- Beam Search Finished ---")
    return current_beam


if __name__ == "__main__":
    # 1. Simulate Medical Knowledge Graph Data
    medical_kg_data = {
        "entities": {
            "Fever": {"type": "Symptom"},
            "Cough": {"type": "Symptom"},
            "Headache": {"type": "Symptom"},
            "Fatigue": {"type": "Symptom"},
            "Sore Throat": {"type": "Symptom"},
            "Influenza": {"type": "Disease", "min_age": 0},
            "Common Cold": {"type": "Disease", "min_age": 0},
            "Pneumonia": {"type": "Disease", "min_age": 1},
            "COVID-19": {"type": "Disease", "min_age": 0},
            "Strep Throat": {"type": "Disease", "min_age": 2},
            "Flu Test": {"type": "Test"},
            "COVID Test": {"type": "Test"},
            "Throat Swab": {"type": "Test"},
            "Antivirals": {"type": "Treatment"},
            "Antibiotics": {"type": "Treatment"},
            "Rest": {"type": "Treatment"}
        },
        "relations": [
            {"source": "Fever", "type": "is_symptom_of", "target": "Influenza"},
            {"source": "Cough", "type": "is_symptom_of", "target": "Influenza"},
            {"source": "Headache", "type": "is_symptom_of", "target": "Influenza"},
            {"source": "Fatigue", "type": "is_symptom_of", "target": "Influenza"},
            {"source": "Sore Throat", "type": "is_symptom_of", "target": "Influenza"},

            {"source": "Fever", "type": "is_symptom_of", "target": "Common Cold"},
            {"source": "Cough", "type": "is_symptom_of", "target": "Common Cold"},
            {"source": "Sore Throat", "type": "is_symptom_of", "target": "Common Cold"},

            {"source": "Fever", "type": "is_symptom_of", "target": "Pneumonia"},
            {"source": "Cough", "type": "is_symptom_of", "target": "Pneumonia"},
            {"source": "Fatigue", "type": "is_symptom_of", "target": "Pneumonia"},

            {"source": "Fever", "type": "is_symptom_of", "target": "COVID-19"},
            {"source": "Cough", "type": "is_symptom_of", "target": "COVID-19"},
            {"source": "Fatigue", "type": "is_symptom_of", "target": "COVID-19"},
            {"source": "Sore Throat", "type": "is_symptom_of", "target": "COVID-19"},

            {"source": "Sore Throat", "type": "is_symptom_of", "target": "Strep Throat"},

            {"source": "Influenza", "type": "diagnosed_by", "target": "Flu Test"},
            {"source": "COVID-19", "type": "diagnosed_by", "target": "COVID Test"},
            {"source": "Strep Throat", "type": "diagnosed_by", "target": "Throat Swab"},

            {"source": "Influenza", "type": "treated_by", "target": "Antivirals"},
            {"source": "Influenza", "type": "treated_by", "target": "Rest"},
            {"source": "Common Cold", "type": "treated_by", "target": "Rest"},
            {"source": "Pneumonia", "type": "treated_by", "target": "Antibiotics"},
            {"source": "COVID-19", "type": "treated_by", "target": "Rest"},
            {"source": "Strep Throat", "type": "treated_by", "target": "Antibiotics"}
        ]
    }

    kg = MedicalKG(medical_kg_data)

    # 2. Initialize LLM Simulator
    llm_simulator = LLMSimulator(kg)

    # 3. Define Patient Information and Question
    patient_symptoms = ["Fever", "Cough", "Fatigue"]
    diagnostic_question = "What could be causing my fever, cough, and fatigue? Should I get a test?"
    patient_profile_data = {"age": 35, "gender": "male", "pre_existing": []}

    # 4. Run LLM-Guided Beam Search
    final_diagnosis_paths = run_llm_guided_beam_search(
        kg,
        llm_simulator,
        initial_symptoms=patient_symptoms,
        question=diagnostic_question,
        patient_profile=patient_profile_data,
        beam_width=3,
        max_iterations=4
    )

    print("\n--- Final Top Reasoning Paths ---")
    for score, path in final_diagnosis_paths:
        path_str = " -> ".join([f"({e['entity']} [{e['type']}])" if isinstance(e, dict) else f"({e[0]['entity']} [{e[0]['type']}] by {e[1]})" for e in path])
        print(f"Score: {score:.2f}, Path: {path_str}")