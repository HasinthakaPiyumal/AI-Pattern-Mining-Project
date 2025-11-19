
class MedicalKnowledgeGraph:
    def __init__(self):
        # A simplified medical knowledge graph using a dictionary structure
        # In a real application, this would be a dedicated KG database (e.g., Neo4j, RDF store)
        self.graph = {
            "Fever": {"is_symptom_of": ["Flu", "Common Cold", "Pneumonia", "COVID-19"]},
            "Cough": {"is_symptom_of": ["Flu", "Common Cold", "Pneumonia", "Bronchitis", "COVID-19"]},
            "Sore Throat": {"is_symptom_of": ["Flu", "Common Cold", "Strep Throat", "COVID-19"]},
            "Fatigue": {"is_symptom_of": ["Flu", "Common Cold", "Pneumonia", "COVID-19", "Anemia"]},
            "Headache": {"is_symptom_of": ["Flu", "Common Cold", "Migraine"]},
            "Shortness of Breath": {"is_symptom_of": ["Pneumonia", "Asthma", "COVID-19"]},
            "Muscle Aches": {"is_symptom_of": ["Flu", "COVID-19"]},

            "Flu": {"has_symptom": ["Fever", "Cough", "Sore Throat", "Fatigue", "Headache", "Muscle Aches"],
                    "has_treatment": ["Antivirals", "Rest", "Fluids"]},
            "Common Cold": {"has_symptom": ["Fever", "Cough", "Sore Throat", "Fatigue", "Headache"],
                            "has_treatment": ["Rest", "Fluids", "Decongestants"]},
            "Pneumonia": {"has_symptom": ["Fever", "Cough", "Shortness of Breath", "Fatigue"],
                          "has_treatment": ["Antibiotics", "Oxygen Therapy"]},
            "Strep Throat": {"has_symptom": ["Sore Throat", "Fever"],
                             "has_treatment": ["Antibiotics"]},
            "COVID-19": {"has_symptom": ["Fever", "Cough", "Sore Throat", "Fatigue", "Shortness of Breath", "Muscle Aches"],
                         "has_treatment": ["Supportive Care", "Antivirals (severe cases)"]},
            "Bronchitis": {"has_symptom": ["Cough", "Fatigue"],
                           "has_treatment": ["Cough Suppressants", "Rest"]},
            "Asthma": {"has_symptom": ["Shortness of Breath", "Cough"],
                       "has_treatment": ["Bronchodilators", "Steroids"]},
            "Migraine": {"has_symptom": ["Headache", "Nausea"],
                         "has_treatment": ["Pain Relievers", "Triptans"]},
            "Anemia": {"has_symptom": ["Fatigue", "Shortness of Breath", "Pale Skin"],
                       "has_treatment": ["Iron Supplements", "Dietary Changes"]},
        }

    def get_related_facts(self, entity, relation=None):
        """
        Retrieves facts related to an entity. If a relation is specified, filters by that relation.
        Returns a list of (subject, predicate, object) triples.
        """
        facts = []
        if entity in self.graph:
            for pred, objs in self.graph[entity].items():
                if relation is None or relation == pred:
                    for obj in objs:
                        facts.append((entity, pred, obj))

        # Also check for relations where the entity is an object
        for sub, predicates in self.graph.items():
            for pred, objs in predicates.items():
                if entity in objs:
                    if relation is None or relation == pred:
                        facts.append((sub, pred, entity))
        return facts

    def find_paths(self, start_entity, end_entity, max_depth=2, current_path=None):
        """
        Finds paths between two entities in the knowledge graph using a depth-first search.
        Returns a list of paths, where each path is a list of (subject, predicate, object) triples.
        """
        if current_path is None:
            current_path = []

        all_paths = []
        visited_nodes = {triple[0] for triple in current_path} | {triple[2] for triple in current_path}
        if not current_path:
            visited_nodes.add(start_entity)

        # Base case: if start_entity is the end_entity (or directly related within one step)
        direct_facts = self.get_related_facts(start_entity)
        for fact in direct_facts:
            if fact[2] == end_entity:
                all_paths.append(current_path + [fact])
            elif fact[0] == end_entity and not current_path:
                 # If start_entity is related TO end_entity (e.g., Fever -> is_symptom_of -> Flu)
                 # and this is the initial call, consider this a path.
                 all_paths.append(current_path + [fact])

        if max_depth <= 0:
            return all_paths

        # Recursive step
        for fact in direct_facts:
            next_entity = fact[2]
            if next_entity not in visited_nodes:
                new_path = current_path + [fact]
                paths_from_next = self.find_paths(next_entity, end_entity, max_depth - 1, new_path)
                all_paths.extend(paths_from_next)

        return all_paths

    def query_kg(self, query_type, entities, relations=None):
        """
        Simulates querying the KG based on a structured query.
        query_type: e.g., "symptoms_of", "causes_of", "treatments_for", "related_facts"
        entities: list of entities (e.g., ["Fever", "Cough"])
        relations: optional list of specific relations to look for
        """
        results = []
        if query_type == "related_facts":
            for entity in entities:
                results.extend(self.get_related_facts(entity, relations[0] if relations else None))
        elif query_type == "symptoms_of":
            for entity in entities:
                results.extend(self.get_related_facts(entity, "has_symptom"))
        elif query_type == "treatments_for":
            for entity in entities:
                results.extend(self.get_related_facts(entity, "has_treatment"))
        elif query_type == "find_diseases_by_symptoms":
            # Find diseases that have ALL specified symptoms
            potential_diseases = set()
            if not entities:
                return []

            # Initialize with diseases for the first symptom
            first_symptom_diseases = {obj for s, p, obj in self.get_related_facts(entities[0], "is_symptom_of")}
            potential_diseases.update(first_symptom_diseases)

            for symptom in entities[1:]:
                diseases_for_symptom = {obj for s, p, obj in self.get_related_facts(symptom, "is_symptom_of")}
                potential_diseases.intersection_update(diseases_for_symptom)

            for disease in potential_diseases:
                results.append((disease, "is_diagnosed_by_symptoms", ", ".join(entities)))
        elif query_type == "paths_between":
            if len(entities) == 2:
                results.extend(self.find_paths(entities[0], entities[1]))
        # Add more query types as needed
        return results

    def get_all_entities(self):
        """
        Returns all entities present in the KG.
        """
        entities = set(self.graph.keys())
        for entity_data in self.graph.values():
            for _, objects in entity_data.items():
                entities.update(objects)
        return list(entities)

