class MedicalKnowledgeGraph:
    def __init__(self):
        # A simplified KG: {entity: {relation: [related_entities]}}
        self.graph = {
            "Fever": {"symptom_of": ["Malaria", "Dengue", "COVID-19", "Influenza"], "related_symptom": ["Fatigue", "Cough"]},
            "Dry Cough": {"symptom_of": ["COVID-19", "Influenza"], "related_symptom": ["Fatigue"]},
            "Fatigue": {"symptom_of": ["Malaria", "Dengue", "COVID-19", "Influenza", "Anemia"], "related_symptom": ["Fever", "Cough"]},
            "Recent travel to Amazon basin": {"risk_factor_for": ["Malaria", "Dengue"]},
            "Malaria": {"has_symptom": ["Fever", "Fatigue"], "treatment": ["Chloroquine"], "diagnosis_method": ["Blood smear"]},
            "Dengue": {"has_symptom": ["Fever", "Fatigue"], "treatment": ["Supportive care"], "diagnosis_method": ["Serology"]},
            "COVID-19": {"has_symptom": ["Fever", "Dry Cough", "Fatigue"], "treatment": ["Antivirals", "Supportive care"], "diagnosis_method": ["PCR test"]},
            "Influenza": {"has_symptom": ["Fever", "Dry Cough", "Fatigue"], "treatment": ["Antivirals"], "diagnosis_method": ["Rapid Flu Test"]},
            "Chloroquine": {"treats": ["Malaria"], "category": ["Antimalarial"]},
            "Supportive care": {"treats": ["Dengue", "COVID-19"], "category": ["General"]},
            "Antivirals": {"treats": ["COVID-19", "Influenza"], "category": ["Antiviral"]},
            "Blood smear": {"diagnoses": ["Malaria"]},
            "Serology": {"diagnoses": ["Dengue"]},
            "PCR test": {"diagnoses": ["COVID-19"]},
            "Rapid Flu Test": {"diagnoses": ["Influenza"]},
        }

    def get_neighbors(self, entity):
        """Retrieves all connected entities and their relations for a given entity."""
        neighbors = []
        if entity in self.graph:
            for relation, targets in self.graph[entity].items():
                for target in targets:
                    neighbors.append({"source": entity, "relation": relation, "target": target})

        # Also check if entity is a target for any relation (reverse lookup)
        for source, relations in self.graph.items():
            for relation, targets in relations.items():
                if entity in targets:
                    # Ensure we don't duplicate if already added as a source-based neighbor
                    # This part could be optimized for a real KG, but for mock, it's fine.
                    if not any(n['source'] == source and n['relation'] == relation and n['target'] == entity for n in neighbors):
                        neighbors.append({"source": source, "relation": relation, "target": entity})
        return neighbors
