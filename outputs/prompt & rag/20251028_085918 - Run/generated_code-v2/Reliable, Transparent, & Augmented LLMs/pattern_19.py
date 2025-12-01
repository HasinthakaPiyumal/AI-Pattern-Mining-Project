"""Python code for a Clinical Diagnostic Assistant with Explainable Reasoning.
This system simulates how an LLM (Language Model) could interact with a Knowledge Graph (KG) to provide diagnoses and explicit reasoning paths, allowing for human review and correction of the underlying knowledge.
"""

# --- 1. Medical Knowledge Graph (Simulated) ---
# In a real-world scenario, this would be a sophisticated graph database (e.g., Neo4j)
# For demonstration, we use a simple dictionary structure where:
# key = entity (e.g., symptom, disease, drug)
# value = dictionary of relations and connected entities
medical_knowledge_graph = {
    "fever": {"causes": "influenza", "is_symptom_of": "common_cold"},
    "cough": {"causes": "influenza", "is_symptom_of": "common_cold", "associated_with": "bronchitis"},
    "sore_throat": {"is_symptom_of": "common_cold", "causes": "strep_throat"},
    "fatigue": {"is_symptom_of": "influenza", "causes": "anemia"},
    "headache": {"is_symptom_of": "influenza", "associated_with": "migraine"},

    "influenza": {"has_symptom": "fever", "has_symptom": "cough", "has_symptom": "fatigue", "treatment": "antivirals"},
    "common_cold": {"has_symptom": "fever", "has_symptom": "cough", "has_symptom": "sore_throat", "treatment": "rest_fluids"},
    "strep_throat": {"has_symptom": "sore_throat", "treatment": "antibiotics"},
    "bronchitis": {"has_symptom": "cough", "treatment": "bronchodilators"},
    "anemia": {"has_symptom": "fatigue", "treatment": "iron_supplements"},
    "migraine": {"has_symptom": "headache", "treatment": "pain_relievers"},

    "antivirals": {"treats": "influenza"},
    "rest_fluids": {"treats": "common_cold"},
    "antibiotics": {"treats": "strep_throat"},
    "bronchodilators": {"treats": "bronchitis"},
    "iron_supplements": {"treats": "anemia"},
    "pain_relievers": {"treats": "migraine"}
}

class ClinicalDiagnosticAssistant:
    """Simulates a diagnostic assistant that provides diagnoses and explainable reasoning paths."""

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def _find_related_triples(self, start_entity, max_depth=2, path=None, visited_nodes=None):
        """Recursively finds related triples in the KG to form a reasoning path."""
        if path is None:
            path = []
        if visited_nodes is None:
            visited_nodes = set()

        if start_entity in visited_nodes or max_depth < 0:
            return []

        visited_nodes.add(start_entity)
        current_entity_path = []

        if start_entity in self.kg:
            for relation, connected_entity in self.kg[start_entity].items():
                triple = (start_entity, relation, connected_entity)
                current_entity_path.append(triple)
                # Explore further from the connected entity
                current_entity_path.extend(self._find_related_triples(
                    connected_entity, max_depth - 1, path, visited_nodes.copy()
                ))
        return current_entity_path

    def diagnose(self, symptoms):
        """Provides a potential diagnosis and its reasoning path based on symptoms.

        Args:
            symptoms (list): A list of patient symptoms (strings).

        Returns:
            tuple: (diagnosis_message: str, reasoning_path: list of (entity, relation, entity) tuples)
        """
        reasoning_path = []
        potential_diagnoses = set()

        print(f"\nProcessing symptoms: {', '.join(symptoms)}")

        # Simulate LLM processing symptoms and querying KG
        for symptom in symptoms:
            print(f"Searching KG for links from symptom: {symptom}")
            path_for_symptom = self._find_related_triples(symptom)
            reasoning_path.extend(path_for_symptom)

            for s, r, e in path_for_symptom:
                # If a symptom 'causes' a disease or 'is_symptom_of' a disease,
                # consider it a potential diagnosis trigger.
                if (r == "causes" or r == "is_symptom_of") and e in self.kg and ("treatment" in self.kg[e] or "has_symptom" in self.kg[e]):
                    potential_diagnoses.add(e)

        # Deduplicate and sort the reasoning path for clarity
        reasoning_path = sorted(list(set(reasoning_path)))

        if not potential_diagnoses:
            return "No specific diagnosis found based on the provided symptoms.", reasoning_path

        diagnosis_message = f"Potential diagnosis: {', '.join(potential_diagnoses)}.\nRecommended treatment(s): "
        treatments = set()
        for diag in potential_diagnoses:
            if diag in self.kg and "treatment" in self.kg[diag]:
                treatments.add(self.kg[diag]["treatment"])
        diagnosis_message += f"{', '.join(treatments) or 'None specified'}."

        return diagnosis_message, reasoning_path

    def correct_knowledge(self, entity, relation, old_value, new_value):
        """Allows an expert to correct or update knowledge in the KG.

        Args:
            entity (str): The subject entity of the triple.
            relation (str): The relation in the triple.
            old_value (str): The current object entity/value.
            new_value (str): The new object entity/value to update to.

        Returns:
            str: A message indicating the result of the correction.
        """
        print(f"\nAttempting to correct knowledge: {entity} {relation} {old_value} -> {new_value}")
        if entity in self.kg and relation in self.kg[entity]:
            if self.kg[entity][relation] == old_value:
                self.kg[entity][relation] = new_value
                return f"SUCCESS: Knowledge corrected: {entity} {relation} {old_value} updated to {new_value}."
            else:
                return f"FAILED: Existing knowledge for {entity} {relation} is {self.kg[entity][relation]}, not {old_value}."
        else:
            # If the specific relation for the entity doesn't exist, we can add it
            if entity not in self.kg:
                self.kg[entity] = {}
            self.kg[entity][relation] = new_value
            return f"SUCCESS: Added new knowledge: {entity} {relation} {new_value}."

# --- Example Usage / Main Application Flow ---
if __name__ == "__main__":
    assistant = ClinicalDiagnosticAssistant(medical_knowledge_graph)

    # --- Scenario 1: Initial Diagnosis ---
    print("### Scenario 1: Initial Diagnosis ###")
    patient_symptoms_1 = ["fever", "cough"]
    diagnosis_1, path_1 = assistant.diagnose(patient_symptoms_1)

    print(diagnosis_1)
    print("\nExplicit Reasoning Path:")
    if path_1:
        for triple in path_1:
            print(f"  - {triple[0]} --({triple[1]})--> {triple[2]}")
    else:
        print("  No explicit reasoning path found.")

    # --- Scenario 2: Identifying and Correcting an Error in Knowledge ---
    print("\n### Scenario 2: Identifying and Correcting an Error ###")
    # Let's assume there's an error: 'fever' is mistakenly linked to 'pneumonia' (not in our KG, but for concept)
    # We'll simulate finding an error in a hypothetical scenario and correcting an existing one.
    print("\nSimulating a hypothetical scenario where an expert identifies an outdated/incorrect link.\n")

    # Example of a perceived error by a clinician: 'fever' shouldn't cause 'influenza' directly, but be a symptom.
    # For this example, let's correct a hypothetical direct 