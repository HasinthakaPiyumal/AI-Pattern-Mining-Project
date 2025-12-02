import collections

class MedKG:
    def __init__(self):
        self.graph = collections.defaultdict(lambda: collections.defaultdict(set))

    def add_entity(self, entity_type, entity_name):
        # In this simplified model, entities are added implicitly via relationships
        pass

    def add_relationship(self, entity1, relation, entity2):
        self.graph[entity1][relation].add(entity2)
        # For some relations, add a reverse relation for easier traversal
        if relation == "symptom_of":
            self.graph[entity2]["has_symptom"].add(entity1)
        elif relation == "treats":
            self.graph[entity2]["treated_by"].add(entity1)
        elif relation == "causes":
            self.graph[entity2]["caused_by"].add(entity1)
        elif relation == "interacts_with":
            self.graph[entity2]["interacts_with"].add(entity1)

    def get_neighbors(self, entity, relation=None):
        if relation:
            return list(self.graph[entity][relation])
        else:
            neighbors = set()
            for rel_type in self.graph[entity]:
                neighbors.update(self.graph[entity][rel_type])
            return list(neighbors)

    def get_relationships(self, entity, relation_type):
        return list(self.graph[entity].get(relation_type, set()))

class LLMAgent:
    def __init__(self, med_kg):
        self.med_kg = med_kg
        self.reasoning_path = []
        self.identified_entities = set()
        self.diagnoses = set()
        self.recommendations = set()

    def _identify_initial_entities(self, patient_input):
        keywords = []
        # Simulate keyword extraction from natural language input
        # In a real system, this would involve NLP models (e.g., spaCy, NLTK)
        if "headache" in patient_input.lower():
            keywords.add("Headache")
        if "fever" in patient_input.lower():
            keywords.add("Fever")
        if "stiff neck" in patient_input.lower():
            keywords.add("Stiff Neck")
        if "nausea" in patient_input.lower():
            keywords.add("Nausea")
        if "vomiting" in patient_input.lower():
            keywords.add("Vomiting")
        if "diabetes" in patient_input.lower():
            keywords.add("Diabetes")
        if "meningitis" in patient_input.lower(): # For direct disease mention
            keywords.add("Meningitis")
        if "influenza" in patient_input.lower():
            keywords.add("Influenza")
        if "antibiotics" in patient_input.lower():
            keywords.add("Antibiotics")
        if "paracetamol" in patient_input.lower():
            keywords.add("Paracetamol")
        if "recent travel" in patient_input.lower() or "endemic area" in patient_input.lower():
            keywords.add("Travel History") # Placeholder for contextual info

        self.identified_entities.update(keywords)
        self.reasoning_path.append(f"Initial observation: Patient reports symptoms including {', '.join(keywords)}.")
        return list(keywords)

    def explore_kg(self, patient_input):
        self.reasoning_path = []
        self.identified_entities = set()
        self.diagnoses = set()
        self.recommendations = set()

        initial_entities = self._identify_initial_entities(patient_input)

        current_entities_to_explore = collections.deque(initial_entities)
        explored_entities = set()

        while current_entities_to_explore:
            entity = current_entities_to_explore.popleft()
            if entity in explored_entities:
                continue
            explored_entities.add(entity)
            self.reasoning_path.append(f"Exploring entity: {entity}")

            # Step 1: Find diseases related to symptoms
            if entity in {"Headache", "Fever", "Stiff Neck", "Nausea", "Vomiting"}:
                diseases = self.med_kg.get_relationships(entity, "symptom_of")
                if diseases:
                    self.reasoning_path.append(f"Found diseases associated with {entity}: {', '.join(diseases)}.")
                    for disease in diseases:
                        self.diagnoses.add(disease)
                        current_entities_to_explore.append(disease) # Add disease for further exploration

            # Step 2: Explore identified diseases for more symptoms or causes
            if entity in self.diagnoses or entity in {"Meningitis", "Influenza"}:
                symptoms = self.med_kg.get_relationships(entity, "has_symptom")
                if symptoms:
                    self.reasoning_path.append(f"Symptoms of {entity}: {', '.join(symptoms)}.")
                    for symptom in symptoms:
                        if symptom not in self.identified_entities:
                            current_entities_to_explore.append(symptom)
                causes = self.med_kg.get_relationships(entity, "caused_by")
                if causes:
                    self.reasoning_path.append(f"Possible causes of {entity}: {', '.join(causes)}.")
                    for cause in causes:
                        current_entities_to_explore.append(cause)

                # Step 3: Find treatments for identified diseases
                treatments = self.med_kg.get_relationships(entity, "treated_by")
                if treatments:
                    self.reasoning_path.append(f"Potential treatments for {entity}: {', '.join(treatments)}.")
                    self.recommendations.update(treatments)
                    for treatment in treatments:
                        current_entities_to_explore.append(treatment)

            # Step 4: Check drug interactions (simplified)
            if entity in {"Antibiotics", "Paracetamol"}: # Example drugs
                interactions = self.med_kg.get_relationships(entity, "interacts_with")
                if interactions:
                    self.reasoning_path.append(f"Warning: {entity} interacts with: {', '.join(interactions)}.")

            # Simulate contextual reasoning for travel history
            if "Travel History" in self.identified_entities and "Meningitis" in self.diagnoses:
                self.reasoning_path.append("Considering recent travel history, specific types of Meningitis (e.g., bacterial strains common in certain regions) might be relevant. Further diagnostics are crucial.")

        return list(self.diagnoses), list(self.recommendations)

    def generate_explanation(self):
        explanation = "\nMedAdvisor Reasoning Process:\n"
        for step in self.reasoning_path:
            explanation += f"- {step}\n"
        
        if self.diagnoses:
            explanation += "\nFinal Diagnostic Hypotheses: " + ", ".join(self.diagnoses) + "\n"
        if self.recommendations:
            explanation += "Treatment Recommendations: " + ", ".join(self.recommendations) + "\n"

        return explanation

# Main application entry point
if __name__ == "__main__":
    med_kg = MedKG()

    # Populate the Medical Knowledge Graph with sample data
    med_kg.add_relationship("Headache", "symptom_of", "Meningitis")
    med_kg.add_relationship("Fever", "symptom_of", "Meningitis")
    med_kg.add_relationship("Stiff Neck", "symptom_of", "Meningitis")
    med_kg.add_relationship("Nausea", "symptom_of", "Meningitis")
    med_kg.add_relationship("Vomiting", "symptom_of", "Meningitis")
    med_kg.add_relationship("Headache", "symptom_of", "Influenza")
    med_kg.add_relationship("Fever", "symptom_of", "Influenza")
    med_kg.add_relationship("Sore Throat", "symptom_of", "Influenza")

    med_kg.add_relationship("Meningitis", "treated_by", "Antibiotics")
    med_kg.add_relationship("Influenza", "treated_by", "Antivirals")
    med_kg.add_relationship("Influenza", "treated_by", "Paracetamol")
    med_kg.add_relationship("Fever", "treated_by", "Paracetamol")

    med_kg.add_relationship("Antibiotics", "interacts_with", "Alcohol") # Example interaction

    med_kg.add_relationship("Diabetes", "has_symptom", "Increased Thirst")
    med_kg.add_relationship("Diabetes", "has_symptom", "Frequent Urination")
    med_kg.add_relationship("Diabetes", "treated_by", "Insulin")

    llm_agent = LLMAgent(med_kg)

    print("Welcome to MedAdvisor - AI-Powered Medical Diagnostic and Recommendation System")
    print("----------------------------------------------------------------------")

    while True:
        patient_input = input("\nEnter patient symptoms and history (e.g., 'severe headache, fever, stiff neck, recent travel'): ")
        if patient_input.lower() == 'exit':
            break

        diagnoses, recommendations = llm_agent.explore_kg(patient_input)
        explanation = llm_agent.generate_explanation()

        print("\n--- MedAdvisor Results ---")
        print(explanation)
        print("--------------------------")

    print("Thank you for using MedAdvisor. Goodbye!")
