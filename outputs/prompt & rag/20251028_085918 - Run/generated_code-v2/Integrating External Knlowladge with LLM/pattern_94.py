
import collections

class MedicalKnowledgeGraph:
    """Simulates a simple medical Knowledge Graph."""
    def __init__(self):
        self.graph = {
            "Fever": {"is_symptom_of": ["Influenza", "Common Cold", "Malaria", "Strep Throat"]},
            "Cough": {"is_symptom_of": ["Influenza", "Common Cold", "Bronchitis"]},
            "Headache": {"is_symptom_of": ["Influenza", "Common Cold", "Migraine"]},
            "Sore Throat": {"is_symptom_of": ["Common Cold", "Strep Throat"]},
            "Muscle Pain": {"is_symptom_of": ["Influenza", "Fibromyalgia"]},
            "Fatigue": {"is_symptom_of": ["Influenza", "Common Cold", "Chronic Fatigue Syndrome"]},
            "Nausea": {"is_symptom_of": ["Migraine", "Food Poisoning"]},
            "Rash": {"is_symptom_of": ["Measles", "Chickenpox"]},
            "Swollen Lymph Nodes": {"is_symptom_of": ["Strep Throat"]},

            "Influenza": {
                "has_symptom": ["Fever", "Cough", "Headache", "Muscle Pain", "Fatigue"],
                "treatment": ["Antivirals", "Rest", "Fluids"],
                "complication": ["Pneumonia", "Bronchitis"]
            },
            "Common Cold": {
                "has_symptom": ["Fever", "Cough", "Headache", "Sore Throat", "Fatigue"],
                "treatment": ["Rest", "Fluids", "Symptom Relief"],
                "complication": []
            },
            "Migraine": {
                "has_symptom": ["Headache", "Nausea", "Sensitivity to Light"],
                "treatment": ["Triptans", "Pain Relievers"],
                "complication": []
            },
            "Strep Throat": {
                "has_symptom": ["Sore Throat", "Fever", "Swollen Lymph Nodes"],
                "treatment": ["Antibiotics"],
                "complication": ["Rheumatic Fever"]
            },
            "Amoxicillin": {
                "treats": ["Strep Throat", "Bacterial Infections"],
                "drug_interaction_with": ["Warfarin"]
            },
            "Warfarin": {
                "drug_interaction_with": ["Amoxicillin", "Aspirin"]
            },
            "Aspirin": {
                "drug_interaction_with": ["Warfarin"]
            }
        }

    def get_related_entities(self, entity, relation_type):
        """Retrieve entities related to a given entity by a specific relation type."""
        return self.graph.get(entity, {}).get(relation_type, [])

    def find_entities_with_relation(self, relation_type, target_entity):
        """Find entities that have a specific relation to a target entity."""
        results = []
        for entity, relations in self.graph.items():
            if target_entity in relations.get(relation_type, []):
                results.append(entity)
        return results


class ClinicalDiagnosticAgent:
    """Simulates an LLM agent tightly coupled with a MedicalKnowledgeGraph.
    It interactively explores the KG to perform diagnostic reasoning.
    """
    def __init__(self, kg: MedicalKnowledgeGraph):
        self.kg = kg
        self.reasoning_steps = []

    def _log_step(self, step_description, kg_query=None, kg_response=None, decision=None):
        """Logs a step in the reasoning process for traceability."""
        log_entry = {
            "step_description": step_description,
            "kg_query": kg_query,
            "kg_response": kg_response,
            "decision": decision
        }
        self.reasoning_steps.append(log_entry)

    def _match_symptoms(self, patient_symptoms):
        """Matches patient symptoms to KG entities and finds potential diseases."""
        possible_diseases = collections.defaultdict(int) # Count how many patient symptoms map to each disease
        self.reasoning_steps = [] # Reset for a new diagnosis

        self._log_step(
            "Initial symptom matching",
            kg_query=f"Finding diseases associated with {patient_symptoms}"
        )

        for symptom in patient_symptoms:
            diseases_for_symptom = self.kg.get_related_entities(symptom, "is_symptom_of")
            if diseases_for_symptom:
                self._log_step(
                    f"Query KG for '{symptom}'",
                    kg_response=f"'{symptom}' is a symptom of: {diseases_for_symptom}"
                )
                for disease in diseases_for_symptom:
                    possible_diseases[disease] += 1
            else:
                self._log_step(
                    f"Query KG for '{symptom}'",
                    kg_response=f"No diseases directly linked to '{symptom}' as a symptom in KG."
                )

        # Filter diseases that match at least one symptom and sort by symptom count
        initial_diagnoses = sorted(
            [(disease, count) for disease, count in possible_diseases.items() if count > 0],
            key=lambda item: item[1], reverse=True
        )
        self._log_step(
            "Generated initial potential diagnoses",
            decision=f"Initial ranked diseases based on matched symptoms: {initial_diagnoses}"
        )
        return initial_diagnoses

    def _refine_diagnoses(self, initial_diagnoses, patient_symptoms, patient_history):
        """Refines proposed diagnoses by checking for additional symptoms, history, etc."""
        refined_diagnoses = []
        for disease, initial_score in initial_diagnoses:
            score = initial_score
            kg_symptoms = self.kg.get_related_entities(disease, "has_symptom")

            self._log_step(
                f"Refining diagnosis for '{disease}'",
                kg_query=f"Getting symptoms for '{disease}'"
            )
            self._log_step(
                f"KG response for '{disease}' symptoms",
                kg_response=f"'{disease}' has symptoms: {kg_symptoms}"
            )

            # Check for symptoms present in patient but not typically for this disease (negative evidence)
            for patient_symptom in patient_symptoms:
                if patient_symptom not in kg_symptoms and patient_symptom in self.kg.graph:
                    # Penalize if a prominent symptom is not associated with this disease in KG
                    score -= 0.5
                    self._log_step(
                        f"Adjusting score for '{disease}'",
                        decision=f"Patient has '{patient_symptom}' which is not a typical symptom of '{disease}'. Score reduced."
                    )

            # Check for symptoms of the disease that are missing in the patient (negative evidence)
            for kg_symptom in kg_symptoms:
                if kg_symptom not in patient_symptoms:
                    # Slightly penalize if key symptom is missing
                    score -= 0.2
                    self._log_step(
                        f"Adjusting score for '{disease}'",
                        decision=f"Patient is missing '{kg_symptom}' which is a typical symptom of '{disease}'. Score reduced."
                    )

            # Incorporate patient history (simple example: if a known condition contradicts/supports)
            if patient_history:
                for condition in patient_history:
                    if condition == disease:
                        score += 2 # Strong support if already diagnosed or strongly indicated
                        self._log_step(
                            f"Adjusting score for '{disease}'",
                            decision=f"'{disease}' is in patient's history. Score increased."
                        )

            refined_diagnoses.append((disease, score))

        refined_diagnoses = sorted(refined_diagnoses, key=lambda item: item[1], reverse=True)
        self._log_step(
            "Final refined diagnoses",
            decision=f"Refined and ranked diseases: {refined_diagnoses}"
        )
        return refined_diagnoses

    def check_complications(self, diagnosis):
        """Checks for potential complications of a given diagnosis."""
        complications = self.kg.get_related_entities(diagnosis, "complication")
        self._log_step(
            f"Checking complications for '{diagnosis}'",
            kg_query=f"Getting complications for '{diagnosis}'",
            kg_response=f"Complications for '{diagnosis}': {complications}"
        )
        return complications

    def check_drug_interactions(self, patient_medications):
        """Checks for drug-drug interactions among patient medications."""
        interactions = []
        for i, drug1 in enumerate(patient_medications):
            related_to_drug1 = self.kg.get_related_entities(drug1, "drug_interaction_with")
            for drug2 in patient_medications[i+1:]:
                if drug2 in related_to_drug1:
                    interactions.append(f"{drug1} and {drug2}")
                    self._log_step(
                        f"Checking drug interaction: {drug1} vs {drug2}",
                        kg_query=f"Checking if '{drug2}' interacts with '{drug1}'",
                        kg_response=f"Interaction found: '{drug1}' interacts with '{drug2}'"
                    )
        return interactions

    def get_treatments(self, diagnosis):
        """Retrieves treatments for a given diagnosis."""
        treatments = self.kg.get_related_entities(diagnosis, "treatment")
        self._log_step(
            f"Getting treatments for '{diagnosis}'",
            kg_query=f"Getting treatments for '{diagnosis}'",
            kg_response=f"Treatments for '{diagnosis}': {treatments}"
        )
        return treatments

    def diagnose_patient(self, patient_data):
        """Main method to perform a diagnostic consultation for a patient."""
        patient_symptoms = patient_data.get("symptoms", [])
        patient_history = patient_data.get("medical_history", [])
        patient_medications = patient_data.get("current_medications", [])

        print("\n--- Starting Diagnostic Process ---")
        print(f"Patient Symptoms: {patient_symptoms}")
        print(f"Patient History: {patient_history}")
        print(f"Patient Medications: {patient_medications}\n")

        # LLM Agent Step 1: Initial symptom matching and hypothesis generation
        initial_diagnoses = self._match_symptoms(patient_symptoms)

        # LLM Agent Step 2: Iterative refinement based on more KG exploration and patient data
        refined_diagnoses = self._refine_diagnoses(initial_diagnoses, patient_symptoms, patient_history)

        # Final top diagnosis (or top few)
        top_diagnosis = refined_diagnoses[0][0] if refined_diagnoses else "Undetermined"

        # LLM Agent Step 3: Explore related knowledge (complications, treatments, drug interactions)
        potential_complications = []
        if top_diagnosis != "Undetermined":
            potential_complications = self.check_complications(top_diagnosis)

        suggested_treatments = []
        if top_diagnosis != "Undetermined":
            suggested_treatments = self.get_treatments(top_diagnosis)

        drug_interactions = self.check_drug_interactions(patient_medications)

        print("\n--- Diagnostic Report ---")
        print(f"Proposed Top Diagnosis: {top_diagnosis} (Confidence Score: {refined_diagnoses[0][1] if refined_diagnoses else 'N/A'})\n")

        print("Differential Diagnoses (Ranked):")
        for diag, score in refined_diagnoses:
            print(f"  - {diag}: {score:.1f}")

        if potential_complications:
            print(f"\nPotential Complications for {top_diagnosis}: {', '.join(potential_complications)}")

        if suggested_treatments:
            print(f"\nSuggested Treatments for {top_diagnosis}: {', '.join(suggested_treatments)}")

        if drug_interactions:
            print(f"\nCritical Drug Interactions: {', '.join(drug_interactions)}")
        else:
            print("\nNo critical drug interactions found.")

        print("\n--- Traceable Reasoning Steps ---")
        for i, step in enumerate(self.reasoning_steps):
            print(f"Step {i+1}: {step['step_description']}")
            if step['kg_query']: print(f"  KG Query: {step['kg_query']}")
            if step['kg_response']: print(f"  KG Response: {step['kg_response']}")
            if step['decision']: print(f"  Agent Decision/Insight: {step['decision']}")

        return {
            "top_diagnosis": top_diagnosis,
            "differential_diagnoses": refined_diagnoses,
            "potential_complications": potential_complications,
            "suggested_treatments": suggested_treatments,
            "drug_interactions": drug_interactions,
            "reasoning_trace": self.reasoning_steps
        }


# Example Usage:
if __name__ == "__main__":
    # Initialize the Knowledge Graph
    medical_kg = MedicalKnowledgeGraph()

    # Initialize the LLM-like Diagnostic Agent with the KG
    agent = ClinicalDiagnosticAgent(medical_kg)

    # Patient Data 1: Common Cold / Flu-like symptoms
    patient_data_1 = {
        "symptoms": ["Fever", "Cough", "Headache", "Sore Throat", "Fatigue"],
        "medical_history": [],
        "current_medications": []
    }
    agent.diagnose_patient(patient_data_1)

    print("\n" + "="*80 + "\n")

    # Patient Data 2: Strep Throat symptoms with medication
    patient_data_2 = {
        "symptoms": ["Sore Throat", "Fever", "Swollen Lymph Nodes"],
        "medical_history": [],
        "current_medications": ["Amoxicillin", "Warfarin"]
    }
    agent.diagnose_patient(patient_data_2)

    print("\n" + "="*80 + "\n")

    # Patient Data 3: Migraine symptoms
    patient_data_3 = {
        "symptoms": ["Headache", "Nausea", "Sensitivity to Light"],
        "medical_history": [],
        "current_medications": []
    }
    agent.diagnose_patient(patient_data_3)

    print("\n" + "="*80 + "\n")

    # Patient Data 4: No clear symptoms (demonstrate lower confidence)
    patient_data_4 = {
        "symptoms": ["Itchy Eye"],
        "medical_history": [],
        "current_medications": []
    }
    agent.diagnose_patient(patient_data_4)
