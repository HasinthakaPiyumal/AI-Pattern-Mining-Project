from pydantic import BaseModel
import spacy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Pydantic Models for Data Structure
class PatientInput(BaseModel):
    patient_history: str

class LLMOutput(BaseModel):
    extracted_symptoms: list[str]
    extracted_medications: list[str]
    extracted_allergies: list[str]
    extracted_demographics: dict

class PGMOutput(BaseModel):
    disease_probabilities: dict[str, float]

class TreatmentPlan(BaseModel):
    recommended_medications: list[str]
    dosage_instructions: dict[str, str]
    contraindications_identified: list[str]
    clinical_guideline_adherence: bool

class SystemOutput(BaseModel):
    diagnosis: dict[str, float]
    treatment_plan: TreatmentPlan
    explanation: str

# Load spaCy model (a small one for demonstration)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'. This may take a moment.")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 2. LLM Component (Simulated with spaCy for NER)
class LLMComponent:
    def process_patient_input(self, patient_history: str) -> LLMOutput:
        doc = nlp(patient_history)
        symptoms = []
        medications = []
        allergies = []
        demographics = {"age": None, "gender": None}

        # Simple keyword extraction for demonstration
        for token in doc:
            if token.text.lower() in ["fever", "cough", "headache", "fatigue", "rash", "nausea"]:
                symptoms.append(token.text.lower())
            if token.text.lower() in ["aspirin", "paracetamol", "amoxicillin"]:
                medications.append(token.text.lower())
            if token.text.lower() in ["penicillin allergy", "nut allergy"]:
                allergies.append(token.text.lower())

        # More sophisticated parsing would be needed for real demographics
        import re
        age_match = re.search(r"age of (\d+)", patient_history, re.IGNORECASE)
        if age_match: demographics["age"] = int(age_match.group(1))
        gender_match = re.search(r"(male|female)", patient_history, re.IGNORECASE)
        if gender_match: demographics["gender"] = gender_match.group(1).lower()

        return LLMOutput(
            extracted_symptoms=list(set(symptoms)),
            extracted_medications=list(set(medications)),
            extracted_allergies=list(set(allergies)),
            extracted_demographics=demographics,
        )

# 3. PGM Component (Simple Bayesian Network Example)
class PGMComponent:
    def __init__(self):
        # Define a simple Bayesian Network for demonstration
        # Nodes: Symptoms (Fever, Cough), Diseases (Flu, Cold)
        self.model = BayesianNetwork([('Flu', 'Fever'), ('Flu', 'Cough'), ('Cold', 'Cough')])

        # Define Conditional Probability Distributions (CPDs)
        cpd_flu = TabularCPD(variable='Flu', variable_card=2, values=[[0.7], [0.3]]) # P(Flu=No), P(Flu=Yes)
        cpd_cold = TabularCPD(variable='Cold', variable_card=2, values=[[0.6], [0.4]]) # P(Cold=No), P(Cold=Yes)

        cpd_fever = TabularCPD(variable='Fever', variable_card=2,
                               values=[[0.9, 0.4], 
                                       [0.1, 0.6]],
                               evidence=['Flu'],
                               evidence_card=[2]) # P(Fever|Flu)
        
        cpd_cough = TabularCPD(variable='Cough', variable_card=2,
                               values=[[0.8, 0.3, 0.7, 0.2], 
                                       [0.2, 0.7, 0.3, 0.8]],
                               evidence=['Flu', 'Cold'],
                               evidence_card=[2, 2]) # P(Cough|Flu, Cold)

        self.model.add_cpds(cpd_flu, cpd_cold, cpd_fever, cpd_cough)
        self.model.check_model()
        self.inference = VariableElimination(self.model)

    def infer_diseases(self, structured_symptoms: list[str]) -> PGMOutput:
        evidence = {}
        if "fever" in structured_symptoms: evidence["Fever"] = 1 # 1 for Yes, 0 for No
        if "cough" in structured_symptoms: evidence["Cough"] = 1

        # If no relevant evidence, return base probabilities
        if not evidence:
            flu_prob = self.inference.query(variables=['Flu']).get_value(Flu=1)
            cold_prob = self.inference.query(variables=['Cold']).get_value(Cold=1)
            return PGMOutput(disease_probabilities={
                "Flu": float(flu_prob),
                "Cold": float(cold_prob)
            })

        query_results = self.inference.query(variables=['Flu', 'Cold'], evidence=evidence)

        flu_prob_yes = query_results['Flu'].get_value(Flu=1)
        cold_prob_yes = query_results['Cold'].get_value(Cold=1)

        return PGMOutput(disease_probabilities={
            "Flu": float(flu_prob_yes),
            "Cold": float(cold_prob_yes)
        })

# 4. Symbolic Logic Component (Rule-Based System)
class SymbolicLogicComponent:
    def __init__(self):
        self.drug_interaction_rules = {
            "aspirin": {"contraindications": ["bleeding disorder", "ulcer"], "interactions": {"paracetamol": "caution"}},
            "amoxicillin": {"allergies": ["penicillin allergy"]}
        }
        self.clinical_guidelines = {
            "Flu": {"recommended_meds": ["paracetamol"], "avoid_meds": ["aspirin"]},
            "Cold": {"recommended_meds": ["paracetamol"]}
        }

    def validate_treatment_plan(self,
                                  disease_probabilities: dict[str, float],
                                  extracted_medications: list[str],
                                  extracted_allergies: list[str],
                                  proposed_medications: list[str]) -> TreatmentPlan:
        
        contraindications = []
        adherence = True
        recommended_meds_final = list(proposed_medications)
        dosage_instructions = {med: "Standard dosage, consult physician" for med in proposed_medications}

        # Check for allergies
        for allergy in extracted_allergies:
            for medication in proposed_medications:
                if allergy.lower() in self.drug_interaction_rules.get(medication.lower(), {}).get("allergies", []):
                    contraindications.append(f"Allergy to {allergy} for {medication}")
                    adherence = False

        # Check drug interactions and general contraindications
        for medication in proposed_medications:
            med_info = self.drug_interaction_rules.get(medication.lower(), {})
            for contra_condition in med_info.get("contraindications", []):
                # This is a simplified check, ideally links to patient's current conditions
                if contra_condition in " ".join(extracted_allergies): # Simulating a broad check
                    contraindications.append(f"{medication} contraindicated due to {contra_condition}")
                    adherence = False

            for other_med, interaction_type in med_info.get("interactions", {}).items():
                if other_med in proposed_medications:
                    contraindications.append(f"Interaction: {medication} and {other_med} ({interaction_type})")
                    adherence = False
        
        # Check clinical guidelines based on most probable disease
        most_probable_disease = max(disease_probabilities, key=disease_probabilities.get) if disease_probabilities else None

        if most_probable_disease:
            guideline = self.clinical_guidelines.get(most_probable_disease)
            if guideline:
                for recommended_med in guideline.get("recommended_meds", []):
                    if recommended_med not in recommended_meds_final: 
                        recommended_meds_final.append(recommended_med)
                for avoid_med in guideline.get("avoid_meds", []):
                    if avoid_med in recommended_meds_final:
                        contraindications.append(f"Clinical guideline for {most_probable_disease} recommends avoiding {avoid_med}")
                        recommended_meds_final.remove(avoid_med)
                        adherence = False

        return TreatmentPlan(
            recommended_medications=list(set(recommended_meds_final)),
            dosage_instructions=dosage_instructions,
            contraindications_identified=list(set(contraindications)),
            clinical_guideline_adherence=adherence,
        )

# 5. Integration and Orchestration (Central System)
class MedicalDiagnosisSystem:
    def __init__(self):
        self.llm_component = LLMComponent()
        self.pgm_component = PGMComponent()
        self.symbolic_logic_component = SymbolicLogicComponent()

    def diagnose_and_plan(self, patient_input: PatientInput) -> SystemOutput:
        # Step 1: LLM processes patient input
        llm_output = self.llm_component.process_patient_input(patient_input.patient_history)
        print(f"LLM Output: {llm_output.model_dump()}")

        # Step 2: PGM infers disease probabilities
        pgm_output = self.pgm_component.infer_diseases(llm_output.extracted_symptoms)
        print(f"PGM Output: {pgm_output.model_dump()}")

        # Step 3: Symbolic Logic validates treatment plan
        # For demonstration, propose some default meds or based on detected symptoms
        proposed_meds = []
        if "fever" in llm_output.extracted_symptoms: proposed_meds.append("paracetamol")
        if "headache" in llm_output.extracted_symptoms: proposed_meds.append("aspirin")

        treatment_plan = self.symbolic_logic_component.validate_treatment_plan(
            disease_probabilities=pgm_output.disease_probabilities,
            extracted_medications=llm_output.extracted_medications,
            extracted_allergies=llm_output.extracted_allergies,
            proposed_medications=proposed_meds
        )
        print(f"Symbolic Logic Output: {treatment_plan.model_dump()}")

        # Step 4: Aggregate and explain
        explanation_parts = [
            "Based on your symptoms:",
            f"  - Extracted Symptoms: {', '.join(llm_output.extracted_symptoms) if llm_output.extracted_symptoms else 'None'}",
            f"  - Most Probable Disease: {max(pgm_output.disease_probabilities, key=pgm_output.disease_probabilities.get)} (Probability: {max(pgm_output.disease_probabilities.values()):.2f})",
            f"  - Other Probabilities: {', '.join([f'{k}: {v:.2f}' for k, v in pgm_output.disease_probabilities.items()])}",
            "Treatment Plan:",
            f"  - Recommended Medications: {', '.join(treatment_plan.recommended_medications)}",
            f"  - Contraindications: {', '.join(treatment_plan.contraindications_identified) if treatment_plan.contraindications_identified else 'None'}",
            f"  - Clinical Guideline Adherence: {treatment_plan.clinical_guideline_adherence}"
        ]

        return SystemOutput(
            diagnosis=pgm_output.disease_probabilities,
            treatment_plan=treatment_plan,
            explanation="\n".join(explanation_parts)
        )

# Example Usage
if __name__ == "__main__":
    system = MedicalDiagnosisSystem()

    patient_case_1 = PatientInput(patient_history="A 30-year-old male with a persistent cough, fever, and fatigue. He has a penicillin allergy.")
    output_1 = system.diagnose_and_plan(patient_case_1)
    print("\n--- Patient Case 1 ---")
    print(output_1.model_dump_json(indent=2))

    patient_case_2 = PatientInput(patient_history="Patient reports a mild headache. No other symptoms.")
    output_2 = system.diagnose_and_plan(patient_case_2)
    print("\n--- Patient Case 2 ---")
    print(output_2.model_dump_json(indent=2))

    patient_case_3 = PatientInput(patient_history="A 45-year-old female experiencing fever and body aches. She is currently taking aspirin for a pre-existing condition.")
    output_3 = system.diagnose_and_plan(patient_case_3)
    print("\n--- Patient Case 3 ---")
    print(output_3.model_dump_json(indent=2))