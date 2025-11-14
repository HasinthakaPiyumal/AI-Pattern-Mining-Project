
import random

class Patient:
    def __init__(self, patient_id, symptoms, demographics, medical_history, cultural_background=None):
        self.patient_id = patient_id
        self.symptoms = symptoms
        self.demographics = demographics  # e.g., {'age': 35, 'gender': 'female', 'ethnicity': 'Asian'}
        self.medical_history = medical_history
        self.cultural_background = cultural_background # e.g., 'East Asian', 'European', 'Latin American'

    def __repr__(self):
        return f"Patient(ID: {self.patient_id}, Symptoms: {self.symptoms})"

class ClinicalDecisionSupportSystem:
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        # In a real system, this would load from databases, ontologies, etc.
        # Simplified for demonstration: mappings of symptoms to rare diseases and treatments.
        return {
            "Fever, Rash, Joint Pain": {
                "disease": "Lupus-like Syndrome (Rare Variant)",
                "treatment_options": ["Immunosuppressants", "Symptomatic Relief"],
                "evidence": {
                    "Immunosuppressants": {
                        "for": "Reduces inflammation, prevents organ damage.",
                        "against": "Potential side effects (infections, long-term complications)."
                    },
                    "Symptomatic Relief": {
                        "for": "Manages acute discomfort, non-invasive.",
                        "against": "Does not address underlying autoimmune process."
                    }
                }
            },
            "Fatigue, Muscle Weakness, Difficulty Swallowing": {
                "disease": "Mitochondrial Myopathy (Rare)",
                "treatment_options": ["CoQ10 Supplementation", "Physical Therapy"],
                "evidence": {
                    "CoQ10 Supplementation": {
                        "for": "May improve mitochondrial function in some patients.",
                        "against": "Efficacy varies widely, not a cure, potential drug interactions."
                    },
                    "Physical Therapy": {
                        "for": "Maintains muscle strength, improves quality of life.",
                        "against": "Does not target the root genetic cause."
                    }
                }
            },
            # Add more rare diseases for diverse scenarios
        }

    def _simulate_llm_response(self, prompt, bias_factor=0):
        # A simple simulated LLM. In reality, this would be an actual LLM call.
        # bias_factor can simulate demographic bias for demonstration.
        print(f"[Simulated LLM Processing Prompt: {prompt[:50]}...]", end=" ")
        for symptom, data in self.knowledge_base.items():
            if all(s in prompt for s in symptom.split(', ')):
                # Introduce a slight bias for demonstration based on bias_factor
                if random.random() < bias_factor: # Simulates a biased response
                    return {"diagnosis": "Common Cold", "confidence": 0.1, "reason": "Simplified for bias demo."} # Incorrect diagnosis due to bias
                return {"diagnosis": data["disease"], "confidence": 0.95, "reason": f"Matches symptoms: {symptom}"}
        return {"diagnosis": "Undetermined", "confidence": 0.5, "reason": "No direct match in knowledge base."}

    def _generate_demonstration_prompt(self, patient, exemplars, cultural_context="None"):
        # Construct a prompt with patient details and few-shot exemplars.
        exemplar_str = "\n".join([f"Example: Patient with {e['symptoms']} -> Diagnosis: {e['diagnosis']}" for e in exemplars])
        cultural_advisory = f"Consider the patient's {cultural_context} cultural background when forming recommendations. " if cultural_context != "None" else ""
        prompt = (
            f"Given the following medical context:\n"
            f"Patient Symptoms: {', '.join(patient.symptoms)}\n"
            f"Patient Demographics: {patient.demographics}\n"
            f"Patient Medical History: {patient.medical_history}\n"
            f"{exemplar_str}\n"
            f"{cultural_advisory}"
            f"Based on this, what is the most likely rare disease diagnosis and initial treatment recommendation?"
        )
        return prompt

    def diagnose_and_recommend(self, patient):
        print(f"\n--- Diagnosing Patient {patient.patient_id} ---")

        # 1. Selecting Balanced Demonstrations (Conceptual)
        # In a real scenario, exemplars would be retrieved from a database of balanced cases
        # Here, we simulate by ensuring variety in 'balanced_exemplars'
        balanced_exemplars = [
            {'symptoms': 'Fatigue, Muscle Weakness, Difficulty Swallowing', 'diagnosis': 'Mitochondrial Myopathy (Rare)'},
            {'symptoms': 'Fever, Rash, Joint Pain', 'diagnosis': 'Lupus-like Syndrome (Rare Variant)'},
        ]
        print(f"[Step 1: Selecting Balanced Demonstrations] Using a diverse set of exemplars.")

        # 2. Cultural Awareness (Prompt Design)
        cultural_context = patient.cultural_background if patient.cultural_background else "general medical practice"
        print(f"[Step 2: Cultural Awareness] Adapting prompt for cultural context: {cultural_context}")

        # 3. Demonstration Ensembling (DENSE)
        num_ensembles = 3
        ensemble_results = []
        print(f"[Step 3: Demonstration Ensembling] Running {num_ensembles} distinct prompts for aggregation.")

        for i in range(num_ensembles):
            # Create distinct exemplar subsets for each prompt in the ensemble
            # For simplicity, we just shuffle or pick a subset of the balanced exemplars
            current_exemplars = random.sample(balanced_exemplars, k=min(2, len(balanced_exemplars)))
            
            # Simulate a slight bias if demographics are 'uncommon' in the simulated LLM's 'training'
            bias_factor = 0.2 if patient.demographics.get('ethnicity') == 'Maori' else 0.0

            prompt = self._generate_demonstration_prompt(patient, current_exemplars, cultural_context)
            llm_output = self._simulate_llm_response(prompt, bias_factor=bias_factor)
            ensemble_results.append(llm_output)
            print(f"Ensemble {i+1} Output: {llm_output['diagnosis']} (Confidence: {llm_output['confidence']:.2f})")

        # Aggregate results (e.g., majority vote or weighted average of confidence)
        diagnoses = [res['diagnosis'] for res in ensemble_results]
        # Simple majority vote for aggregation
        final_diagnosis = max(set(diagnoses), key=diagnoses.count)
        final_confidence = sum(res['confidence'] for res in ensemble_results if res['diagnosis'] == final_diagnosis) / diagnoses.count(final_diagnosis)

        print(f"\n[DENSE Aggregation] Final Ensembled Diagnosis: {final_diagnosis} (Aggregated Confidence: {final_confidence:.2f})")

        # 4. Debate-Style Evidence Aggregation
        print(f"\n[Step 4: Debate-Style Evidence Aggregation] Preparing treatment evidence.")
        treatment_info = self.knowledge_base.get(final_diagnosis, {}).get("evidence")

        if treatment_info:
            print(f"\n--- Treatment Recommendations for {final_diagnosis} ---")
            for treatment, evidence in treatment_info.items():
                print(f"\nTreatment Option: {treatment}")
                print(f"  Pro: {evidence['for']}")
                print(f"  Con: {evidence['against']}")
        else:
            print(f"No specific treatment evidence found for {final_diagnosis} in the knowledge base.")

        print(f"\n--- End of Diagnosis for Patient {patient.patient_id} ---")

# --- Demonstration Usage ---
if __name__ == "__main__":
    cdss = ClinicalDecisionSupportSystem()

    # Patient 1: Standard case, demonstrating DENSE and Balanced Demonstrations
    patient1 = Patient(
        patient_id="P001",
        symptoms=["Fever", "Rash", "Joint Pain"],
        demographics={'age': 40, 'gender': 'female', 'ethnicity': 'Caucasian'},
        medical_history="Recent viral infection",
        cultural_background="Western European"
    )
    cdss.diagnose_and_recommend(patient1)

    # Patient 2: Demonstrating Cultural Awareness and a potentially biased scenario (simulated)
    patient2 = Patient(
        patient_id="P002",
        symptoms=["Fatigue", "Muscle Weakness", "Difficulty Swallowing"],
        demographics={'age': 60, 'gender': 'male', 'ethnicity': 'Maori'},
        medical_history="Long-standing muscle aches",
        cultural_background="Maori"
    )
    cdss.diagnose_and_recommend(patient2)

    # Patient 3: Another case for diverse outputs
    patient3 = Patient(
        patient_id="P003",
        symptoms=["Numbness", "Tingling", "Vision Changes"],
        demographics={'age': 30, 'gender': 'female', 'ethnicity': 'African'},
        medical_history="No significant history",
        cultural_background="North African"
    )
    # Add P003's symptoms to the knowledge base for a positive match
    cdss.knowledge_base["Numbness, Tingling, Vision Changes"] = {
        "disease": "Early Onset Neuropathy (Rare Type)",
        "treatment_options": ["Immunomodulators", "Supportive Care"],
        "evidence": {
            "Immunomodulators": {
                "for": "Slows disease progression, reduces inflammatory response.",
                "against": "Significant side effects, requires careful monitoring."
            },
            "Supportive Care": {
                "for": "Manages symptoms, improves daily functioning.",
                "against": "Does not alter disease course, only palliative."
            }
        }
    }
    cdss.diagnose_and_recommend(patient3)
