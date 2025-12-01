from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class NLUModule:
    def __init__(self):
        self.known_symptoms = [
            "fever", "cough", "sore throat", "headache", "fatigue", 
            "nausea", "vomiting", "diarrhea", "rash", "shortness of breath",
            "chest pain", "muscle pain", "joint pain"
        ]

    def extract_symptoms(self, patient_input: str) -> list:
        extracted = []
        input_lower = patient_input.lower()
        for symptom in self.known_symptoms:
            if symptom in input_lower:
                extracted.append(symptom)
        return extracted

class PGMModule:
    def __init__(self):
        self.model = self._build_bayesian_network()

    def _build_bayesian_network(self):
        model = BayesianModel([
            ('Influenza', 'Fever'), ('Influenza', 'Cough'), ('Influenza', 'Fatigue'),
            ('Common Cold', 'Cough'), ('Common Cold', 'Sore Throat'), ('Common Cold', 'Headache'),
            ('Strep Throat', 'Sore Throat'), ('Strep Throat', 'Fever'),
            ('COVID-19', 'Fever'), ('COVID-19', 'Cough'), ('COVID-19', 'Shortness of Breath'), ('COVID-19', 'Fatigue'),
            ('Fever', 'Headache') # Example of symptom relationship
        ])

        # Define Conditional Probability Distributions (CPDs)
        cpd_influenza = TabularCPD(variable='Influenza', variable_card=2, values=[[0.7], [0.3]]) # P(Influenza=no), P(Influenza=yes)
        cpd_cold = TabularCPD(variable='Common Cold', variable_card=2, values=[[0.8], [0.2]]) # P(Common Cold=no), P(Common Cold=yes)
        cpd_strep = TabularCPD(variable='Strep Throat', variable_card=2, values=[[0.9], [0.1]]) # P(Strep Throat=no), P(Strep Throat=yes)
        cpd_covid = TabularCPD(variable='COVID-19', variable_card=2, values=[[0.95], [0.05]]) # P(COVID-19=no), P(COVID-19=yes)

        cpd_fever = TabularCPD(variable='Fever', variable_card=2, 
                               values=[[0.9, 0.4, 0.6, 0.4, 0.5],
                                       [0.1, 0.6, 0.4, 0.6, 0.5]],
                               evidence=['Influenza', 'Strep Throat', 'COVID-19', 'Common Cold'],
                               evidence_card=[2, 2, 2, 2])

        cpd_cough = TabularCPD(variable='Cough', variable_card=2, 
                               values=[[0.8, 0.3, 0.3, 0.2],
                                       [0.2, 0.7, 0.7, 0.8]],
                               evidence=['Influenza', 'Common Cold', 'COVID-19'],
                               evidence_card=[2, 2, 2])

        cpd_fatigue = TabularCPD(variable='Fatigue', variable_card=2, 
                                 values=[[0.9, 0.4, 0.6],
                                         [0.1, 0.6, 0.4]],
                                 evidence=['Influenza', 'COVID-19'],
                                 evidence_card=[2, 2])

        cpd_sore_throat = TabularCPD(variable='Sore Throat', variable_card=2, 
                                     values=[[0.8, 0.3, 0.3],
                                             [0.2, 0.7, 0.7]],
                                     evidence=['Common Cold', 'Strep Throat'],
                                     evidence_card=[2, 2])

        cpd_headache = TabularCPD(variable='Headache', variable_card=2, 
                                  values=[[0.9, 0.6, 0.4],
                                          [0.1, 0.4, 0.6]],
                                  evidence=['Common Cold', 'Fever'],
                                  evidence_card=[2, 2])

        cpd_shortness_of_breath = TabularCPD(variable='Shortness of Breath', variable_card=2,
                                             values=[[0.99, 0.1],
                                                     [0.01, 0.9]],
                                             evidence=['COVID-19'],
                                             evidence_card=[2])

        model.add_cpds(cpd_influenza, cpd_cold, cpd_strep, cpd_covid, 
                       cpd_fever, cpd_cough, cpd_fatigue, cpd_sore_throat, 
                       cpd_headache, cpd_shortness_of_breath)

        model.check_model()
        return model

    def diagnose(self, symptoms: list) -> dict:
        inference = VariableElimination(self.model)
        evidence = {symptom.replace(' ', '_').title(): 1 for symptom in symptoms if symptom in self.model.nodes()} # Map to PGM variable names (1 for present)

        # Filter out symptoms not in the model to avoid errors
        valid_evidence = {}
        for s, val in evidence.items():
            if s in self.model.nodes():
                valid_evidence[s] = val
        
        if not valid_evidence:
            return {"No specific diagnosis": 1.0}

        # Infer probabilities for diseases
        disease_probabilities = {}
        diseases = ['Influenza', 'Common_Cold', 'Strep_Throat', 'COVID-19'] # Use model's internal names
        for disease in diseases:
            if disease in self.model.nodes():
                try:
                    query_result = inference.query(variables=[disease], evidence=valid_evidence)
                    disease_probabilities[disease.replace('_', ' ')] = query_result.values[1] # Probability of disease being True (index 1)
                except Exception as e:
                    # Handle cases where inference might fail due to lack of connection or insufficient evidence
                    pass
        
        # If no specific disease probability can be inferred, provide a general statement
        if not disease_probabilities:
             # Try to query based on marginals if no evidence is directly applicable to diseases
            for disease in diseases:
                if disease in self.model.nodes():
                    try:
                        query_result = inference.query(variables=[disease])
                        disease_probabilities[disease.replace('_', ' ')] = query_result.values[1]
                    except Exception:
                        pass
            if not disease_probabilities:
                 return {"Could not determine specific diagnoses based on provided symptoms.": 1.0}

        # Sort diagnoses by probability and return top ones
        sorted_diagnoses = sorted(disease_probabilities.items(), key=lambda item: item[1], reverse=True)
        return {diag[0]: diag[1] for diag in sorted_diagnoses}

class MedicalKnowledgeGraph:
    def __init__(self):
        self.knowledge_base = {
            "Influenza": {
                "treatments": ["Antivirals (Oseltamivir)", "Rest", "Fluids", "Pain relievers"],
                "contraindications": {"Oseltamivir": []},
                "diagnostic_steps": ["Nasal swab for flu test"]
            },
            "Common Cold": {
                "treatments": ["Rest", "Fluids", "Decongestants", "Pain relievers"],
                "contraindications": {"Decongestants": ["High blood pressure", "Heart disease"]},
                "diagnostic_steps": []
            },
            "Strep Throat": {
                "treatments": ["Antibiotics (Penicillin, Amoxicillin)", "Pain relievers"],
                "contraindications": {"Penicillin": ["Penicillin allergy"]},
                "diagnostic_steps": ["Rapid strep test", "Throat culture"]
            },
            "COVID-19": {
                "treatments": ["Supportive care", "Antivirals (Remdesivir - for severe cases)", "Monoclonal antibodies"],
                "contraindications": {"Remdesivir": ["Kidney impairment"]},
                "diagnostic_steps": ["PCR test", "Antigen test"]
            }
        }

    def get_treatments(self, disease: str) -> list:
        return self.knowledge_base.get(disease, {}).get("treatments", ["Consult a healthcare professional for treatment."])

    def get_contraindications(self, disease: str, medication: str) -> list:
        return self.knowledge_base.get(disease, {}).get("contraindications", {}).get(medication, [])

    def get_diagnostic_steps(self, disease: str) -> list:
        return self.knowledge_base.get(disease, {}).get("diagnostic_steps", [])

class ReasoningRecommendationEngine:
    def __init__(self):
        self.nlu = NLUModule()
        self.pgm = PGMModule()
        self.kg = MedicalKnowledgeGraph()

    def get_personalized_recommendations(self, patient_input: str, current_medications: list = None) -> dict:
        extracted_symptoms = self.nlu.extract_symptoms(patient_input)
        
        if not extracted_symptoms:
            return {"diagnosis": "Please provide more symptoms or details.", "recommendations": []}

        diagnoses = self.pgm.diagnose(extracted_symptoms)
        
        recommendations = []
        for disease, probability in diagnoses.items():
            if probability > 0.05:  # Consider diagnoses with a reasonable probability
                recommendations.append(f"Potential Diagnosis: {disease} (Probability: {probability:.2f})")
                
                treatments = self.kg.get_treatments(disease)
                recommendations.append(f"  Recommended Treatments: {', '.join(treatments)}")

                if current_medications:
                    for med in current_medications:
                        contraindications = self.kg.get_contraindications(disease, med)
                        if contraindications:
                            recommendations.append(f"  Warning: {med} may have contraindications for {disease} related to: {', '.join(contraindications)}")
                
                diagnostic_steps = self.kg.get_diagnostic_steps(disease)
                if diagnostic_steps:
                    recommendations.append(f"  Further Diagnostic Steps: {', '.join(diagnostic_steps)}")
        
        if not recommendations:
            recommendations.append("Based on the provided information, no specific diagnoses could be confidently made. Please consult a doctor.")

        return {"diagnosis_summary": diagnoses, "recommendations": recommendations}

# Example Usage:
if __name__ == "__main__":
    assistant = ReasoningRecommendationEngine()

    print("--- Scenario 1: Flu-like symptoms ---")
    patient_input_1 = "I have a fever, cough, and feel very tired. My body aches."
    current_meds_1 = []
    output_1 = assistant.get_personalized_recommendations(patient_input_1, current_meds_1)
    print("Patient Input:", patient_input_1)
    for rec in output_1['recommendations']:
        print(rec)
    print("\n")

    print("--- Scenario 2: Sore throat and headache ---")
    patient_input_2 = "My throat is really sore and I have a headache. No fever though."
    current_meds_2 = ["Amoxicillin"] # Example existing medication
    output_2 = assistant.get_personalized_recommendations(patient_input_2, current_meds_2)
    print("Patient Input:", patient_input_2)
    for rec in output_2['recommendations']:
        print(rec)
    print("\n")

    print("--- Scenario 3: Possible COVID-19 symptoms ---")
    patient_input_3 = "I have a high fever, a dry cough, and I'm finding it hard to breathe. Feeling very fatigued."
    current_meds_3 = []
    output_3 = assistant.get_personalized_recommendations(patient_input_3, current_meds_3)
    print("Patient Input:", patient_input_3)
    for rec in output_3['recommendations']:
        print(rec)
    print("\n")

    print("--- Scenario 4: Vague symptoms ---")
    patient_input_4 = "I just don't feel well."
    current_meds_4 = []
    output_4 = assistant.get_personalized_recommendations(patient_input_4, current_meds_4)
    print("Patient Input:", patient_input_4)
    for rec in output_4['recommendations']:
        print(rec)
    print("\n")

    print("--- Scenario 5: Common Cold symptoms with contraindication check ---")
    patient_input_5 = "I have a cough and runny nose, and a mild headache. I have high blood pressure."
    current_meds_5 = ["Decongestants"]
    output_5 = assistant.get_personalized_recommendations(patient_input_5, current_meds_5)
    print("Patient Input:", patient_input_5)
    for rec in output_5['recommendations']:
        print(rec)
    print("\n")
