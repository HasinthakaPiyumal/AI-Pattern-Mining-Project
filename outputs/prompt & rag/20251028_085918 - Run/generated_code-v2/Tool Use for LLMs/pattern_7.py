from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import re

class LLMProcessor:
    def __init__(self):
        self.symptom_keywords = {
            "fever": "Fever",
            "cough": "Cough",
            "headache": "Headache",
            "sore throat": "Sore Throat",
            "fatigue": "Fatigue",
            "nausea": "Nausea",
            "vomiting": "Vomiting"
        }

    def process_input(self, text):
        evidence = {}
        text_lower = text.lower()
        for keyword, symptom in self.symptom_keywords.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                evidence[symptom] = 'present'
            else:
                evidence[symptom] = 'absent'
        return evidence

class PGMModel:
    def __init__(self):
        self.model = BayesianNetwork([
            ('Fever', 'Flu'),
            ('Cough', 'Flu'),
            ('Headache', 'Flu'),
            ('Fever', 'CommonCold'),
            ('Cough', 'CommonCold'),
            ('Sore Throat', 'CommonCold'),
            ('Fatigue', 'Mononucleosis'),
            ('Sore Throat', 'Mononucleosis'),
            ('Nausea', 'FoodPoisoning'),
            ('Vomiting', 'FoodPoisoning')
        ])

        # Define Conditional Probability Distributions (CPDs)
        cpd_fever = TabularCPD(variable='Fever', variable_card=2, 
                               values=[[0.9, 0.1]], # absent, present
                               state_names={'Fever': ['absent', 'present']})
        cpd_cough = TabularCPD(variable='Cough', variable_card=2,
                               values=[[0.8, 0.2]], # absent, present
                               state_names={'Cough': ['absent', 'present']})
        cpd_headache = TabularCPD(variable='Headache', variable_card=2,
                                values=[[0.85, 0.15]], # absent, present
                                state_names={'Headache': ['absent', 'present']})
        cpd_sore_throat = TabularCPD(variable='Sore Throat', variable_card=2,
                                    values=[[0.9, 0.1]], # absent, present
                                    state_names={'Sore Throat': ['absent', 'present']})
        cpd_fatigue = TabularCPD(variable='Fatigue', variable_card=2,
                                values=[[0.8, 0.2]], # absent, present
                                state_names={'Fatigue': ['absent', 'present']})
        cpd_nausea = TabularCPD(variable='Nausea', variable_card=2,
                               values=[[0.95, 0.05]], # absent, present
                               state_names={'Nausea': ['absent', 'present']})
        cpd_vomiting = TabularCPD(variable='Vomiting', variable_card=2,
                                values=[[0.98, 0.02]], # absent, present
                                state_names={'Vomiting': ['absent', 'present']})

        cpd_flu = TabularCPD(variable='Flu', variable_card=2, 
                             values=[[0.99, 0.9, 0.8, 0.1, 0.7, 0.05, 0.6, 0.01], # absent
                                     [0.01, 0.1, 0.2, 0.9, 0.3, 0.95, 0.4, 0.99]], # present
                             evidence=['Fever', 'Cough', 'Headache'],
                             evidence_card=[2, 2, 2],
                             state_names={'Flu': ['absent', 'present'], 
                                          'Fever': ['absent', 'present'], 
                                          'Cough': ['absent', 'present'], 
                                          'Headache': ['absent', 'present']})

        cpd_common_cold = TabularCPD(variable='CommonCold', variable_card=2,
                                     values=[[0.98, 0.8, 0.7, 0.1, 0.6, 0.05, 0.5, 0.01], # absent
                                             [0.02, 0.2, 0.3, 0.9, 0.4, 0.95, 0.5, 0.99]], # present
                                     evidence=['Fever', 'Cough', 'Sore Throat'],
                                     evidence_card=[2, 2, 2],
                                     state_names={'CommonCold': ['absent', 'present'],
                                                  'Fever': ['absent', 'present'],
                                                  'Cough': ['absent', 'present'],
                                                  'Sore Throat': ['absent', 'present']})

        cpd_mononucleosis = TabularCPD(variable='Mononucleosis', variable_card=2,
                                       values=[[0.99, 0.9, 0.8, 0.1], # absent
                                               [0.01, 0.1, 0.2, 0.9]], # present
                                       evidence=['Fatigue', 'Sore Throat'],
                                       evidence_card=[2, 2],
                                       state_names={'Mononucleosis': ['absent', 'present'],
                                                    'Fatigue': ['absent', 'present'],
                                                    'Sore Throat': ['absent', 'present']})

        cpd_food_poisoning = TabularCPD(variable='FoodPoisoning', variable_card=2,
                                        values=[[0.99, 0.9, 0.8, 0.1], # absent
                                                [0.01, 0.1, 0.2, 0.9]], # present
                                        evidence=['Nausea', 'Vomiting'],
                                        evidence_card=[2, 2],
                                        state_names={'FoodPoisoning': ['absent', 'present'],
                                                     'Nausea': ['absent', 'present'],
                                                     'Vomiting': ['absent', 'present']})

        self.model.add_cpds(cpd_fever, cpd_cough, cpd_headache, cpd_sore_throat, cpd_fatigue, 
                            cpd_nausea, cpd_vomiting, cpd_flu, cpd_common_cold, 
                            cpd_mononucleosis, cpd_food_poisoning)
        
        # Check if the model is valid
        # self.model.check_model()

        self.inference = VariableElimination(self.model)

    def infer_diagnosis(self, evidence):
        # Filter evidence to only include nodes present in the PGM
        filtered_evidence = {k: v for k, v in evidence.items() if k in self.model.nodes()}
        
        # Adjust evidence values for pgmpy
        pgmpy_evidence = {}
        for k, v in filtered_evidence.items():
            if v == 'present':
                pgmpy_evidence[k] = 1 # Corresponds to the 'present' state in CPDs
            elif v == 'absent':
                pgmpy_evidence[k] = 0 # Corresponds to the 'absent' state in CPDs
        
        possible_diseases = ['Flu', 'CommonCold', 'Mononucleosis', 'FoodPoisoning']
        results = {}
        for disease in possible_diseases:
            if disease in self.model.nodes():
                try:
                    query_result = self.inference.query(variables=[disease], evidence=pgmpy_evidence)
                    # Assuming 'present' is the second state (index 1)
                    results[disease] = query_result.values[1]
                except Exception as e:
                    # Handle cases where inference might fail due to incompatible evidence
                    results[disease] = 0.0 # Default to 0 probability if inference fails
            else:
                results[disease] = 0.0
        return results

class DiagnosticAssistant:
    def __init__(self):
        self.llm_processor = LLMProcessor()
        self.pgm_model = PGMModel()

    def diagnose(self, patient_input):
        structured_evidence = self.llm_processor.process_input(patient_input)
        diagnosis_probabilities = self.pgm_model.infer_diagnosis(structured_evidence)

        sorted_diagnoses = sorted(diagnosis_probabilities.items(), key=lambda item: item[1], reverse=True)

        recommendations = "\nRecommendations: ", 
        if not sorted_diagnoses or sorted_diagnoses[0][1] < 0.1: # Threshold for low probability
            recommendations += "No clear diagnosis based on the provided symptoms. Please consult a medical professional for a thorough examination."
        else:
            for disease, prob in sorted_diagnoses:
                if prob > 0.1:
                    if disease == 'Flu':
                        recommendations += f"\n - Consider rest, fluids, and over-the-counter flu medication. See a doctor if symptoms worsen."
                    elif disease == 'CommonCold':
                        recommendations += f"\n - Rest, fluids, and symptom relief with cold medications are recommended."
                    elif disease == 'Mononucleosis':
                        recommendations += f"\n - Get plenty of rest and avoid strenuous activities. Consult a doctor for management."
                    elif disease == 'FoodPoisoning':
                        recommendations += f"\n - Stay hydrated with small sips of water. Seek medical attention if symptoms are severe or persist."
                    else:
                        recommendations += f"\n - Consult a medical professional for further evaluation."
        
        return {
            "evidence_extracted": structured_evidence,
            "diagnosis_probabilities": sorted_diagnoses,
            "recommendations": "".join(recommendations)
        }


if __name__ == "__main__":
    assistant = DiagnosticAssistant()
    print("Welcome to the Medical Diagnostic Assistant!")
    print("Tell me about your symptoms (e.g., 'I have a fever and cough'). Type 'exit' to quit.")

    while True:
        user_input = input("\nYour symptoms: ")
        if user_input.lower() == 'exit':
            break

        diagnosis_result = assistant.diagnose(user_input)

        print("\n--- Diagnosis Report ---")
        print("Extracted Symptoms:")
        for symptom, status in diagnosis_result["evidence_extracted"].items():
            print(f"  {symptom}: {status}")

        print("\nProbable Diagnoses:")
        if not diagnosis_result["diagnosis_probabilities"] or diagnosis_result["diagnosis_probabilities"][0][1] < 0.001:
            print("  Unable to determine a clear diagnosis based on the provided information. Consider providing more details or consulting a medical professional.")
        else:
            for disease, prob in diagnosis_result["diagnosis_probabilities"]:
                print(f"  {disease}: {prob:.2f}")
        
        print(diagnosis_result["recommendations"])
