class EHRDataModule:
    def __init__(self):
        self.patients = {
            "patient_001": {
                "name": "Alice Smith",
                "age": 45,
                "gender": "Female",
                "history": ["Hypertension", "Type 2 Diabetes"],
                "current_symptoms": ["fatigue", "frequent urination", "blurred vision"],
                "lab_results": {"blood_glucose": "250 mg/dL", "HbA1c": "9.0%"}
            },
            "patient_002": {
                "name": "Bob Johnson",
                "age": 60,
                "gender": "Male",
                "history": ["Coronary Artery Disease"],
                "current_symptoms": ["chest pain", "shortness of breath", "arm numbness"],
                "lab_results": {"troponin": "elevated", "ECG": "ST depression"}
            }
        }

    def get_patient_data(self, patient_id):
        return self.patients.get(patient_id)


class MedicalKnowledgeGraphModule:
    def __init__(self):
        self.knowledge_graph = {
            "diabetes_mellitus": {
                "symptoms": ["fatigue", "frequent urination", "blurred vision", "unexplained weight loss", "increased thirst"],
                "tests": ["blood glucose test", "HbA1c", "oral glucose tolerance test"],
                "treatments": ["metformin", "insulin", "dietary changes", "exercise"],
                "description": "A chronic condition that affects how your body turns food into energy."
            },
            "hypertension": {
                "symptoms": ["headache", "dizziness", "nosebleeds"], # Often asymptomatic
                "tests": ["blood pressure measurement"],
                "treatments": ["ACE inhibitors", "diuretics", "lifestyle changes"],
                "description": "A common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems."
            },
            "coronary_artery_disease": {
                "symptoms": ["chest pain", "shortness of breath", "fatigue", "arm numbness", "jaw pain"],
                "tests": ["ECG", "stress test", "angiography", "troponin test"],
                "treatments": ["aspirin", "statins", "beta-blockers", "angioplasty", "bypass surgery"],
                "description": "A common heart condition caused by plaque buildup in the heart's arteries."
            }
        }

    def retrieve_facts(self, keywords):
        relevant_facts = {}
        for keyword in keywords:
            for disease, info in self.knowledge_graph.items():
                if keyword.lower() in disease.lower():
                    relevant_facts[disease] = info
                    continue
                for symptom in info.get("symptoms", []):
                    if keyword.lower() in symptom.lower():
                        relevant_facts[disease] = info
                        break
        return relevant_facts


class MockLLM:
    def invoke(self, prompt):
        # Simulate an LLM response based on the prompt content
        if "patient_001" in prompt and "diabetes_mellitus" in prompt:
            return (
                "Based on the patient's symptoms (fatigue, frequent urination, blurred vision) "
                "and lab results (blood glucose 250 mg/dL, HbA1c 9.0%), along with a history of Type 2 Diabetes, "
                "the primary diagnosis is uncontrolled Type 2 Diabetes Mellitus. "
                "Differential diagnoses could include other metabolic disorders, but diabetes is strongly indicated."
                "\n\nRecommended tests: Further metabolic panel, urine analysis. "
                "\nRecommended treatment: Adjust insulin regimen, dietary consultation, increased exercise, consider SGLT2 inhibitors or GLP-1 agonists."
            )
        elif "patient_002" in prompt and "coronary_artery_disease" in prompt:
            return (
                "Given the patient's chief complaints of chest pain, shortness of breath, arm numbness, "
                "and elevated troponin with ST depression on ECG, in a patient with a history of Coronary Artery Disease, "
                "an acute coronary syndrome (ACS), likely a myocardial infarction, is highly suspected. "
                "\n\nRecommended tests: Serial troponin, urgent cardiac catheterization, echocardiogram. "
                "\nRecommended treatment: Immediate antiplatelet therapy (aspirin, clopidogrel), anticoagulation, nitrates, beta-blockers, statins, and preparation for revascularization."
            )
        else:
            return 