import re

class SimulatedLLM:
    def __init__(self):
        pass

    def generate(self, prompt):
        prompt_lower = prompt.lower()
        
        if "what are the symptoms of" in prompt_lower or "tell me about" in prompt_lower or "treatment for" in prompt_lower:
            match = re.search(r"symptoms of (.*?)[.?]" , prompt_lower) or \
                    re.search(r"about (.*?)[.?]" , prompt_lower) or \
                    re.search(r"treatment for (.*?)[.?]", prompt_lower)
            if match:
                condition = match.group(1).strip()
                if "symptoms" in prompt_lower:
                    return f"TOOL_CALL: medical_kb, ACTION: get_symptoms, PARAM: {condition}"
                elif "treatment" in prompt_lower:
                    return f"TOOL_CALL: medical_kb, ACTION: get_treatment, PARAM: {condition}"
                else:
                    return f"TOOL_CALL: medical_kb, ACTION: get_info, PARAM: {condition}"
        
        if "calculate dosage for" in prompt_lower or "drug dosage" in prompt_lower:
            weight_match = re.search(r"(\\d+)\\s*kg", prompt_lower)
            med_match = re.search(r"for (\\w+)\\s*medication", prompt_lower)
            if weight_match and med_match:
                weight = int(weight_match.group(1))
                medication = med_match.group(1)
                return f"TOOL_CALL: medical_calculator, ACTION: calculate_dosage, PARAM: {{'weight_kg': {weight}, 'medication_name': '{medication}'}}"
            return f"I need more information for drug dosage calculation, like weight and medication name."

        if "diagnose based on" in prompt_lower or "what could this be" in prompt_lower:
            symptoms_match = re.search(r"based on (.*?)[.?]", prompt_lower)
            if symptoms_match:
                symptoms = [s.strip() for s in symptoms_match.group(1).split(",")]
                return f"TOOL_CALL: diagnostic_engine, ACTION: diagnose, PARAM: {symptoms}"
            return f"Please provide a list of symptoms for diagnosis."

        if "patient history for" in prompt_lower:
            patient_id_match = re.search(r"patient history for (\\w+)", prompt_lower)
            if patient_id_match:
                patient_id = patient_id_match.group(1)
                return f"TOOL_CALL: patient_record_system, ACTION: get_history, PARAM: {patient_id}"
            return f"Please provide a patient ID."
            
        if "clinical trials for" in prompt_lower:
            condition_match = re.search(r"clinical trials for (.*?)[.?]", prompt_lower)
            if condition_match:
                condition = condition_match.group(1).strip()
                return f"TOOL_CALL: clinical_trial_database, ACTION: search_trials, PARAM: {condition}"
            return f"Please specify a condition to search for clinical trials."

        if "SYNTHESIZE:" in prompt:
            parts = prompt.split("SYNTHESIZE:", 1)
            original_query = parts[0].replace("ORIGINAL_QUERY:", "").strip()
            tool_results = parts[1].strip()
            return f"MediMRKL's comprehensive answer to your query \"{original_query}\": {tool_results}"

        return f"I can help with medical information, diagnosis, calculations, patient history, and clinical trials. Please be specific."

class MedicalKnowledgeBase:
    def __init__(self):
        self.data = {
            "influenza": {
                "symptoms": ["fever", "cough", "sore throat", "muscle aches", "fatigue"],
                "treatment": "rest, fluids, antiviral medication (if severe)",
                "info": "Influenza, commonly known as the flu, is an infectious disease caused by influenza viruses."
            },
            "diabetes": {
                "symptoms": ["frequent urination", "increased thirst", "unexplained weight loss", "blurred vision"],
                "treatment": "diet, exercise, medication (insulin or oral drugs)",
                "info": "Diabetes is a chronic condition that affects how your body turns food into energy."
            },
            "headache": {
                "symptoms": ["pain in head", "pressure"],
                "treatment": "pain relievers, rest",
                "info": "A headache is pain in any region of the head."
            }
        }

    def get_symptoms(self, condition):
        return self.data.get(condition.lower(), {}).get("symptoms", f"Symptoms for {condition} not found.")

    def get_treatment(self, condition):
        return self.data.get(condition.lower(), {}).get("treatment", f"Treatment for {condition} not found.")

    def get_info(self, condition):
        return self.data.get(condition.lower(), {}).get("info", f"Information on {condition} not found.")

class DiagnosticEngine:
    def __init__(self):
        self.disease_symptoms = {
            "influenza": ["fever", "cough", "sore throat"],
            "common cold": ["runny nose", "sneezing", "mild cough"],
            "diabetes": ["frequent urination", "increased thirst"]
        }

    def diagnose(self, symptoms):
        symptoms_lower = [s.lower() for s in symptoms]
        possible_diseases = []
        for disease, known_symptoms in self.disease_symptoms.items():
            if all(s in symptoms_lower for s in known_symptoms):
                possible_diseases.append(disease)
        
        if possible_diseases:
            return f"Possible diagnoses: {', '.join(possible_diseases)}."
        return "No specific diagnosis could be made based on the provided symptoms."

class MedicalCalculator:
    def __init__(self):
        pass

    def calculate_dosage(self, weight_kg, medication_name, params=None):
        if medication_name.lower() == "paracetamol":
            dosage_mg_per_kg = 15
            max_single_dose_mg = 1000
            calculated_dose = min(weight_kg * dosage_mg_per_kg, max_single_dose_mg)
            return f"For {weight_kg}kg patient, a single dose of {medication_name} is approximately {calculated_dose}mg."
        elif medication_name.lower() == "ibuprofen":
            dosage_mg_per_kg = 10
            max_single_dose_mg = 400
            calculated_dose = min(weight_kg * dosage_mg_per_kg, max_single_dose_mg)
            return f"For {weight_kg}kg patient, a single dose of {medication_name} is approximately {calculated_dose}mg."
        return f"Dosage calculation for {medication_name} is not available in this calculator."

    def calculate_bmi(self, weight_kg, height_m):
        if height_m <= 0:
            return "Height must be greater than zero."
        bmi = weight_kg / (height_m ** 2)
        return f"BMI: {bmi:.2f}"

class PatientRecordSystem:
    def __init__(self):
        self.patient_records = {
            "P1001": {"name": "Alice Smith", "age": 35, "conditions": ["hypertension"], "medications": ["lisinopril"]},
            "P1002": {"name": "Bob Johnson", "age": 60, "conditions": ["diabetes type 2"], "medications": ["metformin"]}
        }

    def get_patient_history(self, patient_id):
        record = self.patient_records.get(patient_id.upper())
        if record:
            return f"Patient {patient_id}: Name - {record['name']}, Age - {record['age']}, Conditions - {', '.join(record['conditions'])}, Medications - {', '.join(record['medications'])}"
        return f"Patient {patient_id} not found."

class ClinicalTrialDatabase:
    def __init__(self):
        self.trials = {
            "diabetes": [
                {"title": "Study on New Insulin Regimen", "phase": "Phase 3", "location": "Various"},
                {"title": "Lifestyle Intervention for Type 2 Diabetes", "phase": "Phase 2", "location": "Local Clinic"}
            ],
            "cancer": [
                {"title": "Immunotherapy for Lung Cancer", "phase": "Phase 1", "location": "Major Hospital"}
            ]
        }

    def search_trials(self, condition):
        condition_lower = condition.lower()
        results = self.trials.get(condition_lower, [])
        if results:
            return f"Clinical trials for {condition}: {'; '.join([f'{t['title']} (Phase {t['phase']}, {t['location']})' for t in results])}"
        return f"No clinical trials found for {condition}."

class LLMRouter:
    def __init__(self):
        self.llm = SimulatedLLM()
        self.medical_kb = MedicalKnowledgeBase()
        self.diagnostic_engine = DiagnosticEngine()
        self.medical_calculator = MedicalCalculator()
        self.patient_record_system = PatientRecordSystem()
        self.clinical_trial_database = ClinicalTrialDatabase()
        self.tools = {
            "medical_kb": self.medical_kb,
            "diagnostic_engine": self.diagnostic_engine,
            "medical_calculator": self.medical_calculator,
            "patient_record_system": self.patient_record_system,
            "clinical_trial_database": self.clinical_trial_database,
        }

    def route_query(self, query):
        intent_prompt = f"Analyze the following medical query and identify the most suitable tool and parameters. If no specific tool is needed, indicate 'NO_TOOL'.\nQuery: {query}"
        tool_call_instruction = self.llm.generate(intent_prompt)
        
        tool_results = []
        if tool_call_instruction.startswith("TOOL_CALL:"):
            parts = tool_call_instruction.replace("TOOL_CALL: ", "").split(", ACTION: ", 1)
            tool_name = parts[0].strip()
            action_parts = parts[1].split(", PARAM: ", 1)
            action = action_parts[0].strip()
            params_str = action_parts[1].strip()

            try:
                # Attempt to evaluate parameters if they look like a dict or list
                if params_str.startswith("{") and params_str.endswith("}") or \
                   params_str.startswith("[") and params_str.endswith("]"):
                    params = eval(params_str)
                else:
                    params = params_str # Assume string parameter otherwise

                tool_instance = self.tools.get(tool_name)
                if tool_instance:
                    method = getattr(tool_instance, action, None)
                    if method:
                        if isinstance(params, dict):
                            result = method(**params)
                        elif isinstance(params, list):
                            result = method(params)
                        else:
                            result = method(params)
                        tool_results.append(result)
                    else:
                        tool_results.append(f"Error: Action '{action}' not found for tool '{tool_name}'.")
                else:
                    tool_results.append(f"Error: Tool '{tool_name}' not found.")
            except Exception as e:
                tool_results.append(f"Error processing tool call '{tool_call_instruction}': {e}")
        else:
            tool_results.append("No specific tool identified or direct LLM response simulated.\n" + tool_call_instruction)

        synthesis_prompt = f"ORIGINAL_QUERY: {query}\nSYNTHESIZE: {', '.join(tool_results) if tool_results else 'No specific information found.'}"
        final_response = self.llm.generate(synthesis_prompt)
        return final_response

if __name__ == "__main__":
    router = LLMRouter()

    queries = [
        "What are the symptoms of influenza?",
        "Calculate dosage for a 70kg patient for paracetamol medication.",
        "Diagnose based on fever, cough, and sore throat.",
        "Tell me about diabetes.",
        "What is the patient history for P1001?",
        "Are there any clinical trials for diabetes?",
        "What is the capital of France?" # Query that should not trigger a specific medical tool
    ]

    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1}: {query} ---")
        response = router.route_query(query)
        print(f"MediMRKL Response: {response}")
