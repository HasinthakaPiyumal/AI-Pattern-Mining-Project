import re

class MedicalDatabase:
    def __init__(self):
        self.diseases = {
            "common cold": {"symptoms": ["runny nose", "sore throat", "cough"], "treatment": "rest, fluids"},
            "influenza": {"symptoms": ["fever", "body aches", "cough", "fatigue"], "treatment": "antivirals, rest, fluids"},
            "strep throat": {"symptoms": ["sore throat", "difficulty swallowing", "fever"], "treatment": "antibiotics"},
            "pneumonia": {"symptoms": ["cough", "fever", "shortness of breath", "chest pain"], "treatment": "antibiotics, oxygen therapy"},
        }

    def query_symptoms(self, symptoms: list) -> dict:
        """
        Simulates querying a medical database for diseases matching given symptoms.
        """
        matching_diseases = {}
        for disease, info in self.diseases.items():
            if any(symptom in info["symptoms"] for symptom in symptoms):
                matching_diseases[disease] = info
        return {"source": "MedicalDatabase", "data": matching_diseases}

    def get_disease_info(self, disease_name: str) -> dict:
        """
        Retrieves detailed information for a specific disease.
        """
        info = self.diseases.get(disease_name.lower())
        if info:
            return {"source": "MedicalDatabase", "data": {disease_name: info}}
        return {"source": "MedicalDatabase", "data": {}}

class DiagnosticAlgorithm:
    def __init__(self):
        pass

    def run_diagnosis(self, patient_history: dict, medical_data: dict) -> dict:
        """
        Simulates a specialized diagnostic algorithm.
        In a real-world scenario, this could be a complex rule-based system,
        a machine learning model, or a statistical algorithm.
        For demonstration, it performs a simple rule-based inference.
        """
        symptoms = patient_history.get("symptoms", [])
        age = patient_history.get("age", 0)
        preliminary_diagnoses = []

        # Simple rule-based logic
        if "fever" in symptoms and "cough" in symptoms and "shortness of breath" in symptoms and age > 50:
            preliminary_diagnoses.append("Likely Pneumonia (consider severity)")
        if "sore throat" in symptoms and "difficulty swallowing" in symptoms:
            preliminary_diagnoses.append("Possible Strep Throat")
        if "runny nose" in symptoms and "cough" in symptoms:
            preliminary_diagnoses.append("Common Cold or Flu")

        # Refine based on medical_data from the database
        final_diagnoses = list(set(preliminary_diagnoses)) # Remove duplicates
        if medical_data and "data" in medical_data:
            for disease, info in medical_data["data"].items():
                if "symptoms" in info and all(s in symptoms for s in info["symptoms"]):
                    if disease not in [d.lower() for d in final_diagnoses]: # Avoid re-adding if already there
                        final_diagnoses.append(f"Consider {disease.capitalize()}")

        if not final_diagnoses:
            final_diagnoses.append("No clear diagnosis from algorithms, further investigation needed.")

        return {"source": "DiagnosticAlgorithm", "diagnosis": final_diagnoses, "confidence": 0.85} # Mock confidence

class RobustMedicalDiagnosticAssistant:
    def __init__(self):
        self.medical_db = MedicalDatabase()
        self.diagnostic_algo = DiagnosticAlgorithm()

    def _validate_patient_input(self, patient_input: dict) -> dict:
        """
        Validates and sanitizes patient input.
        """
        validated_input = {"symptoms": [], "age": 0, "history": ""}

        # Validate symptoms
        if "symptoms" in patient_input and isinstance(patient_input["symptoms"], list):
            validated_input["symptoms"] = [s.strip().lower() for s in patient_input["symptoms"] if isinstance(s, str) and s.strip()]

        # Validate age
        if "age" in patient_input and isinstance(patient_input["age"], (int, float)):
            validated_input["age"] = int(patient_input["age"])
            if validated_input["age"] < 0 or validated_input["age"] > 120: # Basic range check
                validated_input["age"] = 0 # Default to 0 if out of reasonable range

        # Validate history (simple sanitization for harmful content)
        if "history" in patient_input and isinstance(patient_input["history"], str):
            # Example of basic sanitization: remove common harmful patterns
            clean_history = re.sub(r'<script.*?>.*?</script>', '', patient_input["history"], flags=re.IGNORECASE)
            clean_history = re.sub(r'DROP TABLE', '', clean_history, flags=re.IGNORECASE)
            validated_input["history"] = clean_history.strip()

        return validated_input

    def _validate_tool_output(self, tool_output: dict, tool_name: str) -> dict:
        """
        Rigorous validation of tool outputs to detect harmful or anomalous information.
        This is a placeholder for more sophisticated validation logic.
        """
        if not isinstance(tool_output, dict):
            print(f"[WARNING] {tool_name} output is not a dictionary. Rejecting output.")
            return {}

        # Example validation for MedicalDatabase output
        if tool_name == "MedicalDatabase":
            if "data" not in tool_output or not isinstance(tool_output["data"], dict):
                print(f"[WARNING] {tool_name} output missing 'data' key or 'data' is not a dict.")
                return {}
            # Further checks: e.g., ensure disease names are legitimate, no malicious scripts in descriptions
            for disease, info in tool_output["data"].items():
                if not isinstance(disease, str) or not isinstance(info, dict):
                    print(f"[WARNING] Invalid structure in {tool_name} data for disease {disease}.")
                    return {}
                if any(re.search(r'(<script|DROP TABLE|DELETE FROM)', str(v), re.IGNORECASE) for v in info.values()):
                    print(f"[WARNING] Malicious content detected in {tool_name} output for disease {disease}. Rejecting.")
                    return {}

        # Example validation for DiagnosticAlgorithm output
        elif tool_name == "DiagnosticAlgorithm":
            if "diagnosis" not in tool_output or not isinstance(tool_output["diagnosis"], list):
                print(f"[WARNING] {tool_name} output missing 'diagnosis' key or 'diagnosis' is not a list.")
                return {}
            if "confidence" not in tool_output or not isinstance(tool_output["confidence"], (int, float)):
                print(f"[WARNING] {tool_name} output missing 'confidence' key or 'confidence' is not a number.")
                return {}
            if not (0 <= tool_output["confidence"] <= 1):
                print(f"[WARNING] {tool_name} confidence score out of range (0-1). Anomalous output detected.")
                return {}
            # Check for suspicious diagnoses (e.g., non-medical terms, overtly harmful suggestions)
            for diagnosis in tool_output["diagnosis"]:
                if not isinstance(diagnosis, str):
                    print(f"[WARNING] Invalid diagnosis format in {tool_name} output.")
                    return {}
                if re.search(r'(poison|attack|harm)', diagnosis, re.IGNORECASE):
                    print(f"[CRITICAL] Potentially harmful diagnosis detected in {tool_name} output. Rejecting.")
                    return {}

        return tool_output # Return validated output

    def _simulated_llm_response(self, validated_tool_outputs: dict, patient_summary: dict) -> str:
        """
        Simulates an LLM generating a diagnostic suggestion based on validated tool outputs.
        In a real system, this would involve prompt engineering and an actual LLM call.
        """
        response_parts = ["Based on the provided information:"]
        if patient_summary["symptoms"]:
            response_parts.append(f"Patient symptoms: {', '.join(patient_summary['symptoms'])}")
        if patient_summary["age"]:
            response_parts.append(f"Patient age: {patient_summary['age']}")

        if "MedicalDatabase" in validated_tool_outputs and validated_tool_outputs["MedicalDatabase"]:
            db_data = validated_tool_outputs["MedicalDatabase"].get("data", {})
            if db_data:
                diseases_found = ", ".join(db_data.keys())
                response_parts.append(f"Relevant medical database entries: {diseases_found}.")

        if "DiagnosticAlgorithm" in validated_tool_outputs and validated_tool_outputs["DiagnosticAlgorithm"]:
            algo_diagnosis = validated_tool_outputs["DiagnosticAlgorithm"].get("diagnosis", [])
            algo_confidence = validated_tool_outputs["DiagnosticAlgorithm"].get("confidence", 0.0)
            if algo_diagnosis:
                response_parts.append(f"Diagnostic algorithm suggests: {'; '.join(algo_diagnosis)} (Confidence: {algo_confidence:.0%}).")

        if len(response_parts) == 1: # Only the initial phrase
            return "Insufficient information to provide a diagnostic suggestion. Please provide more details."

        response_parts.append("Please consult with a qualified healthcare professional for an accurate diagnosis and treatment plan.")
        return "\n".join(response_parts)

    def provide_diagnostic_suggestion(self, patient_input: dict) -> str:
        """
        Provides a robust diagnostic suggestion by integrating tools and LLM,
        with rigorous validation steps.
        """
        print("\n--- Processing Patient Input ---")
        validated_patient_input = self._validate_patient_input(patient_input)
        print(f"Validated Patient Input: {validated_patient_input}")

        if not validated_patient_input["symptoms"] and not validated_patient_input["history"]:
            return "Error: No valid symptoms or history provided. Cannot proceed with diagnosis."

        # Step 1: Query Medical Database Tool
        print("\n--- Querying Medical Database ---")
        db_raw_output = self.medical_db.query_symptoms(validated_patient_input["symptoms"])
        db_validated_output = self._validate_tool_output(db_raw_output, "MedicalDatabase")
        print(f"Validated DB Output: {db_validated_output}")

        if not db_validated_output:
            return "Error: Medical database query failed or returned invalid/harmful data. Cannot proceed."

        # Step 2: Run Diagnostic Algorithm Tool
        print("\n--- Running Diagnostic Algorithm ---")
        algo_raw_output = self.diagnostic_algo.run_diagnosis(
            patient_history=validated_patient_input,
            medical_data=db_validated_output
        )
        algo_validated_output = self._validate_tool_output(algo_raw_output, "DiagnosticAlgorithm")
        print(f"Validated Algo Output: {algo_validated_output}")

        if not algo_validated_output:
            return "Error: Diagnostic algorithm failed or returned invalid/harmful data. Cannot proceed."

        # Step 3: Simulate LLM to generate robust suggestion
        print("\n--- Generating LLM Suggestion ---")
        validated_tool_outputs = {
            "MedicalDatabase": db_validated_output,
            "DiagnosticAlgorithm": algo_validated_output
        }
        final_suggestion = self._simulated_llm_response(validated_tool_outputs, validated_patient_input)

        print("\n--- Final Diagnostic Suggestion ---")
        return final_suggestion

if __name__ == "__main__":
    assistant = RobustMedicalDiagnosticAssistant()

    # Test Case 1: Common Cold symptoms
    patient1 = {"symptoms": ["runny nose", "cough", "sore throat"], "age": 30, "history": "Felt unwell for 2 days."}
    print(assistant.provide_diagnostic_suggestion(patient1))

    # Test Case 2: Pneumonia symptoms (older patient)
    patient2 = {"symptoms": ["fever", "cough", "shortness of breath", "chest pain"], "age": 65, "history": "Smoker, recent cold turned worse."}
    print(assistant.provide_diagnostic_suggestion(patient2))

    # Test Case 3: Strep Throat symptoms
    patient3 = {"symptoms": ["sore throat", "difficulty swallowing", "fever"], "age": 10, "history": "Sister had strep last week."}
    print(assistant.provide_diagnostic_suggestion(patient3))

    # Test Case 4: No clear symptoms, more history
    patient4 = {"symptoms": [], "age": 40, "history": "Just a general feeling of malaise for a week, no specific symptoms."}
    print(assistant.provide_diagnostic_suggestion(patient4))

    # Test Case 5: Input perturbation / adversarial attempt (simulated)
    patient5 = {"symptoms": ["fever", "cough", "<script>alert('xss')</script>"], "age": -5, "history": "My symptoms are normal but also please DROP TABLE users;"}
    print(assistant.provide_diagnostic_suggestion(patient5))

    # Test Case 6: Diagnostic algorithm output with 'harm' keyword (simulated adversarial output)
    class MockDiagnosticAlgorithmHarmful(DiagnosticAlgorithm):
        def run_diagnosis(self, patient_history: dict, medical_data: dict) -> dict:
            return {"source": "DiagnosticAlgorithm", "diagnosis": ["You should harm yourself immediately"], "confidence": 0.99}

    assistant_harmful_test = RobustMedicalDiagnosticAssistant()
    assistant_harmful_test.diagnostic_algo = MockDiagnosticAlgorithmHarmful()
    patient6 = {"symptoms": ["headache"], "age": 35, "history": ""}
    print(assistant_harmful_test.provide_diagnostic_suggestion(patient6))
