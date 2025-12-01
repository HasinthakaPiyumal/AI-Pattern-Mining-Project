medical_knowledge_base = {
    "Common Cold": ["runny nose", "sore throat", "cough", "sneezing", "mild headache"],
    "Influenza (Flu)": ["fever", "body aches", "fatigue", "cough", "sore throat", "headache"],
    "Strep Throat": ["sore throat", "difficulty swallowing", "fever", "red spots on roof of mouth"],
    "Migraine": ["severe headache", "nausea", "sensitivity to light", "sensitivity to sound"],
    "Allergies": ["sneezing", "itchy eyes", "runny nose", "congestion"],
    "Pneumonia": ["cough", "fever", "shortness of breath", "chest pain"],
    "Bronchitis": ["cough", "mucus production", "fatigue", "shortness of breath"]
}

class MedicalDiagnosisAssistant:
    def __init__(self, medical_kb, max_retrieval_iterations=3):
        self.medical_kb = medical_kb
        self.max_retrieval_iterations = max_retrieval_iterations

    def _retrieve_info(self, query):
        query_lower = query.lower()
        relevant_context = []
        identified_conditions = set()

        for condition, symptoms in self.medical_kb.items():
            condition_lower = condition.lower()
            if condition_lower in query_lower:
                relevant_context.append(f"Potential condition: {condition}.")
                identified_conditions.add(condition)
            for symptom in symptoms:
                if symptom in query_lower:
                    if condition not in identified_conditions:
                        relevant_context.append(f"Symptoms matching {condition}: {symptom}.")
                    identified_conditions.add(condition)

        return relevant_context, list(identified_conditions)

    def _evaluate_context(self, context, initial_symptoms):
        sufficient = False
        confidence = 0.0
        reason = "Insufficient context."

        covered_symptoms_count = 0
        for symptom in initial_symptoms:
            if any(symptom.lower() in c.lower() for c in context):
                covered_symptoms_count += 1
        
        symptom_coverage = covered_symptoms_count / len(initial_symptoms) if initial_symptoms else 0
        
        potential_conditions_in_context = set()
        for cond_name in self.medical_kb.keys():
            if any(cond_name.lower() in c.lower() for c in context):
                potential_conditions_in_context.add(cond_name)
        
        condition_identified = len(potential_conditions_in_context) > 0

        if symptom_coverage >= 0.7 and condition_identified:
            sufficient = True
            confidence = 0.8 + (symptom_coverage * 0.2)
            reason = "Context appears sufficient with good symptom coverage and identified conditions."
        elif symptom_coverage >= 0.4 and condition_identified:
            sufficient = False  # Not fully sufficient, but promising
            confidence = 0.5 + (symptom_coverage * 0.2)
            reason = "Context partially sufficient, more information might help refine diagnosis."
        elif condition_identified:
            sufficient = False
            confidence = 0.4
            reason = "A condition was identified, but symptom coverage is low. More details needed."
        else:
            sufficient = False
            confidence = 0.2
            reason = "Context insufficient. No clear conditions or low symptom coverage."
            
        return {"sufficient": sufficient, "confidence": min(1.0, confidence), "reason": reason}

    def _refine_query(self, original_query, current_iteration, initial_symptoms):
        if current_iteration == 1:
            return original_query + ", please describe the intensity or duration of symptoms."
        elif current_iteration == 2:
            return original_query + ", are you experiencing any fever, body aches, or specific pain?"
        else:
            return original_query + ", any other relevant medical history or recent exposures?"

    def _generate_diagnosis(self, context, initial_symptoms):
        possible_conditions = set()
        for cond_name, symptoms in self.medical_kb.items():
            if any(cond_name.lower() in c.lower() for c in context):
                possible_conditions.add(cond_name)
            for symptom in initial_symptoms:
                if symptom.lower() in [s.lower() for s in symptoms]:
                    possible_conditions.add(cond_name)
        
        if possible_conditions:
            primary_candidate = list(possible_conditions)[0] # Simplified selection
            return f"Based on the information, a possible diagnosis is: {primary_candidate}. Please consult a medical professional for a definitive diagnosis."
        else:
            return "Despite comprehensive retrieval, a clear diagnosis cannot be formulated based on the provided symptoms and knowledge base. Please consult a medical professional."

    def diagnose(self, patient_symptoms_list):
        current_query = ", ".join(patient_symptoms_list)
        initial_symptoms_set = set([s.lower() for s in patient_symptoms_list])

        for i in range(self.max_retrieval_iterations):
            retrieved_context, identified_conditions = self._retrieve_info(current_query)
            
            evaluation_result = self._evaluate_context(retrieved_context, patient_symptoms_list)

            print(f"Iteration {i+1}:")
            print(f"  Query: {current_query}")
            print(f"  Retrieved Context: {retrieved_context}")
            print(f"  Evaluation: {evaluation_result}")

            if evaluation_result["sufficient"] and evaluation_result["confidence"] >= 0.75:
                diagnosis = self._generate_diagnosis(retrieved_context, patient_symptoms_list)
                return default_api.CodeGenOutput(
                    filename="medical_diagnosis_assistant_output.txt",
                    code=diagnosis,
                    explanation="Generated a diagnosis after iterative retrieval and confident evaluation."
                )
            elif i < self.max_retrieval_iterations - 1 and evaluation_result["confidence"] < 0.75:
                current_query = self._refine_query(current_query, i + 1, patient_symptoms_list)
                print("  Refining query for next iteration...")
            else:
                break # No more iterations or confidence too low to continue

        abstain_message = "After several attempts, the assistant cannot confidently make a diagnosis based on the provided information. It is recommended to consult a medical professional for further examination."
        return default_api.CodeGenOutput(
            filename="medical_diagnosis_assistant_output.txt",
            code=abstain_message,
            explanation="Abstained from diagnosis due to insufficient confidence or context after all iterations."
        )

# Example Usage:
if __name__ == "__main__":
    assistant = MedicalDiagnosisAssistant(medical_knowledge_base, max_retrieval_iterations=3)

    # Test Case 1: Sufficient context after initial retrieval
    print("\n--- Test Case 1: Sufficient context after initial retrieval ---")
    symptoms1 = ["runny nose", "sore throat", "cough"]
    result1 = assistant.diagnose(symptoms1)
    print(result1["code"])
    print(result1["explanation"])

    # Test Case 2: Requires refinement
    print("\n--- Test Case 2: Requires refinement ---")
    symptoms2 = ["headache"]
    result2 = assistant.diagnose(symptoms2)
    print(result2["code"])
    print(result2["explanation"])

    # Test Case 3: Leads to abstention
    print("\n--- Test Case 3: Leads to abstention ---")
    symptoms3 = ["foot pain", "dizziness"]
    result3 = assistant.diagnose(symptoms3)
    print(result3["code"])
    print(result3["explanation"])

    # Test Case 4: Another sufficient context
    print("\n--- Test Case 4: Another sufficient context ---")
    symptoms4 = ["fever", "body aches", "fatigue"]
    result4 = assistant.diagnose(symptoms4)
    print(result4["code"])
    print(result4["explanation"])

    # Test Case 5: Partial match, then refinement
    print("\n--- Test Case 5: Partial match, then refinement ---")
    symptoms5 = ["sore throat"]
    result5 = assistant.diagnose(symptoms5)
    print(result5["code"])
    print(result5["explanation"])
