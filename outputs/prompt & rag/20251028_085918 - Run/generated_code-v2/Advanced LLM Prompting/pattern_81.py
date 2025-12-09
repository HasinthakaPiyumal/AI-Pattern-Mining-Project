
class MedicalDiagnosticAssistant:
    def __init__(self):
        pass

    def _construct_base_prompt(self, patient_data: dict, medical_query: str) -> str:
        """Constructs the initial prompt from patient data and medical query."""
        prompt_parts = [
            "Analyze the following patient information to provide a differential diagnosis and suggest next steps."
        ]
        prompt_parts.append(f"Patient Symptoms: {patient_data.get('symptoms', 'N/A')}")
        prompt_parts.append(f"Medical History: {patient_data.get('history', 'N/A')}")
        prompt_parts.append(f"Lab Results: {patient_data.get('lab_results', 'N/A')}")
        prompt_parts.append(f"Current Query: {medical_query}")

        return "\n".join(prompt_parts)

    def _simulate_llm_response(self, prompt: str) -> str:
        """Simulates an LLM's response based on the given prompt."""
        print("\n--- LLM Received Prompt (for demonstration of RE2 pattern) ---")
        print(prompt)
        print("----------------------------------------------------------\n")

        if "Read the question again" in prompt:
            # Simulate a more thorough reasoning for complex cases
            if "complex respiratory distress" in prompt.lower() and "Read the question again" in prompt:
                return (
                    "Simulated Diagnosis (RE2 enhanced): Considering the complex respiratory distress, "
                    "re-evaluation suggests a higher probability of atypical pneumonia or acute exacerbation of COPD. "
                    "Recommend immediate chest CT and arterial blood gas analysis."
                )
            return (
                "Simulated Diagnosis (RE2 enhanced): After re-processing, the diagnostic possibilities "
                "are refined. Further investigation into [specific area] is recommended. "
                "Consider [specific test]."
            )
        else:
            return (
                "Simulated Diagnosis (Standard): Based on the initial review, "
                "consider [common condition]. Further tests may include [standard test]."
            )

    def diagnose_patient(self, patient_data: dict, medical_query: str) -> str:
        """Diagnoses a patient using the RE2 pattern for complex queries."""
        print(f"\nDoctor's Query: {medical_query}")

        base_prompt = self._construct_base_prompt(patient_data, medical_query)

        # Implement the Rereading (RE2) pattern
        # Prepend 'Read the question again' and repeat the original full query context
        re2_enhanced_prompt = f"Read the question again. \n{base_prompt}"

        print("Applying Rereading (RE2) pattern for enhanced comprehension...")
        diagnosis = self._simulate_llm_response(re2_enhanced_prompt)

        return diagnosis

# --- Example Usage ---
if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    # Example 1: A complex case where RE2 should ideally improve reasoning
    complex_patient_data = {
        "symptoms": "Severe shortness of breath, persistent cough with frothy sputum, chest pain radiating to the back, low oxygen saturation (SpO2 88%), recent travel history to endemic area.",
        "history": "Smoker for 20 years, controlled hypertension, no prior respiratory issues, recent viral infection.",
        "lab_results": "Elevated D-dimer, normal troponin, slightly elevated WBC count, chest X-ray shows bilateral patchy infiltrates."
    }
    complex_query = "Given this patient's presentation and history, what is the most likely differential diagnosis for complex respiratory distress and what immediate investigations are crucial?"

    print("\n--- Complex Case Diagnosis ---")
    complex_diagnosis = assistant.diagnose_patient(complex_patient_data, complex_query)
    print(f"Final Diagnosis: {complex_diagnosis}")

    print("\n" + "="*80 + "\n")

    # Example 2: A simpler case (though RE2 is still applied per the pattern's logic)
    simple_patient_data = {
        "symptoms": "Mild fever, sore throat, runny nose, body aches.",
        "history": "No significant medical history, vaccinated for flu.",
        "lab_results": "Rapid Strep test negative, CBC normal."
    }
    simple_query = "What is the most probable cause of these symptoms and recommended treatment?"

    print("\n--- Simple Case Diagnosis ---")
    simple_diagnosis = assistant.diagnose_patient(simple_patient_data, simple_query)
    print(f"Final Diagnosis: {simple_diagnosis}")
