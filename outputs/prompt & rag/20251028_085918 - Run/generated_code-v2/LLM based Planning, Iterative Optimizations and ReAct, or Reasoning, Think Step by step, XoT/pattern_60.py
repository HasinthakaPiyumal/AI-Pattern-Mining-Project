
class MedicalDiagnosticAssistant:
    def __init__(self):
        self.diagnostic_steps = [
            "Identify potential organ systems involved based on symptoms.",
            "Suggest and interpret initial diagnostic tests.",
            "Integrate patient's medical history for further insights.",
            "Refine diagnosis and propose initial treatment options."
        ]
        self.conversation_history = []

    def _simulate_llm_response(self, prompt):
        if "Identify potential organ systems" in prompt:
            return "Based on the symptoms, consider respiratory and cardiovascular systems. Further investigation needed."
        elif "Suggest and interpret initial diagnostic tests" in prompt:
            return "Recommend Chest X-ray and ECG. Chest X-ray shows minor consolidation in lower left lobe. ECG is normal. This suggests a respiratory issue like pneumonia."
        elif "Integrate patient's medical history" in prompt:
            return "Patient has a history of asthma, which could exacerbate respiratory symptoms. Current medications are relevant."
        elif "Refine diagnosis and propose initial treatment options" in prompt:
            return "Revised diagnosis: Community-acquired pneumonia with asthma exacerbation. Recommended treatment: Antibiotics (e.g., Azithromycin), bronchodilators, and corticosteroids. Monitor oxygen saturation."
        return "Simulated LLM response: " + prompt[:50] + "..."

    def diagnose_patient(self, initial_symptoms, patient_history=None, test_results=None):
        self.conversation_history = [
            f"Initial Patient Symptoms: {initial_symptoms}"
        ]
        if patient_history:
            self.conversation_history.append(f"Patient Medical History: {patient_history}")
        if test_results:
            self.conversation_history.append(f"Initial Test Results: {test_results}")

        print("\n--- Starting Diagnostic Process ---")

        for step in self.diagnostic_steps:
            current_prompt_context = "\n".join(self.conversation_history)
            prompt = f"{current_prompt_context}\n\nNext Step: {step}\nProvide your analysis and recommendations based on the information so far."
            
            print(f"\n--- Prompting for: {step} ---")
            # Simulate LLM call
            llm_response = self._simulate_llm_response(prompt)
            
            print(f"LLM Response: {llm_response}")
            self.conversation_history.append(f"LLM Analysis for '{step}': {llm_response}")
        
        final_diagnosis_prompt = "\n".join(self.conversation_history) + "\n\nBased on all the above information, provide a final comprehensive diagnosis and detailed treatment plan."
        final_llm_response = self._simulate_llm_response(final_diagnosis_prompt)
        self.conversation_history.append(f"Final Diagnosis and Treatment: {final_llm_response}")

        print("\n--- Final Diagnosis and Treatment ---")
        print(final_llm_response)
        return {"final_diagnosis": final_llm_response, "full_conversation": self.conversation_history}

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()
    
    initial_symptoms = "Patient presents with cough, fever, and shortness of breath for 3 days."
    patient_history = "History of mild asthma. Non-smoker."
    test_results = "No initial test results available yet."

    diagnosis_output = assistant.diagnose_patient(initial_symptoms, patient_history, test_results)
    
    print("\n--- Full Conversation Log ---")
    for entry in diagnosis_output["full_conversation"]:
        print(entry)
