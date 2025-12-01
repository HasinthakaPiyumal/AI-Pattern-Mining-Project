class PatientDataInput:
    def __init__(self, symptoms, medical_history, lab_results):
        self.symptoms = symptoms
        self.medical_history = medical_history
        self.lab_results = lab_results

    def get_patient_data(self):
        return {
            "symptoms": self.symptoms,
            "medical_history": self.medical_history,
            "lab_results": self.lab_results
        }

class CoTPromptGenerator:
    def generate_prompt(self, patient_data):
        symptoms = patient_data["symptoms"]
        medical_history = patient_data["medical_history"]
        lab_results = patient_data["lab_results"]

        prompt = f"""Generate a Chain-of-Thought reasoning process, a differential diagnosis, and a final diagnosis for a patient with the following information:

Symptoms: {symptoms}
Medical History: {medical_history}
Lab Results: {lab_results}

**Thought Process:**
1. Analyze Symptoms:
2. Consider Medical History:
3. Review Lab Results:
4. Formulate Differential Diagnosis:
5. Suggest Further Tests:
6. Synthesize and Conclude:

**Final Diagnosis:**
"""
        return prompt

class LLMSimulator:
    def get_llm_response(self, prompt):
        # This is a simulated LLM response based on the example in the architecture
        # In a real application, this would involve an actual LLM API call.
        if "Fever, cough, shortness of breath" in prompt and "Asthma, Smoker" in prompt and "Elevated D-dimer, low oxygen saturation" in prompt:
            return """**Thought Process:**
1. **Analyze Symptoms:** Fever, cough, and shortness of breath are common respiratory symptoms.
2. **Consider Medical History:** Asthma and smoking history increase risk for respiratory conditions, including exacerbations and lung damage.
3. **Review Lab Results:** Elevated D-dimer suggests a possible thrombotic event like Pulmonary Embolism (PE). Low oxygen saturation indicates impaired respiratory function.
4. **Formulate Differential Diagnosis:**
   * Pneumonia (viral or bacterial)
   * Asthma Exacerbation
   * Chronic Obstructive Pulmonary Disease (COPD) Exacerbation (given smoking history)
   * Pulmonary Embolism (PE) - strongly supported by elevated D-dimer and shortness of breath.
   * Heart Failure (less likely without specific cardiac symptoms/markers but possible).
5. **Suggest Further Tests:** CT Pulmonary Angiogram (CTPA) for PE, Chest X-ray/CT for pneumonia/COPD, ECG/Echocardiogram for cardiac issues, complete blood count, blood cultures.
6. **Synthesize and Conclude:** Given the acute onset of shortness of breath, fever, cough, and crucially, the elevated D-dimer and low oxygen saturation in a patient with a smoking history, Pulmonary Embolism is a high-probability diagnosis that requires urgent investigation. Pneumonia and exacerbation of underlying lung disease remain possibilities, but PE is a critical consideration.

**Final Diagnosis:** High suspicion for Pulmonary Embolism (PE)."""
        else:
            return "Simulated LLM response for: " + prompt

class OutputFormatter:
    def format_output(self, llm_response):
        return f"""--- MedThink Diagnostic Report ---

{llm_response}

---------------------------------------"""

class DiagnosticAssistant:
    def __init__(self):
        self.prompt_generator = CoTPromptGenerator()
        self.llm_simulator = LLMSimulator()
        self.output_formatter = OutputFormatter()

    def diagnose_patient(self, symptoms, medical_history, lab_results):
        patient_data_input = PatientDataInput(symptoms, medical_history, lab_results)
        patient_data = patient_data_input.get_patient_data()

        cot_prompt = self.prompt_generator.generate_prompt(patient_data)

        simulated_llm_response = self.llm_simulator.get_llm_response(cot_prompt)

        formatted_output = self.output_formatter.format_output(simulated_llm_response)

        return formatted_output

if __name__ == "__main__":
    assistant = DiagnosticAssistant()

    # Example Usage
    symptoms = "Fever, cough, shortness of breath"
    medical_history = "Asthma, Smoker"
    lab_results = "Elevated D-dimer, low oxygen saturation"

    report = assistant.diagnose_patient(symptoms, medical_history, lab_results)
    print(report)

    print("\n" + "="*50 + "\n")

    # Another example with different data to show generic simulation
    symptoms_2 = "Headache, nausea"
    medical_history_2 = "Migraines"
    lab_results_2 = "Normal blood work"

    report_2 = assistant.diagnose_patient(symptoms_2, medical_history_2, lab_results_2)
    print(report_2)
