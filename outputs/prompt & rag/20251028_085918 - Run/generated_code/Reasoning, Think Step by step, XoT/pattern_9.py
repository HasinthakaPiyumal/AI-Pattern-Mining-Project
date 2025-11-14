from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from collections import Counter

# Placeholder for a real LLM, using ChatOpenAI as an example.
# In a real application, you would configure this with your API key
# and potentially a fine-tuned model.
class MockLLM:
    def invoke(self, prompt_template):
        # Simulate LLM response with Chain-of-Thought reasoning
        # For demonstration purposes, this is hardcoded.
        # In a real scenario, the LLM would generate this based on the prompt.
        if "fever" in prompt_template.lower() and "cough" in prompt_template.lower():
            return (
                "Thought: Patient presents with fever and cough, common symptoms for respiratory infections. "
                "Consider viral infections like influenza or common cold, and bacterial infections like pneumonia. "
                "No severe symptoms like difficulty breathing or chest pain reported, which would suggest a more severe condition. "
                "Given the general symptoms, a common cold or mild influenza is highly probable. "
                "Action: Recommend rest and hydration. If symptoms worsen, advise further medical consultation. "
                "Diagnosis: Common Cold/Mild Influenza"
            )
        elif "headache" in prompt_template.lower() and "stiff neck" in prompt_template.lower():
            return (
                "Thought: Headache and stiff neck are concerning symptoms, potentially indicative of meningitis. "
                "Other possibilities include severe tension headache or neck strain, but meningitis requires immediate exclusion. "
                "Action: Advise immediate medical attention for further investigation, including lumbar puncture if indicated. "
                "Diagnosis: Suspected Meningitis (requires urgent medical evaluation)"
            )
        elif "abdominal pain" in prompt_template.lower() and "nausea" in prompt_template.lower():
            return (
                "Thought: Abdominal pain and nausea are non-specific symptoms. "
                "Could be gastroenteritis, food poisoning, or more serious conditions like appendicitis depending on location and severity. "
                "Without more specific details (e.g., pain location, duration, fever), it's hard to pinpoint. "
                "Action: Monitor symptoms, advise bland diet and hydration. If pain localizes or worsens, seek medical advice. "
                "Diagnosis: Gastroenteritis/Indigestion"
            )
        else:
            return (
                "Thought: Insufficient information to provide a precise diagnosis. "
                "The symptoms are too general. "
                "Action: Request more detailed patient information and specific symptoms. "
                "Diagnosis: Undetermined (more info needed)"
            )


class MedicalDiagnosticAssistant:
    def __init__(self, llm_model=None, temperature=0.7, num_consistency_checks=3):
        # Initialize the LLM. Use MockLLM if no specific model is provided.
        self.llm = llm_model if llm_model else MockLLM()
        self.temperature = temperature
        self.num_consistency_checks = num_consistency_checks

        # Prompt template incorporating Chain-of-Thought
        self.cot_prompt_template = PromptTemplate.from_template(
            """
            As a highly experienced medical diagnostic AI, analyze the following patient data and provide a step-by-step reasoning process (Chain-of-Thought) leading to a potential diagnosis.
            Then, provide a recommended action plan.

            Patient Data:
            Symptoms: {symptoms}
            Medical History: {medical_history}
            Lab Results: {lab_results}
            Imaging Data: {imaging_data}

            Follow this structure:
            Thought: [Detailed step-by-step reasoning process]
            Action: [Recommended action plan based on the diagnosis]
            Diagnosis: [Most likely diagnosis]
            """
        )

    def _parse_llm_output(self, output):
        # Extract Thought, Action, and Diagnosis from LLM output
        thought = "N/A"
        action = "N/A"
        diagnosis = "N/A"

        lines = output.split('\n')
        for line in lines:
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                action = line[len("Action:"):].strip()
            elif line.startswith("Diagnosis:"):
                diagnosis = line[len("Diagnosis:"):].strip()
        return {"thought": thought, "action": action, "diagnosis": diagnosis}

    def _verify_diagnosis(self, diagnosis_result):
        # Simulate a verification step, e.g., against a knowledge base or guidelines.
        # In a real system, this would involve querying a vector DB (e.g., Chroma) or
        # a structured medical ontology.
        known_conditions = [
            "Common Cold/Mild Influenza",
            "Suspected Meningitis (requires urgent medical evaluation)",
            "Gastroenteritis/Indigestion",
            "Appendicitis",
            "Pneumonia",
            "Migraine",
            "Undetermined (more info needed)"
        ]
        if diagnosis_result["diagnosis"] in known_conditions:
            diagnosis_result["verified"] = True
            diagnosis_result["verification_note"] = "Diagnosis found in known medical conditions list."
        else:
            diagnosis_result["verified"] = False
            diagnosis_result["verification_note"] = "Diagnosis not directly verified against common conditions. Further expert review recommended."
        return diagnosis_result

    def diagnose_patient(self, symptoms, medical_history="None", lab_results="None", imaging_data="None"):
        input_data = {
            "symptoms": symptoms,
            "medical_history": medical_history,
            "lab_results": lab_results,
            "imaging_data": imaging_data,
        }

        full_prompt = self.cot_prompt_template.format(**input_data)

        # Implement Self-Consistency: Query the LLM multiple times
        all_results = []
        for _ in range(self.num_consistency_checks):
            llm_output = self.llm.invoke(full_prompt) # Use temperature for variance if using a real LLM
            parsed_output = self._parse_llm_output(llm_output)
            all_results.append(parsed_output)

        # Aggregate results for self-consistency (e.g., majority vote for diagnosis)
        diagnoses = [res["diagnosis"] for res in all_results]
        most_common_diagnosis = Counter(diagnoses).most_common(1)[0][0] if diagnoses else "Undetermined"

        # Find the full result (thought and action) corresponding to the most common diagnosis
        final_result = None
        for res in all_results:
            if res["diagnosis"] == most_common_diagnosis:
                final_result = res
                break
        if not final_result:
            final_result = {"thought": "No consistent reasoning found.", "action": "Consult a medical professional.", "diagnosis": most_common_diagnosis}

        final_result = self._verify_diagnosis(final_result)

        return {
            "most_consistent_diagnosis": final_result["diagnosis"],
            "reasoning": final_result["thought"],
            "recommended_action": final_result["action"],
            "verification": {
                "status": final_result["verified"],
                "note": final_result["verification_note"]
            },
            "all_raw_llm_outputs": [output["diagnosis"] for output in all_results] # For transparency
        }

# Example Usage (for demonstration)
if __name__ == "__main__":
    # You can pass a real ChatOpenAI instance here if you have an API key
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    # assistant = MedicalDiagnosticAssistant(llm_model=llm, num_consistency_checks=5)

    # Using the MockLLM for demonstration without an API key
    assistant = MedicalDiagnosticAssistant(num_consistency_checks=3)

    print("\n--- Patient 1: Fever and Cough ---")
    patient_data1 = {
        "symptoms": "Patient presents with a persistent fever (101°F/38.3°C) and a dry cough for 3 days. Reports mild fatigue.",
        "medical_history": "No significant medical history.",
        "lab_results": "None available.",
        "imaging_data": "None available."
    }
    diagnosis1 = assistant.diagnose_patient(**patient_data1)
    print(f"Most Consistent Diagnosis: {diagnosis1['most_consistent_diagnosis']}")
    print(f"Reasoning: {diagnosis1['reasoning']}")
    print(f"Recommended Action: {diagnosis1['recommended_action']}")
    print(f"Verification Status: {diagnosis1['verification']['status']} - {diagnosis1['verification']['note']}")
    # print(f"All LLM Outputs: {diagnosis1['all_raw_llm_outputs']}")

    print("\n--- Patient 2: Severe Headache and Stiff Neck ---")
    patient_data2 = {
        "symptoms": "Patient reports sudden onset severe headache, stiff neck, and light sensitivity. Also feels nauseous.",
        "medical_history": "None relevant.",
        "lab_results": "None available.",
        "imaging_data": "None available."
    }
    diagnosis2 = assistant.diagnose_patient(**patient_data2)
    print(f"Most Consistent Diagnosis: {diagnosis2['most_consistent_diagnosis']}")
    print(f"Reasoning: {diagnosis2['reasoning']}")
    print(f"Recommended Action: {diagnosis2['recommended_action']}")
    print(f"Verification Status: {diagnosis2['verification']['status']} - {diagnosis2['verification']['note']}")

    print("\n--- Patient 3: General Abdominal Discomfort ---")
    patient_data3 = {
        "symptoms": "Patient has generalized mild abdominal pain and intermittent nausea for 24 hours.",
    }
    diagnosis3 = assistant.diagnose_patient(**patient_data3)
    print(f"Most Consistent Diagnosis: {diagnosis3['most_consistent_diagnosis']}")
    print(f"Reasoning: {diagnosis3['reasoning']}")
    print(f"Recommended Action: {diagnosis3['recommended_action']}")
    print(f"Verification Status: {diagnosis3['verification']['status']} - {diagnosis3['verification']['note']}")

    print("\n--- Patient 4: Vague Symptoms ---")
    patient_data4 = {
        "symptoms": "Patient feels generally unwell.",
    }
    diagnosis4 = assistant.diagnose_patient(**patient_data4)
    print(f"Most Consistent Diagnosis: {diagnosis4['most_consistent_diagnosis']}")
    print(f"Reasoning: {diagnosis4['reasoning']}")
    print(f"Recommended Action: {diagnosis4['recommended_action']}")
    print(f"Verification Status: {diagnosis4['verification']['status']} - {diagnosis4['verification']['note']}")
