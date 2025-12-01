class MedicalDiagnosisAssistant:

    def __init__(self, specialties: list[str]):
        self.specialties = specialties
        self.specialty_cot_prompts = {
            "infectious_diseases": "Perform a Chain-of-Thought reasoning for infectious diseases based on the following patient data: {patient_data}. Provide a detailed analysis of potential infections.",
            "cardiovascular": "Perform a Chain-of-Thought reasoning for cardiovascular issues based on the following patient data: {patient_data}. Analyze heart-related symptoms and history.",
            "neurological": "Perform a Chain-of-Thought reasoning for neurological conditions based on the following patient data: {patient_data}. Evaluate brain and nerve-related symptoms."
        }
        self.meta_reasoning_prompt = "Review the following diverse Chain-of-Thought reasonings from different medical specialties:\n\n{cot_outputs}\n\nBased on these, provide a consolidated final diagnosis, a confidence level (as a percentage), and recommendations for further investigation or treatment. Format your response as: \"DIAGNOSIS_START: [Diagnosis here] DIAGNOSIS_END. CONFIDENCE_START: [Confidence %] CONFIDENCE_END. RECOMMENDATIONS_START: [Recommendations here] RECOMMENDATIONS_END.\""

    def _mock_llm_call(self, prompt: str) -> str:
        if "DIAGNOSIS_START" in self.meta_reasoning_prompt and "CONFIDENCE_START" in self.meta_reasoning_prompt:
            # This is a meta-reasoning prompt
            diagnosis = "Uncertain; further investigation needed."
            confidence = "50"
            recommendations = "Consult with a general practitioner for comprehensive evaluation."

            if "fever" in prompt.lower() and "body aches" in prompt.lower() and "viral infection" in prompt.lower():
                diagnosis = "Likely Viral Infection"
                confidence = "75"
                recommendations = "Rest, hydration, and over-the-counter medication. Monitor symptoms."
            elif "chest pain" in prompt.lower() and "ECG normal" in prompt.lower() and "anxiety" in prompt.lower():
                diagnosis = "Possible Anxiety or Musculoskeletal Pain"
                confidence = "60"
                recommendations = "Stress management techniques, physical therapy, and follow-up if pain persists."
            elif "headache" in prompt.lower() and "no focal neurological deficits" in prompt.lower() and "tension headache" in prompt.lower():
                diagnosis = "Probable Tension Headache"
                confidence = "80"
                recommendations = "Pain relievers, stress reduction, and ensure adequate sleep."
            elif "chest pain" in prompt.lower() and "shortness of breath" in prompt.lower() and "cardiovascular issues" in prompt.lower() and "viral infection" not in prompt.lower():
                diagnosis = "Requires urgent cardiac evaluation"
                confidence = "90"
                recommendations = "Immediate referral to a cardiologist, perform ECG, cardiac enzymes, and imaging."

            return f"DIAGNOSIS_START: {diagnosis} DIAGNOSIS_END. CONFIDENCE_START: {confidence}% CONFIDENCE_END. RECOMMENDATIONS_START: {recommendations} RECOMMENDATIONS_END."

        else:
            # This is a specialized CoT prompt
            if "infectious diseases" in prompt:
                if "fever" in prompt.lower() and "body aches" in prompt.lower():
                    return "CoT for infectious diseases: Patient presents with classic symptoms of a viral infection, specifically influenza-like illness. No signs of severe bacterial infection currently."
                else:
                    return "CoT for infectious diseases: Initial assessment suggests no clear infectious pathology based on current symptoms. Further tests might be required if symptoms persist."
            elif "cardiovascular issues" in prompt:
                if "chest pain" in prompt.lower() and "ECG normal" in prompt.lower():
                    return "CoT for cardiovascular: ECG is within normal limits. Chest pain might be non-cardiac, possibly related to musculoskeletal strain or anxiety. No immediate signs of myocardial ischemia."
                elif "chest pain" in prompt.lower() and "shortness of breath" in prompt.lower():
                     return "CoT for cardiovascular: Patient reports chest pain and shortness of breath. This raises concern for acute coronary syndrome or other significant cardiac event. Further immediate cardiac workup is critical."
                else:
                    return "CoT for cardiovascular: No specific cardiovascular red flags based on provided data."
            elif "neurological conditions" in prompt:
                if "headache" in prompt.lower() and "no focal neurological deficits" in prompt.lower():
                    return "CoT for neurological: Patient experiences a headache without any focal neurological deficits or signs of increased intracranial pressure. Likely a primary headache disorder such as tension headache or migraine without aura."
                else:
                    return "CoT for neurological: No clear neurological red flags. If symptoms worsen, a more thorough neurological exam and imaging may be warranted."
            return f"Simulated reasoning for: {prompt}"

    def _generate_cot_reasoning(self, patient_data: str, specialty: str) -> str:
        prompt_template = self.specialty_cot_prompts.get(specialty)
        if not prompt_template:
            return f"No CoT prompt defined for specialty: {specialty}"
        formatted_prompt = prompt_template.format(patient_data=patient_data)
        return self._mock_llm_call(formatted_prompt)

    def _meta_reason_and_diagnose(self, cot_outputs: list[str]) -> dict:
        combined_cot_outputs = "\n\n".join(cot_outputs)
        formatted_prompt = self.meta_reasoning_prompt.format(cot_outputs=combined_cot_outputs)
        raw_llm_output = self._mock_llm_call(formatted_prompt)

        diagnosis = "N/A"
        confidence = "N/A"
        recommendations = "N/A"

        diagnosis_start_tag = "DIAGNOSIS_START: "
        diagnosis_end_tag = " DIAGNOSIS_END."
        confidence_start_tag = "CONFIDENCE_START: "
        confidence_end_tag = "% CONFIDENCE_END."
        recommendations_start_tag = "RECOMMENDATIONS_START: "
        recommendations_end_tag = " RECOMMENDATIONS_END."

        if diagnosis_start_tag in raw_llm_output and diagnosis_end_tag in raw_llm_output:
            diag_start_idx = raw_llm_output.find(diagnosis_start_tag) + len(diagnosis_start_tag)
            diag_end_idx = raw_llm_output.find(diagnosis_end_tag)
            diagnosis = raw_llm_output[diag_start_idx:diag_end_idx].strip()

        if confidence_start_tag in raw_llm_output and confidence_end_tag in raw_llm_output:
            conf_start_idx = raw_llm_output.find(confidence_start_tag) + len(confidence_start_tag)
            conf_end_idx = raw_llm_output.find(confidence_end_tag)
            confidence = raw_llm_output[conf_start_idx:conf_end_idx].strip()

        if recommendations_start_tag in raw_llm_output and recommendations_end_tag in raw_llm_output:
            rec_start_idx = raw_llm_output.find(recommendations_start_tag) + len(recommendations_start_tag)
            rec_end_idx = raw_llm_output.find(recommendations_end_tag)
            recommendations = raw_llm_output[rec_start_idx:rec_end_idx].strip()

        return {"diagnosis": diagnosis, "confidence": f"{confidence}%", "recommendations": recommendations}

    def diagnose_patient(self, patient_data: str) -> dict:
        all_cot_outputs = []
        for specialty in self.specialties:
            cot_reasoning = self._generate_cot_reasoning(patient_data, specialty)
            all_cot_outputs.append(cot_reasoning)

        final_diagnosis = self._meta_reason_and_diagnose(all_cot_outputs)
        return final_diagnosis

if __name__ == "__main__":
    assistant = MedicalDiagnosisAssistant(specialties=["infectious_diseases", "cardiovascular", "neurological"])

    patient_data_1 = "Patient reports sudden onset of fever, severe body aches, and fatigue for 2 days. No cough or sore throat. History of seasonal flu."
    diagnosis_1 = assistant.diagnose_patient(patient_data_1)
    print("\n--- Patient 1 Diagnosis ---")
    print(f"Patient Data: {patient_data_1}")
    print(f"Final Diagnosis: {diagnosis_1}")

    patient_data_2 = "Patient complains of recurrent mild chest pain, often after exercise, and occasional shortness of breath. ECG from 6 months ago was normal. Family history of heart disease."
    diagnosis_2 = assistant.diagnose_patient(patient_data_2)
    print("\n--- Patient 2 Diagnosis ---")
    print(f"Patient Data: {patient_data_2}")
    print(f"Final Diagnosis: {diagnosis_2}")

    patient_data_3 = "Patient presents with a persistent headache for a week, localized to the temples, worsened by stress. No visual disturbances, numbness, or weakness. Has a history of migraines."
    diagnosis_3 = assistant.diagnose_patient(patient_data_3)
    print("\n--- Patient 3 Diagnosis ---")
    print(f"Patient Data: {patient_data_3}")
    print(f"Final Diagnosis: {diagnosis_3}")

    patient_data_4 = "Patient reports severe, crushing chest pain radiating to the left arm, accompanied by profuse sweating and dizziness, started 30 minutes ago."
    diagnosis_4 = assistant.diagnose_patient(patient_data_4)
    print("\n--- Patient 4 Diagnosis ---")
    print(f"Patient Data: {patient_data_4}")
    print(f"Final Diagnosis: {diagnosis_4}")