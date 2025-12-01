import streamlit as st
import os
from pydantic import BaseModel, ValidationError
from typing import List, Optional

# Placeholder for OpenAI API key from environment variables
# from openai import OpenAI
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 1. Data Ingestion & Preprocessing Layer (with Pydantic validation)
class PatientData(BaseModel):
    symptoms: str
    medical_history: str
    lab_results: Optional[str] = None
    additional_notes: Optional[str] = None

# 2. Core Reasoning Engine (LLM with CoT Prompting)
class CoTReasoningEngine:
    def __init__(self, llm_model_name: str = "gpt-4"):
        self.llm_model_name = llm_model_name
        # self.client = client # Uncomment if using actual OpenAI client

    def _construct_cot_prompt(self, data: PatientData) -> str:
        prompt = f"""
        As an AI-powered diagnostic assistant, your task is to provide a differential diagnosis for a patient based on the provided information. You must use Chain-of-Thought reasoning, explicitly breaking down your analysis into clear, sequential steps before arriving at the final diagnosis and recommendations.

        Patient Information:
        Symptoms: {data.symptoms}
        Medical History: {data.medical_history}
        Lab Results: {data.lab_results if data.lab_results else 'N/A'}
        Additional Notes: {data.additional_notes if data.additional_notes else 'N/A'}

        Please follow these steps in your reasoning process:

        1.  **Symptom Analysis:** List and briefly analyze each key symptom, considering its potential implications and common associations.
        2.  **Medical History Correlation:** Evaluate how the patient's medical history (past conditions, family history, lifestyle) might influence the current symptoms and potential diagnoses.
        3.  **Lab Result Interpretation:** If provided, interpret any relevant lab results, highlighting abnormal values and their clinical significance.
        4.  **Synthesize Findings & Brainstorm Differential Diagnoses:** Combine insights from symptom analysis, medical history, and lab results. Brainstorm a list of possible conditions that fit the overall clinical picture. Consider common, uncommon, and emergent conditions.
        5.  **Reasoning for Each Diagnosis:** For each proposed differential diagnosis, articulate the logical steps and specific evidence from the patient data that supports it. Also, mention any conflicting information or reasons why a diagnosis might be less likely.
        6.  **Final Differential Diagnosis (Ranked):** Provide a ranked list of the most probable diagnoses.
        7.  **Next Steps/Recommendations:** Suggest appropriate further diagnostic tests, specialist consultations, or initial management strategies.

        Ensure your output is structured clearly with headings for each step.
        """
        return prompt

    def generate_diagnosis_with_cot(self, data: PatientData) -> str:
        cot_prompt = self._construct_cot_prompt(data)

        # Placeholder for LLM call. In a real application, you'd send this to an LLM API.
        # For demonstration, we'll simulate an LLM response.
        # response = self.client.chat.completions.create(
        #     model=self.llm_model_name,
        #     messages=[
        #         {"role": "system", "content": "You are a helpful medical diagnostic assistant."},
        #         {"role": "user", "content": cot_prompt}
        #     ]
        # )
        # return response.choices[0].message.content

        # --- SIMULATED LLM RESPONSE FOR DEMONSTRATION ---
        simulated_response = f"""
        1.  **Symptom Analysis:**
            *   **Fever (102°F):** Indicates an active inflammatory or infectious process.
            *   **Severe headache:** Common in many conditions, including infections (meningitis, sinusitis), migraines, or tension headaches. Severity suggests a more significant underlying cause.
            *   **Stiff neck:** A classic sign of meningeal irritation, highly suspicious for meningitis.
            *   **Photophobia:** Sensitivity to light, further supporting meningeal irritation or severe headache syndromes.
            *   **Rash (non-blanching petechial):** This is a critical symptom, strongly indicative of meningococcemia (sepsis due to Neisseria meningitidis) or other forms of vasculitis. Non-blanching suggests capillary damage/bleeding.

        2.  **Medical History Correlation:**
            *   **No significant past medical history:** Suggests an acute onset illness rather than an exacerbation of a chronic condition.
            *   **Contact with sick individuals:** Increases the likelihood of an infectious etiology, especially for highly transmissible pathogens.

        3.  **Lab Result Interpretation:** (Assuming some hypothetical lab results for demonstration)
            *   **Elevated WBC count (18,000/uL with neutrophil predominance):** Indicates bacterial infection.
            *   **Lumbar Puncture (hypothetical):** Increased CSF white blood cell count (predominantly neutrophils), low glucose, high protein would strongly point to bacterial meningitis.

        4.  **Synthesize Findings & Brainstorm Differential Diagnoses:**
            The combination of high fever, severe headache, stiff neck, photophobia, and especially a non-blanching petechial rash is a medical emergency. The primary concern is bacterial meningitis, potentially with meningococcemia.
            *   Bacterial Meningitis
            *   Viral Meningitis (less likely given rash and severity)
            *   Encephalitis
            *   Sepsis (meningococcemia)
            *   Subarachnoid Hemorrhage (less likely with fever and rash)

        5.  **Reasoning for Each Diagnosis:**
            *   **Bacterial Meningitis:** High likelihood. Supported by fever, severe headache, stiff neck, photophobia (signs of meningeal irritation), and the critical non-blanching rash (suggesting meningococcal etiology). Rapid onset and severity fit. Hypothetical lab results (if available) would confirm.
            *   **Viral Meningitis:** Possible, but less likely to present with such severe systemic symptoms and a non-blanching rash. Lab results would typically show lymphocytic predominance in CSF.
            *   **Meningococcemia (Sepsis):** Very high likelihood, often co-occurs with meningococcal meningitis. The petechial rash is the hallmark symptom, indicating disseminated intravascular coagulation (DIC) and capillary leakage due to bacterial toxins.

        6.  **Final Differential Diagnosis (Ranked):**
            1.  Bacterial Meningitis (most likely meningococcal meningitis)
            2.  Meningococcemia (often concomitant with #1, but considered separately as a severe systemic infection)
            3.  Other forms of Sepsis with CNS involvement

        7.  **Next Steps/Recommendations:**
            *   **Immediate Emergency Medical Evaluation:** This is a life-threatening condition requiring urgent attention.
            *   **Lumbar Puncture:** Essential for definitive diagnosis of meningitis (if not contraindicated).
            *   **Blood Cultures:** To identify the causative pathogen for sepsis.
            *   **Empiric Broad-Spectrum Antibiotics:** Administer immediately without waiting for LP results, especially given the petechial rash.
            *   **Corticosteroids:** May be considered in certain cases of bacterial meningitis.
            *   **Isolation Precautions:** For suspected meningococcal disease.
            *   **Close Monitoring:** For signs of increased intracranial pressure, shock, or organ failure.
        """
        return simulated_response
        # --- END SIMULATED LLM RESPONSE ---


# Streamlit User Interface Layer
st.set_page_config(layout="wide", page_title="Medi-Diagnose CoT Assistant")

st.title("🧠 Medi-Diagnose CoT Assistant")
st.markdown("An AI-powered diagnostic support system utilizing Chain-of-Thought reasoning for healthcare professionals. Enter patient data below to get a detailed differential diagnosis.")

# Input fields
st.header("Patient Information")
with st.form("patient_data_form"):
    symptoms_input = st.text_area("Symptoms (e.g., fever, headache, stiff neck, rash)", height=150)
    medical_history_input = st.text_area("Medical History (e.g., past conditions, allergies, family history)", height=150)
    lab_results_input = st.text_area("Lab Results (optional, e.g., WBC count, CRP, imaging findings)", height=100)
    additional_notes_input = st.text_area("Additional Notes (optional)", height=100)

    submitted = st.form_submit_button("Generate Diagnosis")

    if submitted:
        try:
            patient_data = PatientData(
                symptoms=symptoms_input,
                medical_history=medical_history_input,
                lab_results=lab_results_input if lab_results_input else None,
                additional_notes=additional_notes_input if additional_notes_input else None,
            )
            
            st.subheader("Processing Request...")
            
            # Initialize and run the CoT Reasoning Engine
            engine = CoTReasoningEngine()
            diagnosis_report = engine.generate_diagnosis_with_cot(patient_data)

            st.subheader("🧠 Chain-of-Thought Diagnostic Report")
            st.markdown(diagnosis_report)

            st.info("Disclaimer: This AI assistant is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")

        except ValidationError as e:
            st.error(f"Input Validation Error: {e.errors()}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

