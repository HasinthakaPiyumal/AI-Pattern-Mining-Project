
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv() # Load environment variables from .env file

class MedicalDiagnosisSystem:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        """
        Initializes the MedicalDiagnosisSystem with an LLM.
        Args:
            model_name (str): The name of the LLM model to use (e.g., "gpt-4", "gpt-3.5-turbo").
            temperature (float): The sampling temperature for the LLM.
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.output_parser = StrOutputParser()

        # --- Prompt Templates for Chain-of-Thought Reasoning ---

        self.diagnosis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a highly experienced medical diagnostician. Based on the patient's symptoms and history, provide a comprehensive list of potential differential diagnoses. For each diagnosis, briefly mention why it's being considered. Do not suggest treatments yet. Think step-by-step."),
            ("user", "Patient Information:\nSymptoms: {symptoms}\nMedical History: {medical_history}\nLab Results: {lab_results}\n---\nList potential differential diagnoses and a brief rationale for each:")
        ])

        self.justification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical reasoning assistant. Given a set of potential diagnoses and patient information, provide detailed medical justifications for each diagnosis. Explain the clinical reasoning clearly, referencing symptoms, history, and lab results where relevant. Be thorough and analytical."),
            ("user", "Patient Information:\nSymptoms: {symptoms}\nMedical History: {medical_history}\nLab Results: {lab_results}\n---\nPotential Diagnoses: {diagnoses}\n---\nProvide detailed medical justifications for each of these potential diagnoses:")
        ])

        self.treatment_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical treatment planning assistant. For each confirmed or highly probable diagnosis, suggest an initial, general treatment plan. Include recommended next steps, further tests, or specialist referrals if appropriate. Emphasize that these are suggestions for medical professionals and not definitive prescriptions."),
            ("user", "Patient Information:\nSymptoms: {symptoms}\nMedical History: {medical_history}\nLab Results: {lab_results}\n---\nConfirmed/Probable Diagnoses with Justifications: {diagnoses_with_justifications}\n---\nSuggest initial treatment plans, next steps, and referrals for each:")
        ])

        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a critical medical verifier. Your task is to review a proposed set of differential diagnoses, their justifications, and treatment suggestions for a patient. Assess the consistency, medical soundness, and logical flow of the reasoning. Identify any inconsistencies, illogical leaps, or potential inaccuracies. Provide a concise summary of your findings and highlight any areas requiring further attention or revision. If everything appears sound, state that clearly."),
            ("user", "Patient Information:\nSymptoms: {symptoms}\nMedical History: {medical_history}\nLab Results: {lab_results}\n---\nProposed Diagnoses and Reasoning:\n{full_report}\n---\nPerform a medical verification of this report:")
        ])

        # --- LangChain Chains ---
        self.diagnosis_chain = self.diagnosis_prompt | self.llm | self.output_parser
        self.justification_chain = self.justification_prompt | self.llm | self.output_parser
        self.treatment_chain = self.treatment_prompt | self.llm | self.output_parser
        self.verification_chain = self.verification_prompt | self.llm | self.output_parser

    def run_diagnosis_pipeline(self, patient_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Runs the full diagnostic pipeline using Chain-of-Thought and verification.
        Args:
            patient_data (Dict[str, str]): A dictionary containing 'symptoms', 'medical_history', 'lab_results'.
        Returns:
            Dict[str, Any]: A dictionary containing raw outputs and a verified report.
        """
        symptoms = patient_data.get("symptoms", "")
        medical_history = patient_data.get("medical_history", "")
        lab_results = patient_data.get("lab_results", "")

        print("--- Step 1: Generating Differential Diagnoses (Chain-of-Thought) ---")
        diagnoses = self.diagnosis_chain.invoke({
            "symptoms": symptoms,
            "medical_history": medical_history,
            "lab_results": lab_results
        })
        print(f"\nPotential Diagnoses:\n{diagnoses}\n")

        print("--- Step 2: Generating Detailed Justifications ---")
        justifications = self.justification_chain.invoke({
            "symptoms": symptoms,
            "medical_history": medical_history,
            "lab_results": lab_results,
            "diagnoses": diagnoses
        })
        print(f"\nJustifications:\n{justifications}\n")

        full_diagnosis_report = f"Potential Diagnoses:\n{diagnoses}\n\nDetailed Justifications:\n{justifications}"

        print("--- Step 3: Suggesting Initial Treatment Plans ---")
        treatment_suggestions = self.treatment_chain.invoke({
            "symptoms": symptoms,
            "medical_history": medical_history,
            "lab_results": lab_results,
            "diagnoses_with_justifications": full_diagnosis_report
        })
        print(f"\nTreatment Suggestions:\n{treatment_suggestions}\n")

        full_report = (
            f"Patient Information:\nSymptoms: {symptoms}\nMedical History: {medical_history}\nLab Results: {lab_results}\n\n"
            f"Proposed Differential Diagnoses and Justifications:\n{justifications}\n\n"
            f"Suggested Treatment Plans:\n{treatment_suggestions}"
        )

        print("--- Step 4: Verifying the Entire Report (Self-Correction/Verification) ---")
        verification_result = self.verification_chain.invoke({
            "symptoms": symptoms,
            "medical_history": medical_history,
            "lab_results": lab_results,
            "full_report": full_report
        })
        print(f"\nVerification Result:\n{verification_result}\n")

        return {
            "diagnoses": diagnoses,
            "justifications": justifications,
            "treatment_suggestions": treatment_suggestions,
            "verification_result": verification_result,
            "full_report": full_report
        }

if __name__ == "__main__":
    # --- Mock Patient Data --- 
    # In a real system, this would come from an EMR, user input, or structured forms.
    mock_patient_data_1 = {
        "symptoms": "Fever (102°F) for 3 days, severe headache, stiff neck, sensitivity to light, nausea, vomiting.",
        "medical_history": "No significant past medical history. No recent travel or exposure to sick individuals mentioned.",
        "lab_results": "CSF analysis: elevated white blood cells (neutrophils predominant), low glucose, elevated protein. Blood cultures pending."
    }

    mock_patient_data_2 = {
        "symptoms": "Persistent cough for 4 weeks, shortness of breath on exertion, occasional wheezing, fatigue. Worsens at night.",
        "medical_history": "Smoker (1 pack/day for 20 years), family history of asthma.",
        "lab_results": "Chest X-ray shows hyperinflation. Pulmonary function tests show obstructive pattern, partially reversible with bronchodilator. Eosinophil count slightly elevated."
    }

    print("==== Running Diagnosis System for Patient 1 ====")
    system = MedicalDiagnosisSystem()
    results_1 = system.run_diagnosis_pipeline(mock_patient_data_1)

    print("\n\n==== Full Verified Report for Patient 1 ====")
    print(results_1["full_report"])
    print("\nVerification Summary:")
    print(results_1["verification_result"])

    print("\n\n==== Running Diagnosis System for Patient 2 ====")
    results_2 = system.run_diagnosis_pipeline(mock_patient_data_2)

    print("\n\n==== Full Verified Report for Patient 2 ====")
    print(results_2["full_report"])
    print("\nVerification Summary:")
    print(results_2["verification_result"])
