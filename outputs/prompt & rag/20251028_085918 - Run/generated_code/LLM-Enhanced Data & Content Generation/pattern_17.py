import os
from dotenv import load_dotenv
from llm_client import LLMClient
from data_synthesizer import generate_synthetic_patient_data
from diagnosis_engine import DiagnosisEngine
from treatment_planner import TreatmentPlanner

def main():
    # Load environment variables (e.g., OPENAI_API_KEY)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables. Please set it.")
        return

    # Initialize LLM client
    llm_client = LLMClient(api_key=api_key)

    print("\n--- Generating Synthetic Patient Data ---")
    synthetic_data_list = generate_synthetic_patient_data(
        llm_client=llm_client,
        disease_context="a pediatric patient with neurological symptoms suggestive of a rare genetic disorder",
        num_samples=1
    )
    if synthetic_data_list:
        patient_data = synthetic_data_list[0]
        print("\nGenerated Synthetic Patient Data:")
        print(patient_data)
    else:
        print("Failed to generate synthetic patient data.")
        return

    print("\n--- Diagnosing Patient ---")
    diagnosis_engine = DiagnosisEngine(llm_client=llm_client)
    diagnosis_output = diagnosis_engine.diagnose_patient(patient_data)
    print("\nProposed Diagnosis:")
    print(diagnosis_output)

    print("\n--- Generating Treatment Plan and Patient Education ---")
    treatment_planner = TreatmentPlanner(llm_client=llm_client)

    treatment_plan = treatment_planner.generate_treatment_plan(
        diagnosis=diagnosis_output,
        patient_profile=patient_data
    )
    print("\nPersonalized Treatment Plan:")
    print(treatment_plan)

    patient_education = treatment_planner.generate_patient_education_material(
        diagnosis=diagnosis_output,
        patient_profile=patient_data
    )
    print("\nPatient Education Material:")
    print(patient_education)

if __name__ == "__main__":
    main()