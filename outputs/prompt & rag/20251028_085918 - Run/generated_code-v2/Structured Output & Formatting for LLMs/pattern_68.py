from pydantic import BaseModel
import json

class MedicalReport(BaseModel):
    patient_name: str
    dob: str
    diagnoses: list[str]
    medications: list[str]
    lab_results: dict[str, str]

def generate_llm_prompt(report_text: str) -> str:
    prompt = f"""Extract the following information from the medical report below and format it as a JSON object. Ensure the JSON conforms to the following schema:
{{
    "patient_name": "[Patient's Full Name]",
    "dob": "[Patient's Date of Birth - YYYY-MM-DD]",
    "diagnoses": ["[Diagnosis 1]", "[Diagnosis 2]", ...],
    "medications": ["[Medication 1]", "[Medication 2]", ...],
    "lab_results": {{
        "[Lab Test Name 1]": "[Result 1]",
        "[Lab Test Name 2]": "[Result 2]",
        ...
    }}
}}

Medical Report:
{report_text}

JSON Output:"""
    return prompt

def simulate_llm_response(prompt: str) -> str:
    # In a real application, this would call an actual LLM (e.g., OpenAI, Gemini)
    # For this demonstration, we return a hardcoded JSON string.
    # The content is designed to match the expected schema from the prompt.
    
    # Example of a structured JSON response from an LLM based on a medical report
    simulated_json_output = """{
    "patient_name": "John Doe",
    "dob": "1985-03-15",
    "diagnoses": [
        "Type 2 Diabetes Mellitus",
        "Hypertension"
    ],
    "medications": [
        "Metformin 500mg daily",
        "Lisinopril 10mg daily"
    ],
    "lab_results": {
        "HbA1c": "7.2%",
        "Blood Pressure": "140/90 mmHg",
        "Cholesterol": "200 mg/dL"
    }
} """
    return simulated_json_output

def main():
    sample_medical_report = """Patient Name: John Doe\nDate of Birth: 1985-03-15\nDiagnosis: The patient presents with Type 2 Diabetes Mellitus and Hypertension. \nMedications: Currently on Metformin 500mg daily and Lisinopril 10mg daily. \nLab Results: HbA1c was 7.2%. Blood pressure measured at 140/90 mmHg. Cholesterol levels are 200 mg/dL."""

    # 1. Generate the LLM prompt with instructions for structured output
    llm_prompt = generate_llm_prompt(sample_medical_report)
    print("\n--- Generated LLM Prompt ---")
    print(llm_prompt)

    # 2. Simulate LLM interaction to get structured JSON
    llm_raw_response = simulate_llm_response(llm_prompt)
    print("\n--- Simulated LLM Raw Response (JSON) ---")
    print(llm_raw_response)

    # 3. Parse the JSON output using Pydantic for validation and structured access
    try:
        parsed_report_data = json.loads(llm_raw_response)
        medical_report = MedicalReport(**parsed_report_data)
        print("\n--- Parsed Medical Report (Pydantic Model) ---")
        print(medical_report.model_dump_json(indent=2))
        print("\nPatient Name:", medical_report.patient_name)
        print("Diagnoses:", ", ".join(medical_report.diagnoses))
    except Exception as e:
        print(f"\nError parsing LLM response: {e}")
        print(f"Raw response was: {llm_raw_response}")

if __name__ == "__main__":
    main()