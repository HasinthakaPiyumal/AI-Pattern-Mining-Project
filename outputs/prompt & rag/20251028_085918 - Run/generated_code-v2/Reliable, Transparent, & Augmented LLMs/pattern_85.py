from pydantic import BaseModel
import json

def _mock_llm_response(prompt: str, model_name: str) -> str:
    if "summarize" in prompt.lower():
        return "A detailed summary of the patient's visit, including symptoms, initial assessment, and treatment plan. The patient presented with a persistent cough and fever for 3 days. Diagnosed with acute bronchitis. Prescribed Azithromycin and advised rest." if model_name == "LLM1" else ""
    elif "extract" in prompt.lower():
        if "original report" in prompt.lower() and "llm1 summary" in prompt.lower():
            return (
                "Diagnoses: Acute bronchitis.\n" +
                "Medications: Azithromycin.\n" +
                "Allergies: None reported.\n" +
                "Follow-up Actions: Advised rest, follow-up in 7 days." if model_name == "LLM2" else ""
            )
    return "Mock LLM response."

def summarize_report_llm(report_text: str) -> str:
    prompt = f"Please summarize the following medical report in a comprehensive and freeform manner: {report_text}"
    return _mock_llm_response(prompt, "LLM1")

def extract_info_llm(original_report: str, llm1_summary: str) -> str:
    prompt = (
        f"Given the original medical report and its summary, extract the following key information:\n\n"
        f"Original Report: {original_report}\n\n"
        f"LLM1 Summary: {llm1_summary}\n\n"
        f"Diagnoses: (List all confirmed diagnoses)\n"
        f"Medications: (List all prescribed medications)\n"
        f"Allergies: (List all reported allergies or state 'None reported')\n"
        f"Follow-up Actions: (List all recommended follow-up actions)"
    )
    return _mock_llm_response(prompt, "LLM2")

class ExtractedMedicalInfo(BaseModel):
    diagnoses: list[str]
    medications: list[str]
    allergies: list[str]
    follow_up_actions: list[str]

class MedicalReportProcessor:
    def __init__(self):
        pass

    def process_report(self, report_text: str) -> tuple[str, ExtractedMedicalInfo]:
        summary = summarize_report_llm(report_text)
        extracted_raw = extract_info_llm(report_text, summary)

        # Parse the extracted_raw string into ExtractedMedicalInfo
        diagnoses = []
        medications = []
        allergies = []
        follow_up_actions = []

        for line in extracted_raw.split('\n'):
            if line.startswith("Diagnoses:"):
                diagnoses = [item.strip() for item in line.replace("Diagnoses:", "").split(',') if item.strip()]
            elif line.startswith("Medications:"):
                medications = [item.strip() for item in line.replace("Medications:", "").split(',') if item.strip()]
            elif line.startswith("Allergies:"):
                allergies = [item.strip() for item in line.replace("Allergies:", "").split(',') if item.strip()]
            elif line.startswith("Follow-up Actions:"):
                follow_up_actions = [item.strip() for item in line.replace("Follow-up Actions:", "").split(',') if item.strip()]

        extracted_info = ExtractedMedicalInfo(
            diagnoses=diagnoses,
            medications=medications,
            allergies=allergies,
            follow_up_actions=follow_up_actions
        )
        return summary, extracted_info

if __name__ == "__main__":
    processor = MedicalReportProcessor()
    sample_report = (
        "Patient presented to the emergency department with a chief complaint of persistent cough and mild fever for the past three days. "
        "Physical examination revealed rhonchi in both lung fields. Chest X-ray showed peribronchial thickening consistent with bronchitis. "
        "Patient denies any known drug allergies. Prescribed Azithromycin 500mg once daily for 5 days. Advised to rest and increase fluid intake. "
        "Follow-up with primary care physician in one week or sooner if symptoms worsen."
    )

    summary, extracted_data = processor.process_report(sample_report)

    print("\n--- Original Report ---")
    print(sample_report)
    print("\n--- LLM1 Summary ---")
    print(summary)
    print("\n--- Extracted Key Information (LLM2) ---")
    print(f"Diagnoses: {extracted_data.diagnoses}")
    print(f"Medications: {extracted_data.medications}")
    print(f"Allergies: {extracted_data.allergies}")
    print(f"Follow-up Actions: {extracted_data.follow_up_actions}")