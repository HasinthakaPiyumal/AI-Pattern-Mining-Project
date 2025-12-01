
class MedicalReportProcessor:
    def __init__(self):
        pass

    def _call_primary_llm(self, report_text: str) -> str:
        # Simulated Primary LLM output - complex and verbose
        if "chest pain" in report_text.lower():
            return (
                "Patient presented with acute onset of chest pain radiating to the left arm, "
                "accompanied by dyspnea and diaphoresis. Initial ECG showed ST elevation in leads II, III, aVF. "
                "Cardiac markers are elevated. Differential diagnoses include Myocardial Infarction, Angina Pectoris, "
                "Pericarditis, and Aortic Dissection. Further investigations like echocardiogram and coronary angiography are recommended. "
                "The patient has a history of hypertension and hyperlipidemia. Current status is stable after initial stabilization. "
                "Aspirin and Nitroglycerin were administered. Consider a cardiology consult."
            )
        elif "fever and cough" in report_text.lower():
            return (
                "Patient reports persistent fever for 3 days, non-productive cough, and general malaise. "
                "No significant respiratory distress. Lung auscultation revealed diminished breath sounds in the right lower lobe. "
                "Chest X-ray shows patchy infiltrates consistent with pneumonia. Lab tests indicate elevated CRP. "
                "Consider bacterial pneumonia. Prescribed Azithromycin 500mg once daily for 5 days. "
                "Follow-up in 7 days or sooner if symptoms worsen. Advise rest and hydration."
            )
        else:
            return (
                "Generic detailed medical summary: Patient exhibits various symptoms requiring further analysis. "
                "Complex interplay of factors observed. Additional diagnostic tests are crucial for accurate assessment. "
                "A comprehensive overview of potential conditions is provided for specialist review."
            )

    def _call_secondary_llm(self, primary_output: str, extraction_prompt: str) -> str:
        # Simulated Secondary LLM output - extracts specific info based on trigger
        if "The primary diagnosis is:" in extraction_prompt:
            if "Myocardial Infarction" in primary_output:
                return "Myocardial Infarction"
            elif "bacterial pneumonia" in primary_output:
                return "Bacterial Pneumonia"
            else:
                return "Unspecified Diagnosis"
        elif "Medications:" in extraction_prompt:
            meds = []
            if "Aspirin" in primary_output: meds.append("Aspirin")
            if "Nitroglycerin" in primary_output: meds.append("Nitroglycerin")
            if "Azithromycin" in primary_output: meds.append("Azithromycin")
            return ", ".join(meds) if meds else "None listed"
        elif "Follow-up actions are:" in extraction_prompt:
            if "coronary angiography are recommended" in primary_output:
                return "Echocardiogram, Coronary Angiography, Cardiology Consult"
            elif "Follow-up in 7 days" in primary_output:
                return "Follow-up in 7 days or sooner if symptoms worsen"
            else:
                return "No specific follow-up actions listed"
        elif "A new referral is required (Yes/No):" in extraction_prompt:
            if "cardiology consult" in primary_output:
                return "Yes"
            else:
                return "No"
        return "Extraction Failed"

    def process_report(self, report_text: str) -> dict:
        primary_llm_output = self._call_primary_llm(report_text)

        diagnoses_prompt = "Extract the primary diagnosis from the following medical summary: " + primary_llm_output + ". The primary diagnosis is:"
        medications_prompt = "List all prescribed medications: " + primary_llm_output + ". Medications:"
        followup_prompt = "Identify the recommended follow-up actions: " + primary_llm_output + ". Follow-up actions are:"
        referral_prompt = "State if a new referral is required (Yes/No): " + primary_llm_output + ". A new referral is required (Yes/No):"

        primary_diagnosis = self._call_secondary_llm(primary_llm_output, diagnoses_prompt)
        prescribed_medications = self._call_secondary_llm(primary_llm_output, medications_prompt)
        follow_up_actions = self._call_secondary_llm(primary_llm_output, followup_prompt)
        new_referral_needed = self._call_secondary_llm(primary_llm_output, referral_prompt)

        return {
            "raw_report": report_text,
            "primary_llm_summary": primary_llm_output,
            "extracted_insights": {
                "primary_diagnosis": primary_diagnosis,
                "prescribed_medications": prescribed_medications,
                "follow_up_actions": follow_up_actions,
                "new_referral_needed": new_referral_needed,
            },
        }


if __name__ == "__main__":
    processor = MedicalReportProcessor()

    # Example 1: Chest Pain Report
    report_1 = "Patient reports severe chest pain radiating to left arm, shortness of breath. History of high blood pressure."
    insights_1 = processor.process_report(report_1)
    print("\n--- Report 1 Insights ---")
    print(f"Raw Report: {insights_1['raw_report']}")
    print(f"Primary LLM Summary: {insights_1['primary_llm_summary']}")
    print(f"Extracted Diagnosis: {insights_1['extracted_insights']['primary_diagnosis']}")
    print(f"Extracted Medications: {insights_1['extracted_insights']['prescribed_medications']}")
    print(f"Extracted Follow-up: {insights_1['extracted_insights']['follow_up_actions']}")
    print(f"New Referral Needed: {insights_1['extracted_insights']['new_referral_needed']}")

    # Example 2: Fever and Cough Report
    report_2 = "Patient presents with 3-day fever, dry cough, and feeling unwell. Diminished breath sounds on right side."
    insights_2 = processor.process_report(report_2)
    print("\n--- Report 2 Insights ---")
    print(f"Raw Report: {insights_2['raw_report']}")
    print(f"Primary LLM Summary: {insights_2['primary_llm_summary']}")
    print(f"Extracted Diagnosis: {insights_2['extracted_insights']['primary_diagnosis']}")
    print(f"Extracted Medications: {insights_2['extracted_insights']['prescribed_medications']}")
    print(f"Extracted Follow-up: {insights_2['extracted_insights']['follow_up_actions']}")
    print(f"New Referral Needed: {insights_2['extracted_insights']['new_referral_needed']}")

    # Example 3: Generic Report
    report_3 = "Patient has general discomfort and fatigue. No specific acute symptoms identified."
    insights_3 = processor.process_report(report_3)
    print("\n--- Report 3 Insights ---")
    print(f"Raw Report: {insights_3['raw_report']}")
    print(f"Primary LLM Summary: {insights_3['primary_llm_summary']}")
    print(f"Extracted Diagnosis: {insights_3['extracted_insights']['primary_diagnosis']}")
    print(f"Extracted Medications: {insights_3['extracted_insights']['prescribed_medications']}")
    print(f"Extracted Follow-up: {insights_3['extracted_insights']['follow_up_actions']}")
    print(f"New Referral Needed: {insights_3['extracted_insights']['new_referral_needed']}")
