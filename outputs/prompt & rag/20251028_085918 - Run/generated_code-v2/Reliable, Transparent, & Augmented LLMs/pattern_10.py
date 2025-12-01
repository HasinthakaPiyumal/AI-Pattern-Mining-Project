import re

def mock_llm_call(prompt, llm_type):
    if llm_type == "primary":
        # Simulate a detailed diagnostic report generation
        if "shortness of breath" in prompt.lower() and "chest pain" in prompt.lower():
            return (
                "***Diagnostic Report***\n\n"\
                "Patient presents with acute onset shortness of breath and chest pain, particularly on exertion. "\
                "Medical history includes hypertension and a family history of cardiovascular disease. "\
                "Initial assessment suggests a cardiac event or severe respiratory distress.\n\n"\
                "Potential Conditions:\n"\
                "- Myocardial Infarction (Heart Attack)\n"\
                "- Pulmonary Embolism\n"\
                "- Acute Bronchitis/Pneumonia\n\n"\
                "Reasoning:\n"\
                "The combination of chest pain and shortness of breath, especially with cardiac risk factors, "\
                "strongly points towards an urgent cardiac investigation. Pulmonary embolism is also a serious "\
                "consideration given the symptom presentation. Less likely, but still possible, are severe "\
                "respiratory infections.\n\n"\
                "Suggested Next Steps:\n"\
                "Immediate EKG and cardiac enzyme tests. Chest X-ray and D-dimer test. "\
                "Consultation with a cardiologist is highly recommended. Monitor vital signs closely." 
            )
        elif "fever" in prompt.lower() and "sore throat" in prompt.lower():
            return (
                "***Diagnostic Report***\n\n"\
                "Patient reports a persistent fever for 3 days, accompanied by a sore throat and general malaise. "\
                "No significant past medical history. Physical examination reveals swollen tonsils.\n\n"\
                "Potential Conditions:\n"\
                "- Viral Pharyngitis (Common Cold)\n"\
                "- Bacterial Pharyngitis (Strep Throat)\n"\
                "- Influenza\n\n"\
                "Reasoning:\n"\
                "Symptoms are typical of an upper respiratory tract infection. The presence of swollen tonsils "\
                "could indicate bacterial infection, requiring a strep test. Viral causes are generally more common.\n\n"\
                "Suggested Next Steps:\n"\
                "Perform a rapid strep test. Advise rest, hydration, and over-the-counter pain relievers. "\
                "If strep test is positive, prescribe antibiotics. Follow up in 3-5 days if symptoms worsen or persist." 
            )
        else:
            return (
                "***Diagnostic Report***\n\n"\
                "Insufficient information to provide a detailed diagnosis. Please provide more symptoms and medical history." 
            )
    elif llm_type == "extractor":
        diagnosis = "N/A"
        urgency_level = "N/A"

        # Extract diagnosis
        diagnosis_match = re.search(r"The most probable diagnosis is: ([^\n]+)", prompt)
        if diagnosis_match:
            diagnosis = diagnosis_match.group(1).strip()
        else:
            # Fallback if trigger is not exactly matched, try to infer from report
            if "myocardial infarction" in prompt.lower() or "heart attack" in prompt.lower():
                diagnosis = "Myocardial Infarction (Heart Attack)"
            elif "pulmonary embolism" in prompt.lower():
                diagnosis = "Pulmonary Embolism"
            elif "viral pharyngitis" in prompt.lower():
                diagnosis = "Viral Pharyngitis"
            elif "bacterial pharyngitis" in prompt.lower():
                diagnosis = "Bacterial Pharyngitis"

        # Extract urgency level
        urgency_match = re.search(r"The urgency level is: ([^\n]+)", prompt)
        if urgency_match:
            urgency_level = urgency_match.group(1).strip()
        else:
            # Fallback if trigger is not exactly matched, try to infer from report
            if "immediate ekg" in prompt.lower() or "urgent cardiac investigation" in prompt.lower():
                urgency_level = "Immediate attention required"
            elif "consultation with a cardiologist is highly recommended" in prompt.lower():
                urgency_level = "Immediate attention required"
            elif "follow up in 3-5 days" in prompt.lower() or "advise rest, hydration" in prompt.lower():
                urgency_level = "Routine follow-up"

        return {"diagnosis": diagnosis, "urgency_level": urgency_level}
    return None

def run_medical_diagnostic_assistant(symptoms, medical_history):
    # 1. Primary LLM: Generate Diagnostic Report
    primary_llm_prompt = (
        f"As a medical professional, analyze the following patient information and generate a comprehensive diagnostic report:\n\n"
        f"Symptoms: {symptoms}\n"
        f"Medical History: {medical_history}\n\n"
        f"Please include potential conditions, reasoning, and suggested next steps."
    )
    diagnostic_report = mock_llm_call(primary_llm_prompt, "primary")
    print("\n--- Primary LLM Diagnostic Report ---")
    print(diagnostic_report)

    # 2. Extractor LLM: Extract Diagnosis and Urgency
    extractor_llm_prompt = (
        f"Based on the following diagnostic report, please extract the most probable diagnosis and the urgency level.\n\n"
        f"Diagnostic Report:\n{diagnostic_report}\n\n"
        f"The most probable diagnosis is: "
        f"The urgency level is: "
    )
    extracted_info = mock_llm_call(extractor_llm_prompt, "extractor")
    print("\n--- Extractor LLM Output ---")
    print(f"Extracted Diagnosis: {extracted_info['diagnosis']}")
    print(f"Extracted Urgency Level: {extracted_info['urgency_level']}")

    return extracted_info

if __name__ == "__main__":
    # Example 1: Urgent Case
    print("\n===== Running Example 1 (Urgent Case) =====")
    symptoms_1 = "Severe chest pain, shortness of breath, radiating pain to left arm."
    medical_history_1 = "Hypertension, family history of heart disease, 60 years old."
    run_medical_diagnostic_assistant(symptoms_1, medical_history_1)

    # Example 2: Non-Urgent Case
    print("\n===== Running Example 2 (Non-Urgent Case) =====")
    symptoms_2 = "Persistent fever, sore throat, fatigue, body aches."
    medical_history_2 = "No significant past medical history, 30 years old."
    run_medical_diagnostic_assistant(symptoms_2, medical_history_2)

    # Example 3: Insufficient Information
    print("\n===== Running Example 3 (Insufficient Information) =====")
    symptoms_3 = "Feeling unwell."
    medical_history_3 = ""
    run_medical_diagnostic_assistant(symptoms_3, medical_history_3)