import streamlit as st
import re

def mock_llm(prompt: str) -> str:
    if "severe headache, fever, and stiff neck" in prompt.lower():
        return "Meningitis (Confidence: 9/10), Encephalitis (Confidence: 7/10), Severe Flu (Confidence: 6/10)"
    elif "chest pain, shortness of breath" in prompt.lower():
        return "Myocardial Infarction (Confidence: 8/10), Angina (Confidence: 7/10), Panic Attack (Confidence: 5/10)"
    elif "sore throat, cough, runny nose" in prompt.lower():
        return "Common Cold (Confidence: 9/10), Allergic Rhinitis (Confidence: 7/10), Bronchitis (Confidence: 6/10)"
    else:
        return "Generic Illness (Confidence: 5/10), Unspecified Condition (Confidence: 4/10)"

def parse_llm_output(llm_output: str) -> list:
    diagnoses = []
    # Regex to find patterns like 'Diagnosis Name (Confidence: X/10)'
    pattern = re.compile(r"([A-Za-z0-9 ]+)\(Confidence: (\d)/10\)")
    matches = pattern.finditer(llm_output)
    for match in matches:
        diagnosis_name = match.group(1).strip()
        confidence_score = match.group(2)
        diagnoses.append({"diagnosis": diagnosis_name, "confidence": int(confidence_score)})
    return diagnoses

st.title("Medical Diagnosis Assistant")
st.markdown("Enter patient symptoms and medical history to get potential diagnoses with confidence scores.")

patient_symptoms = st.text_area("Enter patient symptoms and medical history here:", height=150)

if st.button("Get Diagnosis"):
    if patient_symptoms:
        # Constructing the prompt adhering to the ChatPromptTemplate intent
        prompt = f"Based on the following symptoms: {patient_symptoms}, what are the most likely diagnoses? For each diagnosis, provide your confidence level from 1 to 10. Example: 'Diagnosis A (Confidence: 9/10), Diagnosis B (Confidence: 7/10)'."

        st.subheader("Generating Diagnoses...")

        # Invoke the mock LLM
        llm_response = mock_llm(prompt)

        st.subheader("Suggested Diagnoses:")
        parsed_diagnoses = parse_llm_output(llm_response)

        if parsed_diagnoses:
            for diag in parsed_diagnoses:
                st.write(f"- **{diag['diagnosis']}** (Confidence: {diag['confidence']}/10)")
        else:
            st.write("No specific diagnoses found. Please try rephrasing the symptoms.")
    else:
        st.warning("Please enter patient symptoms to get a diagnosis.")