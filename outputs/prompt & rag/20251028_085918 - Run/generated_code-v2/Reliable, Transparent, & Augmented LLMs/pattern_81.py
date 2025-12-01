import streamlit as st
import random

def get_llm_diagnosis_with_confidence(symptoms, history, test_results):
    diagnosis = "No clear diagnosis based on provided information."
    confidence = 0.5

    symptoms_lower = symptoms.lower()
    history_lower = history.lower()
    test_results_lower = test_results.lower()

    if "fever" in symptoms_lower and "cough" in symptoms_lower and "fatigue" in symptoms_lower:
        diagnosis = "Possible Influenza."
        confidence = random.uniform(0.7, 0.95)
    elif "headache" in symptoms_lower and "nausea" in symptoms_lower and "light sensitivity" in symptoms_lower:
        diagnosis = "Suspected Migraine."
        confidence = random.uniform(0.6, 0.85)
    elif "chest pain" in symptoms_lower and "shortness of breath" in symptoms_lower:
        diagnosis = "Cardiovascular issue requires urgent attention."
        confidence = random.uniform(0.8, 0.98)
    elif "abdominal pain" in symptoms_lower and "vomiting" in symptoms_lower:
        diagnosis = "Gastroenteritis."
        confidence = random.uniform(0.65, 0.9)
    elif "elevated crp" in test_results_lower and "joint pain" in symptoms_lower:
        diagnosis = "Inflammatory condition."
        confidence = random.uniform(0.75, 0.92)
    elif "diabetes" in history_lower and "high blood sugar" in test_results_lower:
        diagnosis = "Diabetes management review needed."
        confidence = random.uniform(0.85, 0.99)

    return diagnosis, confidence

st.set_page_config(layout="centered", page_title="Medical Diagnostic Assistant")
st.title("Medical Diagnostic Assistant with Confidence Scoring")
st.write("Enter patient information to get a simulated diagnosis with a confidence score.")

# Input Fields
symptoms_input = st.text_area("Patient Symptoms (e.g., fever, cough, headache)", height=150)
history_input = st.text_area("Medical History (e.g., diabetes, hypertension, allergies)", height=150)
test_results_input = st.text_area("Key Test Results (e.g., elevated CRP, normal blood count)", height=150)

# Action Button
if st.button("Get Diagnosis"):
    if not symptoms_input and not history_input and not test_results_input:
        st.warning("Please enter some patient information to get a diagnosis.")
    else:
        with st.spinner("Generating diagnosis..."):
            diagnosis, confidence = get_llm_diagnosis_with_confidence(symptoms_input, history_input, test_results_input)
            
            st.subheader("Diagnosis Result:")
            st.success(f"**Diagnosis:** {diagnosis}")
            st.info(f"**Confidence Score:** {confidence:.2f} (0.0 = Low, 1.0 = High)")

            if confidence < 0.7:
                st.warning("\n*\n*\nConsider further investigation or human expert consultation due to lower confidence.\n*\n*")
            elif confidence >= 0.9:
                st.success("\n*\n*\nHigh confidence, but always verify with clinical judgment.\n*\n*")

st.markdown("""
---
*Disclaimer: This is a simulated medical diagnostic assistant for demonstration purposes only.
It should NOT be used for actual medical advice or diagnosis. Always consult a qualified medical professional.*
""")