import streamlit as st
import time

def get_initial_diagnosis(symptoms):
    """Simulates a quick, initial LLM diagnosis."""
    if "fever" in symptoms.lower() and "cough" in symptoms.lower():
        return "Initial thought: Common cold or Flu."
    elif "chest pain" in symptoms.lower() and "shortness of breath" in symptoms.lower():
        return "Initial thought: Possible cardiac issue or respiratory infection."
    elif "headache" in symptoms.lower() and "stiff neck" in symptoms.lower():
        return "Initial thought: Migraine or tension headache. Consider meningitis if severe."
    else:
        return "Initial thought: Symptoms are general. More information needed for a precise immediate diagnosis."

def get_refined_diagnosis(symptoms):
    """Simulates a more thorough, refined LLM diagnosis with a delay."""
    time.sleep(4)  # Simulate complex processing
    if "fever" in symptoms.lower() and "cough" in symptoms.lower() and "fatigue" in symptoms.lower():
        return "Refined Diagnosis: Likely Viral Respiratory Infection (e.g., Influenza A or B). Recommend rest, hydration, and OTC symptom management. Consider testing if symptoms worsen or patient is high-risk."
    elif "chest pain" in symptoms.lower() and "shortness of breath" in symptoms.lower() and "left arm pain" in symptoms.lower():
        return "Refined Diagnosis: High suspicion for Acute Myocardial Infarction. Immediate cardiac workup (ECG, Troponins) and emergency referral required."
    elif "headache" in symptoms.lower() and "stiff neck" in symptoms.lower() and "photophobia" in symptoms.lower():
        return "Refined Diagnosis: Strong indication for Meningitis. Immediate lumbar puncture and broad-spectrum antibiotics are critical. Emergency department referral."
    elif "abdominal pain" in symptoms.lower() and "nausea" in symptoms.lower() and "fever" in symptoms.lower() and "right lower quadrant" in symptoms.lower():
        return "Refined Diagnosis: Highly suggestive of Appendicitis. Surgical consultation and imaging (CT scan) recommended immediately."
    else:
        return "Refined Diagnosis: After thorough analysis, the symptoms are still broad. Further diagnostic tests (e.g., blood work, imaging) or specialist consultation may be necessary for a definitive diagnosis."

st.set_page_config(page_title="Medical Diagnostic AI Assistant", layout="centered")
st.title("🩺 Medical Diagnostic AI Assistant")
st.markdown("Enter patient symptoms below to receive an immediate preliminary diagnosis and the option for a more refined analysis.")

symptoms_input = st.text_area("Enter patient symptoms here:", height=150, placeholder="e.g., fever, cough, fatigue, chest pain...")

if symptoms_input:
    st.subheader("Immediate Preliminary Diagnosis")
    initial_diagnosis = get_initial_diagnosis(symptoms_input)
    st.info(initial_diagnosis)

    st.write("--- ")
    st.subheader("Refined Diagnosis")
    st.write("Performing a more thorough analysis, cross-referencing against a broader medical knowledge base and recent research...")

    if st.button("Get Refined Diagnosis"): 
        with st.spinner("Please wait, generating a more accurate diagnosis..."):
            refined_diagnosis = get_refined_diagnosis(symptoms_input)
            st.success("Refined Diagnosis Ready!")
            st.write(refined_diagnosis)
    else:
        st.write("Click 'Get Refined Diagnosis' to wait for a more evidence-backed analysis.")
else:
    st.info("Please enter symptoms to get a diagnosis.")