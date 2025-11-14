
import streamlit as st

# --- 1. Mock Data Management ---

mock_drugs_db = {
    "DrugA": {"interactions": ["DrugB", "DrugC"], "severity": "High"},
    "DrugB": {"interactions": ["DrugA"], "severity": "High"},
    "DrugC": {"interactions": [], "severity": "Low"},
    "Aspirin": {"interactions": ["Warfarin"], "severity": "High"},
    "Warfarin": {"interactions": ["Aspirin", "NSAIDs"], "severity": "High"},
    "Ibuprofen": {"interactions": ["Warfarin"], "severity": "Moderate"}
}

mock_icd10_codes = {
    "Common Cold": "J00",
    "Influenza": "J11.1",
    "Pneumonia": "J18.9",
    "Hypertension": "I10",
    "Diabetes Type 2": "E11.9",
    "Fractured Arm": "S52.90XA"
}

mock_medical_literature = {
    "hypertension": "Recent studies suggest lifestyle modifications are crucial for hypertension management. Refer to ACC/AHA guidelines.",
    "diabetes": "Metformin is a first-line treatment for Type 2 Diabetes. New research explores SGLT2 inhibitors and GLP-1 receptor agonists.",
    "pneumonia": "Antibiotic choice for pneumonia depends on causative agent. Consider local resistance patterns.",
    "fracture": "Treatment for bone fractures involves immobilization, and sometimes surgery. Healing time varies by bone and severity."
}

# --- 2. Specialized Tools (Simulated) ---

def drug_interaction_checker(drugs: list) -> dict:
    """Simulates checking for drug interactions."""
    st.info(f"Tool: Checking for interactions among: {", ".join(drugs)}")
    results = {}
    for i, drug1 in enumerate(drugs):
        drug1_info = mock_drugs_db.get(drug1, {"interactions": [], "severity": "None"})
        for j, drug2 in enumerate(drugs):
            if i != j and drug2 in drug1_info["interactions"]:
                key = tuple(sorted((drug1, drug2)))
                if key not in results:
                    results[key] = f"Interaction between {drug1} and {drug2} (Severity: {drug1_info["severity"]})."
    return {"interactions": list(results.values()) if results else ["No significant interactions found."]}

def medical_imaging_analysis(image_description: str) -> dict:
    """Simulates medical imaging analysis based on text description."""
    st.info(f"Tool: Analyzing image description: \"{image_description}\"")
    image_description_lower = image_description.lower()
    if "fracture" in image_description_lower or "broken bone" in image_description_lower:
        return {"findings": "Evidence of a fracture detected. Further examination needed.", "confidence": "High"}
    elif "pneumonia" in image_description_lower or "lung opacity" in image_description_lower:
        return {"findings": "Suspicion of pneumonia. Recommend follow-up with chest X-ray.", "confidence": "Medium"}
    elif "clear" in image_description_lower or "normal" in image_description_lower:
        return {"findings": "No significant abnormalities detected.", "confidence": "High"}
    return {"findings": "Unable to determine specific findings from description.", "confidence": "Low"}

def medical_literature_search(keywords: list) -> dict:
    """Simulates searching medical literature based on keywords."""
    st.info(f"Tool: Searching medical literature for keywords: {", ".join(keywords)}")
    results = []
    for keyword in keywords:
        summary = mock_medical_literature.get(keyword.lower())
        if summary:
            results.append(f"Relevant Literature for \"{keyword}\": {summary}")
    return {"literature": results if results else ["No specific literature found for the given keywords in mock database."]}

def diagnostic_code_generator(condition: str) -> dict:
    """Simulates generating ICD-10 codes for a given condition."""
    st.info(f"Tool: Generating diagnostic code for condition: \"{condition}\"")
    code = mock_icd10_codes.get(condition.strip())
    return {"icd10_code": code if code else "U07.1 (COVID-19, unspecified)"} # Default for unrecognized or general cases

# --- 3. Core LLM Orchestrator (Simulated) ---

def llm_orchestrator(
    patient_symptoms: str,
    medical_history: str,
    uploaded_image_description: str,
    prescribed_drugs: list
) -> str:
    """
    Simulates an LLM orchestrating specialized tools based on patient information.
    In a real scenario, this would involve prompt engineering and dynamic tool calling
    via a framework like Langchain.
    """
    st.subheader("LLM Orchestrator Activity (Simulated):")
    full_context = f"{patient_symptoms} {medical_history}"
    recommendations = []

    # --- Tool Orchestration Logic (Simplified/Keyword-based) ---

    # 1. Drug Interaction Checker
    if prescribed_drugs:
        st.markdown("**Invoking Drug Interaction Checker...**")
        drug_results = drug_interaction_checker(prescribed_drugs)
        for interaction in drug_results["interactions"]:
            recommendations.append(f"- Drug Interaction Alert: {interaction}")
    else:
        st.info("No drugs provided for interaction check.")

    # 2. Medical Imaging Analysis Tool
    if uploaded_image_description and uploaded_image_description.strip() not in ["", "N/A"]:
        st.markdown("**Invoking Medical Imaging Analysis Tool...**")
        imaging_results = medical_imaging_analysis(uploaded_image_description)
        recommendations.append(f"- Imaging Findings: {imaging_results["findings"]} (Confidence: {imaging_results["confidence"]})")
    else:
        st.info("No image description provided for analysis.")

    # 3. Medical Literature Search Engine
    literature_keywords = []
    if "fever" in patient_symptoms.lower() or "cough" in patient_symptoms.lower():
        literature_keywords.append("pneumonia")
    if "blood pressure" in medical_history.lower() or "hypertension" in medical_history.lower():
        literature_keywords.append("hypertension")
    if "sugar" in medical_history.lower() or "diabetes" in medical_history.lower():
        literature_keywords.append("diabetes")
    if "pain" in patient_symptoms.lower() and ("arm" in patient_symptoms.lower() or "leg" in patient_symptoms.lower()):
        literature_keywords.append("fracture")

    if literature_keywords:
        st.markdown("**Invoking Medical Literature Search Engine...**")
        literature_results = medical_literature_search(list(set(literature_keywords))) # Use set to avoid duplicates
        for lit in literature_results["literature"]:
            recommendations.append(f"- Literature Review: {lit}")
    else:
        st.info("No specific keywords identified for literature search.")

    # 4. Diagnostic Code Generator (after potential diagnosis formed)
    # This is a highly simplified step. A real LLM would deduce a diagnosis first.
    potential_diagnosis = "Common Cold" # Default for demonstration
    if "fever" in patient_symptoms.lower() and "cough" in patient_symptoms.lower() and "shortness of breath" in patient_symptoms.lower():
        potential_diagnosis = "Pneumonia"
    elif "high blood pressure" in medical_history.lower():
        potential_diagnosis = "Hypertension"
    elif "high blood sugar" in medical_history.lower():
        potential_diagnosis = "Diabetes Type 2"
    elif "broken arm" in patient_symptoms.lower() or "arm pain and swelling" in patient_symptoms.lower():
        potential_diagnosis = "Fractured Arm"

    st.markdown(f"**Potential Diagnosis (LLM inferred): {potential_diagnosis}**")
    st.markdown("**Invoking Diagnostic Code Generator...**")
    icd_code_results = diagnostic_code_generator(potential_diagnosis)
    recommendations.append(f"- Suggested ICD-10 Code: {icd_code_results["icd10_code"]} for {potential_diagnosis}")

    # --- Synthesize Final Assessment ---
    final_assessment = f"""
**Comprehensive Diagnostic Assessment:**

Based on the provided patient information, medical history, and augmented tool analyses, here is a synthesized assessment:

**Patient Profile:**
- Symptoms: {patient_symptoms if patient_symptoms else "N/A"}
- Medical History: {medical_history if medical_history else "N/A"}
- Prescribed Drugs: {", ".join(prescribed_drugs) if prescribed_drugs else "N/A"}
- Image Description: {uploaded_image_description if uploaded_image_description else "N/A"}

**Tool-Augmented Insights:**
{"\n".join(recommendations)}

**Overall Recommendation:**
This is a simulated output. In a real-world scenario, this would be a detailed medical recommendation. Given the insights, further investigation into {potential_diagnosis} is recommended. Consider specific tests based on symptoms and imaging findings. Always cross-reference with professional medical judgment.
    """
    return final_assessment

# --- 4. Streamlit User Interface ---

st.set_page_config(layout="wide", page_title="Medical Diagnosis Assistant (LLM-Augmented)")
st.title("👩‍⚕️ Medical Diagnosis Assistant")
st.markdown("Leveraging a simulated LLM with specialized tools for comprehensive patient assessment.")

with st.sidebar:
    st.header("Patient Information Input")
    patient_symptoms = st.text_area("Patient Symptoms", "e.g., severe headache, blurred vision, numbness in left arm", height=100)
    medical_history = st.text_area("Medical History", "e.g., history of hypertension, Type 2 diabetes, takes Warfarin daily", height=100)
    prescribed_drugs_input = st.text_input("Current Medications (comma-separated)", "Aspirin, Warfarin")
    uploaded_image_description = st.text_area("Medical Imaging Description (e.g., X-ray of left arm showing a suspected fracture)", "N/A", height=70)

    prescribed_drugs = [d.strip() for d in prescribed_drugs_input.split(",") if d.strip()]

    if st.button("Get Diagnosis & Recommendations", type="primary"):
        if not patient_symptoms and not medical_history and not prescribed_drugs and uploaded_image_description == "N/A":
            st.warning("Please provide some patient information to get a diagnosis.")
        else:
            with st.spinner("Analyzing patient data and consulting tools..."):             
                diagnosis_report = llm_orchestrator(
                    patient_symptoms,
                    medical_history,
                    uploaded_image_description,
                    prescribed_drugs
                )
            st.success("Analysis Complete!")
            st.markdown("---")
            st.subheader("🤖 AI-Generated Diagnostic Report")
            st.write(diagnosis_report)

st.markdown("""
### How it works (Simulated):
This application demonstrates the concept of a "Tool-Augmented Foundation Model". A simulated Large Language Model (LLM) acts as an orchestrator, taking in patient information and dynamically calling specialized "tools" to gather specific medical insights. The LLM then synthesizes these insights into a comprehensive diagnostic assessment and recommendation.

**Simulated Tools:**
- **Drug Interaction Checker:** Identifies potential drug interactions.
- **Medical Imaging Analysis:** Provides findings based on a textual description of an image.
- **Medical Literature Search:** Retrieves summaries from mock medical literature.
- **Diagnostic Code Generator:** Suggests ICD-10 codes based on inferred conditions.

**Disclaimer:** This is a conceptual demonstration and *not* a real medical diagnostic tool. Do not use for actual medical advice.
""")
