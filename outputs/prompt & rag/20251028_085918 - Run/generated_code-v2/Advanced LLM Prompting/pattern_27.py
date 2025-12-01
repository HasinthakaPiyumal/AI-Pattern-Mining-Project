import streamlit as st
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# 1. Pydantic Models for Exemplars
class LabResult(BaseModel):
    test_name: str
    value: str
    unit: str
    reference_range: str

class PatientExemplar(BaseModel):
    symptoms: List[str]
    lab_results: List[LabResult]
    patient_history_snippets: List[str]
    tentative_diagnosis: str
    rationale: str

# 2. Synthetic Exemplar Generation Module (SEGM)
def mock_llm_generate_exemplars(disease_description: str, num_exemplars: int) -> List[PatientExemplar]:
    generated_exemplars = []
    for i in range(num_exemplars):
        # Simulate a diverse set of symptoms and lab results based on the disease_description
        # In a real scenario, this would be an actual LLM call, e.g., using OpenAI API
        # to generate structured JSON output directly.
        if "Fictional Rare Disease A" in disease_description:
            symptoms = [
                f"Persistent fatigue (exemplar {i+1})",
                f"Unexplained skin rash (exemplar {i+1})",
                f"Intermittent fever (exemplar {i+1})"
            ]
            lab_results = [
                LabResult(test_name="CBC", value="Low WBC", unit="/mcL", reference_range="4.5-11.0"),
                LabResult(test_name="CRP", value="Elevated", unit="mg/L", reference_range="<1.0")
            ]
            history = [
                f"Patient {i+1} reported onset of symptoms 3 months ago.",
                f"Family history negative for similar conditions for patient {i+1}."
            ]
            diagnosis = "Fictional Rare Disease A"
            rationale = "Presence of characteristic rash, fatigue, and elevated inflammatory markers."
        elif "Fictional Rare Disease B" in disease_description:
            symptoms = [
                f"Muscle weakness (exemplar {i+1})",
                f"Difficulty swallowing (exemplar {i+1})",
                f"Respiratory distress (exemplar {i+1})"
            ]
            lab_results = [
                LabResult(test_name="CK", value="Very High", unit="U/L", reference_range="30-200"),
                LabResult(test_name="EMG", value="Abnormal", unit="", reference_range="Normal")
            ]
            history = [
                f"Patient {i+1} experienced rapid progression of muscle weakness.",
                f"No prior neurological issues for patient {i+1}."
            ]
            diagnosis = "Fictional Rare Disease B"
            rationale = "Severe muscle weakness, dysphagia, and elevated CK levels with abnormal EMG findings."
        else:
            symptoms = [
                f"Generic Symptom X (exemplar {i+1})",
                f"Generic Symptom Y (exemplar {i+1})"
            ]
            lab_results = [
                LabResult(test_name="Generic Test", value="Abnormal", unit="", reference_range="Normal")
            ]
            history = [
                f"Patient {i+1} presented with general malaise."
            ]
            diagnosis = f"Undetermined Rare Disease {i+1}"
            rationale = "Symptoms are non-specific, requiring further investigation."
            
        generated_exemplars.append(PatientExemplar(
            symptoms=symptoms,
            lab_results=lab_results,
            patient_history_snippets=history,
            tentative_diagnosis=diagnosis,
            rationale=rationale
        ))
    return generated_exemplars

def generate_synthetic_exemplars(disease_description: str, num_exemplars: int) -> List[PatientExemplar]:
    # In a real application, this would involve a call to an LLM API
    # For this simulation, we use a mock function.
    return mock_llm_generate_exemplars(disease_description, num_exemplars)

# 3. Few-Shot Prompting and Classification Module (FSPCM)
def mock_llm_classify_patient(prompt: str) -> Dict[str, Any]:
    # Simulate LLM's classification capability based on the prompt content.
    # In a real scenario, this would be an actual LLM API call.
    if "Fictional Rare Disease A" in prompt and "Unexplained skin rash" in prompt and "fatigue" in prompt:
        return {"diagnosis": "Highly likely Fictional Rare Disease A", "confidence": 0.95, "reasoning": "Strong correlation with self-generated exemplars for Fictional Rare Disease A, particularly the characteristic rash and persistent fatigue. Elevated CRP supports inflammatory process.", "related_exemplar_matches": ["Exemplar 1", "Exemplar 2"]}
    elif "Fictional Rare Disease B" in prompt and "Muscle weakness" in prompt and "difficulty swallowing" in prompt:
        return {"diagnosis": "Likely Fictional Rare Disease B", "confidence": 0.90, "reasoning": "Symptoms like severe muscle weakness and dysphagia are consistent with Fictional Rare Disease B, as seen in the synthetic exemplars. High CK levels are also indicative.", "related_exemplar_matches": ["Exemplar 1"]}
    else:
        return {"diagnosis": "Further investigation needed or potentially another rare disease", "confidence": 0.60, "reasoning": "Symptoms are not a strong match with the provided rare disease exemplars. Consider a broader differential diagnosis.", "related_exemplar_matches": []}

def classify_patient_case(exemplars: List[PatientExemplar], new_patient_data: Dict[str, Any]) -> Dict[str, Any]:
    prompt_parts = []
    prompt_parts.append("You are a rare disease diagnostic assistant. Below are a few patient exemplars for a specific rare disease, followed by a new patient's data. Provide a most probable diagnosis, a confidence score (0-1), reasoning, and list which exemplars were most relevant.\n\n")

    # Add exemplars to the prompt
    for i, exemplar in enumerate(exemplars):
        prompt_parts.append(f"--- Exemplar {i+1} ---")
        prompt_parts.append(f"Symptoms: {', '.join(exemplar.symptoms)}")
        prompt_parts.append(f"Lab Results: {'; '.join([f'{lr.test_name}: {lr.value} {lr.unit} (Ref: {lr.reference_range})' for lr in exemplar.lab_results])}")
        prompt_parts.append(f"Patient History: {'; '.join(exemplar.patient_history_snippets)}")
        prompt_parts.append(f"Tentative Diagnosis: {exemplar.tentative_diagnosis}")
        prompt_parts.append(f"Rationale: {exemplar.rationale}\n")

    # Add new patient data to the prompt
    prompt_parts.append("--- New Patient Data ---")
    prompt_parts.append(f"Symptoms: {new_patient_data.get('symptoms', 'N/A')}")
    prompt_parts.append(f"Lab Results: {new_patient_data.get('lab_results', 'N/A')}")
    prompt_parts.append(f"Patient History: {new_patient_data.get('patient_history', 'N/A')}\n")

    prompt_parts.append("Based on the above, what is the most probable diagnosis for the new patient? Provide a confidence score (0-1), detailed reasoning, and list any related exemplar matches. Your output should be a JSON object with 'diagnosis', 'confidence', 'reasoning', and 'related_exemplar_matches' keys.")

    full_prompt = "\n".join(prompt_parts)
    
    # In a real application, you'd send `full_prompt` to an LLM API.
    # For this simulation, we use a mock function.
    return mock_llm_classify_patient(full_prompt)

# 4. User Interface (UI) Module using Streamlit
st.set_page_config(layout="wide", page_title="Rare Disease Diagnostic Assistant (RDDA)")
st.title("🧬 Rare Disease Diagnostic Assistant (RDDA)")
st.markdown("---\n### Empowering healthcare professionals with AI-driven insights for rare diseases where data is scarce.")

# Sidebar for information
st.sidebar.header("About RDDA")
st.sidebar.markdown(
    "This assistant leverages generative AI to create synthetic patient exemplars "
    "for rare diseases. These exemplars are then used in few-shot prompts to help "
    "diagnose new patient cases, especially when real-world data is limited."
)
st.sidebar.info("Disclaimer: This is a prototype and should not be used for actual medical diagnosis. Always consult with a qualified healthcare professional.")

# Main application area

st.header("1. Generate Synthetic Patient Exemplars")
disease_description_input = st.text_area(
    "Enter a detailed description of the rare disease (e.g., symptoms, pathophysiology, known markers):",
    "Fictional Rare Disease A: A rare autoimmune disorder characterized by persistent fatigue, unexplained migratory skin rash, and intermittent low-grade fever. Lab markers often show elevated CRP and mild leukopenia."
)
num_exemplars_to_generate = st.slider("Number of exemplars to generate:", 1, 5, 2)

if st.button("Generate Exemplars"):
    if disease_description_input:
        with st.spinner("Generating synthetic exemplars..."):
            try:
                generated_exemplars_list = generate_synthetic_exemplars(disease_description_input, num_exemplars_to_generate)
                st.session_state['generated_exemplars'] = generated_exemplars_list
                st.success(f"Successfully generated {len(generated_exemplars_list)} exemplars.")
                for i, exemplar in enumerate(generated_exemplars_list):
                    st.subheader(f"Exemplar {i+1}")
                    st.json(exemplar.dict())
            except Exception as e:
                st.error(f"Error generating exemplars: {e}")
    else:
        st.warning("Please provide a disease description to generate exemplars.")

st.markdown("--- ")
st.header("2. Diagnose New Patient Case using Few-Shot Prompting")

if 'generated_exemplars' not in st.session_state or not st.session_state['generated_exemplars']:
    st.warning("Please generate synthetic exemplars first before attempting to diagnose a new patient.")
else:
    st.write("Using the following generated exemplars:")
    for i, exemplar in enumerate(st.session_state['generated_exemplars']):
        st.write(f"- **Exemplar {i+1}:** {exemplar.tentative_diagnosis}")

    st.subheader("Enter New Patient Clinical Data")
    new_patient_symptoms = st.text_area(
        "Patient's Symptoms (comma-separated):", 
        "Persistent fatigue, unexplained skin rash, mild fever, muscle aches"
    )
    new_patient_lab_results = st.text_area(
        "Patient's Key Lab Results (e.g., 'CBC: Low WBC; CRP: Elevated'):", 
        "CBC: Low WBC; CRP: Elevated"
    )
    new_patient_history = st.text_area(
        "Patient's Relevant History (e.g., 'Onset 2 months ago'):", 
        "Onset of symptoms 2 months ago, no significant medical history"
    )

    if st.button("Diagnose Patient"):
        if new_patient_symptoms and new_patient_lab_results:
            new_patient_data = {
                "symptoms": [s.strip() for s in new_patient_symptoms.split(',') if s.strip()],
                "lab_results": new_patient_lab_results, # Simplified for text input
                "patient_history": new_patient_history
            }
            with st.spinner("Analyzing patient data and exemplars..."):
                try:
                    diagnosis_result = classify_patient_case(
                        st.session_state['generated_exemplars'],
                        new_patient_data
                    )
                    st.success("Diagnosis complete!")
                    st.subheader("Diagnosis Result")
                    st.json(diagnosis_result)

                    st.write("### Interpretation")
                    st.write(f"**Most Probable Diagnosis:** {diagnosis_result.get('diagnosis', 'N/A')}")
                    st.write(f"**Confidence Score:** {diagnosis_result.get('confidence', 'N/A'):.2f}")
                    st.write(f"**Reasoning:** {diagnosis_result.get('reasoning', 'N/A')}")
                    if diagnosis_result.get('related_exemplar_matches'):
                        st.write(f"**Related Exemplar Matches:** {', '.join(diagnosis_result['related_exemplar_matches'])}")
                    else:
                        st.write("**Related Exemplar Matches:** None found among the generated exemplars.")

                except Exception as e:
                    st.error(f"Error during diagnosis: {e}")
        else:
            st.warning("Please enter patient symptoms and lab results.")
