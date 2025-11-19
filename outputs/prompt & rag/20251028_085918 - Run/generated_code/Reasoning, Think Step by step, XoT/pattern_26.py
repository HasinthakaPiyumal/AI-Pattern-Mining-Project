import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class MockLLM:
    def generate(self, prompt: str) -> str:
        if "diagnosis" in prompt.lower() and "symptoms" in prompt.lower():
            if "fever" in prompt.lower() and "cough" in prompt.lower():
                return "Chain-of-Thought: Patient presents with fever and cough. These are common symptoms for various respiratory infections. Consider flu, common cold, or pneumonia. Further investigation needed. Preliminary Diagnosis: Respiratory Infection."
            elif "headache" in prompt.lower() and "stiff neck" in prompt.lower():
                return "Chain-of-Thought: Patient reports headache and stiff neck. These are concerning symptoms. Rule out meningitis. Need to ask about sensitivity to light. Preliminary Diagnosis: Potential Meningitis."
            else:
                return "Chain-of-Thought: Symptoms provided are general. Cannot provide a specific diagnosis without more information. Preliminary Diagnosis: Undetermined."
        elif "treatment" in prompt.lower() and "diagnosis" in prompt.lower():
            if "respiratory infection" in prompt.lower():
                return "Treatment Recommendation: Rest, hydration, over-the-counter pain relievers. If bacterial infection suspected, antibiotics (e.g., Amoxicillin) after confirmation. Monitor for worsening symptoms."
            elif "potential meningitis" in prompt.lower():
                return "Treatment Recommendation: Immediate medical attention, lumbar puncture for definitive diagnosis. IV antibiotics (e.g., Ceftriaxone) if bacterial meningitis suspected, even before culture results. Steroids might be considered."
            else:
                return "Treatment Recommendation: Consult a specialist for further evaluation."
        return "Mock LLM Response"

mock_llm = MockLLM()

medical_knowledge_base = {
    "fever": ["symptom of flu", "symptom of cold", "symptom of pneumonia", "symptom of meningitis"],
    "cough": ["symptom of flu", "symptom of cold", "symptom of pneumonia"],
    "stiff neck": ["symptom of meningitis"],
    "headache": ["symptom of flu", "symptom of cold", "symptom of meningitis"],
    "respiratory infection": ["diagnosis", "treatment: rest, hydration"],
    "meningitis": ["diagnosis", "urgent treatment: antibiotics, steroids"],
    "amoxicillin": ["antibiotic", "treatment for bacterial infections"],
    "ceftriaxone": ["antibiotic", "treatment for bacterial meningitis"],
    "antibiotics": ["require prescription"],
    "hydration": ["general care"],
    "rest": ["general care"]
}

class MedicalInput(BaseModel):
    symptoms: str = Field(..., min_length=3, description="Comma-separated list of symptoms")
    medical_history: str = Field("", description="Relevant medical history")

class DiagnosisResult(BaseModel):
    preliminary_diagnosis: str
    reasoning_steps: List[str]
    is_verified: bool
    verification_notes: List[str]
    final_diagnosis: str
    treatment_recommendation: str
    explanation: str

def process_input(symptoms: str, medical_history: str) -> Dict[str, Any]:
    processed_symptoms = [s.strip().lower() for s in symptoms.split(",")]
    return {
        "processed_symptoms": processed_symptoms,
        "medical_history": medical_history.strip()
    }

def retrieve_medical_info(query: str, top_k: int = 3) -> List[str]:
    relevant_info = []
    query_words = set(query.lower().split())
    for key, values in medical_knowledge_base.items():
        if any(word in key for word in query_words) or any(word in v for v in values for word in query_words):
            relevant_info.append(f"{key}: {', '.join(values)}")
    return relevant_info[:top_k]

def get_llm_reasoning(processed_input: Dict[str, Any]) -> Dict[str, Any]:
    symptoms_str = ", ".join(processed_input["processed_symptoms"])
    history_str = f"Medical History: {processed_input['medical_history']}." if processed_input["medical_history"] else ""

    prompt = (
        f"Based on the following symptoms: '{symptoms_str}' and medical history: '{history_str}', "
        "generate a step-by-step Chain-of-Thought reasoning for a potential diagnosis. "
        "Conclude with 'Preliminary Diagnosis: [Your Diagnosis]'."
    )
    raw_llm_response = mock_llm.generate(prompt)

    reasoning_steps = [step.strip() for step in raw_llm_response.split("Chain-of-Thought:")[-1].split("Preliminary Diagnosis:")[0].split(". ") if step.strip()]
    preliminary_diagnosis = raw_llm_response.split("Preliminary Diagnosis:")[-1].strip()

    return {
        "raw_llm_response": raw_llm_response,
        "reasoning_steps": reasoning_steps,
        "preliminary_diagnosis": preliminary_diagnosis
    }

def verify_diagnosis(llm_reasoning_output: Dict[str, Any], processed_input: Dict[str, Any]) -> Dict[str, Any]:
    is_verified = True
    verification_notes = []
    final_diagnosis = llm_reasoning_output["preliminary_diagnosis"]

    if llm_reasoning_output["preliminary_diagnosis"] == "Undetermined":
        is_verified = False
        verification_notes.append("Diagnosis is undetermined, indicating LLM uncertainty. Requires more input.")
    elif not any(llm_reasoning_output["preliminary_diagnosis"].lower() in step.lower() for step in llm_reasoning_output["reasoning_steps"]):
        is_verified = False
        verification_notes.append("Preliminary diagnosis is not clearly supported by the generated reasoning steps.")

    symptoms_in_kb = []
    for symptom in processed_input["processed_symptoms"]:
        if symptom in medical_knowledge_base:
            symptoms_in_kb.append(symptom)
        else:
            is_verified = False
            verification_notes.append(f"Symptom '{symptom}' not directly found in knowledge base or requires more context.")

    if symptoms_in_kb:
        found_plausible_link = False
        for symptom in symptoms_in_kb:
            for fact in medical_knowledge_base.get(symptom, []):
                if llm_reasoning_output["preliminary_diagnosis"].lower() in fact.lower():
                    found_plausible_link = True
                    break
            if found_plausible_link:
                break
        if not found_plausible_link and llm_reasoning_output["preliminary_diagnosis"] != "Undetermined":
            is_verified = False
            verification_notes.append(f"Preliminary diagnosis '{llm_reasoning_output['preliminary_diagnosis']}' does not strongly align with symptoms based on knowledge base.")

    if "no definitive" in llm_reasoning_output["raw_llm_response"].lower() and "diagnosis:" in llm_reasoning_output["raw_llm_response"].lower():
        is_verified = False
        verification_notes.append("Potential contradiction: LLM indicates no definitive diagnosis but still provides one.")

    if not is_verified and not verification_notes:
        verification_notes.append("Initial verification passed, but no specific positive checks were performed.")
    if is_verified and not verification_notes:
        verification_notes.append("Diagnosis and reasoning appear consistent and plausible based on available (mocked) knowledge.")

    return {
        "is_verified": is_verified,
        "verification_notes": verification_notes,
        "final_diagnosis": final_diagnosis if is_verified else "Unverified: " + final_diagnosis
    }

def get_llm_treatment_recommendation(final_diagnosis: str, processed_input: Dict[str, Any]) -> str:
    symptoms_str = ", ".join(processed_input["processed_symptoms"])
    history_str = f"Medical History: {processed_input['medical_history']}." if processed_input["medical_history"] else ""

    prompt = (
        f"Based on the verified diagnosis: '{final_diagnosis}', symptoms: '{symptoms_str}', "
        f"and medical history: '{history_str}', recommend appropriate treatment. "
        "Prioritize general recommendations first, then specific ones if applicable."
    )
    return mock_llm.generate(prompt)

def generate_explanation(diagnosis_result: DiagnosisResult) -> str:
    explanation_text = f"**Diagnosis:** {diagnosis_result.final_diagnosis}\n\n"
    explanation_text += "**Reasoning Steps:**\n"
    for step in diagnosis_result.reasoning_steps:
        explanation_text += f"- {step}\n"
    explanation_text += "\n**Verification Status:** "
    explanation_text += "✅ Verified" if diagnosis_result.is_verified else "❌ Unverified"
    if diagnosis_result.verification_notes:
        explanation_text += "\n**Verification Notes:**\n"
        for note in diagnosis_result.verification_notes:
            explanation_text += f"- {note}\n"
    explanation_text += f"\n**Treatment Recommendation:** {diagnosis_result.treatment_recommendation}"
    return explanation_text

st.title("Medical Diagnosis and Treatment Recommendation System (SVR)")
st.markdown("This system uses **Structured and Verified Reasoning (SVR)** to provide robust medical insights.")

with st.sidebar:
    st.header("Patient Input")
    user_symptoms = st.text_area("Enter symptoms (comma-separated):", "fever, cough, body aches")
    user_history = st.text_area("Enter medical history (optional):", "No known allergies, vaccinated for flu last year.")
    
    submit_button = st.button("Get Diagnosis & Recommendation")

if submit_button and user_symptoms:
    st.subheader("Processing Request...")
    
    processed_input = process_input(user_symptoms, user_history)
    st.write("Processed Input:", processed_input)

    llm_reasoning_output = get_llm_reasoning(processed_input)
    st.subheader("LLM's Preliminary Reasoning:")
    st.write(llm_reasoning_output["raw_llm_response"])

    verification_output = verify_diagnosis(llm_reasoning_output, processed_input)
    st.subheader("Verification Results:")
    st.json(verification_output)

    if verification_output["is_verified"]:
        treatment_recommendation = get_llm_treatment_recommendation(
            verification_output["final_diagnosis"], processed_input
        )
    else:
        treatment_recommendation = "Cannot provide a reliable treatment recommendation due to unverified diagnosis. Please consult a medical professional."
    st.subheader("Treatment Recommendation (Preliminary):")
    st.write(treatment_recommendation)

    final_diagnosis_result = DiagnosisResult(
        preliminary_diagnosis=llm_reasoning_output["preliminary_diagnosis"],
        reasoning_steps=llm_reasoning_output["reasoning_steps"],
        is_verified=verification_output["is_verified"],
        verification_notes=verification_output["verification_notes"],
        final_diagnosis=verification_output["final_diagnosis"],
        treatment_recommendation=treatment_recommendation,
        explanation=""
    )
    final_diagnosis_result.explanation = generate_explanation(final_diagnosis_result)

    st.subheader("Final Verified Diagnosis and Recommendation:")
    st.markdown(final_diagnosis_result.explanation)

elif submit_button and not user_symptoms:
    st.error("Please enter some symptoms to get a diagnosis.")
