import streamlit as st
import json
import random
import time

class MockLLM:
    def __init__(self, model_name="MockMedicalLLM"):
        self.model_name = model_name

    def generate_response(self, prompt, task_type="diagnosis"):
        if task_type == "diagnosis":
            return self._mock_diagnosis_response(prompt)
        elif task_type == "verification":
            return self._mock_verification_response(prompt)
        return {"error": "Unknown task type"}

    def _mock_diagnosis_response(self, prompt):
        # Simulate complex reasoning based on keywords in prompt
        symptoms_keywords = {
            "fever": ["infection", "flu", "malaria"],
            "cough": ["bronchitis", "pneumonia", "asthma"],
            "headache": ["migraine", "tension headache", "sinusitis"],
            "fatigue": ["anemia", "hypothyroidism", "chronic fatigue syndrome"],
            "rash": ["allergy", "measles", "dermatitis"],
            "abdominal pain": ["appendicitis", "gastritis", "IBS"],
        }

        detected_symptoms = []
        for keyword in symptoms_keywords.keys():
            if keyword in prompt.lower():
                detected_symptoms.append(keyword)

        possible_diagnoses = []
        for symptom in detected_symptoms:
            possible_diagnoses.extend(symptoms_keywords[symptom])

        # Remove duplicates and ensure at least one diagnosis
        possible_diagnoses = list(set(possible_diagnoses))
        if not possible_diagnoses:
            possible_diagnoses = ["General malaise", "Observation recommended"]

        # Generate structured reasoning and confidence
        diagnoses_output = []
        for diag in possible_diagnoses[:3]: # Limit to top 3 for simplicity
            confidence = round(random.uniform(0.6, 0.95), 2)
            reasoning_steps = [
                f"Step 1: Patient presented with {' and '.join(detected_symptoms)}.",
                f"Step 2: {diag} is a common condition associated with these symptoms.",
                f"Step 3: Further tests (e.g., blood work, imaging) would be required to confirm {diag} and rule out others."
            ]
            if "fever" in detected_symptoms and diag == "Malaria":
                reasoning_steps.insert(2, "Step 2.5: Travel history to endemic areas should be inquired for Malaria.")

            diagnoses_output.append({
                "diagnosis": diag,
                "confidence": confidence,
                "reasoning": "\n".join(reasoning_steps)
            })

        # Simulate potential contradictions for some cases
        potential_risks = []
        if "fever" in detected_symptoms and "cold intolerance" in prompt.lower():
            potential_risks.append("Contradiction: Fever typically indicates warmth, but patient reports cold intolerance. Investigate further for atypical presentations or co-existing conditions.")

        return {
            "diagnoses": diagnoses_output,
            "potential_risks_contradictions": potential_risks,
            "raw_prompt": prompt
        }

    def _mock_verification_response(self, reasoning_data):
        # Simulate a verification step that checks for obvious inconsistencies
        risks = reasoning_data.get("potential_risks_contradictions", [])
        for diag_info in reasoning_data.get("diagnoses", []):
            if diag_info["confidence"] < 0.7 and "definitive" in diag_info["reasoning"].lower():
                risks.append(f"Verification Flag: Low confidence ({diag_info['confidence']}) for '{diag_info['diagnosis']}' but reasoning implies definitive conclusion. Review reasoning steps.")

        if not risks and random.random() < 0.1: # 10% chance to find a minor additional risk
             risks.append("Verification Check: No major contradictions found, but suggest considering rare genetic predispositions based on symptom pattern.")

        return {
            "verified_risks_contradictions": risks if risks else ["No significant contradictions or risks identified upon review."],
            "verification_status": "Verified with observations" if risks else "Verified Clean"
        }


class MedicalDiagnosisCoPilot:
    def __init__(self, llm_model):
        self.llm = llm_model

    def get_diagnosis(self, symptoms, medical_history, test_results):
        full_patient_data = f"Patient Symptoms: {symptoms}\nMedical History: {medical_history}\nTest Results: {test_results}\n\nBased on this information, provide differential diagnoses, structured step-by-step reasoning for each, a confidence score (0.0-1.0), and highlight any potential risks or contradictions in the data or your initial reasoning. Think step by step."

        st.subheader("Phase 1: Initial LLM Reasoning (Chain-of-Thought)")
        st.text_area("LLM Input Prompt (for initial diagnosis):", full_patient_data, height=200, disabled=True)
        with st.spinner("LLM is generating initial diagnoses and reasoning..."):
            initial_llm_output = self.llm.generate_response(full_patient_data, task_type="diagnosis")
            time.sleep(1) # Simulate LLM processing time

        st.write("### Proposed Diagnoses and Reasoning:")
        for diag_info in initial_llm_output.get("diagnoses", []):
            st.success(f"**Diagnosis: {diag_info['diagnosis']} (Confidence: {diag_info['confidence']:.2f})**")
            st.markdown(f"**Reasoning:**\n```\n{diag_info['reasoning']}\n```")
            st.markdown("--- ---")

        st.subheader("Phase 2: Reasoning Verification")
        st.info("The system is now performing a self-correction/external verification step to evaluate the initial reasoning.")
        with st.spinner("Verifying reasoning..."):
            verification_output = self.llm.generate_response(initial_llm_output, task_type="verification")
            time.sleep(0.8) # Simulate verification time

        st.write("### Verification Report:")
        for risk in initial_llm_output.get("potential_risks_contradictions", []) + verification_output.get("verified_risks_contradictions", []):
            st.warning(f"**❗ Identified Risk/Contradiction:** {risk}")
        if not (initial_llm_output.get("potential_risks_contradictions", []) or verification_output.get("verified_risks_contradictions", [])):
            st.info("✅ No significant risks or contradictions identified during verification.")

        return initial_llm_output, verification_output


# --- Streamlit Application --- #
st.set_page_config(layout="wide", page_title="Medical Diagnosis Co-pilot")
st.title("🩺 Medical Diagnosis Co-pilot")
st.markdown("This AI co-pilot assists doctors by providing differential diagnoses with structured and verified reasoning, leveraging the Structured and Verified Reasoning (SVR) pattern.")

mock_llm_instance = MockLLM()
co_pilot = MedicalDiagnosisCoPilot(mock_llm_instance)

st.header("Patient Information Input")

symptoms = st.text_area("Enter Patient Symptoms (e.g., 'fever, cough, body aches, fatigue'):", "fever, cough, sore throat")
medical_history = st.text_area("Enter Medical History (e.g., 'no significant past medical history, smokes occasionally'):", "no significant past medical history")
test_results = st.text_area("Enter Relevant Test Results (e.g., 'CBC normal, Rapid Flu Test positive'):", "Rapid Flu Test positive")

if st.button("Get Diagnosis and Verified Reasoning", type="primary"):
    if symptoms.strip() or medical_history.strip() or test_results.strip():
        with st.expander("Detailed Processing Flow"): # Collapsible section for verbose output
            initial_output, verification_output = co_pilot.get_diagnosis(symptoms, medical_history, test_results)

        st.subheader("Final Co-pilot Summary")
        st.markdown("--- ---")
        for diag_info in initial_output.get("diagnoses", []):
            st.success(f"**Proposed Diagnosis: {diag_info['diagnosis']} (Confidence: {diag_info['confidence']:.2f})**")
            st.markdown(f"**Structured Reasoning:**\n```\n{diag_info['reasoning']}\n```")
        
        st.markdown("--- ---")
        st.markdown("### Key Verification Insights for Doctor Review:")
        all_risks = initial_output.get("potential_risks_contradictions", []) + verification_output.get("verified_risks_contradictions", [])
        if all_risks:
            for risk in all_risks:
                st.error(f"**Attention Needed:** {risk}")
        else:
            st.info("✅ The initial reasoning appears robust and consistent based on verification.")

        st.markdown("--- ---")
        st.markdown("**:bulb: Note:** This is a simulated co-pilot. Always rely on professional medical judgment.")
    else:
        st.warning("Please enter some patient information to get a diagnosis.")
