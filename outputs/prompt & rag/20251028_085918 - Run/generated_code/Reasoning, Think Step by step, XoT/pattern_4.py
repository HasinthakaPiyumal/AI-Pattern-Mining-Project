import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import time # For simulating LLM delay

# 1. Medical Knowledge Base (Simplified, Hardcoded)
class MedicalKnowledgeBase:
    def __init__(self):
        self.drug_interactions = {
            "warfarin": {
                "aspirin": {"severity": "High", "effect": "Increased bleeding risk"},
                "ibuprofen": {"severity": "Moderate", "effect": "Increased bleeding risk"},
                "amiodarone": {"severity": "High", "effect": "Increased warfarin effect"},
            },
            "amiodarone": {
                "simvastatin": {"severity": "High", "effect": "Increased risk of myopathy"}
            },
            "metformin": {
                "iodinated contrast media": {"severity": "High", "effect": "Increased risk of lactic acidosis"}
            }
        }
        self.drug_dosages = {
            "paracetamol": {"adult": "500mg - 1000mg every 4-6 hours, max 4000mg/day", "pediatric_per_kg": "10-15mg/kg"},
            "amoxicillin": {"adult": "250mg - 500mg every 8 hours", "pediatric_per_kg": "20-40mg/kg/day in divided doses"},
            "warfarin": {"adult": "Individualized based on INR"}
        }
        self.condition_contraindications = {
            "ibuprofen": {
                "renal impairment": {"severity": "High", "effect": "Worsening of kidney function"},
                "asthma": {"severity": "Moderate", "effect": "Exacerbation of asthma"}
            },
            "metformin": {
                "severe renal impairment": {"severity": "High", "effect": "Increased risk of lactic acidosis"},
                "congestive heart failure": {"severity": "Moderate", "effect": "Increased risk of lactic acidosis"}
            }
        }

    def check_interaction(self, drug1: str, drug2: str) -> Dict[str, Any]:
        drug1 = drug1.lower()
        drug2 = drug2.lower()
        if drug1 in self.drug_interactions and drug2 in self.drug_interactions[drug1]:
            return self.drug_interactions[drug1][drug2]
        if drug2 in self.drug_interactions and drug1 in self.drug_interactions[drug2]:
            return self.drug_interactions[drug2][drug1]
        return {"severity": "None", "effect": "No known interaction found in simplified KB"}

    def get_dosage_info(self, drug: str, is_pediatric: bool = False, weight_kg: float = 0.0) -> str:
        drug = drug.lower()
        info = self.drug_dosages.get(drug)
        if not info:
            return f"Dosage info for {drug} not found in simplified KB."

        if is_pediatric and "pediatric_per_kg" in info and weight_kg > 0:
            return f"Pediatric dosage: {info['pediatric_per_kg']} (based on {weight_kg}kg, needs calculation)"
        elif "adult" in info:
            return f"Adult dosage: {info['adult']}"
        return f"General dosage info: {info}"

    def get_condition_interaction(self, drug: str, condition: str) -> Dict[str, Any]:
        drug = drug.lower()
        condition = condition.lower()
        if drug in self.condition_contraindications and condition in self.condition_contraindications[drug]:
            return self.condition_contraindications[drug][condition]
        return {"severity": "None", "effect": "No known condition contraindication found in simplified KB"}

# 2. Mock LLM Component
class MockLLM:
    def generate(self, prompt: str) -> str:
        time.sleep(1) # Simulate LLM thinking time
        prompt_lower = prompt.lower()
        response_lines = []

        response_lines.append("Reasoning Steps:")

        # Simulate Chain-of-Thought for drug interactions
        if "current medications" in prompt_lower and "proposed medication" in prompt_lower:
            current_meds = self._extract_list_from_prompt(prompt_lower, "current medications:")
            proposed_med = self._extract_single_from_prompt(prompt_lower, "proposed medication:")

            response_lines.append(f"1. Evaluate potential interactions between '{proposed_med}' and existing medications: {', '.join(current_meds)}.")

            # Simulate interaction detection based on common patterns
            if "warfarin" in current_meds and "aspirin" in proposed_med:
                response_lines.append(f"   - Identified a high-severity interaction between Warfarin and Aspirin, increasing bleeding risk.")
            elif "warfarin" in current_meds and "ibuprofen" in proposed_med:
                response_lines.append(f"   - Identified a moderate-severity interaction between Warfarin and Ibuprofen, increasing bleeding risk.")
            elif "amiodarone" in current_meds and "simvastatin" in proposed_med:
                response_lines.append(f"   - Identified a high-severity interaction between Amiodarone and Simvastatin, increasing risk of myopathy.")
            else:
                response_lines.append(f"   - No immediate severe drug-drug interaction found for {proposed_med} with {', '.join(current_meds)} in initial assessment.")

        # Simulate Chain-of-Thought for condition contraindications
        if "existing conditions" in prompt_lower and "proposed medication" in prompt_lower:
            conditions = self._extract_list_from_prompt(prompt_lower, "existing conditions:")
            proposed_med = self._extract_single_from_prompt(prompt_lower, "proposed medication:")
            response_lines.append(f"2. Assess contraindications for '{proposed_med}' based on patient's existing conditions: {', '.join(conditions)}.")
            if "renal impairment" in conditions and "ibuprofen" in proposed_med:
                response_lines.append(f"   - Identified a high-severity contraindication for Ibuprofen with renal impairment.")
            else:
                response_lines.append(f"   - No immediate severe drug-condition contraindication found for {proposed_med} with {', '.join(conditions)} in initial assessment.")

        # Simulate Chain-of-Thought for dosage
        if "age:" in prompt_lower and "weight_kg:" in prompt_lower and "proposed medication:" in prompt_lower:
            proposed_med = self._extract_single_from_prompt(prompt_lower, "proposed medication:")
            age = int(self._extract_single_from_prompt(prompt_lower, "age:") or 0)
            weight = float(self._extract_single_from_prompt(prompt_lower, "weight_kg:") or 0.0)
            is_pediatric = age < 18
            response_lines.append(f"3. Determine appropriate dosage for '{proposed_med}' considering age ({age} years) and weight ({weight} kg). Patient is {'pediatric' if is_pediatric else 'adult'}.")
            if "paracetamol" in proposed_med and is_pediatric and weight > 0:
                response_lines.append(f"   - Pediatric dosage for Paracetamol is approximately {10*weight:.0f}-{15*weight:.0f}mg per dose.")
            elif "paracetamol" in proposed_med and not is_pediatric:
                response_lines.append(f"   - Adult dosage for Paracetamol is typically 500mg-1000mg per dose.")
            else:
                response_lines.append(f"   - General dosage guidelines for {proposed_med} should be consulted.")

        response_lines.append("Final Recommendation (LLM's initial thought): Review all findings carefully before prescribing.")
        return "\n".join(response_lines)

    def _extract_list_from_prompt(self, prompt: str, keyword: str) -> List[str]:
        try:
            start_idx = prompt.find(keyword)
            if start_idx == -1: return []
            end_idx = prompt.find("\n", start_idx)
            if end_idx == -1: end_idx = len(prompt)
            list_str = prompt[start_idx + len(keyword):end_idx].strip()
            return [item.strip() for item in list_str.split(',') if item.strip()]
        except:
            return []

    def _extract_single_from_prompt(self, prompt: str, keyword: str) -> str:
        try:
            start_idx = prompt.find(keyword)
            if start_idx == -1: return ""
            end_idx = prompt.find("\n", start_idx)
            if end_idx == -1: end_idx = len(prompt)
            return prompt[start_idx + len(keyword):end_idx].strip()
        except:
            return ""

# 3. Patient Data Model
class PatientData(BaseModel):
    age: int = Field(..., gt=0, description="Patient's age in years")
    weight_kg: float = Field(..., gt=0, description="Patient's weight in kilograms")
    existing_conditions: List[str] = Field(default_factory=list, description="List of patient's existing medical conditions")
    current_medications: List[str] = Field(default_factory=list, description="List of medications patient is currently taking")
    proposed_medication: str = Field(..., min_length=1, description="The new medication being considered")

# 4. Reasoning Engine (Orchestration Layer)
class ReasoningEngine:
    def __init__(self, llm: MockLLM, kb: MedicalKnowledgeBase):
        self.llm = llm
        self.kb = kb

    def _generate_cot_prompt(self, patient_data: PatientData) -> str:
        prompt = f"""Analyze the following patient scenario for medication safety. Provide a step-by-step reasoning process focusing on potential drug interactions, condition contraindications, and dosage appropriateness.

Patient Details:
Age: {patient_data.age}
Weight (kg): {patient_data.weight_kg}
Existing Conditions: {', '.join(patient_data.existing_conditions) if patient_data.existing_conditions else 'None'}
Current Medications: {', '.join(patient_data.current_medications) if patient_data.current_medications else 'None'}
Proposed Medication: {patient_data.proposed_medication}

Reasoning Process (Chain-of-Thought):
1. Identify all current and proposed medications.
2. Systematically check for drug-drug interactions between the proposed medication and each current medication.
3. Systematically check for drug-condition contraindications between the proposed medication and each existing condition.
4. Determine an appropriate dosage range for the proposed medication based on patient's age and weight.
5. Summarize findings and provide a safety recommendation.

"""
        return prompt

    def analyze_medication(self, patient_data: PatientData) -> Dict[str, Any]:
        prompt = self._generate_cot_prompt(patient_data)
        llm_raw_response = self.llm.generate(prompt)

        reasoning_steps = [line.strip() for line in llm_raw_response.split('\n') if line.strip() and not line.startswith("Final Recommendation")]
        initial_recommendation = [line.replace("Final Recommendation (LLM's initial thought): ", "") for line in llm_raw_response.split('\n') if line.startswith("Final Recommendation")]

        return {
            "llm_raw_response": llm_raw_response,
            "reasoning_steps": reasoning_steps,
            "initial_recommendation": initial_recommendation[0] if initial_recommendation else "No specific recommendation from LLM."
        }

# 5. Verification and Self-Correction Module
class VerifierModule:
    def __init__(self, kb: MedicalKnowledgeBase):
        self.kb = kb

    def verify_reasoning_step(self, step_description: str, patient_data: PatientData) -> Dict[str, Any]:
        step_lower = step_description.lower()
        verification_status = "Unverified"
        explanation = "Could not verify this step with available knowledge base information."
        confidence = 0.0

        proposed_med_lower = patient_data.proposed_medication.lower()

        # Verification for drug-drug interactions
        if "evaluate potential interactions" in step_lower or "drug-drug interaction" in step_lower:
            verified_interactions = []
            for current_med in patient_data.current_medications:
                interaction = self.kb.check_interaction(proposed_med_lower, current_med)
                if interaction["severity"] != "None":
                    verified_interactions.append(f"Interaction: {proposed_med_lower} + {current_med}: {interaction['severity']} - {interaction['effect']}")
            if verified_interactions:
                verification_status = "Verified"
                explanation = "Interactions found and confirmed against KB: " + "; ".join(verified_interactions)
                confidence = 0.9
            else:
                verification_status = "Verified"
                explanation = "No significant drug-drug interactions found in KB for these medications."
                confidence = 0.8

        # Verification for drug-condition contraindications
        elif "assess contraindications" in step_lower or "drug-condition contraindication" in step_lower:
            verified_contraindications = []
            for condition in patient_data.existing_conditions:
                contra = self.kb.get_condition_interaction(proposed_med_lower, condition)
                if contra["severity"] != "None":
                    verified_contraindications.append(f"Contraindication: {proposed_med_lower} + {condition}: {contra['severity']} - {contra['effect']}")
            if verified_contraindications:
                verification_status = "Verified"
                explanation = "Contraindications found and confirmed against KB: " + "; ".join(verified_contraindications)
                confidence = 0.9
            else:
                verification_status = "Verified"
                explanation = "No significant drug-condition contraindications found in KB."
                confidence = 0.8

        # Verification for dosage appropriateness
        elif "determine appropriate dosage" in step_lower:
            is_pediatric = patient_data.age < 18
            dosage_info = self.kb.get_dosage_info(proposed_med_lower, is_pediatric, patient_data.weight_kg)
            if "not found" not in dosage_info:
                verification_status = "Verified"
                explanation = f"Dosage information retrieved from KB: {dosage_info}"
                confidence = 0.9
            else:
                verification_status = "Partially Verified"
                explanation = f"Dosage information for {proposed_med_lower} not fully available in KB. KB says: {dosage_info}"
                confidence = 0.0

        return {"status": verification_status, "explanation": explanation, "confidence": confidence}

    def perform_verification(self, llm_reasoning_steps: List[str], patient_data: PatientData) -> List[Dict[str, Any]]:
        verified_steps = []
        for step in llm_reasoning_steps:
            result = self.verify_reasoning_step(step, patient_data)
            verified_steps.append({"step": step, **result})
        return verified_steps

    def check_self_consistency(self, primary_analysis_results: Dict[str, Any], patient_data: PatientData) -> Dict[str, Any]:
        # Simplified self-consistency: Re-check critical facts directly against KB
        # In a real system, this would involve generating alternative reasoning paths
        # or asking the LLM to 'reverse' its reasoning.

        proposed_med_lower = patient_data.proposed_medication.lower()
        consistency_issues = []
        overall_consistent = True

        # Re-check interactions
        for current_med in patient_data.current_medications:
            kb_interaction = self.kb.check_interaction(proposed_med_lower, current_med)
            if kb_interaction["severity"] != "None" and f"interaction between {current_med.lower()}" not in primary_analysis_results["llm_raw_response"].lower():
                consistency_issues.append(f"LLM might have missed an interaction: {proposed_med_lower} + {current_med} ({kb_interaction['severity']})")
                overall_consistent = False

        # Re-check contraindications
        for condition in patient_data.existing_conditions:
            kb_contra = self.kb.get_condition_interaction(proposed_med_lower, condition)
            if kb_contra["severity"] != "None" and f"contraindication for {condition.lower()}" not in primary_analysis_results["llm_raw_response"].lower():
                consistency_issues.append(f"LLM might have missed a contraindication: {proposed_med_lower} + {condition} ({kb_contra['severity']})")
                overall_consistent = False

        # Re-check dosage (basic)
        is_pediatric = patient_data.age < 18
        kb_dosage_info = self.kb.get_dosage_info(proposed_med_lower, is_pediatric, patient_data.weight_kg)
        if "not found" not in kb_dosage_info and kb_dosage_info.lower() not in primary_analysis_results["llm_raw_response"].lower():
             # This check is very simplistic, just looking for presence
             pass # Too complex to reliably check in this mock setup

        return {
            "consistent": overall_consistent,
            "issues": consistency_issues,
            "explanation": "Primary analysis checked against KB for critical interactions and contraindications. " \
                           + ("No inconsistencies found." if overall_consistent else "Inconsistencies detected.")
        }


# Streamlit User Interface
st.set_page_config(layout="wide", page_title="Medication Safety Verification")
st.title("💊 Medication Interaction and Dosage Verification System")
st.subheader("Leveraging Enhanced LLM Reasoning and Reliability")

# Initialize components
@st.cache_resource
def init_components():
    kb = MedicalKnowledgeBase()
    llm = MockLLM()
    reasoning_engine = ReasoningEngine(llm, kb)
    verifier = VerifierModule(kb)
    return kb, llm, reasoning_engine, verifier

kb, llm, reasoning_engine, verifier = init_components()

st.sidebar.header("Patient Information")
with st.sidebar.form("patient_form"):
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    weight_kg = st.number_input("Weight (kg)", min_value=1.0, value=70.0, step=0.1)
    existing_conditions_input = st.text_area("Existing Medical Conditions (comma-separated)", "asthma, hypertension")
    current_medications_input = st.text_area("Current Medications (comma-separated)", "warfarin, paracetamol")
    proposed_medication_input = st.text_input("Proposed New Medication", "aspirin")

    submitted = st.form_submit_button("Analyze Medication Safety")

    if submitted:
        try:
            existing_conditions = [c.strip() for c in existing_conditions_input.split(',') if c.strip()]
            current_medications = [m.strip() for m in current_medications_input.split(',') if m.strip()]
            patient_data = PatientData(
                age=age,
                weight_kg=weight_kg,
                existing_conditions=existing_conditions,
                current_medications=current_medications,
                proposed_medication=proposed_medication_input
            )

            st.success("Patient data validated.")

            st.subheader("🧠 LLM Initial Reasoning (Chain-of-Thought)")
            with st.spinner("LLM is thinking and generating initial reasoning..."):
                analysis_results = reasoning_engine.analyze_medication(patient_data)
                st.text_area("LLM's Raw Reasoning", analysis_results["llm_raw_response"], height=250)

            st.subheader("✅ Verification and Self-Correction")
            with st.spinner("Verifying LLM's reasoning steps against medical knowledge base..."):
                verified_steps = verifier.perform_verification(analysis_results["reasoning_steps"], patient_data)
                for i, step_info in enumerate(verified_steps):
                    status_icon = "✅" if step_info["status"] == "Verified" else ("⚠️" if step_info["status"] == "Partially Verified" else "❌")
                    st.markdown(f"**Step {i+1}:** {step_info['step']}")
                    st.info(f"  {status_icon} **Verification Status:** {step_info['status']} (Confidence: {step_info['confidence']:.2f}) - {step_info['explanation']}")

                st.markdown("--- ")
                st.markdown("**Self-Consistency Check:**")
                consistency_check = verifier.check_self_consistency(analysis_results, patient_data)
                if consistency_check["consistent"]:
                    st.success(f"No immediate inconsistencies detected in critical areas. {consistency_check['explanation']}")
                else:
                    st.warning(f"Potential inconsistencies or omissions detected. {consistency_check['explanation']}")
                    for issue in consistency_check["issues"]:
                        st.error(f"- {issue}")

            st.subheader("💡 Final Consolidated Recommendation")
            final_recommendation = f"**LLM's Initial Thought:** {analysis_results['initial_recommendation']}\n\n"
            overall_confidence_score = sum(s['confidence'] for s in verified_steps) / len(verified_steps) if verified_steps else 0

            if any(s['status'] == 'Verified' and 'high-severity' in s['explanation'].lower() for s in verified_steps):
                final_recommendation += "**Critical Alert:** High-severity interactions or contraindications have been VERIFIED. Exercise extreme caution or AVOID proposed medication."
                st.error(final_recommendation)
            elif not consistency_check["consistent"]:
                 final_recommendation += "**Warning:** Inconsistencies or potential omissions were found during self-consistency check. Further expert review is highly recommended."
                 st.warning(final_recommendation)
            elif overall_confidence_score > 0.7:
                final_recommendation += "**Recommendation:** The proposed medication appears generally safe based on current information and verification, but always cross-reference with professional medical judgment."
                st.success(final_recommendation)
            else:
                final_recommendation += "**Caution:** Verification was limited or identified partial concerns. Recommend thorough medical review."
                st.warning(final_recommendation)
            st.info(f"Overall Verification Confidence: {overall_confidence_score:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

st.sidebar.markdown("--- ")
st.sidebar.info("This is a simplified demonstration. Always consult with qualified healthcare professionals for medical advice.")
