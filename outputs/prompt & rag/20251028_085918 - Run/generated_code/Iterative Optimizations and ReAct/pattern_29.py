import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from loguru import logger
from collections import defaultdict
import time

# --- 1. Logging Configuration ---
logger.add("file.log", rotation="500 MB", level="INFO")

# --- 2. models.py content ---
class PatientProfile(BaseModel):
    name: str = ""
    age: int = 45
    gender: str = "Female"
    chronic_conditions: List[str] = Field(default_factory=lambda: ["Diabetes Type 2", "Hypertension"])
    existing_medications: List[str] = Field(default_factory=lambda: ["Metformin", "Lisinopril"])
    allergies: List[str] = Field(default_factory=lambda: ["Penicillin"])
    lifestyle_factors: str = "Sedentary, high-sugar diet"
    current_symptoms: str = "Fatigue, occasional blurred vision"
    lab_results: Dict[str, Any] = Field(default_factory=lambda: {
        "HbA1c": "8.2%",
        "Blood Pressure": "145/95 mmHg",
        "Cholesterol": "LDL 150 mg/dL"
    })

class TreatmentPlan(BaseModel):
    medications: List[str] = Field(default_factory=list)
    lifestyle_recommendations: List[str] = Field(default_factory=list)
    monitoring_schedule: str = ""
    goals: List[str] = Field(default_factory=list)
    rationale: str = ""
    risk_assessment: str = ""
    feedback_history: List[str] = Field(default_factory=list)

# --- 3. medical_kb.py content (Simulated ChromaDB) ---
class MockChromaDB:
    def __init__(self):
        self.knowledge = {
            "diabetes type 2": [
                "First-line treatment for Type 2 Diabetes often includes Metformin.",
                "Lifestyle modifications like diet and exercise are crucial for diabetes management.",
                "Regular monitoring of HbA1c is essential. Target HbA1c < 7%.",
                "Insulin may be required if oral medications are insufficient."
            ],
            "hypertension": [
                "Lisinopril is a common ACE inhibitor for hypertension.",
                "Dietary sodium restriction and regular exercise help manage hypertension.",
                "Blood pressure target for most adults is <130/80 mmHg.",
                "Diuretics or calcium channel blockers might be added if current treatment is inadequate."
            ],
            "metformin": [
                "Metformin side effects can include gastrointestinal upset.",
                "Contraindicated in severe renal impairment."
            ],
            "lisinopril": [
                "Lisinopril side effects can include dry cough, dizziness.",
                "Avoid in pregnancy. Monitor potassium levels."
            ],
            "penicillin": [
                "Penicillin allergy requires avoiding all penicillin-derived antibiotics."
            ],
            "high blood pressure": [
                "High blood pressure requires careful management to prevent cardiovascular events."
            ],
            "high hba1c": [
                "Elevated HbA1c indicates poor glycemic control and increased risk of complications."
            ]
        }

    def query(self, text: str, top_k: int = 3) -> List[str]:
        results = []
        lower_text = text.lower()
        for keyword, facts in self.knowledge.items():
            if keyword in lower_text:
                results.extend(facts)
        return list(set(results))[:top_k]

mock_medical_kb = MockChromaDB()

def retrieve_medical_info(query: str) -> str:
    info = mock_medical_kb.query(query)
    return "\n".join(info) if info else "No specific medical information found for this query."

# --- 4. tools.py content ---
def check_drug_interactions(medications: List[str], conditions: List[str]) -> str:
    logger.info(f"Tool: check_drug_interactions called with meds={medications}, conditions={conditions}")
    interactions = []
    # Simplified logic for demonstration
    if "Metformin" in medications and "severe renal impairment" in [c.lower() for c in conditions]:
        interactions.append("Potential contraindication: Metformin in severe renal impairment.")
    if "Lisinopril" in medications and "pregnancy" in [c.lower() for c in conditions]:
        interactions.append("Warning: Lisinopril should be avoided in pregnancy.")
    if "Penicillin" in medications and "Penicillin" in [c.lower() for c in conditions]: # Simulating allergy info
        interactions.append("Severe Warning: Patient has Penicillin allergy, avoid Penicillin-based drugs.")
    
    if not interactions:
        return "No significant drug-drug or drug-condition interactions detected based on current knowledge."
    return "Interactions found: " + "; ".join(interactions)

def evaluate_plan_risk(patient: PatientProfile, plan: TreatmentPlan) -> str:
    logger.info(f"Tool: evaluate_plan_risk called for patient={patient.name}, plan={plan.medications}")
    risks = []

    # Example: Check for high BP with current plan if BP is high
    if "Hypertension" in patient.chronic_conditions and "145/95 mmHg" in patient.lab_results.get("Blood Pressure", ""):
        if not any(m in plan.medications for m in ["Lisinopril", "Amlodipine", "Hydrochlorothiazide"]):
            risks.append("High blood pressure not adequately addressed by current medications in plan.")
    
    # Example: Check for high HbA1c with current plan if HbA1c is high
    if "Diabetes Type 2" in patient.chronic_conditions and "8.2%" in patient.lab_results.get("HbA1c", ""):
        if not any(m in plan.medications for m in ["Metformin", "Insulin", "Glyburide"]):
            risks.append("High HbA1c not adequately addressed by current medications in plan.")
    
    for med in plan.medications:
        if med in patient.allergies:
            risks.append(f"Warning: Plan includes {med}, but patient has an allergy to it.")

    if not risks:
        return "Overall risk for the proposed plan appears low based on available information."
    return "Potential risks identified: " + "; ".join(risks)

# --- 5. agent.py content (Simulated LLM and Adaptive Agent) ---
def simulate_llm_response(prompt: str) -> str:
    logger.info(f"LLM Prompt: {prompt[:200]}...")
    # This is a highly simplified LLM simulation
    if "initial treatment plan" in prompt.lower():
        return "Initial Plan: Start Metformin 500mg BID, Lisinopril 10mg QD. Recommend low-carb diet, 30 min exercise daily. Monitor BP, HbA1c monthly. Goals: HbA1c <7%, BP <130/80. Rationale: Standard for Diabetes/Hypertension."
    elif "refine treatment plan" in prompt.lower() and "drug interaction" in prompt.lower():
        return "Refined Plan: Metformin 500mg BID. *Replace Lisinopril with Amlodipine 5mg QD due to potential pregnancy risk.* Low-carb diet, 30 min exercise daily. Monitor BP, HbA1c monthly. Goals: HbA1c <7%, BP <130/80. Rationale: Adjusted based on drug interaction feedback."
    elif "refine treatment plan" in prompt.lower() and "risk identified" in prompt.lower():
        return "Refined Plan: Metformin 500mg BID, Amlodipine 5mg QD. Add emphasis on gradual increase in exercise and personalized dietary plan from nutritionist due to sedentary lifestyle. Monitor BP, HbA1c monthly. Goals: HbA1c <7%, BP <130/80. Rationale: Incorporated lifestyle and risk feedback for better adherence and safety."
    elif "self-reflection" in prompt.lower():
        return "Self-reflection: The plan seems robust but could benefit from more personalized lifestyle recommendations."
    else:
        return "Generic response: This is a placeholder LLM output. The plan needs further refinement based on specific instructions."

class AdaptiveTreatmentAgent:
    def __init__(self, medical_kb: MockChromaDB):
        self.medical_kb = medical_kb
        self.tools = {
            "check_drug_interactions": check_drug_interactions,
            "evaluate_plan_risk": evaluate_plan_risk,
        }
        self.history = []

    def _call_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        return f"Error: Tool '{tool_name}' not found."

    def generate_initial_plan(self, patient: PatientProfile) -> TreatmentPlan:
        context_info = retrieve_medical_info(f"{patient.chronic_conditions} {patient.existing_medications}")
        prompt = f"Given patient profile: {patient.model_dump_json(indent=2)}\n\nMedical Context: {context_info}\n\nGenerate an initial treatment plan for this patient. Focus on medications, lifestyle, monitoring, and goals. Provide rationale."
        llm_output = simulate_llm_response(prompt)
        
        # Simple parsing for initial plan from LLM output (highly simplified)
        plan = TreatmentPlan(
            medications = [m.strip() for m in llm_output.split('Medications:')[1].split('Lifestyle:')[0].split(',') if m.strip()] if 'Medications:' in llm_output else (["Metformin", "Lisinopril"] if "Lisinopril" in patient.existing_medications else ["Metformin"]),
            lifestyle_recommendations = [ls.strip() for ls in llm_output.split('Lifestyle:')[1].split('Monitoring:')[0].split(',') if ls.strip()] if 'Lifestyle:' in llm_output else ["Low-carb diet", "30 min exercise daily"],
            monitoring_schedule = llm_output.split('Monitoring:')[1].split('Goals:')[0].strip() if 'Monitoring:' in llm_output else "Monthly BP, HbA1c",
            goals = [g.strip() for g in llm_output.split('Goals:')[1].split('Rationale:')[0].split(',') if g.strip()] if 'Goals:' in llm_output else ["HbA1c <7%", "BP <130/80"],
            rationale = llm_output.split('Rationale:')[1].strip() if 'Rationale:' in llm_output else "Standard care based on conditions."
        )
        logger.info(f"Initial Plan Generated: {plan.model_dump_json(indent=2)}")
        return plan

    def refine_plan(self, patient: PatientProfile, current_plan: TreatmentPlan, feedback: str) -> TreatmentPlan:
        prompt = f"Patient Profile: {patient.model_dump_json(indent=2)}\nCurrent Treatment Plan: {current_plan.model_dump_json(indent=2)}\nFeedback received: {feedback}\n\nRefine the current treatment plan based on this feedback. Provide updated medications, lifestyle recommendations, monitoring, goals, and a new rationale."
        llm_output = simulate_llm_response(prompt)

        # Simple parsing for refined plan from LLM output (highly simplified)
        updated_plan_data = defaultdict(list)
        lines = llm_output.split('\n')
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("Refined Plan:"):
                line = line[len("Refined Plan:"):].strip()
            
            if line.startswith("Medications:"): current_section = "medications"
            elif line.startswith("Lifestyle:"): current_section = "lifestyle_recommendations"
            elif line.startswith("Monitoring:"): current_section = "monitoring_schedule"
            elif line.startswith("Goals:"): current_section = "goals"
            elif line.startswith("Rationale:"): current_section = "rationale"
            else:
                if current_section == "medications": updated_plan_data["medications"].extend([m.strip() for m in line.split(',') if m.strip()])
                elif current_section == "lifestyle_recommendations": updated_plan_data["lifestyle_recommendations"].extend([ls.strip() for ls in line.split(',') if ls.strip()])
                elif current_section == "monitoring_schedule": updated_plan_data["monitoring_schedule"] = line
                elif current_section == "goals": updated_plan_data["goals"].extend([g.strip() for g in line.split(',') if g.strip()])
                elif current_section == "rationale": updated_plan_data["rationale"] = line

        # Fallback to current plan if parsing fails for a section
        refined_plan = TreatmentPlan(
            medications=list(set(updated_plan_data.get("medications", current_plan.medications))),
            lifestyle_recommendations=list(set(updated_plan_data.get("lifestyle_recommendations", current_plan.lifestyle_recommendations))),
            monitoring_schedule=updated_plan_data.get("monitoring_schedule", current_plan.monitoring_schedule),
            goals=list(set(updated_plan_data.get("goals", current_plan.goals))),
            rationale=updated_plan_data.get("rationale", current_plan.rationale)
        )
        refined_plan.feedback_history.append(feedback)
        logger.info(f"Refined Plan: {refined_plan.model_dump_json(indent=2)}")
        return refined_plan

    def self_reflect(self, patient: PatientProfile, plan: TreatmentPlan) -> str:
        prompt = f"Patient Profile: {patient.model_dump_json(indent=2)}\nProposed Treatment Plan: {plan.model_dump_json(indent=2)}\n\nCritique this plan. Identify potential gaps, areas for improvement, or ambiguities. Provide a concise self-reflection."
        llm_output = simulate_llm_response(prompt)
        logger.info(f"Self-reflection: {llm_output}")
        return f"Self-reflection: {llm_output}"

    def run_iteration(self, patient: PatientProfile, current_plan: TreatmentPlan, user_feedback: str = "") -> TreatmentPlan:
        all_feedback = []
        st.session_state.logs.append("--- Agent Iteration Started ---")
        logger.info("--- Agent Iteration Started ---")

        # 1. Tool Calls
        drug_interaction_feedback = self._call_tool(
            "check_drug_interactions", 
            medications=current_plan.medications, 
            conditions=patient.chronic_conditions + patient.allergies
        )
        all_feedback.append(f"Tool Feedback (Drug Interactions): {drug_interaction_feedback}")
        st.session_state.logs.append(f"Tool Feedback (Drug Interactions): {drug_interaction_feedback}")

        plan_risk_feedback = self._call_tool(
            "evaluate_plan_risk", 
            patient=patient, 
            plan=current_plan
        )
        all_feedback.append(f"Tool Feedback (Plan Risk): {plan_risk_feedback}")
        st.session_state.logs.append(f"Tool Feedback (Plan Risk): {plan_risk_feedback}")

        # 2. Self-Reflection
        self_reflection_feedback = self.self_reflect(patient, current_plan)
        all_feedback.append(self_reflection_feedback)
        st.session_state.logs.append(self_reflection_feedback)

        # 3. User Feedback
        if user_feedback: 
            all_feedback.append(f"User Feedback: {user_feedback}")
            st.session_state.logs.append(f"User Feedback: {user_feedback}")

        synthesized_feedback = "\n".join(all_feedback)
        st.session_state.logs.append(f"Synthesized Feedback:\n{synthesized_feedback}")

        # 4. Refine Plan
        refined_plan = self.refine_plan(patient, current_plan, synthesized_feedback)
        st.session_state.logs.append("--- Agent Iteration Finished ---")
        logger.info("--- Agent Iteration Finished ---")
        return refined_plan

# --- 6. main.py (Streamlit UI) ---
st.set_page_config(layout="wide", page_title="AI-Powered Personalized Treatment Plan Assistant")
st.title("AI-Powered Personalized Treatment Plan Assistant")

# Initialize session state for patient, plan, and logs
if 'patient' not in st.session_state:
    st.session_state.patient = PatientProfile()
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = None
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'iteration_count' not in st.session_state:
    st.session_state.iteration_count = 0

# Initialize the agent
agent = AdaptiveTreatmentAgent(medical_kb=mock_medical_kb)

# Sidebar for Patient Profile Input
with st.sidebar:
    st.header("Patient Profile Input")
    patient_name = st.text_input("Name", st.session_state.patient.name)
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.patient.age)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=(["Male", "Female", "Other"].index(st.session_state.patient.gender)))
    patient_conditions_str = st.text_area("Chronic Conditions (comma-separated)", ", ".join(st.session_state.patient.chronic_conditions))
    patient_meds_str = st.text_area("Existing Medications (comma-separated)", ", ".join(st.session_state.patient.existing_medications))
    patient_allergies_str = st.text_area("Allergies (comma-separated)", ", ".join(st.session_state.patient.allergies))
    patient_lifestyle = st.text_area("Lifestyle Factors", st.session_state.patient.lifestyle_factors)
    patient_symptoms = st.text_area("Current Symptoms", st.session_state.patient.current_symptoms)
    
    # Simplified lab results input
    st.subheader("Lab Results (Key: Value)")
    lab_results_input = st.text_area(
        "e.g., HbA1c: 8.2%, Blood Pressure: 145/95 mmHg", 
        ", ".join([f"{k}: {v}" for k,v in st.session_state.patient.lab_results.items()])
    )
    parsed_lab_results = {}
    try:
        for item in lab_results_input.split(','):
            if ':' in item:
                key, value = item.split(':', 1)
                parsed_lab_results[key.strip()] = value.strip()
    except Exception as e:
        st.error(f"Error parsing lab results: {e}. Please use 'Key: Value' format.")

    if st.button("Update Patient Profile"):
        st.session_state.patient = PatientProfile(
            name=patient_name,
            age=patient_age,
            gender=patient_gender,
            chronic_conditions=[c.strip() for c in patient_conditions_str.split(',') if c.strip()],
            existing_medications=[m.strip() for m in patient_meds_str.split(',') if m.strip()],
            allergies=[a.strip() for a in patient_allergies_str.split(',') if a.strip()],
            lifestyle_factors=patient_lifestyle,
            current_symptoms=patient_symptoms,
            lab_results=parsed_lab_results
        )
        st.session_state.current_plan = None # Reset plan if patient profile changes significantly
        st.session_state.logs.append("Patient profile updated. Plan reset.")
        st.session_state.iteration_count = 0
        st.experimental_rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Treatment Plan Generation")
    st.write(f"**Current Patient:** {st.session_state.patient.name} (Age: {st.session_state.patient.age}, Gender: {st.session_state.patient.gender})")
    st.json(st.session_state.patient.model_dump_json(indent=2))

    if st.session_state.current_plan is None:
        if st.button("Generate Initial Plan"):
            st.session_state.logs.append("Generating initial plan...")
            st.session_state.current_plan = agent.generate_initial_plan(st.session_state.patient)
            st.session_state.iteration_count = 1
            st.experimental_rerun()
    else:
        st.subheader(f"Iteration {st.session_state.iteration_count} - Proposed Treatment Plan")
        st.json(st.session_state.current_plan.model_dump_json(indent=2))

        user_feedback_input = st.text_area("Provide Feedback (e.g., 'Concerns about Metformin side effects', 'Plan seems good, approve', 'Consider adding exercise details')", key="user_feedback")
        
        if st.button("Refine Plan (Next Iteration)"):
            if st.session_state.current_plan:
                st.session_state.logs.append(f"Running iteration {st.session_state.iteration_count + 1}...")
                st.session_state.current_plan = agent.run_iteration(st.session_state.patient, st.session_state.current_plan, user_feedback_input)
                st.session_state.iteration_count += 1
                st.experimental_rerun()
            else:
                st.warning("Please generate an initial plan first.")

        if st.button("Approve Plan and Finish"):
            st.success("Treatment plan approved!")
            st.session_state.logs.append("Treatment plan approved by user.")
            # You might want to save the final plan here

with col2:
    st.header("Agent Logs and Reasoning")
    for log_entry in reversed(st.session_state.logs):
        st.write(f"- {log_entry}")


# Optional: Display final plan details if approved
if st.session_state.current_plan and "approved" in " ".join(st.session_state.logs).lower():
    st.subheader("Final Approved Treatment Plan Details")
    st.json(st.session_state.current_plan.model_dump_json(indent=2))


