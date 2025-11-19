import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import uuid
import datetime


# --- 1. Pydantic Models for Structured Output ---

class Diagnosis(BaseModel):
    name: str = Field(..., description="Name of the potential diagnosis.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="AI's self-rated confidence in this diagnosis (0.0 to 1.0).")
    reasoning_path: str = Field(..., description="Step-by-step reasoning for this diagnosis.")

class LLMResponse(BaseModel):
    primary_diagnosis: Diagnosis
    differential_diagnoses: List[Diagnosis] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    raw_llm_output: str = Field(..., description="The raw text output from the LLM.")

class Feedback(BaseModel):
    session_id: str
    timestamp: str
    patient_input: Dict[str, Any]
    llm_response: LLMResponse
    human_action: str = Field(..., description="'accept', 'reject', 'more_info'")
    human_comments: Optional[str] = None
    modified_diagnosis: Optional[Diagnosis] = None


# --- 2. External Tool Simulators (Placeholders) ---

def search_medical_knowledge_base(query: str) -> str:
    """Simulates searching a medical knowledge base."""
    st.info(f"Simulating: Searching medical knowledge base for '{query}'...")
    if "fever" in query.lower() and "rash" in query.lower():
        return "Relevant info: Measles, Rubella, Dengue Fever. Measles often presents with high fever, cough, runny nose, red eyes, and a characteristic rash that appears a few days later. Rubella is milder with similar symptoms. Dengue has fever, rash, joint pain."
    elif "chest pain" in query.lower() and "shortness of breath" in query.lower():
        return "Relevant info: Myocardial Infarction (heart attack), Angina, Pneumonia, Pleurisy. MI typically involves severe chest pain radiating to arm/jaw, shortness of breath, sweating. Pneumonia can cause chest pain and difficulty breathing along with cough and fever."
    return f"No direct matches found for '{query}' in simplified knowledge base."

def retrieve_ehr_data(patient_id: str) -> Dict[str, Any]:
    """Simulates retrieving patient EHR data."""
    st.info(f"Simulating: Retrieving EHR for patient ID '{patient_id}'...")
    if patient_id == "P123":
        return {"patient_id": "P123", "age": 45, "gender": "Male", "allergies": ["Penicillin"], "medications": ["Lisinopril"], "past_medical_history": ["Hypertension", "Type 2 Diabetes"]}
    return {"patient_id": patient_id, "message": "EHR not found for this ID (simulated)."}

def search_clinical_guidelines(condition: str) -> str:
    """Simulates searching clinical guidelines."""
    st.info(f"Simulating: Searching clinical guidelines for '{condition}'...")
    if "hypertension" in condition.lower():
        return "Guideline: JNC 8 guidelines for management of hypertension recommend lifestyle modifications and pharmacotherapy based on age, comorbidities."
    return f"No specific guidelines found for '{condition}' in simplified set."

# --- 3. LLM Integration (Simulator/Placeholder) ---

def call_llm(prompt: str, model_name: str = "Simulated-LLM") -> LLMResponse:
    """Simulates an LLM call and returns structured output. 
    In a real system, this would interact with an actual LLM API (e.g., OpenAI, Gemini).
    """
    st.info(f"Simulating LLM call using model: {model_name}")
    st.code(f"LLM Prompt:\n{prompt}", language="text")

    # Simulate different responses based on keywords in the prompt
    if "fever and rash" in prompt.lower():
        raw_output = "The patient's symptoms of fever and rash, especially given a potential exposure, strongly suggest Measles. Rubella is a differential. Consider Dengue in endemic areas. Always confirm with serology. Why is vaccination status unknown?"
        return LLMResponse(
            primary_diagnosis=Diagnosis(name="Measles", confidence_score=0.9, reasoning_path="Patient has high fever, maculopapular rash, and cough. Medical knowledge base indicates these are classic Measles symptoms. Lack of vaccination history increases likelihood."),
            differential_diagnoses=[
                Diagnosis(name="Rubella", confidence_score=0.7, reasoning_path="Similar rash and fever, but generally milder course. Distinguished by different viral cause."),
                Diagnosis(name="Dengue Fever", confidence_score=0.5, reasoning_path="If in endemic region, presents with fever, rash, and joint pain. Requires travel history.")
            ],
            follow_up_questions=["What is the patient's vaccination history?", "Has the patient traveled recently to dengue endemic areas?"],
            raw_llm_output=raw_output
        )
    elif "chest pain and shortness of breath" in prompt.lower():
        raw_output = "Acute chest pain with dyspnea is concerning for cardiac events like MI or pulmonary issues like pneumonia. Patient's age and history of hypertension increase cardiac risk. Further investigation with ECG, cardiac markers, and chest X-ray is vital. What about the pain's character?"
        return LLMResponse(
            primary_diagnosis=Diagnosis(name="Myocardial Infarction (MI)", confidence_score=0.85, reasoning_path="Sudden onset chest pain radiating to left arm, shortness of breath, and past medical history of hypertension and diabetes strongly suggest MI. EHR data confirms risk factors."),
            differential_diagnoses=[
                Diagnosis(name="Pneumonia", confidence_score=0.6, reasoning_path="Pleuritic chest pain and shortness of breath can indicate pneumonia, especially if cough and fever are present. Less likely given sudden onset without typical infection symptoms."),
                Diagnosis(name="Angina", confidence_score=0.75, reasoning_path="Could be unstable angina given risk factors and symptoms, but MI is more acute and severe.")
            ],
            follow_up_questions=["What is the character of the chest pain (sharp, dull, crushing)?", "Any recent history of respiratory infection?"],
            raw_llm_output=raw_output
        )
    else:
        raw_output = "Based on the provided input, I am generating a general diagnostic assessment. Further details would improve accuracy. The system notes that explicit reasoning and confidence scores are crucial for user trust."
        return LLMResponse(
            primary_diagnosis=Diagnosis(name="Undetermined Condition", confidence_score=0.4, reasoning_path="Insufficient specific symptoms or context provided to confidently narrow down a primary diagnosis."),
            differential_diagnoses=[],
            follow_up_questions=["Please provide more specific symptoms.", "What is the duration of symptoms?"],
            raw_llm_output=raw_output
        )

# --- 4. Guardrails (Placeholder) ---

def run_guardrails(llm_output: LLMResponse) -> bool:
    """Simulates running guardrails for safety and accuracy."""
    st.info("Simulating: Running Guardrails check...")
    # In a real scenario, this would check for hallucinations, harmful content, medical inaccuracies.
    # For this simulation, we'll assume it passes unless a specific keyword triggers a 'fail'.
    if "unethical" in llm_output.raw_llm_output.lower(): # Example of a simplistic check
        st.warning("Guardrails detected potential issue: 'unethical' keyword. Flagging for review.")
        return False
    return True


# --- 5. Evaluation & Feedback Module (Simplified Storage) ---

def store_feedback(feedback_data: Feedback):
    """Stores feedback in Streamlit session state for demonstration."""
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []
    st.session_state.feedback_log.append(feedback_data.dict())
    st.success("Feedback recorded!")


# --- 6. Agentic Controller / Orchestration Layer ---

def diagnose_patient(patient_input: Dict[str, Any], session_id: str) -> Optional[LLMResponse]:
    st.header("Agentic Controller: Orchestrating Diagnosis")

    # Step 1: Tool Execution - Gather context
    symptoms = patient_input.get("symptoms", "")
    patient_id = patient_input.get("patient_id", "")

    medical_kb_info = search_medical_knowledge_base(symptoms)
    ehr_info = retrieve_ehr_data(patient_id)
    clinical_guidelines_info = search_clinical_guidelines(symptoms.split(',')[0] if symptoms else "")

    context = f"Patient Symptoms: {symptoms}\n"
    context += f"EHR Data: {ehr_info}\n"
    context += f"Medical Knowledge Base Info: {medical_kb_info}\n"
    context += f"Clinical Guidelines Info: {clinical_guidelines_info}\n"
    context += "Based on this information, provide a primary diagnosis, differential diagnoses with confidence scores (0.0-1.0), and a step-by-step reasoning path for each. Also, suggest any crucial follow-up questions."

    # Step 2: LLM Processing
    try:
        llm_response = call_llm(context)
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return None

    # Step 3: Guardrails Check
    if not run_guardrails(llm_response):
        st.warning("LLM output flagged by guardrails. Reviewing and potentially stopping further processing.")
        # In a real system, this would trigger human review or regeneration.
        return None

    return llm_response


# --- 7. Streamlit UI ---

st.set_page_config(layout="wide", page_title="AI Diagnostic Assistant")
st.title("🩺 AI-Powered Diagnostic Assistant")
st.markdown("--- Request more Info button not implemented in this version ---")

# Initialize session state for consistent UI and feedback
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'current_llm_response' not in st.session_state:
    st.session_state.current_llm_response = None
if 'patient_input' not in st.session_state:
    st.session_state.patient_input = {}

with st.sidebar:
    st.header("Patient Information")
    patient_id_input = st.text_input("Patient ID", value="P123")
    symptoms_input = st.text_area("Symptoms and Presenting Complaint", height=150, 
                                  placeholder="e.g., High fever, widespread red rash, cough, runny nose, red eyes. Started 3 days ago.")
    history_input = st.text_area("Relevant Medical History", height=100, 
                                 placeholder="e.g., No known allergies. Childhood vaccinations complete (unconfirmed).")
    
    if st.button("Get AI Diagnosis", type="primary"):
        if not symptoms_input:
            st.error("Please enter patient symptoms to get a diagnosis.")
        else:
            st.session_state.patient_input = {
                "patient_id": patient_id_input,
                "symptoms": symptoms_input,
                "medical_history": history_input
            }
            with st.spinner("AI is processing the patient data..."):
                st.session_state.current_llm_response = diagnose_patient(st.session_state.patient_input, st.session_state.session_id)


st.header("AI Diagnosis")

if st.session_state.current_llm_response:
    response: LLMResponse = st.session_state.current_llm_response

    st.subheader("Primary Diagnosis")
    st.metric(label="Diagnosis", value=response.primary_diagnosis.name)
    st.progress(response.primary_diagnosis.confidence_score, text=f"Confidence: {response.primary_diagnosis.confidence_score:.0%}")
    with st.expander("Reasoning Path"): # Progressive disclosure
        st.write(response.primary_diagnosis.reasoning_path)

    if response.differential_diagnoses:
        st.subheader("Differential Diagnoses")
        for i, diff_diag in enumerate(response.differential_diagnoses):
            st.write(f"**{i+1}. {diff_diag.name}**")
            st.progress(diff_diag.confidence_score, text=f"Confidence: {diff_diag.confidence_score:.0%}")
            with st.expander(f"Reasoning for {diff_diag.name}"):
                st.write(diff_diag.reasoning_path)

    if response.follow_up_questions:
        st.subheader("Follow-up Questions")
        for q in response.follow_up_questions:
            st.write(f"- {q}")

    st.subheader("Provide Feedback")
    col1, col2, col3 = st.columns(3)
    
    if col1.button("✅ Accept Diagnosis"):
        feedback = Feedback(
            session_id=st.session_state.session_id,
            timestamp=datetime.datetime.now().isoformat(),
            patient_input=st.session_state.patient_input,
            llm_response=response,
            human_action="accept",
            human_comments=st.text_input("Optional comments for acceptance:", key="accept_comments")
        )
        store_feedback(feedback)
    
    if col2.button("❌ Reject Diagnosis"):
        feedback = Feedback(
            session_id=st.session_state.session_id,
            timestamp=datetime.datetime.now().isoformat(),
            patient_input=st.session_state.patient_input,
            llm_response=response,
            human_action="reject",
            human_comments=st.text_area("Reason for rejection (required):", key="reject_comments")
        )
        if feedback.human_comments:
            store_feedback(feedback)
        else:
            st.error("Please provide a reason for rejecting the diagnosis.")

    # This button would trigger a deeper dive, potentially re-running the agent with a refined query
    # For this example, it's just a placeholder button.
    if col3.button("🔎 Request More Info (Not Implemented)"):
        st.warning("This feature would trigger a deeper dive into specific aspects. (Placeholder)")
        # In a full implementation, this would involve sending a new query to the agent based on user selection

else:
    st.info("Enter patient information in the sidebar and click 'Get AI Diagnosis' to begin.")

st.markdown("--- User Feedback Log ---")
if 'feedback_log' in st.session_state and st.session_state.feedback_log:
    st.json(st.session_state.feedback_log)
else:
    st.write("No feedback recorded yet.")

