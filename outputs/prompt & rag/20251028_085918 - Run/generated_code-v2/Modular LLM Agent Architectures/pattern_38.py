import gradio as gr
import time # For simulating delays

# --- Stage 1: Information Collection ---

def simulate_medical_database_query(symptoms: str, medical_history: str) -> str:
    """
    Simulates querying external medical databases for relevant information.
    In a real application, this would involve API calls to actual databases
    or sophisticated RAG systems.
    """
    print(f"[{time.ctime()}] Initiating medical database query...")
    time.sleep(2) # Simulate network latency and processing
    info = f"Gathered latest research on '{symptoms}' and common treatments related to '{medical_history}'. "
    info += "Found potential drug interactions and relevant clinical guidelines."
    print(f"[{time.ctime()}] Medical database query complete.")
    return info

def simulate_patient_record_access(patient_id: str) -> dict:
    """
    Simulates accessing a patient's electronic health records (EHR).
    """
    print(f"[{time.ctime()}] Accessing patient records for ID: {patient_id}...")
    time.sleep(1.5) # Simulate record retrieval time
    if patient_id == "PAT001":
        records = {
            "allergies": ["Penicillin"],
            "current_medications": ["Metformin (for Type 2 Diabetes)"],
            "diagnosed_conditions": ["Type 2 Diabetes", "Hypertension"],
            "recent_lab_results": "Glucose: 180 mg/dL, Blood Pressure: 140/90 mmHg"
        }
    else:
        records = {
            "allergies": [],
            "current_medications": [],
            "diagnosed_conditions": [],
            "recent_lab_results": "Normal"
        }
    print(f"[{time.ctime()}] Patient records retrieved.")
    return records

def collect_and_synthesize_information(patient_id: str, symptoms: str, medical_history: str) -> dict:
    """
    Orchestrates the information collection stage, combining data from various sources.
    """
    print(f"[{time.ctime()}] Starting Information Collection Stage...")
    db_data = simulate_medical_database_query(symptoms, medical_history)
    patient_data = simulate_patient_record_access(patient_id)

    synthesized_info = {
        "summary_symptoms": symptoms,
        "summary_medical_history": medical_history,
        "patient_id": patient_id,
        "external_medical_info": db_data,
        "patient_ehr": patient_data
    }
    print(f"[{time.ctime()}] Information Collection Stage complete. Synthesized data.")
    return synthesized_info

# --- Stage 2: Planning Stage ---

def generate_treatment_plan(synthesized_info: dict) -> str:
    """
    Generates a comprehensive treatment plan based on the synthesized information.
    This function represents the 'cognitive load' of planning.
    """
    print(f"[{time.ctime()}] Starting Planning Stage...")
    time.sleep(3) # Simulate complex planning and reasoning

    patient_id = synthesized_info.get("patient_id", "N/A")
    symptoms = synthesized_info.get("summary_symptoms", "N/A")
    medical_history = synthesized_info.get("summary_medical_history", "N/A")
    ehr = synthesized_info.get("patient_ehr", {})
    external_info = synthesized_info.get("external_medical_info", "")

    plan_parts = [
        f"--- Treatment Plan for Patient ID: {patient_id} ---",
        f"**Presenting Symptoms:** {symptoms}",
        f"**Medical History Summary:** {medical_history}",
        f"**Patient EHR Snippet:**",
        f"  - Allergies: {', '.join(ehr.get('allergies', ['None']))}",
        f"  - Current Meds: {', '.join(ehr.get('current_medications', ['None']))}",
        f"  - Diagnosed Conditions: {', '.join(ehr.get('diagnosed_conditions', ['None']))}",
        f"  - Recent Lab Results: {ehr.get('recent_lab_results', 'N/A')}",
        "",
        "**External Medical Insights (from database query):**",
        f"- {external_info}",
        "",
        "**Proposed Treatment Plan:**"
    ]

    # Simple rule-based planning for demonstration
    if "diabetes" in medical_history.lower() or "glucose: 180" in ehr.get('recent_lab_results', '').lower():
        plan_parts.append("- Adjust Metformin dosage (if applicable) or consider new oral hypoglycemic agent.")
        plan_parts.append("- Diet and exercise regimen focusing on blood sugar control.")
        plan_parts.append("- Regular blood glucose monitoring.")
    if "hypertension" in medical_history.lower() or "blood pressure: 140/90" in ehr.get('recent_lab_results', '').lower():
        plan_parts.append("- Recommend DASH diet and sodium restriction.")
        plan_parts.append("- Consider Antihypertensive medication adjustment/initiation.")
        plan_parts.append("- Regular blood pressure monitoring.")
    if "fever" in symptoms.lower():
        plan_parts.append("- Prescribe antipyretic (e.g., Acetaminophen 500mg PRN).")
        plan_parts.append("- Advise rest and hydration.")
    if "cough" in symptoms.lower():
        plan_parts.append("- Recommend cough suppressant if persistent.")
    if "Penicillin" in ehr.get('allergies', []):
        plan_parts.append("- **ALERT:** Patient has Penicillin allergy. Avoid all penicillin-derived medications.")

    plan_parts.append("- Follow-up appointment in 2 weeks.")
    plan_parts.append("- Refer to an endocrinologist if diabetes control remains suboptimal.")
    plan_parts.append("\n**Disclaimer:** This is an AI-generated draft plan and requires review and approval by a qualified medical professional.")

    final_plan = "\n".join(plan_parts)
    print(f"[{time.ctime()}] Planning Stage complete. Treatment plan generated.")
    return final_plan

# --- Main Agent Orchestrator ---

def medical_treatment_agent(patient_id: str, symptoms: str, medical_history: str) -> str:
    """
    The main orchestrator for the Medical Treatment Plan Generator agent.
    It manages the cognitive load by separating information collection and planning.
    """
    print("\n--- Starting Medical Treatment Plan Generation Process ---")

    # Stage 1: Information Collection
    collected_data = collect_and_synthesize_information(patient_id, symptoms, medical_history)

    # Stage 2: Planning Stage
    treatment_plan = generate_treatment_plan(collected_data)

    print("--- Medical Treatment Plan Generation Process Complete ---\n")
    return treatment_plan

# --- Gradio Interface ---

if __name__ == "__main__":
    print("Launching Gradio interface for Medical Treatment Plan Generator...")
    iface = gr.Interface(
        fn=medical_treatment_agent,
        inputs=[
            gr.Textbox(label="Patient ID (e.g., PAT001)", placeholder="Enter patient identifier"),
            gr.Textbox(label="Current Symptoms", placeholder="e.g., persistent cough, fever, fatigue"),
            gr.Textbox(label="Relevant Medical History", placeholder="e.g., Type 2 Diabetes, Hypertension, asthma")
        ],
        outputs=gr.Markdown(label="Generated Treatment Plan"),
        title="👩‍⚕️ AI Medical Treatment Plan Generator (Cognitive Load Management Demo)",
        description="This AI agent generates a treatment plan in two distinct stages: first, it gathers all necessary information, and then it formulates the plan. This separation helps manage the agent's 'cognitive load' for more accurate and reliable outputs."
    )
    iface.launch()