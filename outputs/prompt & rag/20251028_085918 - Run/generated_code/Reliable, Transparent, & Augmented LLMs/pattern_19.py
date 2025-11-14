
import gradio as gr
import pandas as pd
import random
import time

# --- 1. Simulated Medical Knowledge Base and Patient Records ---

MEDICAL_KNOWLEDGE_BASE = {
    "fever": "Fever is a temporary increase in your body temperature, often due to an illness. Causes include infections (viral, bacterial), inflammation, and certain medications. Common associated symptoms are chills, sweating, headache, and muscle aches. Treatment often involves rest, fluids, and fever-reducing medications.",
    "cough": "A cough is a reflex action to clear your airway of irritants and mucus. It can be acute or chronic. Causes range from common cold and flu to allergies, asthma, and more serious conditions like bronchitis or pneumonia. Depending on the cause, treatments include cough suppressants, expectorants, and antibiotics.",
    "headache": "A headache is pain in any region of the head. Headaches can be a symptom of a wide range of conditions, from stress and dehydration to more serious neurological issues. Common types include tension headaches, migraines, and cluster headaches. Management involves pain relievers, rest, and identifying triggers.",
    "fatigue": "Fatigue is a feeling of extreme tiredness, lack of energy, and motivation. It can be caused by lack of sleep, stress, poor diet, certain medical conditions (e.g., anemia, thyroid issues), and medications. Lifestyle changes, addressing underlying conditions, and rest are key to managing fatigue.",
    "sore throat": "A sore throat is pain or irritation of the throat, often made worse by swallowing. Common causes are viral infections (like the common cold), bacterial infections (like strep throat), allergies, and dry air. Treatment usually involves pain relievers, warm liquids, and sometimes antibiotics for bacterial infections.",
    "chest pain": "Chest pain can arise from various conditions, ranging from benign muscle strain to life-threatening heart attacks or lung issues. It's crucial to seek immediate medical attention for severe or persistent chest pain. Other symptoms like shortness of breath, radiating pain, and sweating can indicate serious conditions.",
    "shortness of breath": "Shortness of breath, or dyspnea, is a feeling of not being able to get enough air. It can be a symptom of heart or lung problems, anxiety, or high altitude. Acute onset often requires immediate medical evaluation, especially if accompanied by chest pain or dizziness.",
    "nausea": "Nausea is an uneasy sensation in the stomach, often leading to vomiting. It can be caused by food poisoning, motion sickness, pregnancy, migraines, or certain medications. Rest, bland foods, and hydration are common remedies."
}

MOCK_PATIENT_RECORDS_DF = pd.DataFrame({
    "patient_id": ["P001", "P002", "P003"],
    "name": ["Alice Smith", "Bob Johnson", "Carol White"],
    "age": [45, 62, 31],
    "gender": ["Female", "Male", "Female"],
    "past_conditions": ["Hypertension, Type 2 Diabetes", "Asthma, High Cholesterol", "Seasonal Allergies"],
    "medications": ["Lisinopril, Metformin", "Albuterol, Atorvastatin", "Loratadine"],
    "last_visit": ["2023-10-15", "2023-09-20", "2023-11-01"]
})

def retrieve_medical_info(query):
    """Simulates retrieval from a medical knowledge base."""
    retrieved_docs = []
    query_lower = query.lower()
    for keyword, info in MEDICAL_KNOWLEDGE_BASE.items():
        if keyword in query_lower:
            retrieved_docs.append(f"**From Knowledge Base (Relevant to {keyword}):** {info}")
    return "\n".join(retrieved_docs) if retrieved_docs else "No specific medical knowledge found for the given symptoms."

def retrieve_patient_history(patient_id):
    """Simulates retrieving patient-specific information."""
    record = MOCK_PATIENT_RECORDS_DF[MOCK_PATIENT_RECORDS_DF["patient_id"] == patient_id]
    if not record.empty:
        return record.iloc[0].to_dict()
    return None

# --- 2. Agentic LLM Core (Simulated) ---

def simulate_llm_diagnosis(
    symptoms: str,
    patient_id: str = None,
    retrieved_kb_info: str = "",
    patient_history: dict = None
):
    """Simulates LLM behavior for diagnosis, reasoning, and confidence.
    This is a rule-based simulation, not an actual LLM call.
    """
    diagnosis = "Uncertain Diagnosis"
    reasoning = "Based on the provided symptoms, the AI considered several possibilities."
    confidence = 0.5 # Default confidence
    probabilistic_diagnoses = {}

    symptoms_lower = symptoms.lower()
    keywords_present = [kw for kw in MEDICAL_KNOWLEDGE_BASE.keys() if kw in symptoms_lower]

    if "fever" in symptoms_lower and "cough" in symptoms_lower:
        diagnosis = "Common Cold/Flu"
        reasoning = f"The combination of fever and cough strongly suggests a common viral infection like a cold or flu. {retrieved_kb_info}"
        confidence = random.uniform(0.7, 0.9)
        probabilistic_diagnoses = {"Common Cold/Flu": 0.85, "Bronchitis": 0.10, "Allergies": 0.05}
    elif "headache" in symptoms_lower and "fatigue" in symptoms_lower:
        diagnosis = "Tension Headache/Stress-related Fatigue"
        reasoning = f"Headache and fatigue often co-occur due to stress, lack of sleep, or underlying tension. {retrieved_kb_info}"
        confidence = random.uniform(0.6, 0.85)
        probabilistic_diagnoses = {"Tension Headache": 0.70, "Migraine": 0.15, "Dehydration": 0.10, "Other": 0.05}
    elif "chest pain" in symptoms_lower and "shortness of breath" in symptoms_lower:
        diagnosis = "Potential Cardiac or Pulmonary Issue - *Immediate Medical Attention Recommended*"
        reasoning = f"Chest pain combined with shortness of breath is a red flag and requires urgent medical evaluation to rule out serious cardiac or pulmonary conditions. {retrieved_kb_info}"
        confidence = 0.95 # High confidence in *recommendation for urgent care*
        probabilistic_diagnoses = {"Cardiac Event (e.g., Angina, MI)": 0.45, "Pneumonia/Pleurisy": 0.30, "Anxiety Attack": 0.15, "Other Serious Cause": 0.10}
    elif "sore throat" in symptoms_lower:
        diagnosis = "Pharyngitis (Sore Throat)"
        reasoning = f"Sore throat is a common symptom of viral pharyngitis, often associated with a cold. {retrieved_kb_info}"
        confidence = random.uniform(0.6, 0.8)
        probabilistic_diagnoses = {"Viral Pharyngitis": 0.75, "Bacterial Strep Throat": 0.20, "Allergies": 0.05}
    elif keywords_present:
        diagnosis = f"General symptoms related to: {', '.join(keywords_present)}"
        reasoning = f"Based on the detected keywords, here is some general information. Further details would require more context. {retrieved_kb_info}"
        confidence = random.uniform(0.5, 0.7)
        probabilistic_diagnoses = {diagnosis: confidence, "Undetermined": 1 - confidence}
    else:
        diagnosis = "Non-specific Symptoms. More information needed."
        reasoning = "The AI could not confidently identify a specific condition based on the provided symptoms alone. Please provide more details or consult a human expert."
        confidence = 0.4
        probabilistic_diagnoses = {"Undetermined": 0.8, "Minor Ailment": 0.2}

    # Incorporate patient history if available
    if patient_history:
        history_str = f"Patient ID: {patient_id}, Age: {patient_history['age']}, Gender: {patient_history['gender']}, Past Conditions: {patient_history['past_conditions']}, Medications: {patient_history['medications']}.\n"
        reasoning = history_str + reasoning
        if "Hypertension" in patient_history.get("past_conditions", "") and "headache" in symptoms_lower:
            reasoning += "\nConsider the patient's history of hypertension in relation to the headache."
            confidence = min(1.0, confidence + 0.05)

    return {
        "diagnosis": diagnosis,
        "reasoning": reasoning,
        "confidence": round(confidence, 2),
        "probabilistic_diagnoses": probabilistic_diagnoses
    }

# --- 3. Feedback and Continuous Improvement Loop (Simulated) ---

feedback_data = []

def submit_feedback(doctor_id, patient_id, symptoms, suggested_diagnosis, actual_diagnosis, feedback_text, override_reason):
    """Simulates storing user feedback for future model improvement."""
    feedback_entry = {
        "timestamp": time.ctime(),
        "doctor_id": doctor_id,
        "patient_id": patient_id,
        "symptoms": symptoms,
        "ai_suggested_diagnosis": suggested_diagnosis,
        "doctor_actual_diagnosis": actual_diagnosis,
        "feedback_text": feedback_text,
        "override_reason": override_reason
    }
    feedback_data.append(feedback_entry)
    print(f"Feedback received: {feedback_entry}")
    return "Thank you for your feedback! It has been recorded for continuous improvement."

# --- Main Application Logic ---

def medical_diagnosis_agent(
    symptoms: str,
    patient_id: str,
    doctor_id: str,
    feedback_diagnosis: str = "",
    feedback_text: str = "",
    override_reason: str = ""
):
    """The main agent function orchestrating diagnosis and feedback.
    """
    # 1. Retrieve information (RAG)
    retrieved_kb_info = retrieve_medical_info(symptoms)
    patient_history = retrieve_patient_history(patient_id)
    patient_history_display = "No patient record found." if patient_history is None else \
                               f"Patient ID: {patient_history['patient_id']}, Name: {patient_history['name']}, " \
                               f"Age: {patient_history['age']}, Gender: {patient_history['gender']}, " \
                               f"Past Conditions: {patient_history['past_conditions']}, " \
                               f"Medications: {patient_history['medications']}, Last Visit: {patient_history['last_visit']}"

    # 2. Simulate LLM diagnosis
    llm_output = simulate_llm_diagnosis(symptoms, patient_id, retrieved_kb_info, patient_history)
    diagnosis = llm_output["diagnosis"]
    reasoning = llm_output["reasoning"]
    confidence = llm_output["confidence"]
    prob_diagnoses = llm_output["probabilistic_diagnoses"]

    # Format probabilistic diagnoses
    prob_diag_str = "\n".join([f"- {d}: {p*100:.1f}%" for d, p in prob_diagnoses.items()])

    # 3. Controlled Abstention
    abstention_message = ""
    if confidence < 0.6:
        abstention_message = (f"\n\n**AI Confidence is Low ({confidence*100:.1f}%).** The AI recommends further tests or consultation with a specialist for a definitive diagnosis.")
        diagnosis = "Requires further evaluation/Low Confidence"

    full_reasoning = f"**Patient History:**\n{patient_history_display}\n\n**Retrieved Medical Knowledge:**\n{retrieved_kb_info}\n\n**AI Reasoning Path:**\n{reasoning}"

    # 4. Handle feedback submission
    if feedback_diagnosis or feedback_text or override_reason:
        feedback_message = submit_feedback(
            doctor_id,
            patient_id,
            symptoms,
            diagnosis, # AI's suggestion at time of feedback
            feedback_diagnosis, # Doctor's actual/chosen diagnosis
            feedback_text,
            override_reason
        )
    else:
        feedback_message = ""

    return (
        f"**Probable Diagnosis:** {diagnosis}{abstention_message}\n\n" \
        f"**Confidence Score:** {confidence*100:.1f}%\n\n" \
        f"**Probabilistic Diagnoses:**\n{prob_diag_str}\n\n" \
        f"**Detailed Reasoning:**\n{full_reasoning}",
        feedback_message
    )

# --- Gradio UI ---

with gr.Blocks() as demo:
    gr.Markdown("# Agentic and Trustworthy Medical Diagnosis Assistant")
    gr.Markdown(
        "This assistant helps doctors with preliminary diagnoses by providing reasoning, "
        "confidence scores, and integrating with simulated medical knowledge and patient records. "
        "Doctors can provide feedback to improve the system."
    )

    with gr.Row():
        with gr.Column():
            symptoms_input = gr.Textbox(label="Patient Symptoms (e.g., 'fever, cough, fatigue')", lines=5, placeholder="Enter symptoms here...")
            patient_id_input = gr.Textbox(label="Patient ID (e.g., P001)", placeholder="Enter patient ID for history retrieval")
            doctor_id_input = gr.Textbox(label="Doctor ID", placeholder="Enter your ID for feedback")
            diagnose_btn = gr.Button("Get Diagnosis")

        with gr.Column():
            diagnosis_output = gr.Markdown(label="Diagnosis and Reasoning")
            feedback_output = gr.Markdown(label="Feedback Status")

    with gr.Accordion("Feedback and Override", open=False):
        gr.Markdown("## Provide Feedback / Override AI Suggestion")
        actual_diagnosis_input = gr.Textbox(label="Doctor's Actual Diagnosis (if different)", placeholder="e.g., 'Viral Bronchitis'")
        override_reason_input = gr.Textbox(label="Reason for Override (if applicable)", placeholder="e.g., 'Patient history indicated asthma exacerbation'")
        feedback_text_input = gr.Textbox(label="General Feedback / Comments", lines=3, placeholder="e.g., 'Reasoning was helpful but missed a key detail.'")
        submit_feedback_btn = gr.Button("Submit Feedback")

    # Define the primary diagnosis flow
    diagnose_btn.click(
        medical_diagnosis_agent,
        inputs=[symptoms_input, patient_id_input, doctor_id_input],
        outputs=[diagnosis_output, feedback_output]
    )

    # Define the feedback submission flow
    submit_feedback_btn.click(
        medical_diagnosis_agent,
        inputs=[
            symptoms_input,
            patient_id_input,
            doctor_id_input,
            actual_diagnosis_input,
            feedback_text_input,
            override_reason_input
        ],
        outputs=[diagnosis_output, feedback_output] # Output fields remain the same to show feedback status
    )

demo.launch()
