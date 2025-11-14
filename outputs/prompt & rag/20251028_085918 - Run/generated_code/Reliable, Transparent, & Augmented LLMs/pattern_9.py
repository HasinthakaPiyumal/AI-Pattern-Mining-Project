
import gradio as gr
import re
import string

# --- 1. Mock External Tools/Services --- 

# Mock Medical Knowledge Base Search
MEDICAL_KNOWLEDGE_BASE = {
    "fever": "Fever is a temporary increase in your body temperature, often due to an illness.",
    "cough": "A cough is a reflex action to clear your airways of mucus and irritants.",
    "headache": "A headache is pain in any region of the head.",
    "fatigue": "Fatigue is a feeling of extreme tiredness, lack of energy, and motivation.",
    "sore throat": "A sore throat is pain or irritation of the throat, often due to a viral infection.",
    "influenza": "Influenza (flu) is a contagious respiratory illness caused by flu viruses.",
    "common cold": "The common cold is a viral infection of your nose and throat (upper respiratory tract).",
    "migraine": "A migraine is a type of headache that can cause severe throbbing pain or a pulsing sensation.",
    "pneumonia": "Pneumonia is an infection that inflames air sacs in one or both lungs.",
    "bronchitis": "Bronchitis is an inflammation of the lining of your bronchial tubes."
}

def search_medical_knowledge_base(query: str) -> str:
    """Simulates searching a medical knowledge base for relevant information."""
    query_lower = query.lower()
    results = [v for k, v in MEDICAL_KNOWLEDGE_BASE.items() if k in query_lower or query_lower in v.lower()]
    if results:
        return "\n".join(results)
    return "No specific information found for this query in the knowledge base."

# Mock Diagnostic Algorithm Integration
def run_diagnostic_algorithm(symptoms: list) -> dict:
    """Simulates running a diagnostic algorithm based on symptoms.
    Returns a hypothetical diagnosis and its probability.
    """
    symptoms_str = " ".join(symptoms).lower()

    if "fever" in symptoms_str and "cough" in symptoms_str and "fatigue" in symptoms_str:
        if "sore throat" in symptoms_str:
            return {"diagnosis": "Influenza", "probability": 0.85, "evidence": "Presence of fever, cough, fatigue, and sore throat strongly suggests influenza."}
        else:
            return {"diagnosis": "Common Cold or early Influenza", "probability": 0.70, "evidence": "Fever, cough, and fatigue are common to both common cold and influenza. Further symptoms needed for differentiation."}
    elif "headache" in symptoms_str and "throbbing" in symptoms_str:
        return {"diagnosis": "Migraine", "probability": 0.90, "evidence": "Severe throbbing headache is characteristic of migraine."}
    elif "cough" in symptoms_str and "difficulty breathing" in symptoms_str:
        return {"diagnosis": "Pneumonia or Bronchitis", "probability": 0.75, "evidence": "Cough and breathing difficulty are indicative of respiratory infections like pneumonia or bronchitis."}
    elif "fever" in symptoms_str:
        return {"diagnosis": "Unspecified Viral Infection", "probability": 0.60, "evidence": "Fever is a general symptom of many viral infections."}
    
    return {"diagnosis": "Undetermined", "probability": 0.40, "evidence": "Insufficient specific symptoms for a confident diagnosis at this time."}

# Mock Patient Record Access
PATIENT_RECORDS = {
    "patient_A": {"age": 45, "gender": "male", "history": "mild asthma", "lab_results": {"WBC": "normal"}},
    "patient_B": {"age": 30, "gender": "female", "history": "no significant medical history", "lab_results": {"CRP": "elevated"}},
}

def access_patient_records(patient_id: str) -> dict:
    """Simulates secure access to anonymized patient records."""
    return PATIENT_RECORDS.get(patient_id, {"error": "Patient ID not found or unauthorized access."})

# --- 2. Robust Processing & Input Validation --- 

def nlp_preprocess(text: str) -> str:
    """Performs basic NLP pre-processing on input text."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # Normalize whitespace
    return text

def detect_adversarial_input(text: str) -> bool:
    """Basic heuristic to detect potentially adversarial or highly unusual inputs."""
    # Example: Look for very long strings of random characters, SQL injection patterns, etc.
    if len(text) > 500 or not any(char.isalpha() for char in text): # Very long or no alphabetic chars
        return True
    return False

# --- 3. LLM Agent (Orchestrator) --- 

class LLMAgent:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def _mock_llm_reasoning(self, preprocessed_symptoms: str, kb_results: str, algo_results: dict, patient_info: dict) -> dict:
        """Simulates LLM's reasoning and response generation, including confidence.
        In a real system, this would involve complex prompt engineering and actual LLM calls.
        """
        diagnosis = algo_results.get("diagnosis", "Undetermined")
        probability = algo_results.get("probability", 0.0)
        evidence_from_algo = algo_results.get("evidence", "No specific evidence from diagnostic algorithm.")

        reasoning_steps = [
            f"1. Analyzed preprocessed symptoms: '{preprocessed_symptoms}'.",
            f"2. Consulted medical knowledge base: '{kb_results[:100]}...'",
            f"3. Applied diagnostic algorithms, yielding a primary diagnosis of '{diagnosis}' with a probability of {probability:.2f}.",
            f"4. Considered patient information (if available): {patient_info.get('history', 'N/A')}.",
            f"5. Based on the aggregated information, the most likely diagnosis is {diagnosis}."
        ]
        
        final_confidence = probability # Simple mapping for mock

        if final_confidence < self.confidence_threshold:
            return {
                "diagnosis": "Uncertain - Further Investigation Needed",
                "reasoning": "\n".join(reasoning_steps) + f"\nConfidence ({final_confidence:.2f}) is below threshold ({self.confidence_threshold:.2f}).",
                "confidence": final_confidence,
                "recommendation": "Recommend additional tests or specialist consultation due to low confidence."
            }

        return {
            "diagnosis": diagnosis,
            "reasoning": "\n".join(reasoning_steps) + f"\nSupporting evidence: {evidence_from_algo}.",
            "confidence": final_confidence,
            "recommendation": f"Consider treatment plan for {diagnosis}."
        }

    def orchestrate_diagnosis(self, patient_symptoms: str, patient_id: str = None) -> dict:
        """Orchestrates the diagnostic process using external tools and LLM reasoning."""
        if detect_adversarial_input(patient_symptoms):
            return {"error": "Potential adversarial input detected. Please provide a valid description of symptoms.", "confidence": 0.0}

        preprocessed_symptoms = nlp_preprocess(patient_symptoms)
        symptom_list = preprocessed_symptoms.split()

        # Tool Call 1: Medical Knowledge Base Search
        kb_results = search_medical_knowledge_base(preprocessed_symptoms)

        # Tool Call 2: Diagnostic Algorithm Integration
        algo_results = run_diagnostic_algorithm(symptom_list)

        # Tool Call 3: Patient Record Access (if patient_id is provided)
        patient_info = {} 
        if patient_id:
            patient_info = access_patient_records(patient_id)

        # LLM Reasoning and Response Generation
        llm_response = self._mock_llm_reasoning(preprocessed_symptoms, kb_results, algo_results, patient_info)

        return llm_response

# --- 4. Human-AI Interaction Layer (Gradio UI) --- 

agent = LLMAgent()

def diagnostic_interface(symptoms: str, patient_id: str = "") -> tuple:
    if not symptoms.strip():
        return "Please enter patient symptoms.", "", "", "", ""

    result = agent.orchestrate_diagnosis(symptoms, patient_id if patient_id.strip() else None)

    if "error" in result:
        return result["error"], "N/A", "N/A", "N/A", ""

    diagnosis = result.get("diagnosis", "N/A")
    reasoning = result.get("reasoning", "N/A")
    confidence = f"{result.get('confidence', 0.0)*100:.2f}%"
    recommendation = result.get("recommendation", "No specific recommendation.")

    # Display patient info if available
    patient_display = ""
    if patient_id and patient_id.strip() and "error" not in access_patient_records(patient_id):
        patient_data = access_patient_records(patient_id)
        patient_display = f"**Patient ID:** {patient_id}\n**Age:** {patient_data.get('age')}\n**Gender:** {patient_data.get('gender')}\n**History:** {patient_data.get('history')}\n**Lab Results:** {patient_data.get('lab_results')}"
    elif patient_id and patient_id.strip():
        patient_display = f"Patient ID '{patient_id}' not found or invalid."

    return diagnosis, reasoning, confidence, recommendation, patient_display

with gr.Blocks(title="AI Medical Diagnostic Assistant") as demo:
    gr.Markdown(
        """
        # AI-Powered Medical Diagnostic Assistant
        This assistant helps healthcare professionals by suggesting diagnoses, providing reasoning, and estimating confidence.
        **Note: This is a demo for architectural illustration and should NOT be used for actual medical advice.**
        """
    )

    with gr.Row():
        symptoms_input = gr.Textbox(label="Patient Symptoms (e.g., 'fever, cough, severe headache for 2 days')", lines=5, placeholder="Describe the patient's symptoms...")
        patient_id_input = gr.Textbox(label="Patient ID (Optional, e.g., 'patient_A')", placeholder="Enter patient ID if available...")
    
    diagnose_button = gr.Button("Get Diagnosis")

    with gr.Column():
        diagnosis_output = gr.Textbox(label="Suggested Diagnosis", interactive=False)
        confidence_output = gr.Textbox(label="Confidence", interactive=False)
        reasoning_output = gr.Textbox(label="Reasoning Path", interactive=False, lines=10)
        recommendation_output = gr.Textbox(label="Recommendations", interactive=False)
        patient_info_output = gr.Markdown(label="Patient Information", interactive=False)

    diagnose_button.click(
        fn=diagnostic_interface,
        inputs=[symptoms_input, patient_id_input],
        outputs=[diagnosis_output, reasoning_output, confidence_output, recommendation_output, patient_info_output]
    )

demo.launch()
