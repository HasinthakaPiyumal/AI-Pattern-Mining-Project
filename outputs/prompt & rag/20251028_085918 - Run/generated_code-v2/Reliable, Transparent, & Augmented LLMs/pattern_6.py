import gradio as gr
import random

def simulate_llm_diagnosis(symptoms: str):
    # Simulate LLM behavior: generate a diagnosis and a random confidence score
    # In a real application, this would be an actual LLM call
    possible_diagnoses = [
        "Common Cold",
        "Influenza",
        "Bacterial Pneumonia",
        "Asthma Exacerbation",
        "Migraine",
        "Tension Headache",
        "Gastroenteritis",
        "Appendicitis",
        "Type 2 Diabetes",
        "Hypertension"
    ]
    diagnosis = random.choice(possible_diagnoses)
    confidence = round(random.uniform(0.1, 0.95), 2)  # Simulate varied confidence
    return diagnosis, confidence

confidence_threshold = 0.7

def diagnose_patient(symptoms: str):
    diagnosis, confidence = simulate_llm_diagnosis(symptoms)
    
    review_recommendation = ""
    if confidence < confidence_threshold:
        review_recommendation = "Recommendation: Low confidence. Human medical professional review strongly recommended or consider further diagnostic tests."
    else:
        review_recommendation = "Recommendation: High confidence. Proceed with caution and standard medical protocols."
        
    return diagnosis, confidence, review_recommendation


iface = gr.Interface(
    fn=diagnose_patient,
    inputs=gr.Textbox(lines=5, label="Enter Patient Symptoms (e.g., 'fever, cough, fatigue, shortness of breath')"),
    outputs=[
        gr.Textbox(label="AI Diagnosis"),
        gr.Number(label="Confidence Score (0-1)"),
        gr.Textbox(label="Review Recommendation")
    ],
    title="Medical Diagnostic AI Assistant with Confidence Estimation",
    description="Input patient symptoms and receive an AI diagnosis with a self-rated confidence score. Diagnoses with low confidence will be flagged for human review."
)

iface.launch(share=False)