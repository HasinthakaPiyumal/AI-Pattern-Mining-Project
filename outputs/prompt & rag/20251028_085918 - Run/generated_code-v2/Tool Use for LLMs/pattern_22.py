"""
This script implements a Medical Diagnosis Assistant using FastAPI and Gradio.
It combines Natural Language Understanding, a simulated Reasoning Engine, Symbolic Computation,
and a Chain-of-Thought (CoT) Generator to provide diagnostic probabilities and explanations.
"""

import spacy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import random

# --- 0. Simulated Medical Knowledge Base ---

symptoms_to_diseases = {
    "fever": ["flu", "pneumonia", "malaria"],
    "cough": ["flu", "pneumonia", "bronchitis"],
    "sore throat": ["flu", "strep throat"],
    "headache": ["flu", "migraine"],
    "fatigue": ["flu", "anemia"],
    "shortness of breath": ["pneumonia", "asthma"],
    "chest pain": ["pneumonia", "heart attack"],
    "nausea": ["flu", "food poisoning"],
    "vomiting": ["flu", "food poisoning"],
    "rash": ["chickenpox", "measles"],
    "joint pain": ["arthritis", "lupus"],
}

disease_prevalence = {
    "flu": 0.3,
    "pneumonia": 0.1,
    "malaria": 0.01,
    "strep throat": 0.05,
    "migraine": 0.08,
    "anemia": 0.07,
    "bronchitis": 0.06,
    "heart attack": 0.02,
    "food poisoning": 0.04,
    "chickenpox": 0.005,
    "measles": 0.001,
    "arthritis": 0.03,
    "lupus": 0.002,
}

drug_interactions_db = {
    ("warfarin", "ibuprofen"): "Increased bleeding risk",
    ("antibiotics", "oral contraceptives"): "Decreased effectiveness of contraceptives",
    ("statins", "grapefruit juice"): "Increased statin levels, muscle pain risk",
}

# --- 1. Natural Language Understanding (NLU) Module ---
nlp = spacy.load("en_core_web_sm")

def extract_medical_entities(text: str) -> list[str]:
    doc = nlp(text.lower())
    extracted_symptoms = []
    for symptom in symptoms_to_diseases.keys():
        if symptom in text.lower():
            extracted_symptoms.append(symptom)
    # A more sophisticated NER would use spacy's entity recognition on medical vocab
    # For this example, a simple keyword matching is used.
    return extracted_symptoms

# --- 2. Reasoning Engine / Diagnostic Model ---

# Simulate training data for a diagnostic model
def generate_simulated_data(num_samples=1000):
    data = []
    labels = []
    all_symptoms = list(symptoms_to_diseases.keys())
    all_diseases = list(disease_prevalence.keys())

    for _ in range(num_samples):
        patient_symptoms = random.sample(all_symptoms, k=random.randint(1, 5))
        present_symptoms = {symptom: 1 for symptom in patient_symptoms}
        for s in all_symptoms:
            if s not in present_symptoms:
                present_symptoms[s] = 0

        # Simple rule-based labeling for simulation
        possible_diseases = set()
        for s in patient_symptoms:
            possible_diseases.update(symptoms_to_diseases.get(s, []))

        if possible_diseases:
            # Randomly pick one of the possible diseases, biased by prevalence
            disease_choices = list(possible_diseases)
            prevalences = [disease_prevalence.get(d, 0.001) for d in disease_choices]
            total_prevalence = sum(prevalences)
            probabilities = [p / total_prevalence for p in prevalences]
            diagnosis = random.choices(disease_choices, weights=probabilities, k=1)[0]
        else:
            diagnosis = random.choice(all_diseases) # Default if no specific match

        data.append(" ".join(patient_symptoms))
        labels.append(diagnosis)
    return data, labels

sim_data, sim_labels = generate_simulated_data(2000)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(sim_data)
y_train = sim_labels

diagnostic_model = LogisticRegression(max_iter=1000, solver='liblinear')
diagnostic_model.fit(X_train, y_train)

def predict_diagnosis(symptoms: list[str]) -> tuple[dict, str]:
    if not symptoms:
        return {}, "No specific symptoms provided for diagnosis."

    symptoms_text = " ".join(symptoms)
    X_test = vectorizer.transform([symptoms_text])

    if hasattr(diagnostic_model, 'predict_proba'):
        probabilities = diagnostic_model.predict_proba(X_test)[0]
        disease_probabilities = {
            diagnostic_model.classes_[i]: prob for i, prob in enumerate(probabilities)
        }
        sorted_diagnoses = sorted(disease_probabilities.items(), key=lambda item: item[1], reverse=True)
        top_diagnosis = sorted_diagnoses[0][0]

        reasoning = f"Based on the reported symptoms: {', '.join(symptoms)}, the most probable diagnosis is '{top_diagnosis}' with a probability of {sorted_diagnoses[0][1]:.2f}. Other possibilities include: "
        other_diagnoses = [f"{d[0]} ({d[1]:.2f})" for d in sorted_diagnoses[1:3]]
        reasoning += ", ".join(other_diagnoses) + "."

        return disease_probabilities, reasoning
    else:
        return {}, "Diagnostic model cannot provide probabilities."

# --- 3. Symbolic Computation Module ---

def calculate_simplified_cvd_risk(age: int, cholesterol: float, systolic_bp: float, smoker: bool) -> float:
    # A very simplified cardiovascular disease risk score for demonstration
    risk = 0.0
    if age > 50: risk += (age - 50) * 0.5
    if cholesterol > 200: risk += (cholesterol - 200) * 0.1
    if systolic_bp > 140: risk += (systolic_bp - 140) * 0.2
    if smoker: risk += 5.0
    return round(risk, 2)

def check_drug_interactions(drugs: list[str]) -> list[str]:
    interactions = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            drug1, drug2 = sorted([drugs[i].lower(), drugs[j].lower()])
            interaction = drug_interactions_db.get((drug1, drug2))
            if interaction:
                interactions.append(f"Interaction between {drug1.capitalize()} and {drug2.capitalize()}: {interaction}")
    return interactions

def recommend_dosage(drug_name: str, weight_kg: float, age_years: int) -> str:
    drug_name_lower = drug_name.lower()
    if "ibuprofen" in drug_name_lower:
        if age_years < 12:
            return f"For {drug_name}, consult a pediatrician for child-specific dosage. Typical adult dose: 200-400mg every 4-6 hours."
        else:
            return f"For {drug_name}, typical adult dosage is 200-400mg every 4-6 hours as needed. Max 1200mg/day."
    elif "paracetamol" in drug_name_lower or "acetaminophen" in drug_name_lower:
        if age_years < 12:
            return f"For {drug_name}, consult a pediatrician for child-specific dosage. Typical adult dose: 500-1000mg every 4-6 hours."
        else:
            return f"For {drug_name}, typical adult dosage is 500-1000mg every 4-6 hours as needed. Max 4000mg/day."
    else:
        return f"Dosage recommendation for {drug_name} is not available in the current knowledge base. Consult drug's prescribing information."


# --- 4. Chain-of-Thought (CoT) Generator / Explanation Module ---

def generate_faithful_cot(patient_data: dict, diagnosis_reasoning: str, symbolic_results: dict) -> str:
    cot_explanation = f"### Faithful Chain of Thought Explanation\n\n"
    cot_explanation += f"**1. Patient Information and Initial Assessment:**\n"
    cot_explanation += f"   - Patient reported symptoms: {patient_data.get('symptoms', 'None')}\n"
    cot_explanation += f"   - Other relevant data: Age={patient_data.get('age', 'N/A')}, Cholesterol={patient_data.get('cholesterol', 'N/A')}, Systolic BP={patient_data.get('systolic_bp', 'N/A')}, Smoker={patient_data.get('smoker', 'N/A')}, Medications={patient_data.get('medications', 'N/A')}\n\n"

    cot_explanation += f"**2. Diagnostic Reasoning (Natural Language):**\n"
    cot_explanation += f"   - {diagnosis_reasoning}\n\n"

    if symbolic_results:
        cot_explanation += f"**3. Verifiable Symbolic Computations (Python Logic):**\n"
        if 'cvd_risk_score' in symbolic_results:
            cot_explanation += f"   - **Cardiovascular Risk Score Calculation:**\n"
            cot_explanation += f"     - Inputs: Age={patient_data.get('age')}, Cholesterol={patient_data.get('cholesterol')}, Systolic BP={patient_data.get('systolic_bp')}, Smoker={patient_data.get('smoker')}\n"
            cot_explanation += f"     - Calculation Logic (simplified): \
                                 risk = 0.0; \
                                 if age > 50: risk += (age - 50) * 0.5; \
                                 if cholesterol > 200: risk += (cholesterol - 200) * 0.1; \
                                 if systolic_bp > 140: risk += (systolic_bp - 140) * 0.2; \
                                 if smoker: risk += 5.0; \
                                 Result: {symbolic_results['cvd_risk_score']}\n"

        if 'drug_interactions' in symbolic_results and symbolic_results['drug_interactions']:
            cot_explanation += f"   - **Drug Interaction Check:**\n"
            cot_explanation += f"     - Medications: {', '.join(patient_data.get('medications', []))}\n"
            cot_explanation += f"     - Identified Interactions: {'; '.join(symbolic_results['drug_interactions'])}\n"

        if 'dosage_recommendation' in symbolic_results:
            cot_explanation += f"   - **Dosage Recommendation:**\n"
            cot_explanation += f"     - Drug: {patient_data.get('medications', [''])[0] if patient_data.get('medications') else 'N/A'}, Weight: {patient_data.get('weight_kg', 'N/A')}, Age: {patient_data.get('age', 'N/A')}\n"
            cot_explanation += f"     - Recommendation: {symbolic_results['dosage_recommendation']}\n"
    else:
        cot_explanation += f"**3. Verifiable Symbolic Computations (Python Logic):**\n"
        cot_explanation += f"   - No specific symbolic computations were required or performed based on the provided data.\n"

    cot_explanation += f"\n**Conclusion:** This integrated approach combines human-readable reasoning with precise computational verification to aid in medical decision-making.\n"
    return cot_explanation

# --- 5. User Interface (UI) / API (FastAPI & Gradio) ---

app = FastAPI()

class PatientData(BaseModel):
    symptoms: str
    age: int = None
    cholesterol: float = None
    systolic_bp: float = None
    smoker: bool = False
    medications: list[str] = []
    weight_kg: float = None

@app.post("/diagnose")
async def diagnose_patient(data: PatientData):
    # NLU
    extracted_symptoms = extract_medical_entities(data.symptoms)

    # Reasoning Engine
    disease_probabilities, diagnosis_reasoning = predict_diagnosis(extracted_symptoms)

    # Symbolic Computation
    symbolic_results = {}
    if data.age and data.cholesterol and data.systolic_bp is not None:
        symbolic_results['cvd_risk_score'] = calculate_simplified_cvd_risk(data.age, data.cholesterol, data.systolic_bp, data.smoker)

    if data.medications:
        symbolic_results['drug_interactions'] = check_drug_interactions(data.medications)
        if data.weight_kg and data.age and len(data.medications) == 1: # Only recommend for single drug for simplicity
            symbolic_results['dosage_recommendation'] = recommend_dosage(data.medications[0], data.weight_kg, data.age)

    # CoT Generator
    patient_data_for_cot = data.dict()
    patient_data_for_cot['extracted_symptoms'] = extracted_symptoms # Add for clearer CoT

    faithful_cot_explanation = generate_faithful_cot(patient_data_for_cot, diagnosis_reasoning, symbolic_results)

    return {
        "diagnosis_probabilities": disease_probabilities,
        "natural_language_reasoning": diagnosis_reasoning,
        "symbolic_computation_results": symbolic_results,
        "faithful_chain_of_thought_explanation": faithful_cot_explanation,
    }

# Gradio Interface
def gradio_interface_fn(
    symptoms: str,
    age: int = None,
    cholesterol: float = None,
    systolic_bp: float = None,
    smoker: bool = False,
    medications: str = "", # Comma-separated string for Gradio
    weight_kg: float = None
):
    med_list = [m.strip() for m in medications.split(',') if m.strip()]
    patient_data = PatientData(
        symptoms=symptoms,
        age=age,
        cholesterol=cholesterol,
        systolic_bp=systolic_bp,
        smoker=smoker,
        medications=med_list,
        weight_kg=weight_kg
    )
    response = diagnose_patient(patient_data)
    return (
        response["natural_language_reasoning"],
        str(response["diagnosis_probabilities"]),
        str(response["symbolic_computation_results"]),
        response["faithful_chain_of_thought_explanation"]
    )

iface = gr.Interface(
    fn=gradio_interface_fn,
    inputs=[
        gr.Textbox(label="Symptoms (e.g., fever, cough, headache)"),
        gr.Number(label="Age (years)", optional=True),
        gr.Number(label="Cholesterol (mg/dL)", optional=True),
        gr.Number(label="Systolic Blood Pressure (mmHg)", optional=True),
        gr.Checkbox(label="Smoker"),
        gr.Textbox(label="Current Medications (comma-separated, e.g., Warfarin, Ibuprofen)", optional=True),
        gr.Number(label="Weight (kg)", optional=True),
    ],
    outputs=[
        gr.Textbox(label="Natural Language Diagnostic Reasoning"),
        gr.Textbox(label="Diagnostic Probabilities"),
        gr.Textbox(label="Symbolic Computation Results"),
        gr.Markdown(label="Faithful Chain of Thought Explanation")
    ],
    title="Medical Diagnosis Assistant with Faithful Chain of Thought",
    description="Enter patient symptoms and data to receive a diagnosis, relevant calculations, and a combined natural language and symbolic explanation."
)

# To run the FastAPI app: uvicorn medical_diagnosis_assistant:app --reload
# To run the Gradio app: iface.launch()

# Example of how to run the Gradio interface directly for testing:
# if __name__ == "__main__":
#     print("Launching Gradio interface...")
#     iface.launch(share=True)
