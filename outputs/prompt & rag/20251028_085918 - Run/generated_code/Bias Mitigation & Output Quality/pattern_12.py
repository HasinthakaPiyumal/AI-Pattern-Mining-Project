import gradio as gr
from symptom_processor import SymptomProcessor
from medical_knowledge_base import MedicalKnowledgeBase
from exemplar_selector import ExemplarSelector
from prompt_generator import PromptGenerator
from llm_integrator import LLMIntegrator
from ensembler import Ensembler

# Initialize components
symptom_processor = SymptomProcessor()
medical_kb = MedicalKnowledgeBase()
medical_kb.add_medical_cases([
    {"symptoms": "fever, cough, sore throat", "diagnosis": "Common Cold"},
    {"symptoms": "fever, body aches, fatigue, headache", "diagnosis": "Influenza"},
    {"symptoms": "chest pain, shortness of breath, left arm pain", "diagnosis": "Heart Attack"},
    {"symptoms": "severe headache, stiff neck, fever", "diagnosis": "Meningitis"},
    {"symptoms": "fever, cough, difficulty breathing, chills", "diagnosis": "Pneumonia"},
    {"symptoms": "abdominal pain, nausea, vomiting, loss of appetite", "diagnosis": "Appendicitis"},
    {"symptoms": "joint pain, swelling, stiffness in the morning", "diagnosis": "Rheumatoid Arthritis"},
    {"symptoms": "sudden weakness on one side of the body, difficulty speaking, vision changes", "diagnosis": "Stroke"},
    {"symptoms": "frequent urination, increased thirst, unexplained weight loss", "diagnosis": "Diabetes Mellitus Type 2"},
    {"symptoms": "skin rash, itching, redness, swelling", "diagnosis": "Allergic Reaction"},
    {"symptoms": "persistent cough, weight loss, night sweats, fatigue", "diagnosis": "Tuberculosis"},
    {"symptoms": "severe back pain, numbness or tingling in legs, muscle weakness", "diagnosis": "Herniated Disc"},
    {"symptoms": "dizziness, spinning sensation, nausea", "diagnosis": "Vertigo"},
    {"symptoms": "difficulty sleeping, low mood, loss of interest, fatigue", "diagnosis": "Depression"},
    {"symptoms": "racing heart, shortness of breath, sweating, chest pain (panic attack)", "diagnosis": "Anxiety Disorder"}
])
exemplar_selector = ExemplarSelector(medical_kb)
prompt_generator = PromptGenerator()
llm_integrator = LLMIntegrator()
ensembler = Ensembler()

def diagnose_patient(raw_symptoms, num_exemplar_subsets=3):
    # 1. Symptom Processing
    processed_symptoms = symptom_processor.process_symptoms(raw_symptoms)
    print(f"Processed Symptoms: {processed_symptoms}")

    # 2. Exemplar Selection
    relevant_exemplars = exemplar_selector.get_relevant_exemplars(processed_symptoms)
    if not relevant_exemplars:
        return "No relevant medical cases found in the knowledge base. Please provide more detailed symptoms."
    
    exemplar_subsets = exemplar_selector.create_exemplar_subsets(relevant_exemplars, num_exemplar_subsets)
    print(f"Generated {len(exemplar_subsets)} exemplar subsets.")

    # 3. Prompt Generation and LLM Integration
    llm_diagnoses = []
    for i, subset in enumerate(exemplar_subsets):
        prompt = prompt_generator.generate_prompt(processed_symptoms, subset)
        print(f"\n--- Prompt {i+1} ---\n{prompt}")
        diagnosis = llm_integrator.get_llm_diagnosis(prompt)
        llm_diagnoses.append(diagnosis)
        print(f"LLM Diagnosis {i+1}: {diagnosis}")
    
    # 4. Ensembling and Aggregation
    final_diagnosis = ensembler.ensemble_diagnoses(llm_diagnoses)
    print(f"\nFinal Ensembled Diagnosis: {final_diagnosis}")

    return final_diagnosis

# Gradio Interface
iface = gr.Interface(
    fn=diagnose_patient,
    inputs=[
        gr.Textbox(label="Enter Patient Symptoms", placeholder="e.g., fever, cough, sore throat, fatigue", lines=5),
        gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Number of Exemplar Subsets for DENSE")
    ],
    outputs=gr.Textbox(label="Preliminary Diagnosis"),
    title="Few-Shot Medical Diagnosis Assistant (DENSE)",
    description="Input patient symptoms to receive a preliminary diagnosis leveraging Demonstration Ensembling for improved robustness and accuracy."
)

if __name__ == "__main__":
    iface.launch()