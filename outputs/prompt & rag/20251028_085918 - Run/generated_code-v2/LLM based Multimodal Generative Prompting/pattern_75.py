import pandas as pd
from PIL import Image
import io
import base64
import gradio as gr

def preprocess_image(image_bytes):
    if image_bytes is None:
        return None
    image = Image.open(io.BytesIO(image_bytes))
    return image.resize((256, 256)) # Example preprocessing

def preprocess_text(text):
    if text is None:
        return ""
    return text.strip()

def preprocess_lab_results(lab_results_csv):
    if lab_results_csv is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(lab_results_csv))
        return df
    except Exception:
        return pd.DataFrame()

def decompose_questions(symptoms, doctor_notes, has_image, has_lab_results):
    questions = []
    if has_image:
        questions.append("What abnormalities are visible in the medical image?")
    if symptoms:
        questions.append("What are the likely conditions based on the patient's symptoms?")
    if doctor_notes:
        questions.append("What additional insights can be gathered from the doctor's notes?")
    if has_lab_results:
        questions.append("How do the lab results correlate with potential diagnoses?")
    
    if not questions:
        questions.append("Please provide more information for a comprehensive analysis.")
    return questions

def analyze_image(processed_image):
    if processed_image is None:
        return "No image provided for analysis."
    # Simulate image analysis results
    return "Possible signs of inflammation in the lung area (simulated result)."

def analyze_symptoms_notes(symptoms, doctor_notes):
    if not symptoms and not doctor_notes:
        return "No symptoms or doctor's notes provided for NLP analysis."
    
    combined_text = f"Symptoms: {symptoms}. Doctor's Notes: {doctor_notes}"
    # Simulate NLP extraction
    extracted_entities = []
    if "fever" in combined_text.lower():
        extracted_entities.append("fever")
    if "cough" in combined_text.lower():
        extracted_entities.append("cough")
    if "pneumonia" in combined_text.lower():
        extracted_entities.append("suspected pneumonia")
    
    return f"Extracted medical entities: {', '.join(extracted_entities) or 'None'}."

def analyze_lab_results(lab_results_df):
    if lab_results_df.empty:
        return "No lab results provided for analysis."
    
    findings = []
    if 'WBC' in lab_results_df.columns and not lab_results_df.empty:
        wbc_count = lab_results_df['WBC'].iloc[0]
        if wbc_count > 10.0: # Example threshold
            findings.append(f"Elevated WBC count ({wbc_count}), possibly indicating infection.")
        elif wbc_count < 4.0:
            findings.append(f"Low WBC count ({wbc_count}), possibly indicating weakened immune system.")
        else:
            findings.append(f"WBC count ({wbc_count}) within normal range.")
    
    if 'CRP' in lab_results_df.columns and not lab_results_df.empty:
        crp_level = lab_results_df['CRP'].iloc[0]
        if crp_level > 5.0: # Example threshold
            findings.append(f"Elevated CRP level ({crp_level}), indicating inflammation.")
        else:
            findings.append(f"CRP level ({crp_level}) within normal range.")

    return "Lab result interpretations: " + ("; ".join(findings) or "No significant findings based on provided data.")

def integrate_and_reason(image_analysis_output, nlp_output, lab_results_output, sub_questions):
    reasoning_steps = []
    diagnostic_hypothesis = ""
    suggested_actions = []
    explanation = ""

    reasoning_steps.append(f"Based on sub-questions: {', '.join(sub_questions)}")

    if "lung" in image_analysis_output.lower() and "inflammation" in image_analysis_output.lower():
        reasoning_steps.append("Image analysis suggests lung inflammation.")
        suggested_actions.append("Recommend further imaging (e.g., CT scan) for detailed lung assessment.")

    if "fever" in nlp_output.lower() or "cough" in nlp_output.lower():
        reasoning_steps.append("Symptoms include fever and/or cough.")
        if "suspected pneumonia" in nlp_output.lower():
            reasoning_steps.append("NLP analysis indicates suspected pneumonia.")
            suggested_actions.append("Consider antibiotic treatment and sputum culture.")
        else:
            suggested_actions.append("Recommend viral panel and rest.")

    if "elevated wbc" in lab_results_output.lower() or "elevated crp" in lab_results_output.lower():
        reasoning_steps.append("Lab results show markers of infection/inflammation.")
        suggested_actions.append("Monitor inflammatory markers.")

    if "lung inflammation" in ' '.join(reasoning_steps).lower() and \
       ("fever" in ' '.join(reasoning_steps).lower() or "cough" in ' '.join(reasoning_steps).lower()) and \
       ("elevated wbc" in ' '.join(reasoning_steps).lower() or "elevated crp" in ' '.join(reasoning_steps).lower()):
        diagnostic_hypothesis = "High likelihood of bacterial pneumonia."
        explanation = "Multiple sources (image, symptoms, labs) point towards a bacterial lung infection."
    elif "inflammation" in ' '.join(reasoning_steps).lower():
        diagnostic_hypothesis = "Likely inflammatory process, cause unclear."
        explanation = "Image and/or lab findings suggest inflammation, but symptoms are non-specific."
    else:
        diagnostic_hypothesis = "Insufficient data for a definitive diagnosis or no obvious pathology detected."
        explanation = "Please provide more information for a comprehensive diagnosis."
    
    if not suggested_actions:
        suggested_actions.append("Observe patient and follow up as needed.")

    return {
        "diagnostic_hypothesis": diagnostic_hypothesis,
        "suggested_further_actions": list(set(suggested_actions)), # Remove duplicates
        "explanation": explanation,
        "reasoning_steps": reasoning_steps
    }

def medical_diagnosis_assistant(image_file, symptoms_text, lab_results_csv_file, doctor_notes_text):
    # 1. Input Layer & Preprocessing
    image_bytes = image_file.read() if image_file else None
    processed_image = preprocess_image(image_bytes)
    processed_symptoms = preprocess_text(symptoms_text)
    processed_doctor_notes = preprocess_text(doctor_notes_text)
    lab_results_csv_data = lab_results_csv_file.read().decode('utf-8') if lab_results_csv_file else None
    processed_lab_results = preprocess_lab_results(lab_results_csv_data)

    # 2. Sub-question Decomposition Module
    sub_questions = decompose_questions(
        processed_symptoms, processed_doctor_notes,
        processed_image is not None, not processed_lab_results.empty
    )

    # 3. Specialized AI Modules (Solvers)
    image_analysis_output = analyze_image(processed_image)
    nlp_output = analyze_symptoms_notes(processed_symptoms, processed_doctor_notes)
    lab_results_output = analyze_lab_results(processed_lab_results)

    # 4. Integration & Reasoning Module
    final_diagnosis_output = integrate_and_reason(
        image_analysis_output, nlp_output, lab_results_output, sub_questions
    )

    # 5. Output Layer
    output_str = f"Diagnostic Hypothesis: {final_diagnosis_output['diagnostic_hypothesis']}\n\n"
    output_str += f"Suggested Further Actions: {', '.join(final_diagnosis_output['suggested_further_actions'])}\n\n"
    output_str += f"Explanation: {final_diagnosis_output['explanation']}\n\n"
    output_str += "--- Detailed Analysis ---\n"
    output_str += f"Sub-questions generated: {'; '.join(sub_questions)}\n"
    output_str += f"Image Analysis: {image_analysis_output}\n"
    output_str += f"NLP (Symptoms & Notes): {nlp_output}\n"
    output_str += f"Lab Results Analysis: {lab_results_output}\n"
    output_str += f"Reasoning Steps: {' '.join(final_diagnosis_output['reasoning_steps'])}"

    return output_str

# Gradio Interface
iface = gr.Interface(
    fn=medical_diagnosis_assistant,
    inputs=[
        gr.File(type="bytes", label="Upload X-ray/MRI Image (Optional)", file_count="single", accept=[".png", ".jpg", ".jpeg"]),
        gr.Textbox(lines=5, label="Patient Symptoms (e.g., 'fever, cough, shortness of breath')", placeholder="Enter symptoms here..."),
        gr.File(type="bytes", label="Upload Lab Results (CSV, Optional)", file_count="single", accept=[".csv"]),
        gr.Textbox(lines=5, label="Doctor's Notes (Optional)", placeholder="Enter doctor's notes here...")
    ],
    outputs=gr.Textbox(label="Diagnostic Report"),
    title="Multimodal Medical Diagnosis Assistant (DDCoT Demo)",
    description="Upload medical images, provide symptoms, lab results, and doctor's notes to receive a comprehensive diagnostic hypothesis based on decomposed reasoning."
)

if __name__ == "__main__":
    iface.launch()
