from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from transformers import pipeline
import gradio as gr
import os

# --- Pydantic Models for Structured Output ---
class DiagnosisOutput(BaseModel):
    diagnosis: str = Field(..., description="The preliminary medical diagnosis.")
    reasoning: str = Field(..., description="Step-by-step reasoning for the diagnosis.")
    confidence_score: float = Field(..., description="Confidence score (0.0 to 1.0) for the diagnosis.")
    suggested_tests: List[str] = Field(default_factory=list, description="Suggested follow-up tests.")
    disclaimer: str = Field("This is an AI-generated preliminary diagnosis and should not replace professional medical advice.", description="Standard medical disclaimer.")

# --- Mock External Medical Database (for demonstration) ---
def mock_medical_database_lookup(query: str) -> List[str]:
    """Simulates looking up information in a medical database."""
    medical_data = {
        "fever": ["Common cold", "Flu", "Infection"],
        "cough": ["Common cold", "Bronchitis", "Allergies"],
        "headache": ["Migraine", "Tension headache", "Sinusitis"],
        "fatigue": ["Anemia", "Hypothyroidism", "Chronic fatigue syndrome"],
        "sore throat": ["Strep throat", "Tonsillitis", "Common cold"],
        "chest pain": ["Heartburn", "Anxiety", "Pneumonia", "Angina"],
        "shortness of breath": ["Asthma", "Anxiety", "Heart failure", "COPD"]
    }
    results = []
    for key, values in medical_data.items():
        if key in query.lower():
            results.extend(values)
    return list(set(results))

# --- LLM Integration (using Hugging Face Transformers pipeline) ---
# Using a generic text generation model for demonstration. 
# In a real application, this would be a specialized medical LLM.
# Ensure you have 'torch' or 'tensorflow' installed for transformers.
# You might need to install: pip install transformers torch pydantic gradio

try:
    # Using 'distilgpt2' as a lightweight example. For better results, use a larger model like 'gpt2' or a specialized medical LLM.
    llm_pipeline = pipeline("text-generation", model="distilgpt2", device=-1) # -1 for CPU, 0 for GPU
except Exception as e:
    print(f"Warning: Could not load LLM pipeline (distilgpt2). Falling back to dummy responses. Error: {e}")
    llm_pipeline = None

def get_llm_response(prompt: str, max_new_tokens: int = 200) -> str:
    if llm_pipeline:
        response = llm_pipeline(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, 
                                do_sample=True, temperature=0.7, top_p=0.9, truncation=True)
        return response[0]['generated_text'].strip()
    else:
        return "Error: LLM pipeline not loaded. Cannot generate response."

# --- Core Diagnostic Function ---
def diagnose_patient(symptoms: str) -> DiagnosisOutput:
    """Generates a preliminary diagnosis, reasoning, and confidence score using LLM and mock DB."""
    
    # Step 1: Query external medical database (simulated)
    db_info = mock_medical_database_lookup(symptoms)
    db_info_str = f"Based on medical knowledge, symptoms like {symptoms} can be associated with: {', '.join(db_info) if db_info else 'no specific conditions found'}."

    # Step 2: Construct LLM prompt for diagnosis, reasoning, and confidence
    prompt = f"""You are an AI medical assistant. Analyze the following patient symptoms and provide a preliminary diagnosis, detailed reasoning, a confidence score from 0.0 to 1.0, and suggest relevant follow-up tests. 
{db_info_str}
Patient symptoms: {symptoms}

Provide your response in the following structured format:
Diagnosis: <Your diagnosis>
Reasoning: <Detailed reasoning for your diagnosis, explaining how symptoms lead to it, referencing medical knowledge, and considering differentials.>
Confidence Score: <A single float number between 0.0 and 1.0 indicating your certainty.>
Suggested Tests: <A comma-separated list of tests, e.g., 'Blood Test', 'X-ray'>
"""

    llm_raw_output = get_llm_response(prompt)
    
    # Step 3: Parse LLM output using Pydantic (or manual parsing if LLM is less structured)
    diagnosis = "No diagnosis generated."
    reasoning = "No reasoning provided."
    confidence_score = 0.0
    suggested_tests = []

    # Simple parsing logic - ideally, fine-tune LLM for perfect structured output or use a more robust parser.
    try:
        lines = llm_raw_output.split('\n')
        output_dict = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                output_dict[key.strip()] = value.strip()
        
        diagnosis = output_dict.get('Diagnosis', 'No diagnosis generated.').replace('Diagnosis:', '').strip()
        reasoning = output_dict.get('Reasoning', 'No reasoning provided.').replace('Reasoning:', '').strip()
        
        try:
            confidence_str = output_dict.get('Confidence Score', '0.0')
            # Extract the first float found in the string
            import re
            match = re.search(r'\d+\.?\d*', confidence_str)
            if match:
                confidence_score = float(match.group(0))
                confidence_score = max(0.0, min(1.0, confidence_score)) # Ensure it's between 0 and 1
            else:
                confidence_score = 0.0
        except ValueError:
            confidence_score = 0.0
        
        tests_str = output_dict.get('Suggested Tests', '')
        suggested_tests = [test.strip() for test in tests_str.split(',') if test.strip()]

    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        print(f"Raw LLM output:\n{llm_raw_output}")
        # Fallback to a generic error message
        reasoning = f"Error parsing LLM output: {e}. Raw output: {llm_raw_output}"

    return DiagnosisOutput(
        diagnosis=diagnosis,
        reasoning=reasoning,
        confidence_score=confidence_score,
        suggested_tests=suggested_tests
    )

# --- Gradio Interface ---
def gradio_interface(symptoms: str, feedback: str = ""):
    if not symptoms:
        return "Please enter symptoms.", "", 0.0, [], ""

    # Simulate feedback storage (in a real app, this would go to a DB)
    if feedback:
        with open("user_feedback.log", "a") as f:
            f.write(f"Symptoms: {symptoms}\nFeedback: {feedback}\n---\n")
        print("Feedback received and logged.")

    output = diagnose_patient(symptoms)
    
    # Format output for Gradio
    formatted_tests = "\n- " + "\n- ".join(output.suggested_tests) if output.suggested_tests else "None"
    
    explanation_text = f"**Reasoning:**\n{output.reasoning}\n\n**Suggested Tests:**{formatted_tests}\n\n**Disclaimer:** {output.disclaimer}"

    return output.diagnosis, explanation_text, output.confidence_score, output.suggested_tests, "Feedback logged successfully!" if feedback else ""


# Create Gradio Blocks interface for better layout control
with gr.Blocks() as demo:
    gr.Markdown("# AI Medical Diagnostic Assistant")
    gr.Markdown("Enter patient symptoms to receive a preliminary diagnosis, reasoning, and confidence score. This system is for informational purposes only and does not replace professional medical advice.")
    
    with gr.Row():
        symptoms_input = gr.Textbox(label="Patient Symptoms", placeholder="e.g., fever, cough, fatigue, headache, body aches")
    
    with gr.Row():
        diagnose_btn = gr.Button("Get Diagnosis")

    with gr.Row():
        with gr.Column(scale=1):
            diagnosis_output = gr.Textbox(label="Preliminary Diagnosis", interactive=False, lines=2)
            confidence_output = gr.Number(label="Confidence Score (0.0 - 1.0)", interactive=False, precision=2)
        with gr.Column(scale=2):
            reasoning_output = gr.Textbox(label="Reasoning & Suggestions", interactive=False, lines=10)
    
    gr.Markdown("### Provide Feedback")
    feedback_input = gr.Textbox(label="Your Feedback (e.g., 'Diagnosis correct', 'Confidence too high', 'Missing test')", lines=3)
    feedback_btn = gr.Button("Submit Feedback")
    feedback_status = gr.Textbox(label="Feedback Status", interactive=False)

    diagnose_btn.click(
        fn=gradio_interface,
        inputs=[symptoms_input],
        outputs=[diagnosis_output, reasoning_output, confidence_output, gr.State([]), feedback_status] # gr.State([]) for suggested_tests to avoid display issues if not needed
    )

    feedback_btn.click(
        fn=gradio_interface,
        inputs=[symptoms_input, feedback_input],
        outputs=[diagnosis_output, reasoning_output, confidence_output, gr.State([]), feedback_status]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=False) # Set share=True to get a public link (for testing/sharing)

