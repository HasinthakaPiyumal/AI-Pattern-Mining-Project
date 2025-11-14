
import gradio as gr
from llm_reasoning_engine import LLMReasoningEngine
from knowledge_base_verifier import KnowledgeBaseVerifier
from prompt_manager import PromptManager
import os

# Initialize modules
# For demonstration, we'll use a placeholder for an actual LLM. 
# In a real application, you'd configure your OpenAI, Anthropic, or other LLM client here.
# Ensure you have your API key set as an environment variable, e.g., OPENAI_API_KEY
llm_engine = LLMReasoningEngine(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_LLM_API_KEY"))
verifier = KnowledgeBaseVerifier()
prompt_manager = PromptManager()

def diagnose_patient(symptoms: str, medical_history: str):
    """
    Orchestrates the diagnostic process using LLM reasoning and verification.
    """
    if not os.environ.get("OPENAI_API_KEY") and "YOUR_LLM_API_KEY" in llm_engine.api_key:
        return "Error: LLM API Key not set. Please set OPENAI_API_KEY environment variable or replace 'YOUR_LLM_API_KEY'.", "", ""

    # 1. Construct initial prompt
    initial_prompt = prompt_manager.construct_initial_diagnostic_prompt(symptoms, medical_history)

    # 2. Get LLM to generate initial diagnosis and Chain-of-Thought reasoning
    try:
        llm_output, detailed_reasoning_steps = llm_engine.generate_diagnosis_and_reasoning(initial_prompt)
    except Exception as e:
        return f"Error during LLM generation: {e}", "", ""

    diagnosis = llm_output.get("diagnosis", "N/A")
    treatment_suggestions = llm_output.get("treatment_suggestions", "N/A")

    # 3. Perform verification
    verification_results = verifier.verify_diagnosis_and_reasoning(diagnosis, detailed_reasoning_steps)
    
    # Combine reasoning and verification for explanation
    full_explanation = f"**Detailed Reasoning Steps:**\n{detailed_reasoning_steps}\n\n**Verification Against Medical Knowledge:**\n{verification_results}"

    return diagnosis, treatment_suggestions, full_explanation

# Gradio Interface
iface = gr.Interface(
    fn=diagnose_patient,
    inputs=[
        gr.Textbox(label="Patient Symptoms (e.g., 'fever, cough, fatigue')"),
        gr.Textbox(label="Medical History (e.g., '2-day history of illness, no allergies')")
    ],
    outputs=[
        gr.Textbox(label="Differential Diagnosis"),
        gr.Textbox(label="Treatment Suggestions"),
        gr.Markdown(label="Reasoning and Verification")
    ],
    title="AI Medical Diagnostic Assistant with Explainable Reasoning",
    description="Enter patient symptoms and medical history to receive a differential diagnosis, treatment suggestions, and a detailed, verifiable reasoning process."
)

if __name__ == "__main__":
    iface.launch(share=False)
