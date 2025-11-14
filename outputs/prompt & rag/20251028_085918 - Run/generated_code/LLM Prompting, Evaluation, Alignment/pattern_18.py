import os
import openai
import gradio as gr
import json
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment or a .env file.")

# --- 1. LLM Integration Helper ---
def get_llm_response(prompt: str, model: str = "gpt-4o", temperature: float = 0.1) -> str:
    """Helper function to get response from an LLM."""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error communicating with LLM: {e}"

# --- 2. Prompt Engineering Strategies ---

def generate_zero_shot_prompt(report_text: str) -> str:
    return f"Summarize the following medical report concisely and accurately, highlighting key diagnoses, treatments, and patient status:\n\nReport:\n{report_text}\n\nSummary:"

def generate_few_shot_prompt(report_text: str) -> str:
    # Example of a well-summarized report
    example_report = """Patient: John Doe, DOB: 1970-05-15. Admission Date: 2023-10-26. Discharge Date: 2023-10-29. Reason for Admission: Acute appendicitis. History: Presented with severe periumbilical pain radiating to the right lower quadrant, nausea, and fever (101.5°F). Physical Exam: Rebound tenderness, guarding in RLQ. Labs: WBC 18,000. Imaging: CT scan confirmed acute appendicitis. Treatment: Laparoscopic appendectomy performed on 2023-10-27. Post-op Course: Unremarkable. Discharged with pain medication and follow-up instructions for 1 week. Diagnoses: Acute appendicitis. Plan: Follow-up with surgeon in 1 week, pain management as needed."""
    example_summary = """John Doe, a 53-year-old male, was admitted on 2023-10-26 with acute appendicitis. He underwent a laparoscopic appendectomy on 2023-10-27 and had an unremarkable post-operative course. Discharged on 2023-10-29 with pain medication and a follow-up scheduled in 1 week."""

    return f"""Summarize medical reports concisely and accurately, focusing on diagnoses, treatments, and patient status, like the following example:

Report: {example_report}
Summary: {example_summary}

Now, summarize this report:

Report:\n{report_text}\n\nSummary:"""

def generate_role_based_prompt(report_text: str, role: str = "experienced medical doctor") -> str:
    return f"""You are an {role}. Your task is to summarize the following medical report for another healthcare professional. Focus on critical information, diagnoses, interventions, and patient outcomes. Maintain a neutral and objective tone.

Report:\n{report_text}\n\nSummary:"""

def generate_template_driven_prompt(report_text: str) -> str:
    return f"""Generate a structured summary for the medical report below, adhering to the following format:

Patient Name: [Patient's Full Name]
DOB: [Date of Birth]
Admission/Discharge: [Admission Date] - [Discharge Date]
Primary Diagnosis: [Main Diagnosis]
Key Treatments/Procedures: [List of Treatments/Procedures]
Major Findings: [Significant clinical/lab/imaging findings]
Outcome/Plan: [Patient's current status and future plan]

Report:\n{report_text}"""

def generate_dynamic_prompt(report_text: str) -> str:
    # This is a simplified dynamic prompt generation. In a real system,
    # it might analyze the report for keywords or structure to pick the best prompt template.
    if "cancer" in report_text.lower() or "oncology" in report_text.lower():
        return generate_role_based_prompt(report_text, role="oncologist")
    elif "surgery" in report_text.lower() or "operation" in report_text.lower():
        return generate_role_based_prompt(report_text, role="surgeon")
    else:
        return generate_zero_shot_prompt(report_text)

# --- 3. Summarization Logic ---
def summarize_report(report_text: str, prompt_strategy: str) -> str:
    """Generates a summary using the specified prompt strategy."""
    prompt = ""
    if prompt_strategy == "Zero-shot":
        prompt = generate_zero_shot_prompt(report_text)
    elif prompt_strategy == "Few-shot":
        prompt = generate_few_shot_prompt(report_text)
    elif prompt_strategy == "Role-based":
        prompt = generate_role_based_prompt(report_text)
    elif prompt_strategy == "Template-driven":
        prompt = generate_template_driven_prompt(report_text)
    elif prompt_strategy == "Dynamic":
        prompt = generate_dynamic_prompt(report_text)
    else:
        return "Invalid prompt strategy selected."

    return get_llm_response(prompt)

# --- 4. Evaluation Frameworks ---

def llm_autorate_summary(original_report: str, generated_summary: str) -> str:
    """Uses an LLM to evaluate the quality of the generated summary."""
    rating_prompt = f"""You are an expert medical reviewer. Evaluate the following summary based on its accuracy, completeness, conciseness, and relevance to the original medical report. Provide a score from 1 to 5 (1 = poor, 5 = excellent) and a brief justification.

Original Report:\n{original_report}\n
Generated Summary:\n{generated_summary}\n
Evaluation (Score: /5, Justification:)"""
    return get_llm_response(rating_prompt, temperature=0.2)

def round_trip_consistency_check(original_report: str, generated_summary: str) -> str:
    """Checks consistency by asking the LLM to reconstruct key facts from the summary and comparing to original."""
    reconstruction_prompt = f"""Based ONLY on the following medical summary, extract the patient's primary diagnosis, main treatments, and current status. If information is not present in the summary, state 'Not available'.

Summary:\n{generated_summary}\n
Extracted Info (Diagnosis: ..., Treatments: ..., Status: ...):"""
    reconstructed_info = get_llm_response(reconstruction_prompt)

    # A more robust check would involve parsing both original and reconstructed info
    # and comparing structured entities. For this example, we do a basic qualitative check.
    consistency_check_prompt = f"""Compare the following extracted information with the original report to determine consistency. Is the extracted information accurately derivable from the original report and does it align with the original? Answer 'Yes' or 'No' and provide a brief explanation.

Original Report Key Info (manual extraction for comparison):
  - Primary Diagnosis: [Assume we know this from original for a true check]
  - Main Treatments: [Assume we know this from original for a true check]
  - Current Status: [Assume we know this from original for a true check]

Extracted Info from Summary:\n{reconstructed_info}\n
Does the extracted info from the summary align consistently with the original report? (Yes/No, Explanation):"""

    # For a real system, you'd extract structured data from original_report and compare with reconstructed_info
    # This part is simplified for demonstration.
    return f"Reconstructed from summary: {reconstructed_info}\n\nNote: A full consistency check would compare this extracted info with a parsed version of the original report for discrepancies. LLM check for alignment is: {get_llm_response(consistency_check_prompt)}"

def adversarial_evaluation(original_report: str, generated_summary: str) -> str:
    """Tests the summary against potential misinformation or omitted critical details."""
    adversarial_prompt = f"""Given the original medical report and its summary, identify if the summary omits any critical safety information, introduces any factual inaccuracies, or could be misleading. If found, provide examples. If not, state 'No issues found'.

Original Report:\n{original_report}\n
Generated Summary:\n{generated_summary}\n
Adversarial Analysis:"""
    return get_llm_response(adversarial_prompt, temperature=0.3)

def ethical_alignment_check(generated_summary: str) -> str:
    """Checks the summary for ethical considerations like bias, privacy breaches, or inappropriate language."""
    ethical_prompt = f"""Review the following medical summary for any potential biases (e.g., gender, race), breaches of patient confidentiality (e.g., exposing sensitive identifiers unnecessarily), or inappropriate/non-medical language. Provide 'No issues found' or describe the ethical concerns.

Generated Summary:\n{generated_summary}\n
Ethical Review:"""
    return get_llm_response(ethical_prompt, temperature=0.1)

# --- Main Application Logic (Gradio Interface) ---
def process_medical_report(report_text: str, prompt_strategy: str):
    if not report_text:
        return "", "", "", "", "", "Please provide a medical report to summarize."

    # 1. Generate Summary
    summary = summarize_report(report_text, prompt_strategy)

    # 2. Perform Evaluations
    autorating = llm_autorate_summary(report_text, summary)
    consistency = round_trip_consistency_check(report_text, summary)
    adversarial = adversarial_evaluation(report_text, summary)
    ethical = ethical_alignment_check(summary)

    return summary, autorating, consistency, adversarial, ethical, "Summary and evaluations generated successfully."

# --- Gradio Interface Setup ---

example_report_1 = """Patient: Jane Smith, DOB: 1965-11-20. Admission Date: 2024-01-10. Discharge Date: 2024-01-15. Reason for Admission: Unstable Angina. History: 58-year-old female with history of hypertension, hyperlipidemia, and type 2 diabetes. Presented with crushing chest pain radiating to the left arm, unrelieved by rest. ECG showed ST depression in leads V4-V6. Troponin I elevated. Treatment: Admitted to CCU, started on IV nitrates, beta-blockers, antiplatelet therapy. Underwent cardiac catheterization on 2024-01-12 revealing 90% stenosis in LAD. Placed on drug-eluting stent. Post-procedure, stable with no chest pain. Discharged on Aspirin, Clopidogrel, Metoprolol, Atorvastatin, Lisinopril, and Metformin. Plan: Follow-up with Cardiology in 2 weeks."""

example_report_2 = """Patient: Robert Johnson, DOB: 1980-03-01. Admission Date: 2023-12-01. Reason for Admission: Community-Acquired Pneumonia. History: 43-year-old male, non-smoker, presenting with cough, fever (102°F), chills, and dyspnea for 3 days. Chest X-ray showed right lower lobe infiltrate. Labs: WBC 15,000. Treatment: Started on Azithromycin and Ceftriaxone empirically. Responded well to treatment, fever subsided. Discharged 2023-12-05 with a 7-day course of oral antibiotics. Plan: Complete antibiotic course, follow-up with PCP in 1 month if symptoms persist."""


with gr.Blocks(title="Medical Report Summarization and Validation") as demo:
    gr.Markdown("# Medical Report Summarization and Validation System")
    gr.Markdown("This system summarizes medical reports using various prompt engineering techniques and validates the output using robust evaluation frameworks.")

    with gr.Row():
        report_input = gr.Textbox(label="Medical Report Text", lines=10, placeholder="Paste medical report text here...")
        strategy_selector = gr.Radio(
            ["Zero-shot", "Few-shot", "Role-based", "Template-driven", "Dynamic"],
            label="Prompt Engineering Strategy",
            value="Zero-shot"
        )

    submit_btn = gr.Button("Generate Summary and Run Evaluations")

    with gr.Accordion("Generated Summary", open=True):
        summary_output = gr.Textbox(label="Generated Summary", lines=5, interactive=False)

    with gr.Accordion("Evaluation Results", open=False):
        autorating_output = gr.Textbox(label="LLM-based Autorating", lines=3, interactive=False)
        consistency_output = gr.Textbox(label="Round-trip Consistency Check", lines=5, interactive=False)
        adversarial_output = gr.Textbox(label="Adversarial Evaluation", lines=5, interactive=False)
        ethical_output = gr.Textbox(label="Ethical Alignment Check", lines=3, interactive=False)

    status_message = gr.Textbox(label="Status", interactive=False)

    submit_btn.click(
        process_medical_report,
        inputs=[report_input, strategy_selector],
        outputs=[summary_output, autorating_output, consistency_output, adversarial_output, ethical_output, status_message]
    )

    gr.Examples(
        examples=[
            [example_report_1, "Zero-shot"],
            [example_report_2, "Role-based"]
        ],
        inputs=[report_input, strategy_selector]
    )

demo.launch()
