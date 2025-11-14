import streamlit as st
import pypdf
import io
import os
from dotenv import load_dotenv
from loguru import logger
import uuid # For unique session IDs for tracking

# Load environment variables
load_dotenv()

# Configure logger
logger.add("file_{time}.log", rotation="1 day")

# --- Mock LLM Integration ---
class MockLLM:
    def __init__(self, api_key=None):
        self.api_key = api_key # In a real scenario, this would be used to initialize the LLM client
        logger.info("MockLLM initialized.")

    def generate_summary(self, prompt, medical_report_text):
        logger.info(f"MockLLM generating summary with prompt: {prompt[:50]}...")
        # Simulate LLM response
        if "summarize for a general physician" in prompt.lower():
            summary = f"[GP Summary of Medical Report]: This report generally indicates a patient with certain conditions and treatments. Key details from the original report: {medical_report_text[:200]}..."
        elif "extract key diagnoses for a specialist" in prompt.lower():
            summary = f"[Specialist Diagnosis Extract]: Primary diagnosis: Condition X. Secondary diagnosis: Condition Y. Relevant findings: {medical_report_text[:150]}..."
        else:
            summary = f"[General Summary]: A concise overview of the medical report. Main points: {medical_report_text[:250]}..."
        
        # Simulate a small delay
        import time
        time.sleep(1)
        logger.info("MockLLM summary generated.")
        return summary

    def evaluate_text(self, text_to_evaluate, evaluation_criteria):
        logger.info(f"MockLLM evaluating text with criteria: {evaluation_criteria[:50]}...")
        # Simulate LLM-based evaluation
        if "accuracy" in evaluation_criteria.lower() and "error" in text_to_evaluate.lower():
            return {"score": 2, "feedback": "Potential factual inconsistency detected."}
        elif "bias" in evaluation_criteria.lower() and "male patient only" in text_to_evaluate.lower():
            return {"score": 3, "feedback": "Possible gender bias in phrasing."}
        return {"score": 5, "feedback": "Looks good and meets criteria."}

# Initialize Mock LLM (replace with actual LLM client in production)
llm_api_key = os.getenv("OPENAI_API_KEY", "mock_api_key") # Example for OpenAI
llm = MockLLM(api_key=llm_api_key)

# --- Data Management & Preprocessing ---
def extract_text_from_pdf(pdf_file):
    logger.info("Extracting text from PDF.")
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    logger.info(f"Extracted {len(text)} characters from PDF.")
    return text

def basic_text_cleaning(text):
    logger.info("Performing basic text cleaning.")
    # A very basic cleaning for demonstration
    cleaned_text = text.replace("\n", " ").replace("  ", " ").strip()
    logger.info(f"Cleaned text length: {len(cleaned_text)}.")
    return cleaned_text

# --- Prompt Engineering ---
def create_dynamic_prompt(report_context, summary_purpose):
    logger.info(f"Creating dynamic prompt for purpose: {summary_purpose}.")
    if summary_purpose == "general_physician":
        prompt = f"You are a highly experienced medical summarizer. Summarize the following medical report for a general physician, focusing on key findings, diagnoses, and treatment plans. Be concise and use clear medical terminology.\n\nReport: {report_context}"
    elif summary_purpose == "specialist_diagnosis":
        prompt = f"You are a meticulous medical specialist extracting critical information. From the following medical report, extract only the primary and secondary diagnoses, relevant lab results, and any significant prognostic indicators. \n\nReport: {report_context}"
    elif summary_purpose == "patient_friendly":
        prompt = f"You are a compassionate healthcare communicator. Explain the following medical report in simple, easy-to-understand language for a non-medical patient, avoiding jargon where possible.\n\nReport: {report_context}"
    else:
        prompt = f"Summarize the following medical report, ensuring all critical information is retained.\n\nReport: {report_context}"
    logger.info("Dynamic prompt created.")
    return prompt

# --- Fact-Checking & Quality Assurance Module ---
def apply_guardrails(llm_output):
    logger.info("Applying output guardrails.")
    # In a real scenario, Guardrails AI or Pydantic would define a schema
    # For this mock, we'll check for basic validity and presence of key terms
    if not llm_output or len(llm_output) < 50:
        return False, "Summary is too short or empty."
    if "financial" in llm_output.lower() or "billing" in llm_output.lower():
        return False, "Summary contains irrelevant financial/billing information."
    return True, "Guardrails passed."

def perform_llm_autorating(summary, original_text):
    logger.info("Performing LLM-based autorating.")
    # Use LLM to evaluate the summary based on criteria
    criteria = "Evaluate the following medical summary for accuracy, completeness, conciseness, and absence of bias. Assign a score out of 5 and provide feedback.\nSummary: {summary}\nOriginal Report Snippet: {original_text[:500]}..."
    evaluation_result = llm.evaluate_text(summary, criteria)
    logger.info(f"Autorating result: {evaluation_result}.")
    return evaluation_result

def perform_round_trip_consistency(summary, original_text):
    logger.info("Performing round-trip consistency check.")
    # A simplified round-trip: ask LLM if summary is consistent with original
    check_prompt = f"Is the following summary consistent with the key facts presented in the original medical report? Answer Yes or No, and explain any inconsistencies.\nSummary: {summary}\nOriginal Report Snippet: {original_text[:500]}..."
    # In a real system, you'd send this to the LLM and parse its response.
    # For mock, we'll assume consistency if summary is reasonable length.
    if len(summary) > 100 and "[error]" not in summary.lower():
        return {"consistent": True, "explanation": "Summary appears consistent with the original report (mock check)."}
    return {"consistent": False, "explanation": "Summary might be inconsistent or too brief (mock check)."}

def perform_adversarial_evaluation(summary):
    logger.info("Performing adversarial evaluation.")
    # Simulate injecting an adversarial prompt or checking for specific vulnerabilities
    # For example, check if the LLM hallucinates sensitive patient data not present in original.
    if "patient_secret_data" in summary.lower(): # Hypothetical sensitive data
        return {"vulnerable": True, "finding": "Hallucinated sensitive data detected!"}
    return {"vulnerable": False, "finding": "No obvious adversarial vulnerabilities found (mock check)."}

# --- Ethical Alignment Module ---
def apply_ethical_alignment(summary, original_text):
    logger.info("Applying ethical alignment principles.")
    # Incorporate Constitutional AI-like self-correction prompts
    ethical_prompt = f"Review the following medical summary for empathy, avoidance of alarmist language, and patient data privacy adherence. Rewrite if necessary to align with ethical medical communication standards.\nSummary: {summary}\nOriginal Report Snippet: {original_text[:300]}..."
    
    # Mock ethical evaluation/correction
    if "catastrophic" in summary.lower() and "not confirmed" in original_text.lower():
        corrected_summary = summary.replace("catastrophic", "potentially serious but unconfirmed")
        return corrected_summary, "Rewritten to avoid alarmist language."
    return summary, "Ethical alignment passed (mock check)."

# --- Streamlit UI ---
st.set_page_config(page_title="Medical Report Summarizer & Fact-Checker", layout="wide")
st.title("🩺 AI-powered Medical Report Summarization and Fact-Checking System")

st.markdown("Upload a medical report (PDF) to get an AI-generated summary and quality assurance checks.")

session_id = st.session_state.get("session_id", str(uuid.uuid4()))
st.session_state["session_id"] = session_id
logger.info(f"New session started: {session_id}")

uploaded_file = st.file_uploader("Choose a PDF medical report", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Read PDF as bytes
        pdf_bytes = uploaded_file.read()
        pdf_file_obj = io.BytesIO(pdf_bytes)
        
        # Extract and clean text
        original_text = extract_text_from_pdf(pdf_file_obj)
        cleaned_text = basic_text_cleaning(original_text)
        
        if not cleaned_text:
            st.error("Could not extract text from the PDF. Please ensure it's a searchable PDF.")
        else:
            st.success("PDF processed successfully!")
            st.subheader("Original Report Snippet (First 500 chars)")
            st.text_area("", cleaned_text[:500] + "...", height=150, disabled=True)

            st.subheader("Summarization Settings")
            summary_purpose = st.selectbox(
                "Select Summary Purpose:",
                ["General Physician", "Specialist Diagnosis", "Patient-Friendly", "General"],
                key="summary_purpose"
            )
            
            # Convert purpose to internal key
            purpose_key = summary_purpose.lower().replace(" ", "_").replace("-", "_")

            if st.button("Generate Summary and Perform Checks"):
                with st.spinner("Generating summary and performing quality checks..."):
                    # 1. Prompt Engineering & LLM Core
                    prompt = create_dynamic_prompt(cleaned_text, purpose_key)
                    generated_summary = llm.generate_summary(prompt, cleaned_text)
                    
                    st.subheader("Generated Summary")
                    st.info(generated_summary)

                    # 2. Fact-Checking & Quality Assurance Module
                    st.subheader("Quality Assurance Checks")

                    # Guardrails
                    guardrails_passed, guardrails_msg = apply_guardrails(generated_summary)
                    if guardrails_passed:
                        st.success(f"Guardrails: {guardrails_msg}")
                    else:
                        st.error(f"Guardrails: {guardrails_msg}")

                    # LLM-based Autorating
                    autorating_result = perform_llm_autorating(generated_summary, original_text)
                    st.write(f"Autorating Score: {autorating_result['score']}/5")
                    st.write(f"Autorating Feedback: {autorating_result['feedback']}")

                    # Round-Trip Consistency Check
                    consistency_result = perform_round_trip_consistency(generated_summary, original_text)
                    if consistency_result['consistent']:
                        st.success(f"Consistency Check: {consistency_result['explanation']}")
                    else:
                        st.warning(f"Consistency Check: {consistency_result['explanation']}")

                    # Adversarial Evaluation
                    adversarial_result = perform_adversarial_evaluation(generated_summary)
                    if adversarial_result['vulnerable']:
                        st.error(f"Adversarial Evaluation: {adversarial_result['finding']}")
                    else:
                        st.success(f"Adversarial Evaluation: {adversarial_result['finding']}")

                    # 3. Ethical Alignment Module
                    st.subheader("Ethical Alignment")
                    ethically_aligned_summary, ethical_feedback = apply_ethical_alignment(generated_summary, original_text)
                    if ethically_aligned_summary != generated_summary:
                        st.warning(f"Ethical Alignment Feedback: {ethical_feedback}")
                        st.info(f"Ethically Aligned Summary: {ethically_aligned_summary}")
                    else:
                        st.success(f"Ethical Alignment: {ethical_feedback}")

                    # 4. Monitoring & Logging (conceptual)
                    logger.info(f"Session {session_id} - Summary generated and checks completed for file: {uploaded_file.name}")
                    # In a real app, wandb/langsmith would log metrics here
                    st.markdown("--- \n _*Note: Logging details can be found in `file_<timestamp>.log`*_")


# How to run:
# 1. Save the code as medical_report_system.py
# 2. Create a .env file in the same directory (optional, for OPENAI_API_KEY if using real LLM)
# 3. Install necessary libraries: pip install streamlit pypdf python-dotenv loguru
# 4. Run from your terminal: streamlit run medical_report_system.py
