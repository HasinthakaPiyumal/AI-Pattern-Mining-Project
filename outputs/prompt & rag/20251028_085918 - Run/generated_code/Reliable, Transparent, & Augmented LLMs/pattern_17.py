"""
MediTrust AI Diagnostic Assistant

This script combines all components of the MediTrust AI Diagnostic Assistant
into a single runnable Python file, featuring a Gradio-based user interface.
It demonstrates the Agentic and Trustworthy AI System Design pattern by integrating
LLM capabilities with external tools, providing explicit reasoning and confidence
estimations, and incorporating a feedback loop for continuous improvement.
"""

import json
import os
from datetime import datetime
import gradio as gr

# --- 1. External Tool Integration Layer (from external_tools.py) ---
# Simulates access to various external medical data sources and processing tools.

class EHRAccessor:
    """
    Simulates access to an Electronic Health Record (EHR) system.
    In a real system, this would involve secure API calls to an actual EHR.
    """
    def get_patient_record(self, patient_id: str) -> dict:
        print(f"Fetching EHR for patient ID: {patient_id}")
        # Mock data for demonstration
        if patient_id == "P12345":
            return {
                "patient_id": patient_id,
                "name": "Jane Doe",
                "age": 45,
                "gender": "Female",
                "allergies": ["Penicillin"],
                "medications": ["Aspirin", "Lisinopril"],
                "history": "Hypertension, previous appendectomy, family history of diabetes.",
                "symptoms": "Recent onset of fever, cough, fatigue."
            }
        else:
            return {"patient_id": patient_id, "error": "Patient record not found."}

class MedicalKnowledgeBase:
    """
    Simulates querying a medical knowledge base (e.g., PubMed, UpToDate).
    """
    def query_knowledge_base(self, query: str) -> list:
        print(f"Querying medical knowledge base for: \'{query[:50]}...\'")
        # Mock data based on query keywords
        if "fever" in query.lower() and "cough" in query.lower():
            return [
                "Common causes of fever and cough include influenza, common cold, bronchitis, pneumonia, and COVID-19.",
                "Smoking significantly increases the risk of respiratory infections and COPD exacerbations.",
                "Elevated CRP can indicate inflammation or infection, while normal WBC count may suggest a viral infection or early bacterial infection."
            ]
        elif "hypertension" in query.lower():
            return [
                "Lisinopril is an ACE inhibitor commonly used to treat hypertension."
            ]
        return ["No specific knowledge found for the query."]

class ImageAnalyzer:
    """
    Simulates an external service for analyzing medical images (e.g., X-rays, MRIs).
    """
    def analyze_image(self, image_data: str) -> dict:
        print(f"Sending image data for analysis (first 50 chars): \'{image_data[:50]}...\'")
        # In a real system, this would send image_data to a specialized ML model endpoint.
        # Mock data for demonstration
        if "base64_encoded_chest_xray_image" in image_data:
            return {
                "image_type": "Chest X-Ray",
                "findings": "Possible infiltrates in lower left lobe, suggestive of pneumonia.",
                "confidence": 0.85
            }
        return {"image_type": "Unknown", "findings": "No specific findings detected or unable to process image.", "confidence": 0.0}

class DrugInteractionChecker:
    """
    Simulates an external service for checking drug-drug interactions.
    """
    def check_interactions(self, medications: list) -> list:
        print(f"Checking drug interactions for: {medications}")
        # Mock data for demonstration
        if "Aspirin" in medications and "Lisinopril" in medications:
            return [
                "Potential interaction: NSAIDs (like Aspirin) can reduce the antihypertensive effect of ACE inhibitors (like Lisinopril). Monitor blood pressure."
            ]
        if "Aspirin" in medications:
            return ["No significant interactions found for Aspirin with common medications."]
        return ["No interactions found or no medications provided."]


# --- 2. Data Processor Layer (from data_processor.py) ---
# Handles validation and preprocessing of patient data.

class DataProcessor:
    """
    Handles validation and preprocessing of patient data.
    """
    def validate_patient_data(self, patient_data: dict) -> tuple[bool, dict]:
        """
        Validates the structure and content of patient data.
        Returns a tuple: (is_valid, errors_dict)
        """
        errors = {}

        # Check for essential fields
        required_fields = ["patient_id", "symptoms", "history"]
        for field in required_fields:
            if not patient_data.get(field):
                errors[field] = f"\'{field}\' is a required field."

        # Basic type checking and non-empty string checks
        if "patient_id" in patient_data and (not isinstance(patient_data["patient_id"], str) or not patient_data["patient_id"].strip()):
            errors["patient_id"] = "Patient ID must be a non-empty string."
        if "symptoms" in patient_data and (not isinstance(patient_data["symptoms"], str) or not patient_data["symptoms"].strip()):
            errors["symptoms"] = "Symptoms must be a non-empty string."
        if "history" in patient_data and (not isinstance(patient_data["history"], str) or not patient_data["history"].strip()):
            errors["history"] = "History must be a non-empty string."
        
        if "lab_results" in patient_data and not isinstance(patient_data["lab_results"], dict):
            errors["lab_results"] = "Lab results must be a dictionary."

        if "current_medications" in patient_data and not isinstance(patient_data["current_medications"], list):
            errors["current_medications"] = "Current medications must be a list."
        
        if errors:
            return False, errors
        return True, {}

    def preprocess_data(self, raw_data: dict) -> dict:
        """
        Placeholder for future data preprocessing steps like tokenization, normalization.
        """
        print("Preprocessing data (placeholder): No specific preprocessing applied yet.")
        return raw_data


# --- 3. Feedback & Finetuning Layer (from feedback_finetuning.py) ---
# Captures user feedback, logs interactions, and prepares data for continuous LLM improvement.

class FeedbackLogger:
    """
    Logs AI interaction details and user feedback for auditing and finetuning purposes.
    """
    def __init__(self, log_dir="feedback_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _generate_interaction_id(self) -> str:
        """Generates a unique ID for each interaction."""
        return datetime.now().strftime("%Y%m%d%H%M%S%f")

    def log_interaction(self, patient_id: str, input_context: dict, ai_output: dict) -> str:
        """
        Logs the full interaction between user input, tool outputs, and AI's response.
        """
        interaction_id = self._generate_interaction_id()
        log_entry = {
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
            "input_context": input_context,
            "ai_output": ai_output,
            "feedback": None # Placeholder for later feedback
        }
        log_filepath = os.path.join(self.log_dir, f"interaction_{interaction_id}.json")
        with open(log_filepath, "w") as f:
            json.dump(log_entry, f, indent=2)
        print(f"Interaction logged: {log_filepath}")
        return interaction_id

    def record_feedback(self, interaction_id: str, feedback_type: str, details: str, correct_diagnosis: str = None):
        """
        Records user feedback against a specific interaction.
        """
        log_filepath = os.path.join(self.log_dir, f"interaction_{interaction_id}.json")
        if not os.path.exists(log_filepath):
            print(f"Error: Interaction ID {interaction_id} not found for feedback.")
            return False

        with open(log_filepath, "r+") as f:
            log_entry = json.load(f)
            log_entry["feedback"] = {
                "timestamp": datetime.now().isoformat(),
                "type": feedback_type,
                "details": details,
                "correct_diagnosis": correct_diagnosis
            }
            f.seek(0) # Rewind to the beginning
            json.dump(log_entry, f, indent=2)
            f.truncate() # Trim any remaining old content
        print(f"Feedback recorded for interaction: {interaction_id}")
        return True

class FinetuningManager:
    """
    Conceptual manager for collecting and preparing data for LLM finetuning.
    In a real system, this would involve more sophisticated data pipelines and model training code.
    """
    def __init__(self, feedback_logger: FeedbackLogger, finetune_data_dir="finetune_data"):
        self.feedback_logger = feedback_logger
        self.finetune_data_dir = finetune_data_dir
        os.makedirs(self.finetune_data_dir, exist_ok=True)

    def collect_finetuning_data(self):
        """
        Collects interactions with recorded feedback to prepare for finetuning.
        Filters for interactions where feedback indicates an incorrect diagnosis or hallucination,
        and a corrected diagnosis is provided. This data can then be used to create
        a finetuning dataset.
        """
        finetuning_samples = []
        for filename in os.listdir(self.feedback_logger.log_dir):
            if filename.startswith("interaction_") and filename.endswith(".json"):
                filepath = os.path.join(self.feedback_logger.log_dir, filename)
                with open(filepath, "r") as f:
                    log_entry = json.load(f)
                    feedback = log_entry.get("feedback")
                    
                    if feedback and feedback.get("type") in ["Incorrect Diagnosis", "Hallucination Detected"] and feedback.get("correct_diagnosis"):
                        # Extract relevant input and corrected output for finetuning
                        finetuning_samples.append({
                            "input": log_entry["input_context"],
                            "target_output": {
                                "diagnosis": feedback["correct_diagnosis"],
                                # You might also want to include corrected reasoning if available
                                # and other elements of the desired output format.
                            }
                        })
        
        if finetuning_samples:
            finetune_filepath = os.path.join(self.finetune_data_dir, f"finetune_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl")
            with open(finetune_filepath, "w") as f:
                for sample in finetuning_samples:
                    f.write(json.dumps(sample) + "\n")
            print(f"Collected {len(finetuning_samples)} samples for finetuning: {finetune_filepath}")
            return finetune_filepath
        else:
            print("No suitable samples found for finetuning.")
            return None

    def trigger_llm_finetuning(self, finetuning_data_path: str):
        """
        Conceptual method to trigger an LLM finetuning job.
        In practice, this would involve using an LLM provider's API (e.g., OpenAI's finetuning API)
        or a custom training script.
        """
        if finetuning_data_path:
            print(f"Triggering LLM finetuning with data from: {finetuning_data_path}")
            print("This is a conceptual step. Actual finetuning implementation would go here.")
            # Example: openai.FineTuningJob.create(training_file=finetuning_data_path, model="gpt-3.5-turbo")
        else:
            print("Cannot trigger finetuning: No finetuning data available.")


# --- 4. LLM Agent Layer (from llm_agent.py) ---
# Interfaces with the Large Language Model to generate diagnostic hypotheses.

class LLMAgent:
    def __init__(self, model_name="gpt-3.5-turbo", api_key="YOUR_OPENAI_API_KEY"):
        # In a real application, use a proper LLM client (e.g., from openai library)
        # For demonstration, we'll simulate LLM responses.
        self.model_name = model_name
        self.api_key = api_key # This would be used to initialize the actual LLM client
        self.confidence_threshold = 0.65 # Threshold for controlled abstention

    def _call_llm(self, prompt: str) -> str:
        """
        Simulates an API call to a Large Language Model.
        In a real scenario, this would interface with OpenAI, Google Gemini, etc.
        """
        # This is a placeholder for actual LLM inference.
        # The response structure is designed to be parsed by generate_diagnosis.
        if "fever, cough, shortness of breath" in prompt and "smoker" in prompt:
            return json.dumps({
                "diagnosis": "Possible Viral Pneumonia or Bronchitis",
                "reasoning": "Patient presents with common respiratory symptoms (fever, cough, shortness of breath) and a history of smoking, which increases susceptibility to respiratory infections. Lab results (elevated CRP, normal WBC) are consistent with a viral process or early bacterial infection. Drug interaction check for Aspirin is clear.",
                "confidence": 0.75,
                "recommended_actions": ["Recommend chest X-ray for consolidation.", "Prescribe symptomatic relief.", "Advise follow-up in 48 hours."]
            })
        elif "invalid input" in prompt:
            return json.dumps({
                "diagnosis": "Insufficient or invalid data.",
                "reasoning": "The provided patient data was incomplete or contained errors, preventing a reliable diagnosis.",
                "confidence": 0.05,
                "recommended_actions": ["Review patient data for accuracy and completeness."]
            })
        else:
            return json.dumps({
                "diagnosis": "Further investigation needed.",
                "reasoning": "The provided information is too general or ambiguous to confidently determine a specific diagnosis. More detailed symptoms, history, or diagnostic test results are required.",
                "confidence": 0.4,
                "recommended_actions": ["Request more specific symptoms.", "Order additional lab tests (e.g., blood culture, viral panel).", "Consider specialist consultation."]
            })

    def generate_diagnosis(self, patient_context: dict) -> dict:
        """
        Generates a diagnosis, reasoning, confidence score, and recommended actions
        based on the comprehensive patient context.
        Implements controlled abstention.
        """
        # Construct a detailed prompt for the LLM
        prompt = f"""
        As a highly experienced medical diagnostic AI, analyze the following patient data and provide a concise diagnosis, detailed reasoning, a confidence score (0.0-1.0), and actionable recommendations.
        
        Patient ID: {patient_context.get("patient_id")}
        Symptoms: {patient_context.get("symptoms")}
        History: {patient_context.get("history")}
        Lab Results: {patient_context.get("lab_results")}
        Image Analysis Report: {patient_context.get("image_analysis")}
        Drug Interactions: {patient_context.get("drug_interactions")}
        EHR Data: {patient_context.get("ehr_data")}
        Medical Knowledge Base Info: {patient_context.get("medical_knowledge")}

        Based on this information, provide your best diagnostic hypothesis, a clear step-by-step reasoning path citing relevant facts, your confidence in this diagnosis (as a float between 0.0 and 1.0), and actionable recommendations. If you are uncertain or the data is insufficient, state that you abstain from a specific diagnosis and suggest further investigations. Ensure the output is in JSON format with keys: "diagnosis", "reasoning", "confidence", "recommended_actions".
        """
        
        # Call the simulated LLM
        llm_raw_response = self._call_llm(prompt)
        
        try:
            llm_output = json.loads(llm_raw_response)
        except json.JSONDecodeError:
            # Handle cases where LLM might not return valid JSON
            return {
                "diagnosis": "Error in AI processing.",
                "reasoning": "The AI could not process the request correctly or returned an invalid format.",
                "confidence": 0.0,
                "recommended_actions": ["Review input format or contact support."]
            }
        
        confidence = llm_output.get("confidence", 0.0)
        
        # Controlled Abstention Logic
        if confidence < self.confidence_threshold:
            llm_output["diagnosis"] = "Abstain: Insufficient confidence for a definitive diagnosis."
            llm_output["reasoning"] = f"The AI's confidence score ({confidence:.2f}) is below the threshold ({self.confidence_threshold:.2f}).\n" + llm_output.get("reasoning", "Further investigation is recommended.")
            llm_output["recommended_actions"].insert(0, "Consider additional diagnostic tests or specialist consultation due to low confidence.")

        return llm_output


# --- 5. Orchestration Layer (from meditrust_assistant.py) ---
# Coordinates data flow, manages interactions, and implements core pattern logic.

class MediTrustAssistant:
    def __init__(self):
        self.llm_agent = LLMAgent()
        self.ehr_accessor = EHRAccessor()
        self.medical_kb = MedicalKnowledgeBase()
        self.image_analyzer = ImageAnalyzer()
        self.drug_checker = DrugInteractionChecker()
        self.data_processor = DataProcessor()
        self.feedback_logger = FeedbackLogger()

    def _orchestrate_tools(self, patient_id, image_data=None, current_medications=None):
        """Orchestrates calls to external tools to gather comprehensive patient data."""
        ehr_data = self.ehr_accessor.get_patient_record(patient_id)
        
        # Simulate fetching relevant medical knowledge based on EHR data
        relevant_kb_info = self.medical_kb.query_knowledge_base(ehr_data.get("symptoms", "") + " " + ehr_data.get("history", ""))

        image_analysis_report = None
        if image_data:
            image_analysis_report = self.image_analyzer.analyze_image(image_data)

        drug_interaction_report = None
        if current_medications:
            drug_interaction_report = self.drug_checker.check_interactions(current_medications)

        return {
            "ehr_data": ehr_data,
            "medical_knowledge": relevant_kb_info,
            "image_analysis": image_analysis_report,
            "drug_interactions": drug_interaction_report
        }

    def diagnose_patient(self, patient_id: str, symptoms: str, history: str, lab_results: dict = None, image_data: str = None, current_medications: list = None):
        """
        Main function to diagnose a patient using LLM and integrated tools.
        Returns a dictionary containing diagnosis, reasoning, confidence, and recommended actions.
        """
        patient_data = {
            "patient_id": patient_id,
            "symptoms": symptoms,
            "history": history,
            "lab_results": lab_results if lab_results is not None else {},
            "image_data": image_data,
            "current_medications": current_medications if current_medications is not None else []
        }

        # 1. Input Validation
        is_valid, validation_errors = self.data_processor.validate_patient_data(patient_data)
        if not is_valid:
            return {
                "diagnosis": "Unable to provide diagnosis due to invalid input.",
                "reasoning": "Validation Errors: " + json.dumps(validation_errors),
                "confidence": 0.0,
                "recommended_actions": ["Please correct the input data and try again."]
            }
        
        # 2. Orchestrate External Tools
        tool_outputs = self._orchestrate_tools(patient_id, image_data, current_medications)
        
        # Combine all relevant information for the LLM
        full_patient_context = {
            **patient_data,
            **tool_outputs
        }

        # 3. Generate Diagnosis, Reasoning, and Confidence using LLM Agent
        diagnosis_result = self.llm_agent.generate_diagnosis(full_patient_context)
        
        # 4. Log interaction for potential finetuning and auditing
        self.feedback_logger.log_interaction(patient_id, full_patient_context, diagnosis_result)

        return diagnosis_result

    def provide_feedback(self, interaction_id: str, feedback_type: str, details: str, correct_diagnosis: str = None):
        """
        Allows medical professionals to provide feedback on an AI's diagnosis.
        """
        self.feedback_logger.record_feedback(interaction_id, feedback_type, details, correct_diagnosis)
        return {"status": "Feedback recorded successfully."}


# --- 6. User Interface Layer (from gradio_interface.py) ---
# Provides an intuitive web-based interface for medical professionals.

# Global instance of the assistant for Gradio functions to access
assistant_instance = None

def get_diagnosis(patient_id, symptoms, history, lab_results_json, image_data, current_medications_str):
    """
    Function to be called by Gradio interface to get a diagnosis.
    Parses JSON inputs from Gradio textboxes and calls the MediTrustAssistant.
    """
    global assistant_instance
    if assistant_instance is None:
        return "Error: Assistant not initialized.", "", "0.00", "Please restart the application."

    # Convert JSON string inputs to Python objects
    try:
        lab_results = json.loads(lab_results_json) if lab_results_json else {}
    except json.JSONDecodeError:
        return "Error: Invalid JSON for Lab Results.", "Please provide valid JSON.", "0.00", []

    try:
        current_medications = json.loads(current_medications_str) if current_medications_str else []
    except json.JSONDecodeError:
        return "Error: Invalid JSON for Current Medications.", "Please provide valid JSON array.", "0.00", []

    # Simulate image data if a checkbox is enabled
    processed_image_data = "base64_encoded_chest_xray_image" if image_data else None

    diagnosis_output = assistant_instance.diagnose_patient(
        patient_id=patient_id,
        symptoms=symptoms,
        history=history,
        lab_results=lab_results,
        image_data=processed_image_data,
        current_medications=current_medications
    )

    diagnosis = diagnosis_output.get("diagnosis", "N/A")
    reasoning = diagnosis_output.get("reasoning", "No reasoning provided.")
    confidence = diagnosis_output.get("confidence", 0.0)
    recommended_actions = diagnosis_output.get("recommended_actions", ["No actions recommended."])

    return (
        diagnosis,
        reasoning,
        f"{confidence:.2f}", # Format confidence as a string
        "\n".join(recommended_actions)
    )

def submit_feedback(interaction_id, feedback_type, feedback_details, correct_diagnosis):
    """
    Function to submit feedback.
    """
    global assistant_instance
    if assistant_instance is None:
        return "Error: Assistant not initialized."

    if not interaction_id:
        return "Please enter an Interaction ID for feedback."
    
    response = assistant_instance.provide_feedback(interaction_id, feedback_type, feedback_details, correct_diagnosis)
    return response["status"]


# --- Main Application Entry Point ---
if __name__ == "__main__":
    # Initialize the main assistant instance
    assistant_instance = MediTrustAssistant()
    
    # Gradio Interface setup
    with gr.Blocks(title="MediTrust AI Diagnostic Assistant") as demo:
        gr.Markdown("# MediTrust AI Diagnostic Assistant")
        gr.Markdown("Input patient information below to receive an AI-powered diagnostic suggestion.")

        with gr.Tab("Diagnosis"):
            with gr.Row():
                with gr.Column():
                    patient_id_input = gr.Textbox(label="Patient ID", placeholder="e.g., P12345", value="P12345")
                    symptoms_input = gr.Textbox(label="Symptoms", lines=3, placeholder="e.g., Fever, cough, shortness of breath", value="Fever, cough, shortness of breath")
                    history_input = gr.Textbox(label="Patient History", lines=3, placeholder="e.g., Smoker for 10 years, no known allergies.", value="Smoker for 10 years, no known allergies, recent travel to endemic area.")
                    lab_results_input = gr.Textbox(label="Lab Results (JSON)", lines=2, placeholder='{"CRP": "elevated", "WBC": "normal"}', value='{"CRP": "elevated", "WBC": "normal"}')
                    current_medications_input = gr.Textbox(label="Current Medications (JSON Array)", lines=1, placeholder='["Aspirin", "Lisinopril"]', value='["Aspirin"]')
                    image_data_input = gr.Checkbox(label="Simulate Chest X-Ray (Check to enable)", value=False)
                    
                    diagnose_btn = gr.Button("Get Diagnosis")

                with gr.Column():
                    diagnosis_output = gr.Textbox(label="AI Diagnosis", interactive=False)
                    reasoning_output = gr.Textbox(label="Reasoning Path", lines=5, interactive=False)
                    confidence_output = gr.Textbox(label="Confidence Score", interactive=False)
                    recommended_actions_output = gr.Textbox(label="Recommended Actions", lines=3, interactive=False)

            diagnose_btn.click(
                get_diagnosis,
                inputs=[
                    patient_id_input, symptoms_input, history_input, 
                    lab_results_input, image_data_input, current_medications_input
                ],
                outputs=[
                    diagnosis_output, reasoning_output, confidence_output, recommended_actions_output
                ]
            )
        
        with gr.Tab("Provide Feedback"):
            gr.Markdown("Help us improve the AI by providing feedback on its performance.")
            with gr.Row():
                with gr.Column():
                    feedback_interaction_id = gr.Textbox(label="Interaction ID", placeholder="Enter the ID from the diagnosis session")
                    feedback_type_input = gr.Radio(
                        ["Correct Diagnosis", "Incorrect Diagnosis", "Incomplete Reasoning", "Hallucination Detected", "Other"], 
                        label="Feedback Type"
                    )
                    feedback_details_input = gr.Textbox(label="Details", lines=3, placeholder="Describe the issue or provide suggestions.")
                    correct_diagnosis_input = gr.Textbox(label="Correct Diagnosis (if applicable)", placeholder="e.g., Bacterial Pneumonia")
                    submit_feedback_btn = gr.Button("Submit Feedback")
                with gr.Column():
                    feedback_status_output = gr.Textbox(label="Feedback Status", interactive=False)

            submit_feedback_btn.click(
                submit_feedback,
                inputs=[
                    feedback_interaction_id, feedback_type_input, 
                    feedback_details_input, correct_diagnosis_input
                ],
                outputs=feedback_status_output
            )

    demo.launch()
