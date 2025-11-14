import os
from typing import List, Dict, Any
import random
import time

# --- 1. LLM Core (Placeholder) ---
# In a real application, you would load a model from Hugging Face transformers
# e.g., from transformers import pipeline
# self.llm_pipeline = pipeline("text-generation", model="path/to/medical/llm")
class LLMCore:
    def __init__(self, model_name: str = "MedicalLLM-v1"):
        self.model_name = model_name
        print(f"LLM Core initialized with model: {self.model_name}")

    def generate_response(self, prompt: str) -> str:
        # Simulate LLM thinking and generating a response
        time.sleep(1) # Simulate processing time
        if "diagnosis" in prompt.lower() and "step-by-step" in prompt.lower():
            return f"**LLM Reasoning (CoT Diagnosis):**\n1. Analyze patient symptoms.\n2. Cross-reference with medical history.\n3. Consider differential diagnoses: {random.choice(['Influenza', 'Pneumonia', 'Bronchitis'])}, {random.choice(['Asthma', 'Allergies'])}, etc.\n4. Rule out conditions based on test results.\n5. Conclude with a likely diagnosis: {random.choice(['Acute Bronchitis', 'Seasonal Allergies', 'Mild Asthma Exacerbation'])}\n\n**Proposed Diagnosis:** {random.choice(['Acute Bronchitis', 'Seasonal Allergies', 'Mild Asthma Exacerbation'])}"
        elif "treatment plan" in prompt.lower() and "verify" in prompt.lower():
            return f"**LLM Reasoning (CoVe Treatment):**\n1. Based on diagnosis, propose initial treatment: {random.choice(['Rest, fluids, OTC meds', 'Inhaler, antihistamines', 'Antibiotics if bacterial infection suspected'])}\n2. Check for drug interactions: No major interactions found for proposed meds.\n3. Review patient allergies: Patient has no known allergies to proposed meds.\n4. Adherence to guidelines: Treatment aligns with standard guidelines.\n\n**Recommended Treatment:** {random.choice(['Rest, fluids, paracetamol for symptoms.', 'Prescribe salbutamol inhaler and antihistamines.', 'Consider a 5-day course of azithromycin.'])} (Confidence: 0.{random.randint(70,99)})\n**Potential Side Effects:** {random.choice(['Drowsiness', 'Mild stomach upset', 'None significant'])}"
        elif "follow-up questions" in prompt.lower():
            return f"**LLM Suggestion (Active Prompting):**\nTo refine diagnosis, consider asking:\n1. How long have the symptoms been present?\n2. Are symptoms worse at night or during the day?\n3. Any recent travel history or exposure to sick individuals?"
        elif "reverse engineer" in prompt.lower():
             return f"**LLM Reasoning (Reversing CoT):**\nTo arrive at '{prompt.split('from')[-1].strip()}', the following medical conditions or patient history would be essential:\n1. ...\n2. ...\nThis plan logically supports the diagnosis."
        else:
            return f"LLM processed: '{prompt[:100]}...' and generated a general medical response."


# --- 2. Prompt Engineering Module ---
class PromptEngineering:
    def __init__(self, llm_core: LLMCore):
        self.llm_core = llm_core

    def generate_cot_diagnosis_prompt(self, patient_data: Dict[str, Any]) -> str:
        symptoms = patient_data.get("symptoms", "N/A")
        history = patient_data.get("medical_history", "N/A")
        test_results = patient_data.get("test_results", "N/A")
        prompt = (
            f"Given the patient's symptoms: {symptoms},\n"
            f"medical history: {history},\n"
            f"and test results: {test_results}.\n"
            "Please provide a step-by-step chain-of-thought diagnosis, considering differential diagnoses and ruling out conditions before concluding with the most likely diagnosis. Explain your reasoning clearly."
        )
        return prompt

    def generate_stepback_prompt(self, current_diagnosis_context: str) -> str:
        prompt = (
            f"Considering the current diagnostic context: '{current_diagnosis_context}',\n"
            "step back and think about broader medical principles or common diagnostic pathways that might be relevant before finalizing a specific diagnosis."
        )
        return prompt

    def generate_tabular_cot_prompt(self, patient_data: Dict[str, Any]) -> str:
        symptoms = patient_data.get("symptoms", "N/A")
        prompt = (
            f"For the patient with symptoms: {symptoms}.\n"
            "Generate a tabular chain-of-thought analysis. For each potential condition, list key indicators and evaluate their presence/absence in the patient's data. Conclude with a likely diagnosis based on this structured reasoning.\n"
            "Format: | Condition | Indicator 1 | Indicator 2 | ... | Evaluation |\n"
            "        |-----------|-------------|-------------|-----|------------|"
        )
        return prompt

    def generate_cove_treatment_prompt(self, diagnosis: str, patient_data: Dict[str, Any]) -> str:
        allergies = patient_data.get("allergies", "None")
        medications = patient_data.get("current_medications", "None")
        prompt = (
            f"Based on the diagnosis: '{diagnosis}', propose a detailed treatment plan.\n"
            "Then, critically verify this plan by considering potential drug interactions with existing medications ({medications}),\n"
            f"patient allergies ({allergies}), and adherence to medical guidelines. Identify any risks or inconsistencies.\n"
            "Explain your verification steps."
        )
        return prompt
    
    def generate_reversing_cot_prompt(self, treatment_plan: str, target_diagnosis: str) -> str:
        prompt = (
            f"Given the proposed treatment plan: '{treatment_plan}'.\n"
            f"Reverse engineer a chain of thought to justify how this treatment plan logically leads from or supports the diagnosis of '{target_diagnosis}'.\n"
            "Highlight the key medical principles or conditions that would necessitate such a treatment."
        )
        return prompt

    def generate_active_prompt_for_refinement(self, current_information: str) -> str:
        prompt = (
            f"Based on the current patient information and preliminary findings: '{current_information}'.\n"
            "What crucial follow-up questions should be asked to the healthcare professional to gather more specific information and refine the diagnosis or treatment plan? Provide 3-5 concise questions."
        )
        return prompt


# --- 3. Knowledge Base & Verifier Module ---
class KnowledgeBaseVerifier:
    def __init__(self, db_path: str = "medical_kb.json"):
        self.db_path = db_path
        self.medical_knowledge = self._load_knowledge_base()
        # In a real system, you'd initialize a vector DB client (e.g., Chroma, Pinecone)
        # and an embedding model (e.g., from sentence_transformers)
        # self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Knowledge Base Verifier initialized. Loaded {len(self.medical_knowledge)} medical entries.")

    def _load_knowledge_base(self) -> List[Dict[str, str]]:
        # Simulate loading a medical knowledge base
        # In reality, this would involve parsing medical documents, guidelines, drug databases
        return [
            {"topic": "Influenza", "info": "Common viral respiratory infection. Symptoms include fever, cough, sore throat... Treatment: rest, fluids, antivirals (oseltamivir) in some cases."},
            {"topic": "Pneumonia", "info": "Lung infection, can be bacterial or viral. Symptoms: cough, fever, chills, shortness of breath... Treatment: antibiotics for bacterial, supportive care for viral."},
            {"topic": "Acute Bronchitis", "info": "Inflammation of bronchial tubes, often viral. Symptoms: cough, mucus, chest discomfort... Treatment: usually self-limiting, symptom relief."},
            {"topic": "Salbutamol", "info": "Bronchodilator, relaxes muscles in the airways. Used for asthma, COPD. Side effects: tremor, headache, palpitations."},
            {"topic": "Azithromycin", "info": "Antibiotic, treats bacterial infections. Side effects: nausea, diarrhea, abdominal pain. Contraindicated with certain heart conditions."},
            {"topic": "Paracetamol", "info": "Pain reliever and fever reducer. Side effects: liver damage in overdose."},
            {"topic": "Medical Guideline: Asthma Exacerbation", "info": "Mild exacerbation: short-acting beta-agonist (SABA). Moderate: SABA + oral corticosteroids. Severe: Hospitalization, oxygen, systemic steroids."}
        ]

    def retrieve_info(self, query: str) -> List[str]:
        # Simulate RAG: simple keyword matching for demonstration
        # In a real system, this would involve vector similarity search
        results = [entry["info"] for entry in self.medical_knowledge if query.lower() in entry["topic"].lower() or query.lower() in entry["info"].lower()]
        return results if results else ["No specific information found in knowledge base for this query."]

    def verify_treatment_plan(self, diagnosis: str, treatment_plan: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        verification_results = {
            "drug_interactions_check": "N/A",
            "allergy_check": "N/A",
            "guideline_adherence": "N/A",
            "overall_consistency": "N/A",
            "issues_found": []
        }

        # Simulate drug interaction check
        current_meds = patient_data.get("current_medications", "").lower()
        if "azithromycin" in treatment_plan.lower() and "certain heart conditions" in current_meds:
            verification_results["drug_interactions_check"] = "Potential severe interaction detected with Azithromycin and heart condition. Consult cardiologist."
            verification_results["issues_found"].append("Drug interaction")
        else:
            verification_results["drug_interactions_check"] = "No immediate critical drug interactions found (simulated)."
        
        # Simulate allergy check
        patient_allergies = patient_data.get("allergies", "").lower()
        if any(med.lower() in patient_allergies for med in ["salbutamol", "azithromycin", "paracetamol"]):
            verification_results["allergy_check"] = "Potential allergy to one of the proposed medications. Re-evaluate treatment."
            verification_results["issues_found"].append("Allergy")
        else:
            verification_results["allergy_check"] = "No known allergies to proposed medications (simulated)."

        # Simulate guideline adherence
        if "asthma" in diagnosis.lower() and "inhaler" in treatment_plan.lower() and "corticosteroids" not in treatment_plan.lower():
            verification_results["guideline_adherence"] = "Consider adding corticosteroids for moderate asthma exacerbation (guideline check)."
            verification_results["issues_found"].append("Guideline deviation")
        else:
            verification_results["guideline_adherence"] = "Treatment generally aligns with guidelines (simulated)."

        verification_results["overall_consistency"] = "Consistent" if not verification_results["issues_found"] else "Inconsistent"
        return verification_results


# --- 4. Data Ingestion & Preprocessing ---
class DataProcessor:
    def __init__(self):
        print("Data Processor initialized.")

    def load_medical_data(self, source_path: str) -> List[Dict[str, Any]]:
        # Simulate loading data from various sources
        print(f"Loading data from {source_path}...")
        # In reality, this would parse PDFs, EMR data, etc.
        return [
            {"text": "Patient presented with cough, fever, and shortness of breath. No known drug allergies. History of asthma.", "label": "Pneumonia"},
            {"text": "Patient has seasonal allergies, runny nose, and itchy eyes. No fever.", "label": "Allergies"}
        ]

    def anonymize_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("Anonymizing data...")
        # Simple placeholder for anonymization. Real anonymization is complex.
        anonymized_data = []
        for entry in data:
            anonymized_text = entry["text"].replace("Patient", "Individual").replace("John Doe", "Anon Person")
            anonymized_data.append({"text": anonymized_text, "label": entry["label"]})
        return anonymized_data

    def preprocess_for_llm(self, data: List[Dict[str, Any]]) -> List[str]:
        print("Preprocessing data for LLM finetuning...")
        # In reality, this involves tokenization, formatting for specific LLM tasks
        return [entry["text"] for entry in data]


# --- 5. Finetuning Module ---
class FinetuningModule:
    def __init__(self, llm_core: LLMCore):
        self.llm_core = llm_core
        print("Finetuning Module initialized.")

    def finetune_llm(self, preprocessed_data: List[str], finetuning_type: str = "domain-specific"):
        print(f"Starting {finetuning_type} finetuning on {len(preprocessed_data)} samples...")
        # This is a placeholder. Real finetuning involves training loops, optimizers, etc.
        time.sleep(3) # Simulate finetuning time
        self.llm_core.model_name = f"MedicalLLM-v1-Finetuned-{finetuning_type}"
        print(f"LLM finetuning complete. Model updated to: {self.llm_core.model_name}")


# --- 6. User Interface (UI) - using Gradio for simplicity ---
# This will be called externally to run the UI
import gradio as gr

class MedVerifyUI:
    def __init__(self, orchestrator_instance: 'MedVerifyOrchestrator'):
        self.orchestrator = orchestrator_instance

    def process_patient_data(self, symptoms, medical_history, test_results, allergies, current_medications):
        patient_data = {
            "symptoms": symptoms,
            "medical_history": medical_history,
            "test_results": test_results,
            "allergies": allergies,
            "current_medications": current_medications
        }
        
        full_output = []

        # 1. Initial CoT Diagnosis
        full_output.append("### Step 1: Initial Chain-of-Thought Diagnosis")
        diagnosis_prompt = self.orchestrator.prompt_engine.generate_cot_diagnosis_prompt(patient_data)
        diagnosis_response = self.orchestrator.llm_core.generate_response(diagnosis_prompt)
        full_output.append(f"**Prompt:** {diagnosis_prompt}\n\n**LLM Response:** {diagnosis_response}")
        diagnosis = diagnosis_response.split("**Proposed Diagnosis:**")[-1].strip().split("\n")[0].strip()
        full_output.append(f"**Extracted Diagnosis:** {diagnosis}\n")

        # 2. StepBack Prompting (Example if ambiguity detected)
        if "ambiguity" in diagnosis_response.lower() or "consider differential" in diagnosis_response.lower():
            full_output.append("\n### Step 2: StepBack Prompting (for ambiguity)")
            stepback_prompt = self.orchestrator.prompt_engine.generate_stepback_prompt(diagnosis_response)
            stepback_response = self.orchestrator.llm_core.generate_response(stepback_prompt)
            full_output.append(f"**Prompt:** {stepback_prompt}\n\n**LLM Response:** {stepback_response}\n")
        
        # 3. CoVe Treatment Plan
        full_output.append("\n### Step 3: Chain-of-Verification Treatment Plan")
        treatment_prompt = self.orchestrator.prompt_engine.generate_cove_treatment_prompt(diagnosis, patient_data)
        treatment_response = self.orchestrator.llm_core.generate_response(treatment_prompt)
        full_output.append(f"**Prompt:** {treatment_prompt}\n\n**LLM Response:** {treatment_response}")
        treatment_plan = treatment_response.split("**Recommended Treatment:**")[-1].strip().split("Confidence")[0].strip()
        full_output.append(f"**Extracted Treatment Plan:** {treatment_plan}\n")

        # 4. Verifier Module Check
        full_output.append("\n### Step 4: Knowledge Base & Verifier Module Check")
        verification_results = self.orchestrator.kb_verifier.verify_treatment_plan(diagnosis, treatment_plan, patient_data)
        full_output.append(f"**Verification Results:** {verification_results}\n")
        if verification_results["issues_found"]:
            full_output.append("**ACTION REQUIRED:** Issues detected in treatment plan. Review immediately.\n")

        # 5. Active Prompting for Refinement
        full_output.append("\n### Step 5: Active Prompting for Refinement")
        refinement_prompt = self.orchestrator.prompt_engine.generate_active_prompt_for_refinement(f"Diagnosis: {diagnosis}, Treatment: {treatment_plan}, Verification: {verification_results}")
        refinement_response = self.orchestrator.llm_core.generate_response(refinement_prompt)
        full_output.append(f"**Prompt:** {refinement_prompt}\n\n**LLM Response (Follow-up Questions):** {refinement_response}\n")

        # 6. Reversing CoT (Example)
        full_output.append("\n### Step 6: Reversing Chain-of-Thought (Justification)")
        reverse_cot_prompt = self.orchestrator.prompt_engine.generate_reversing_cot_prompt(treatment_plan, diagnosis)
        reverse_cot_response = self.orchestrator.llm_core.generate_response(reverse_cot_prompt)
        full_output.append(f"**Prompt:** {reverse_cot_prompt}\n\n**LLM Response:** {reverse_cot_response}\n")

        self.orchestrator.eval_monitor.log_interaction(patient_data, diagnosis, treatment_plan, verification_results, "success")

        return "\n".join(full_output)

    def create_interface(self):
        iface = gr.Interface(
            fn=self.process_patient_data,
            inputs=[
                gr.Textbox(label="Patient Symptoms (e.g., cough, fever, shortness of breath)", lines=3, value="Persistent cough, mild fever (100.5F), fatigue."),
                gr.Textbox(label="Medical History (e.g., asthma, diabetes, heart conditions)", lines=2, value="History of seasonal allergies."),
                gr.Textbox(label="Test Results (e.g., X-ray, blood tests, no results)", lines=2, value="Rapid Flu Test: Negative. CBC: Normal."),
                gr.Textbox(label="Allergies (e.g., Penicillin, dust, pollen)", lines=1, value="Dust, Pollen."),
                gr.Textbox(label="Current Medications (e.g., Lisinopril, Ibuprofen)", lines=1, value="Multivitamin.")
            ],
            outputs=gr.Markdown(label="MedVerify AI Diagnosis & Treatment Recommendation"),
            title="MedVerify AI: Enhanced Diagnostic & Treatment System",
            description="Enter patient details to get AI-powered diagnosis and verified treatment recommendations. This system uses Chain-of-Thought, Self-Correction, and RAG for reliability.",
            allow_flagging="manual",
            flagging_callback=self.orchestrator.eval_monitor.log_feedback
        )
        return iface


# --- 7. API/Backend (using FastAPI for simplicity) ---
# This will be part of the main application file but run separately if needed
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class PatientInput(BaseModel):
    symptoms: str
    medical_history: str
    test_results: str
    allergies: str = "None"
    current_medications: str = "None"

class MedVerifyAPI:
    def __init__(self, orchestrator_instance: 'MedVerifyOrchestrator'):
        self.app = FastAPI(title="MedVerify AI API")
        self.orchestrator = orchestrator_instance
        self._setup_routes()
        print("FastAPI backend initialized.")

    def _setup_routes(self):
        @self.app.post("/diagnose_and_treat")
        async def diagnose_and_treat(patient_data: PatientInput):
            try:
                # This is a simplified call, in a real API, the orchestration
                # would handle the full multi-step process and return structured data.
                # For now, we'll mimic the UI's `process_patient_data` flow.
                full_output_markdown = self.orchestrator.process_full_patient_workflow(patient_data.dict())
                return {"status": "success", "recommendations": full_output_markdown}
            except Exception as e:
                self.orchestrator.eval_monitor.log_error(patient_data.dict(), str(e))
                raise HTTPException(status_code=500, detail=str(e))


# --- 8. Orchestration & Workflow Management ---
class MedVerifyOrchestrator:
    def __init__(self):
        self.llm_core = LLMCore()
        self.prompt_engine = PromptEngineering(self.llm_core)
        self.kb_verifier = KnowledgeBaseVerifier()
        self.data_processor = DataProcessor()
        self.finetuning_module = FinetuningModule(self.llm_core)
        self.eval_monitor = EvaluatorMonitor()
        print("MedVerify Orchestrator initialized.")

    def process_full_patient_workflow(self, patient_data: Dict[str, Any]) -> str:
        # This method orchestrates the full logic as seen in the UI's process_patient_data
        full_output = []

        # 1. Initial CoT Diagnosis
        full_output.append("### Step 1: Initial Chain-of-Thought Diagnosis")
        diagnosis_prompt = self.prompt_engine.generate_cot_diagnosis_prompt(patient_data)
        diagnosis_response = self.llm_core.generate_response(diagnosis_prompt)
        full_output.append(f"**Prompt:** {diagnosis_prompt}\n\n**LLM Response:** {diagnosis_response}")
        diagnosis = diagnosis_response.split("**Proposed Diagnosis:**")[-1].strip().split("\n")[0].strip()
        full_output.append(f"**Extracted Diagnosis:** {diagnosis}\n")

        # 2. StepBack Prompting (Example if ambiguity detected)
        if "ambiguity" in diagnosis_response.lower() or "consider differential" in diagnosis_response.lower():
            full_output.append("\n### Step 2: StepBack Prompting (for ambiguity)")
            stepback_prompt = self.prompt_engine.generate_stepback_prompt(diagnosis_response)
            stepback_response = self.llm_core.generate_response(stepback_prompt)
            full_output.append(f"**Prompt:** {stepback_prompt}\n\n**LLM Response:** {stepback_response}\n")
        
        # 3. CoVe Treatment Plan
        full_output.append("\n### Step 3: Chain-of-Verification Treatment Plan")
        treatment_prompt = self.prompt_engine.generate_cove_treatment_prompt(diagnosis, patient_data)
        treatment_response = self.llm_core.generate_response(treatment_prompt)
        full_output.append(f"**Prompt:** {treatment_prompt}\n\n**LLM Response:** {treatment_response}")
        treatment_plan = treatment_response.split("**Recommended Treatment:**")[-1].strip().split("Confidence")[0].strip()
        full_output.append(f"**Extracted Treatment Plan:** {treatment_plan}\n")

        # 4. Verifier Module Check
        full_output.append("\n### Step 4: Knowledge Base & Verifier Module Check")
        verification_results = self.kb_verifier.verify_treatment_plan(diagnosis, treatment_plan, patient_data)
        full_output.append(f"**Verification Results:** {verification_results}\n")
        if verification_results["issues_found"]:
            full_output.append("**ACTION REQUIRED:** Issues detected in treatment plan. Review immediately.\n")

        # 5. Active Prompting for Refinement
        full_output.append("\n### Step 5: Active Prompting for Refinement")
        refinement_prompt = self.prompt_engine.generate_active_prompt_for_refinement(f"Diagnosis: {diagnosis}, Treatment: {treatment_plan}, Verification: {verification_results}")
        refinement_response = self.llm_core.generate_response(refinement_prompt)
        full_output.append(f"**Prompt:** {refinement_prompt}\n\n**LLM Response (Follow-up Questions):** {refinement_response}\n")

        # 6. Reversing CoT (Example)
        full_output.append("\n### Step 6: Reversing Chain-of-Thought (Justification)")
        reverse_cot_prompt = self.prompt_engine.generate_reversing_cot_prompt(treatment_plan, diagnosis)
        reverse_cot_response = self.llm_core.generate_response(reverse_cot_prompt)
        full_output.append(f"**Prompt:** {reverse_cot_prompt}\n\n**LLM Response:** {reverse_cot_response}\n")

        self.eval_monitor.log_interaction(patient_data, diagnosis, treatment_plan, verification_results, "success")

        return "\n".join(full_output)



# --- 9. Evaluation & Monitoring ---
class EvaluatorMonitor:
    def __init__(self):
        self.logs = []
        self.feedback_logs = []
        print("Evaluator and Monitor initialized.")

    def log_interaction(self, input_data: Dict, diagnosis: str, treatment: str, verification: Dict, status: str):
        log_entry = {
            "timestamp": time.time(),
            "input": input_data,
            "diagnosis": diagnosis,
            "treatment": treatment,
            "verification": verification,
            "status": status
        }
        self.logs.append(log_entry)
        print(f"[Monitor] Logged interaction: {status}")
        # In a real system, this would push to WandB, TruLens, or a dedicated logging system.

    def log_feedback(self, label: str, inputs: List[Any]):
        # Gradio's flagging callback provides label (e.g., "Yes" for good, "No" for bad) and inputs
        feedback_entry = {
            "timestamp": time.time(),
            "label": label,
            "input_symptoms": inputs[0],
            "input_history": inputs[1],
            "input_tests": inputs[2],
            "input_allergies": inputs[3],
            "input_meds": inputs[4],
            # The full output is not directly available here, would need to store previous state if needed
            "comment": "User flagged output"
        }
        self.feedback_logs.append(feedback_entry)
        print(f"[Monitor] Logged user feedback: {label}")
        # This feedback could be used for RLHF or dataset curation.

    def log_error(self, input_data: Dict, error_message: str):
        error_entry = {
            "timestamp": time.time(),
            "input": input_data,
            "error": error_message
        }
        self.logs.append(error_entry)
        print(f"[Monitor] Logged error: {error_message}")


# --- Main Application Entry Point ---
if __name__ == "__main__":
    print("Starting MedVerify AI System...")

    # Initialize the Orchestrator which in turn initializes all other modules
    orchestrator = MedVerifyOrchestrator()

    # Example of Finetuning (simulated)
    print("\n--- Simulating Data Ingestion and Finetuning ---")
    raw_data = orchestrator.data_processor.load_medical_data("mock_clinical_notes.txt")
    anonymized_data = orchestrator.data_processor.anonymize_data(raw_data)
    preprocessed_data = orchestrator.data_processor.preprocess_for_llm(anonymized_data)
    orchestrator.finetuning_module.finetune_llm(preprocessed_data, "domain-specific")
    print("--- Finetuning Simulation Complete ---\n")


    # --- Run the Gradio UI ---
    print("Starting Gradio UI... Access it in your browser (usually http://127.0.0.1:7860)")
    ui_app = MedVerifyUI(orchestrator)
    iface = ui_app.create_interface()
    
    # To run the Gradio app, use the following:
    # iface.launch(share=False) # share=True for public link, but be cautious with medical data
    print("To run the Gradio UI, uncomment `iface.launch()` and execute this script.")
    # iface.launch()

    # --- Run the FastAPI Backend (optional, can be run in a separate process/server) ---
    print("\n--- FastAPI Backend Setup ---")
    api_app = MedVerifyAPI(orchestrator)
    print("FastAPI app created. To run it, you would typically use:\n")
    print("  uvicorn medverify_ai_system:api_app.app --reload")
    print("Make sure to install uvicorn: `pip install uvicorn`")
    print("Access API endpoints like /diagnose_and_treat (POST request).")
    
    # Example API usage (programmatic call simulation)
    print("\n--- Simulating API Call ---")
    sample_patient_input = PatientInput(
        symptoms="Severe headache, stiff neck, sensitivity to light.",
        medical_history="None relevant.",
        test_results="CSF analysis pending."
    )
    # In a real scenario, this would be an HTTP POST request
    # For simulation, directly call the orchestrator method
    try:
        print("Simulating API call to /diagnose_and_treat...")
        api_response = orchestrator.process_full_patient_workflow(sample_patient_input.dict())
        print("\n--- API Call Simulation Result ---")
        print(api_response)
    except Exception as e:
        print(f"API Call Simulation Failed: {e}")

    print("\nMedVerify AI System startup complete. Ready for interactions.")
    print("Monitoring logs:", orchestrator.eval_monitor.logs)
    print("Feedback logs:", orchestrator.eval_monitor.feedback_logs)
