import pandas as pd
import numpy as np
import random

# --- 1. Data Ingestion and Preprocessing (Simulated) ---
class PatientDataHandler:
    def __init__(self):
        pass

    def ingest_patient_data(self, raw_data):
        processed_data = {
            "patient_id": raw_data.get("patient_id", "P001"),
            "symptoms": raw_data.get("symptoms", []),
            "lab_results": raw_data.get("lab_results", {})
        }
        return processed_data

# --- 2. Tool Integration (Simulated Tools) ---
class MedicalKnowledgeBaseTool:
    def query(self, topic):
        if "diabetes" in topic.lower():
            return "Diabetes is a chronic metabolic disease characterized by high blood sugar levels. Treatment often involves insulin, diet, and exercise."
        elif "hypertension" in topic.lower():
            return "Hypertension (high blood pressure) can lead to heart disease. Management includes lifestyle changes and medication."
        else:
            return "Information on '{}' is limited in the current knowledge base.".format(topic)

class ClinicalDataAnalysisTool:
    def analyze(self, patient_record):
        analysis_results = {}
        if "lab_results" in patient_record and "glucose" in patient_record["lab_results"]:
            glucose_level = patient_record["lab_results"]["glucose"]
            if glucose_level > 125:
                analysis_results["high_glucose_alert"] = True
            else:
                analysis_results["high_glucose_alert"] = False
        analysis_results["risk_score"] = random.randint(30, 90) # Simulated risk score
        return analysis_results

class ImagingAnalysisTool:
    def analyze_image(self, image_data):
        return "Simulated imaging report: No major anomalies detected. (placeholder)"

# --- 3. Core Diagnostic LLM Agent (Simulated) ---
class MockLLM:
    def generate_response(self, prompt, history=None):
        if "symptoms" in prompt and "fever" in prompt and "cough" in prompt:
            if random.random() < 0.7:
                return "Based on the symptoms, a common cold or flu is likely. Consider rest and hydration." , 0.8
            else:
                return "The symptoms suggest a viral infection. Further tests might be needed to rule out other conditions.", 0.6
        elif "high glucose" in prompt:
            return "Elevated glucose levels are concerning. Consider diabetes screening and lifestyle modifications.", 0.9
        elif "refine diagnosis" in prompt:
            return "Re-evaluating based on new information. My refined diagnosis is...", 0.75
        else:
            return "I am still processing the information. Please provide more context.", 0.5

class DiagnosticAgent:
    def __init__(self, llm, medical_kb_tool, clinical_data_tool, imaging_tool):
        self.llm = llm
        self.medical_kb = medical_kb_tool
        self.clinical_data_tool = clinical_data_tool
        self.imaging_tool = imaging_tool
        self.memory = []
        self.max_iterations = 3
        self.diagnosis_history = []

    def _generate_prompt(self, patient_data, current_state, feedback=None):
        prompt = f"Patient ID: {patient_data['patient_id']}\n"
        prompt += f"Symptoms: {', '.join(patient_data['symptoms'])}\n"
        prompt += f"Lab Results: {patient_data['lab_results']}\n"
        prompt += f"Current State: {current_state}\n"
        if self.memory:
            prompt += f"Previous Interactions: {'\n'.join(self.memory)}\n"
        if feedback:
            prompt += f"Clinician Feedback: {feedback}\n"
            prompt += "Please refine the diagnosis and treatment plan based on this feedback.\n"
        else:
            prompt += "Please provide an initial diagnosis and suggest further steps or tests.\n"
        return prompt

    def _use_tool(self, tool_name, query_or_data):
        if tool_name == "medical_kb":
            return self.medical_kb.query(query_or_data)
        elif tool_name == "clinical_data_analysis":
            return self.clinical_data_tool.analyze(query_or_data)
        elif tool_name == "imaging_analysis":
            return self.imaging_tool.analyze_image(query_or_data)
        return "Tool not found or invalid usage."

    def _self_correct(self, current_diagnosis, confidence, patient_data):
        if confidence < 0.7:
            print("Agent: Low confidence detected. Attempting self-correction...")
            # Simulate querying tools for more info
            if "glucose" in patient_data["lab_results"] and patient_data["lab_results"]["glucose"] > 125:
                kb_info = self._use_tool("medical_kb", "diabetes")
                analysis_info = self._use_tool("clinical_data_analysis", patient_data)
                correction_prompt = f"I initially diagnosed with '{current_diagnosis}' but confidence is low. Patient has high glucose. Medical KB info: {kb_info}. Clinical data analysis: {analysis_info}. Re-evaluate."
                corrected_diagnosis, new_confidence = self.llm.generate_response(correction_prompt)
                if new_confidence > confidence:
                    print(f"Agent: Self-corrected. New confidence: {new_confidence}")
                    return corrected_diagnosis, new_confidence
        return current_diagnosis, confidence

    def _assess_confidence(self, llm_response):
        # Mock LLM returns confidence directly. In real world, this could be from LLM's logprobs or a separate model.
        return llm_response[1]

    def diagnose_patient(self, patient_data):
        print(f"\n--- Starting Diagnosis for Patient {patient_data['patient_id']} ---")
        current_diagnosis = ""
        current_confidence = 0.0
        iteration = 0
        self.memory = []
        self.diagnosis_history = []

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}:")
            prompt = self._generate_prompt(patient_data, current_diagnosis, feedback=None if iteration == 1 else self.memory[-1])
            
            llm_response, confidence = self.llm.generate_response(prompt)
            current_diagnosis = llm_response
            current_confidence = confidence

            print(f"Agent (Initial/Refined): {current_diagnosis} (Confidence: {current_confidence:.2f})")
            self.diagnosis_history.append((current_diagnosis, current_confidence))
            self.memory.append(f"Agent Response {iteration}: {current_diagnosis} (Conf: {current_confidence:.2f})")

            # Self-correction
            corrected_diagnosis, corrected_confidence = self._self_correct(current_diagnosis, current_confidence, patient_data)
            if corrected_confidence > current_confidence:
                current_diagnosis = corrected_diagnosis
                current_confidence = corrected_confidence
                print(f"Agent (Self-Corrected): {current_diagnosis} (Confidence: {current_confidence:.2f})")
                self.diagnosis_history[-1] = (current_diagnosis, current_confidence) # Update history

            # --- Simulate Clinician Feedback ---
            clinician_feedback = input("Clinician Feedback (e.g., 'correct', 'incorrect, consider X', 'needs more info'): ")
            self.memory.append(f"Clinician Feedback {iteration}: {clinician_feedback}")

            if "correct" in clinician_feedback.lower() and current_confidence > 0.7:
                print("Agent: Clinician feedback positive and high confidence. Finalizing diagnosis.")
                break
            elif "incorrect" in clinician_feedback.lower() or current_confidence < 0.6:
                print("Agent: Incorporating clinician feedback and low confidence for next iteration.")
            elif "more info" in clinician_feedback.lower():
                print("Agent: Clinician requested more info. Querying tools...")
                analysis_result = self._use_tool("clinical_data_analysis", patient_data)
                kb_result = self._use_tool("medical_kb", "general medical condition")
                self.memory.append(f"Tool Results: Clinical Analysis: {analysis_result}, Medical KB: {kb_result}")

            if iteration == self.max_iterations:
                print("Agent: Maximum iterations reached.")

        final_diagnosis = self.diagnosis_history[-1][0] if self.diagnosis_history else "No diagnosis reached."
        final_confidence = self.diagnosis_history[-1][1] if self.diagnosis_history else 0.0
        explanation = f"The final diagnosis of '{final_diagnosis}' was reached after {iteration} iterations, considering patient data, medical knowledge, and clinician feedback. Confidence score: {final_confidence:.2f}."

        print(f"\n--- Final Diagnosis for Patient {patient_data['patient_id']} ---")
        print(f"Diagnosis: {final_diagnosis}")
        print(f"Confidence: {final_confidence:.2f}")
        print(f"Explanation: {explanation}")
        return final_diagnosis, final_confidence, explanation

# --- Main Application Loop (Simulated Streamlit/Gradio interaction) ---
def run_diagnostic_app():
    data_handler = PatientDataHandler()
    medical_kb = MedicalKnowledgeBaseTool()
    clinical_analyzer = ClinicalDataAnalysisTool()
    imaging_tool = ImagingAnalysisTool()
    mock_llm = MockLLM()

    agent = DiagnosticAgent(mock_llm, medical_kb, clinical_analyzer, imaging_tool)

    print("\nWelcome to the AI Diagnostic Assistant (Simulated Interface)\n")

    while True:
        print("\n--- New Patient Session ---")
        patient_id = input("Enter Patient ID (e.g., P001): ")
        symptoms_input = input("Enter symptoms (comma-separated, e.g., fever, cough, fatigue): ")
        lab_results_glucose = input("Enter Glucose Lab Result (numeric, e.g., 150) or leave blank: ")

        raw_patient_data = {
            "patient_id": patient_id,
            "symptoms": [s.strip() for s in symptoms_input.split(',')] if symptoms_input else [],
            "lab_results": {}
        }
        if lab_results_glucose.strip():
            try:
                raw_patient_data["lab_results"]["glucose"] = float(lab_results_glucose)
            except ValueError:
                print("Invalid glucose value. Ignoring.")

        patient_data = data_handler.ingest_patient_data(raw_patient_data)

        final_diagnosis, final_confidence, explanation = agent.diagnose_patient(patient_data)

        another_patient = input("\nProcess another patient? (yes/no): ")
        if another_patient.lower() != 'yes':
            break

if __name__ == "__main__":
    run_diagnostic_app()
