
import json

# 1. Memory Module (Patient Records)
class MemoryModule:
    def __init__(self):
        self.patient_records = {}

    def add_patient_record(self, patient_id, record):
        self.patient_records[patient_id] = record
        print(f"[Memory] Added record for patient {patient_id}")

    def get_patient_record(self, patient_id):
        record = self.patient_records.get(patient_id)
        print(f"[Memory] Retrieving record for patient {patient_id}: {record is not None}")
        return record

# 2. External Tool Interfaces
class MedicalDatabaseAPI:
    def query(self, symptoms, patient_history=None):
        print(f"[Tool] Querying medical database for symptoms: {symptoms}")
        # Mock database response
        if "fever" in symptoms and "cough" in symptoms:
            return {"diagnosis_suggestions": ["Influenza", "Common Cold", "Bronchitis"], "common_treatments": ["Rest", "Fluids"]}
        elif "headache" in symptoms and "nausea" in symptoms:
            return {"diagnosis_suggestions": ["Migraine", "Tension Headache"], "common_treatments": ["Painkillers", "Rest in dark room"]}
        return {"diagnosis_suggestions": ["Unknown Condition"], "common_treatments": ["Symptomatic treatment"]}

class ImagingAnalysisModule:
    def analyze_image(self, image_data):
        print(f"[Tool] Analyzing image data...")
        # Mock imaging analysis
        if "lung_xray" in image_data:
            return {"findings": "Possible pneumonia in right lung"}
        return {"findings": "No significant findings"}

class LabResultInterpretationModule:
    def interpret_results(self, lab_results):
        print(f"[Tool] Interpreting lab results: {lab_results}")
        # Mock lab result interpretation
        if "WBC" in lab_results and lab_results["WBC"] > 15000:
            return {"interpretation": "High White Blood Cell count, indicating infection"}
        return {"interpretation": "Lab results within normal limits"}

# 3. Planning Module (Treatment Pathways)
class PlanningModule:
    def suggest_treatment_plan(self, diagnosis, patient_record):
        print(f"[Planning] Suggesting treatment for: {diagnosis}")
        plan = {"diagnosis": diagnosis, "steps": []}
        if diagnosis == "Influenza":
            plan["steps"] = ["Antiviral medication (if early)", "Rest", "Hydration", "Over-the-counter fever reducers"]
        elif diagnosis == "Pneumonia":
            plan["steps"] = ["Antibiotics", "Rest", "Oxygen therapy (if needed)", "Follow-up imaging"]
        elif diagnosis == "Migraine":
            plan["steps"] = ["Migraine-specific medication", "Avoid triggers", "Dark, quiet room"]
        else:
            plan["steps"] = ["Symptomatic relief", "Further investigation"]

        if patient_record and patient_record.get("allergies"): # Example of considering patient record
            plan["considerations"] = f"Check for allergies: {', '.join(patient_record['allergies'])}"
        return plan

# 4. Context Management System
class ContextManagementSystem:
    def __init__(self):
        self.conversation_history = []
        self.current_context = {}

    def add_to_history(self, role, message):
        self.conversation_history.append({"role": role, "message": message})

    def update_context(self, key, value):
        self.current_context[key] = value

    def get_context(self, key=None):
        if key:
            return self.current_context.get(key)
        return self.current_context

    def clear_current_context(self):
        self.current_context = {}

# 5. Core LLM (Diagnosis & Orchestration)
class CoreLLM:
    def __init__(self, memory_module, medical_db, imaging_analysis, lab_interpretation, planning_module, context_manager):
        self.memory = memory_module
        self.medical_db = medical_db
        self.imaging_analysis = imaging_analysis
        self.lab_interpretation = lab_interpretation
        self.planning = planning_module
        self.context = context_manager

    def process_symptoms(self, patient_id, symptoms):
        self.context.clear_current_context()
        self.context.add_to_history("user", f"Patient {patient_id} reports symptoms: {symptoms}")
        self

        # 1. Retrieve patient history
        patient_record = self.memory.get_patient_record(patient_id)
        self.context.update_context("patient_record", patient_record)
        
        llm_thought = f"Considering symptoms: {symptoms}"
        if patient_record:
            llm_thought += f" and patient history: {patient_record}"

        # 2. Query external medical database
        db_response = self.medical_db.query(symptoms, patient_history=patient_record)
        self.context.update_context("db_suggestions", db_response.get("diagnosis_suggestions"))
        llm_thought += f"\nMedical database suggests: {db_response.get('diagnosis_suggestions')}"

        # Simulate LLM reasoning for further tool use (e.g., if symptoms suggest need for imaging)
        if "shortness of breath" in symptoms or "chest pain" in symptoms:
            llm_thought += "\nSymptoms suggest potential need for imaging analysis."
            imaging_data = {"type": "lung_xray", "patient_id": patient_id} # Mock data
            imaging_results = self.imaging_analysis.analyze_image(imaging_data)
            self.context.update_context("imaging_results", imaging_results)
            llm_thought += f"\nImaging analysis findings: {imaging_results.get('findings')}"
        
        # Simulate LLM reasoning for lab tests
        if "fever" in symptoms and "fatigue" in symptoms:
            llm_thought += "\nSymptoms suggest potential need for lab tests."
            lab_results_data = {"WBC": 18000, "CRP": 12} # Mock data
            lab_interpretation = self.lab_interpretation.interpret_results(lab_results_data)
            self.context.update_context("lab_interpretation", lab_interpretation)
            llm_thought += f"\nLab interpretation: {lab_interpretation.get('interpretation')}"

        # 3. Generate diagnostic hypothesis (simplified LLM output)
        final_diagnosis = "" 
        if self.context.get("imaging_results", {}).get("findings") == "Possible pneumonia in right lung":
            final_diagnosis = "Pneumonia"
        elif "Influenza" in self.context.get("db_suggestions", []) and "High White Blood Cell count" in self.context.get("lab_interpretation", {}).get("interpretation", ""):
            final_diagnosis = "Severe Influenza (likely with bacterial co-infection)"
        elif self.context.get("db_suggestions") and self.context.get("db_suggestions")[0] != "Unknown Condition":
            final_diagnosis = self.context.get("db_suggestions")[0] # Take the first suggestion for simplicity
        else:
            final_diagnosis = "Undetermined. Further investigation needed."
            
        self.context.update_context("final_diagnosis", final_diagnosis)
        llm_thought += f"\nBased on all information, the diagnostic hypothesis is: {final_diagnosis}"
        self.context.add_to_history("llm_thought", llm_thought)

        # 4. Suggest treatment plan
        treatment_plan = self.planning.suggest_treatment_plan(final_diagnosis, patient_record)
        self.context.update_context("treatment_plan", treatment_plan)
        self.context.add_to_history("assistant", f"Proposed treatment plan: {json.dumps(treatment_plan, indent=2)}")
        
        response = f"Diagnostic Hypothesis: {final_diagnosis}\n\nTreatment Plan:\n"
        for step in treatment_plan["steps"]:
            response += f"- {step}\n"
        if treatment_plan.get("considerations"):
            response += f"Considerations: {treatment_plan['considerations']}\n"

        return response

# Main Application Orchestrator
class MedicalDiagnosticAssistant:
    def __init__(self):
        self.memory_module = MemoryModule()
        self.medical_db = MedicalDatabaseAPI()
        self.imaging_analysis = ImagingAnalysisModule()
        self.lab_interpretation = LabResultInterpretationModule()
        self.planning_module = PlanningModule()
        self.context_manager = ContextManagementSystem()
        self.core_llm = CoreLLM(
            self.memory_module, 
            self.medical_db, 
            self.imaging_analysis, 
            self.lab_interpretation, 
            self.planning_module, 
            self.context_manager
        )

    def run_diagnosis(self, patient_id, symptoms):
        print(f"\n--- Starting Diagnosis for Patient {patient_id} with symptoms: {symptoms} ---")
        response = self.core_llm.process_symptoms(patient_id, symptoms)
        print(f"--- Diagnosis Complete ---")
        print("\nAssistant Response:\n", response)
        print("\n--- Current Context ---")
        print(json.dumps(self.context_manager.get_context(), indent=2))
        print("\n--- Conversation History ---")
        print(json.dumps(self.context_manager.conversation_history, indent=2))
        return response

# Example Usage
if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    # Add a patient record to memory
    assistant.memory_module.add_patient_record(
        "patient_001", 
        {"name": "Alice Smith", "age": 45, "allergies": ["Penicillin"], "past_conditions": ["Hypertension"]}
    )

    assistant.memory_module.add_patient_record(
        "patient_002",
        {"name": "Bob Johnson", "age": 30, "allergies": [], "past_conditions": ["Asthma"]}
    )

    # Scenario 1: Common Cold/Flu symptoms
    assistant.run_diagnosis("patient_001", ["fever", "cough", "sore throat"])

    # Scenario 2: Symptoms requiring imaging
    assistant.run_diagnosis("patient_002", ["shortness of breath", "chest pain", "fever"])

    # Scenario 3: Symptoms suggesting lab tests
    assistant.run_diagnosis("patient_001", ["fever", "fatigue", "body aches"])

    # Scenario 4: Simple headache
    assistant.run_diagnosis("patient_002", ["headache", "mild nausea"])
