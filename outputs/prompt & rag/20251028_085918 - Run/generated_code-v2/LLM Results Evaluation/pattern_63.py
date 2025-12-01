from pydantic import BaseModel
from typing import List, Dict, Any


class WorkingMemory(BaseModel):
    dialog_history: List[str] = []
    current_user_query: str = ""
    external_evidence_cache: str = ""
    feedback_log: str = ""

    def update_dialog_history(self, new_turn: str):
        self.dialog_history.append(new_turn)

    def update_query(self, query: str):
        self.current_user_query = query

    def update_evidence(self, evidence: str):
        self.external_evidence_cache = evidence

    def update_feedback(self, feedback: str):
        self.feedback_log = feedback


class KnowledgeConsolidator:
    def fetch_ehr_data(self, patient_id: str) -> str:
        # Simulate fetching data from an EHR system
        if patient_id == "P123":
            return "Patient P123 has a history of hypertension and diabetes. Latest lab results show elevated blood sugar levels (HbA1c 8.5%)."
        return "No EHR data found for this patient."

    def query_medical_database(self, query: str) -> str:
        # Simulate querying external medical databases
        if "elevated blood sugar" in query.lower() or "diabetes management" in query.lower():
            return "Clinical guidelines for diabetes suggest metformin as first-line treatment, followed by SGLT2 inhibitors or GLP-1 receptor agonists if targets are not met. Regular monitoring of HbA1c is crucial."
        return "No specific medical database information found for this query."

    def synthesize_evidence(self, ehr_data: str, db_data: str) -> str:
        # Combine and summarize the retrieved evidence
        if not ehr_data and not db_data:
            return "No relevant evidence available."
        return f"Patient History: {ehr_data}\nMedical Guidelines: {db_data}"


class UtilityModule:
    def assess_relevance(self, llm_response: str, user_query: str) -> str:
        # Simulate relevance scoring
        if "diagnosis" in llm_response.lower() and "treatment" in llm_response.lower() and user_query.lower() in llm_response.lower():
            return "Highly relevant."
        return "Moderately relevant, could be more specific."

    def check_grounding(self, llm_response: str, evidence: str) -> str:
        # Simulate grounding check against provided evidence
        if "HbA1c 8.5%" in llm_response and "metformin" in llm_response and "hypertension" in evidence:
            return "Well-grounded in provided evidence."
        return "Partially grounded, some statements lack direct evidence."

    def generate_feedback(self, relevance_score: str, grounding_check: str) -> str:
        # Formulate actionable feedback for the Prompt Engine
        feedback = []
        if "Moderately relevant" in relevance_score:
            feedback.append("Response lacked specificity related to the initial query.")
        if "Partially grounded" in grounding_check:
            feedback.append("Ensure all diagnostic claims are directly supported by evidence.")
        return "\n".join(feedback) if feedback else "No specific feedback required."


class PromptEngine:
    def __init__(self, task_instruction: str):
        self.task_instruction = task_instruction
        self.prompt_template = """
{task_instruction}

Dialog History:
{dialog_history}

Current User Query:
{current_user_query}

External Evidence:
{external_evidence_cache}

Automated Feedback:
{feedback_log}

Please provide a diagnostic assessment and recommended next steps based on the above information.
"""

    def construct_prompt(self, working_memory: WorkingMemory) -> str:
        dialog_history_str = "\n".join(working_memory.dialog_history) if working_memory.dialog_history else "N/A"
        
        return self.prompt_template.format(
            task_instruction=self.task_instruction,
            dialog_history=dialog_history_str,
            current_user_query=working_memory.current_user_query,
            external_evidence_cache=working_memory.external_evidence_cache,
            feedback_log=working_memory.feedback_log
        )


class LLMInteractionModule:
    def send_prompt_to_llm(self, prompt: str) -> str:
        # Simulate interaction with an LLM
        # In a real application, this would involve an API call (e.g., to OpenAI, a local Hugging Face model)
        print("\n--- Sending Prompt to LLM ---")
        print(prompt)
        print("-----------------------------")
        
        # This is a highly simplified simulated response
        if "elevated blood sugar" in prompt and "P123" in prompt:
            return "Based on patient P123's elevated HbA1c (8.5%) and history of diabetes, the diagnosis is uncontrolled Type 2 Diabetes Mellitus. Recommend initiating/adjusting Metformin, consider adding an SGLT2 inhibitor. Advise regular glucose monitoring and lifestyle modifications. Review in 3 months."
        elif "hypertension" in prompt:
            return "Patient shows signs of hypertension. Further investigation and medication review are recommended."
        else:
            return "I need more information to provide a comprehensive diagnosis. Please provide additional details about the patient's symptoms, history, and relevant lab results."

    def parse_llm_response(self, raw_response: str) -> str:
        # Simple parsing for demonstration; real parsing might extract specific entities or sections
        return raw_response


class MedicalDiagnosisAssistant:
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.knowledge_consolidator = KnowledgeConsolidator()
        self.utility_module = UtilityModule()
        self.prompt_engine = PromptEngine(task_instruction="Act as a highly experienced medical diagnostic AI. Provide comprehensive diagnostic assessments and clear treatment recommendations based *only* on the provided context.")
        self.llm_interaction_module = LLMInteractionModule()

    def conduct_consultation(self, patient_id: str, doctor_query: str) -> str:
        self.working_memory.update_query(doctor_query)
        self.working_memory.update_dialog_history(f"Doctor: {doctor_query}")

        # 1. Knowledge Consolidation
        ehr_data = self.knowledge_consolidator.fetch_ehr_data(patient_id)
        medical_db_data = self.knowledge_consolidator.query_medical_database(doctor_query)
        consolidated_evidence = self.knowledge_consolidator.synthesize_evidence(ehr_data, medical_db_data)
        self.working_memory.update_evidence(consolidated_evidence)

        # 2. Prompt Construction
        prompt = self.prompt_engine.construct_prompt(self.working_memory)

        # 3. LLM Interaction
        raw_llm_response = self.llm_interaction_module.send_prompt_to_llm(prompt)
        assistant_response = self.llm_interaction_module.parse_llm_response(raw_llm_response)
        self.working_memory.update_dialog_history(f"Assistant: {assistant_response}")

        # 4. Utility Module Feedback (for next turn or internal refinement)
        relevance = self.utility_module.assess_relevance(assistant_response, doctor_query)
        grounding = self.utility_module.check_grounding(assistant_response, consolidated_evidence)
        feedback = self.utility_module.generate_feedback(relevance, grounding)
        self.working_memory.update_feedback(feedback)
        
        print(f"\n--- Assistant's Response ---\n{assistant_response}\n----------------------------")
        print(f"--- Automated Feedback (for internal use) ---\nRelevance: {relevance}\nGrounding: {grounding}\nFeedback: {feedback}\n--------------------------------------------")

        return assistant_response

# Example Usage:
if __name__ == "__main__":
    assistant = MedicalDiagnosisAssistant()
    
    print("\n--- First Consultation Turn ---")
    response1 = assistant.conduct_consultation(
        patient_id="P123",
        doctor_query="What is the diagnosis for patient P123 given their current lab results and history, and what are the initial treatment recommendations?"
    )
    
    print("\n--- Second Consultation Turn (simulating follow-up or refinement) ---")
    response2 = assistant.conduct_consultation(
        patient_id="P123",
        doctor_query="Considering the previous discussion, what are the long-term management strategies for patient P123's diabetes and hypertension?"
    )
    
    print("\n--- Final Working Memory State ---")
    print(assistant.working_memory.model_dump_json(indent=2))

    print("\n--- New Patient Scenario ---")
    response3 = assistant.conduct_consultation(
        patient_id="P999",
        doctor_query="Patient P999 presents with sudden chest pain and shortness of breath. What could be the potential causes?"
    )
    
    print("\n--- Final Working Memory State (after new patient, should reflect new context) ---")
    print(assistant.working_memory.model_dump_json(indent=2))

