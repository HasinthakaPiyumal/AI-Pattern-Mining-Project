
import gradio as gr
import pandas as pd
import os
from typing import List, Dict, Any

# --- 1. Simplified Medical Knowledge Base (In-memory) ---
# In a real application, this would be a much larger, structured database or a vector store.
medical_knowledge_base = {
    "conditions": {
        "Influenza (Flu)": {
            "symptoms": ["fever", "cough", "sore throat", "body aches", "fatigue", "headache"],
            "treatments": ["antivirals", "rest", "fluids"]
        },
        "Common Cold": {
            "symptoms": ["runny nose", "sore throat", "cough", "sneezing", "congestion"],
            "treatments": ["rest", "fluids", "decongestants"]
        },
        "Pneumonia": {
            "symptoms": ["cough", "fever", "chills", "difficulty breathing", "chest pain"],
            "treatments": ["antibiotics", "antivirals", "oxygen therapy"]
        },
        "Bronchitis": {
            "symptoms": ["cough", "mucus production", "fatigue", "shortness of breath", "chest discomfort"],
            "treatments": ["bronchodilators", "cough suppressants", "rest"]
        }
    },
    "symptoms": {
        "fever": {"causes": ["Influenza (Flu)", "Pneumonia", "Common Cold"]},
        "cough": {"causes": ["Influenza (Flu)", "Pneumonia", "Common Cold", "Bronchitis"]},
        "sore throat": {"causes": ["Influenza (Flu)", "Common Cold"]},
        "body aches": {"causes": ["Influenza (Flu)"]},
        "fatigue": {"causes": ["Influenza (Flu)", "Bronchitis"]},
        "headache": {"causes": ["Influenza (Flu)"]},
        "runny nose": {"causes": ["Common Cold"]},
        "sneezing": {"causes": ["Common Cold"]},
        "congestion": {"causes": ["Common Cold"]},
        "chills": {"causes": ["Pneumonia"]},
        "difficulty breathing": {"causes": ["Pneumonia"]},
        "chest pain": {"causes": ["Pneumonia"]},
        "mucus production": {"causes": ["Bronchitis"]},
        "shortness of breath": {"causes": ["Bronchitis"]},
        "chest discomfort": {"causes": ["Bronchitis"]}
    }
}

# --- 2. Verifier Component ---
class Verifier:
    def __init__(self, knowledge_base: Dict[str, Any]):
        self.knowledge_base = knowledge_base

    def check_factual_accuracy(self, statement: str) -> Dict[str, Any]:
        # This is a very simplistic factual checker.
        # A real system would use NLP techniques, semantic search, or external APIs.
        statement_lower = statement.lower()
        feedback = {"is_accurate": True, "explanation": "Factually accurate or unverified."}

        for condition, details in self.knowledge_base["conditions"].items():
            condition_lower = condition.lower()
            if condition_lower in statement_lower:
                for symptom in details["symptoms"]:
                    if f"symptom of {symptom.lower()}" in statement_lower or f"caused by {symptom.lower()}" in statement_lower:
                        # This logic needs refinement for proper NLP parsing
                        # For now, let's assume if it mentions a condition and a known symptom, it's plausible.
                        pass
                if "symptoms include" in statement_lower or "characterized by" in statement_lower:
                    # Check if stated symptoms are actually in the knowledge base for this condition
                    for symptom in details["symptoms"]:
                        if symptom.lower() in statement_lower:
                            feedback["is_accurate"] = True # Found a match
                            break
                    else:
                        feedback = {"is_accurate": False, "explanation": f"Statement about {condition} contains unverified symptoms."}
                        return feedback # Return immediately if inaccuracy found

        # Example of a negative check
        if "cancer" in statement_lower and "common cold" in statement_lower:
            feedback = {"is_accurate": False, "explanation": "Common cold is not typically associated with cancer."}
            return feedback

        return feedback

    def check_logical_consistency(self, current_step: str, previous_step: str = None) -> Dict[str, Any]:
        feedback = {"is_consistent": True, "explanation": "Logically consistent."}

        # Simple contradiction check example
        if previous_step and "not" in previous_step.lower() and previous_step.lower().replace("not", "").strip() in current_step.lower():
            feedback = {"is_consistent": False, "explanation": "Potential contradiction with the previous step."}
            return feedback

        if "no fever" in current_step.lower() and "high fever" in current_step.lower():
            feedback = {"is_consistent": False, "explanation": "Contradictory information about fever in the same step."}
            return feedback

        return feedback

    def verify_reasoning_chain(self, reasoning_chain: List[str]) -> List[Dict[str, Any]]:
        verification_results = []
        previous_step = None

        for i, step in enumerate(reasoning_chain):
            step_results = {"step": step, "factual_accuracy": {}, "logical_consistency": {}}

            step_results["factual_accuracy"] = self.check_factual_accuracy(step)
            step_results["logical_consistency"] = self.check_logical_consistency(step, previous_step)

            verification_results.append(step_results)
            previous_step = step

        return verification_results

# Initialize the Verifier
verifier = Verifier(medical_knowledge_base)

# --- 3. LLM Orchestration and Diagnosis Generation (Simulated) ---
# In a real application, you would use Langchain and an actual LLM provider (e.g., OpenAI).
# For demonstration, we simulate the LLM's response.

def simulate_llm_diagnosis(symptoms: str, medical_history: str) -> Dict[str, Any]:
    # This function would interact with Langchain and an actual LLM
    # For now, it provides a hardcoded response based on keywords

    diagnosis = "Uncertain Diagnosis"
    reasoning_steps = []

    symptoms_lower = symptoms.lower()
    history_lower = medical_history.lower()

    if "fever" in symptoms_lower and "cough" in symptoms_lower and "body aches" in symptoms_lower:
        diagnosis = "Probable Influenza (Flu)"
        reasoning_steps = [
            "Patient presents with fever, cough, and body aches, which are classic symptoms of Influenza.",
            "Influenza is a viral infection that commonly causes these respiratory and systemic symptoms.",
            "Given the symptom profile, an influenza diagnosis is strongly indicated. Further testing (e.g., flu test) could confirm."
        ]
    elif "runny nose" in symptoms_lower and "sore throat" in symptoms_lower and "sneezing" in symptoms_lower:
        diagnosis = "Probable Common Cold"
        reasoning_steps = [
            "Symptoms like runny nose, sore throat, and sneezing are highly indicative of a Common Cold.",
            "The Common Cold is a mild viral infection of the nose and throat.",
            "These symptoms, without severe systemic signs like high fever or extreme fatigue, point towards a common cold."
        ]
    elif "cough" in symptoms_lower and "difficulty breathing" in symptoms_lower and "chest pain" in symptoms_lower:
        diagnosis = "Possible Pneumonia"
        reasoning_steps = [
            "The combination of cough, difficulty breathing, and chest pain raises suspicion for Pneumonia.",
            "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus, leading to respiratory distress.",
            "Medical history of smoking or recent illness could further increase the likelihood of pneumonia. Immediate medical evaluation is recommended."
        ]
    else:
        diagnosis = "Further investigation required"
        reasoning_steps = [
            "Based on the provided symptoms, a definitive diagnosis is not immediately apparent.",
            "More detailed information, potentially including lab tests or imaging, would be necessary."
        ]
    
    # Simulate an 