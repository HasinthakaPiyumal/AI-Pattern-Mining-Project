from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import random

# Placeholder for a base LLM model
class FoundationLLM:
    def __init__(self, model_name: str = "dummy-llm"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        # Simulate LLM response
        if "diagnostic support for a cough" in prompt.lower():
            return "It could be a common cold, flu, or pneumonia. Consider chest X-ray for pneumonia. Avoid specific drug names without further examination."
        elif "treatment for diabetes" in prompt.lower():
            return "For diabetes, treatment typically involves lifestyle changes, oral medications like metformin, or insulin therapy. Specific dosages depend on individual patient factors and should be prescribed by a doctor."
        elif "patient X's symptoms" in prompt.lower():
            return "Patient X is experiencing flu-like symptoms. Recommend rest and fluids."
        return f"Based on your query: '{prompt}', the LLM suggests general medical information. Please consult a healthcare professional for personalized advice."

# The Constitution: Ethical principles
CONSTITUTION = [
    "Factuality: All medical information must be evidence-based and accurate.",
    "Non-Maleficence: Do no harm; avoid generating dangerous or misleading advice.",
    "Beneficence: Act in the best interest of the patient/user; provide helpful and constructive information.",
    "Patient Privacy: Do not process or generate responses that could compromise patient data or privacy.",
    "Fairness: Avoid biases in diagnostic support or treatment recommendations.",
    "Transparency: Clearly state limitations, sources, and uncertainties.",
    "Dignity: Maintain a respectful and empathetic tone."
]

class CritiqueModel:
    def __init__(self, constitution: List[str]):
        self.constitution = constitution

    def evaluate(self, generated_text: str, query: str) -> Dict[str, List[str]]:
        violations = []
        # Simulate critique based on some keywords or random chance
        if "avoid specific drug names" in generated_text.lower():
            violations.append("Non-Maleficence: Directly suggesting medication without patient context.")
        if "patient x's symptoms" in query.lower() and "patient x" in generated_text.lower():
            violations.append("Patient Privacy: Mentioning patient identifiers.")
        if random.random() < 0.2: # 20% chance of a random factual violation
            violations.append("Factuality: Potential for unverified information. Always cross-reference.")
        
        return {"violations_found": violations}

class RevisionModel:
    def revise(self, original_text: str, critique: Dict[str, List[str]]) -> str:
        revised_text = original_text
        for violation in critique.get("violations_found", []):
            if "Non-Maleficence" in violation:
                revised_text = revised_text.replace("Avoid specific drug names without further examination.", "Always consult with a qualified healthcare professional before administering any medication.")
            if "Patient Privacy" in violation:
                revised_text = revised_text.replace("Patient X is experiencing flu-like symptoms.", "The patient is experiencing flu-like symptoms.")
            if "Factuality" in violation:
                revised_text = f"[Review Required for Factuality] {revised_text} (Note: Always verify information with up-to-date medical sources.)"
        return revised_text

class SLAIFModule:
    def __init__(self):
        self.feedback_data = [] # Stores (original_output, revised_output) pairs

    def collect_feedback(self, original_output: str, revised_output: str):
        self.feedback_data.append({"original": original_output, "revised": revised_output})
        print(f"SLAIF: Collected feedback. Current data points: {len(self.feedback_data)}")

    def get_feedback_data(self):
        return self.feedback_data

class EthicalMedicalAssistant:
    def __init__(self):
        self.foundation_llm = FoundationLLM()
        self.critique_model = CritiqueModel(constitution=CONSTITUTION)
        self.revision_model = RevisionModel()
        self.slaif_module = SLAIFModule()

    def process_query(self, query: str) -> Dict[str, str]:
        original_response = self.foundation_llm.generate(query)
        critique_result = self.critique_model.evaluate(original_response, query)

        if critique_result["violations_found"]:
            revised_response = self.revision_model.revise(original_response, critique_result)
            self.slaif_module.collect_feedback(original_response, revised_response)
            return {"response": revised_response, "status": "revised", "violations": critique_result["violations_found"]}
        else:
            return {"response": original_response, "status": "original", "violations": []}

app = FastAPI()
ethical_assistant = EthicalMedicalAssistant()

class QueryRequest(BaseModel):
    query: str

@app.post("/medical-assistant/query")
async def query_medical_assistant(request: QueryRequest):
    result = ethical_assistant.process_query(request.query)
    return result

@app.get("/medical-assistant/slaif-data")
async def get_slaif_data():
    return {"slaif_feedback": ethical_assistant.slaif_module.get_feedback_data()}
