from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

app = FastAPI()

# --- Simulated External Tools ---
class MedicalKnowledgeBase:
    def get_disease_info(self, disease_name: str) -> str:
        # Simulate API call to a medical knowledge base
        if "diabetes" in disease_name.lower():
            return "Diabetes is a chronic condition that affects how your body turns food into energy. It requires careful management of blood sugar levels."
        elif "hypertension" in disease_name.lower():
            return "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease."
        else:
            return f"Could not find detailed information for '{disease_name}' in the medical knowledge base."

class DrugInteractionChecker:
    def check_interactions(self, drug1: str, drug2: str) -> str:
        # Simulate API call to a drug interaction checker
        d1_lower = drug1.lower()
        d2_lower = drug2.lower()

        if ("ibuprofen" in d1_lower and "warfarin" in d2_lower) or ("warfarin" in d1_lower and "ibuprofen" in d2_lower):
            return "Severe interaction: Ibuprofen can increase the risk of bleeding when taken with Warfarin. Consult a doctor immediately."
        elif ("acetaminophen" in d1_lower and "alcohol" in d2_lower) or ("alcohol" in d1_lower and "acetaminophen" in d2_lower):
            return "Moderate interaction: Combining Acetaminophen with alcohol can increase the risk of liver damage."
        else:
            return f"No significant interactions found between {drug1} and {drug2} based on our simulated data."

class MedicalCalculator:
    def calculate(self, expression: str) -> str:
        # Simulate a simple medical calculation or a general calculator API
        try:
            # Basic evaluation for demonstration. In a real app, use a safer math parser.
            result = eval(expression.replace('x', '*')) # Replace 'x' with '*' for multiplication
            return f"The result of '{expression}' is: {result}"
        except Exception as e:
            return f"Could not perform calculation for '{expression}': {e}"

class SymptomChecker:
    def get_possible_conditions(self, symptoms: str) -> str:
        # Simulate API call to a symptom checker
        symptoms_lower = symptoms.lower()
        possible_conditions = []

        if "fever" in symptoms_lower and "cough" in symptoms_lower and "sore throat" in symptoms_lower:
            possible_conditions.append("Common Cold or Flu")
        if "chest pain" in symptoms_lower and "shortness of breath" in symptoms_lower:
            possible_conditions.append("Cardiac Issue (e.g., Angina, Heart Attack) - Seek immediate medical attention!")
        if "headache" in symptoms_lower and "nausea" in symptoms_lower:
            possible_conditions.append("Migraine or Tension Headache")
        if "fatigue" in symptoms_lower and "weight loss" in symptoms_lower:
            possible_conditions.append("Thyroid Dysfunction or other chronic condition")

        if possible_conditions:
            return f"Based on symptoms '{symptoms}', possible conditions include: {'; '.join(possible_conditions)}. Please consult a healthcare professional for a proper diagnosis."
        else:
            return f"No immediate matches for symptoms '{symptoms}'. Please provide more details or consult a healthcare professional."

# --- LLM Router (Simulated using keyword matching) ---
class LLMRouter:
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()
        self.drug_checker = DrugInteractionChecker()
        self.calculator = MedicalCalculator()
        self.symptom_checker = SymptomChecker()

    def route_query(self, query: str) -> str:
        query_lower = query.lower()

        if "symptom" in query_lower or "i feel" in query_lower or "my symptoms are" in query_lower:
            # Extract symptoms (simple heuristic)
            match = re.search(r"symptoms (are|include)?\s*(.*?)(?=\.|$)", query_lower)
            symptoms = match.group(2).strip() if match else query
            return self.symptom_checker.get_possible_conditions(symptoms)
        
        elif "drug interaction" in query_lower or "interact with" in query_lower:
            # Extract drugs (simple heuristic, could be improved with NLP)
            drugs = re.findall(r'\b(?!drug|interact|with)\w+\b', query_lower)
            drugs = [d for d in drugs if d not in ['how', 'do', 'will', 'my', 'is', 'affect', 'what', 'between']][:2]
            if len(drugs) >= 2:
                return self.drug_checker.check_interactions(drugs[0], drugs[1])
            else:
                return "Please specify at least two drugs to check for interactions."

        elif "calculate" in query_lower or re.search(r'[\d\s\+\-\*\/x]+', query_lower):
            # Extract expression (simple heuristic)
            match = re.search(r'calculate\s*(.*?)(?=\.|\?|$)', query_lower) or re.search(r'is\s*(.*?)(?=\.|\?|$)', query_lower)
            expression = match.group(1).strip() if match else query_lower.replace('calculate', '').strip()
            # Further refinement to extract only the mathematical part
            math_expression = re.search(r'[\d\s\+\-\*\/x\(\).]+', expression)
            if math_expression:
                return self.calculator.calculate(math_expression.group(0))
            else:
                return "Please provide a valid mathematical expression to calculate."

        elif "what is" in query_lower or "info about" in query_lower or "tell me about" in query_lower:
            # Extract disease name (simple heuristic)
            match = re.search(r'(what is|info about|tell me about)\s*(.*?)(?=\.|$)', query_lower)
            disease = match.group(2).strip() if match else query_lower.replace('what is', '').replace('info about', '').replace('tell me about', '').strip()
            return self.knowledge_base.get_disease_info(disease)
        
        else:
            return "I'm sorry, I couldn't understand your request. Please ask about symptoms, drug interactions, calculations, or disease information."

# --- FastAPI Endpoints ---
class QueryRequest(BaseModel):
    query: str

@app.post("/diagnose")
async def diagnose(request: QueryRequest):
    router = LLMRouter()
    response = router.route_query(request.query)
    return {"response": response}

# To run this application:
# 1. Save the code as healthcare_diagnostic_assistant.py
# 2. Install uvicorn: pip install uvicorn fastapi pydantic
# 3. Run from your terminal: uvicorn healthcare_diagnostic_assistant:app --reload
# 4. Access the API at http://127.0.0.1:8000/diagnose (POST request with JSON body: {"query": "Your question here"})