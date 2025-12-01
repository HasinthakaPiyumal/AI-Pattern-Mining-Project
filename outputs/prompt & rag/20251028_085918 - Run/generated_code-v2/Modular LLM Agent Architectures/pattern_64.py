
class MedicalDatabaseTool:
    def get_disease_info(self, disease_name: str) -> str:
        # Simulate fetching disease information from a database
        disease_data = {
            "diabetes": "Diabetes is a chronic condition that affects how your body turns food into energy.",
            "hypertension": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems.",
            "asthma": "Asthma is a condition in which your airways narrow and swell and may produce extra mucus."
        }
        info = disease_data.get(disease_name.lower(), "Disease information not found.")
        return f"Information about {disease_name}: {info}"

    def get_symptom_info(self, symptom_name: str) -> str:
        # Simulate fetching symptom information
        symptom_data = {
            "fever": "Fever is a temporary increase in your body temperature, often due to an illness.",
            "cough": "A cough is a reflex action to clear your airways of mucus and irritants.",
            "fatigue": "Fatigue is extreme tiredness resulting from mental or physical exertion or illness."
        }
        info = symptom_data.get(symptom_name.lower(), "Symptom information not found.")
        return f"Information about {symptom_name}: {info}"

    def get_drug_info(self, drug_name: str) -> str:
        # Simulate fetching drug information
        drug_data = {
            "paracetamol": "Paracetamol (Acetaminophen) is a pain reliever and a fever reducer.",
            "insulin": "Insulin is a hormone that helps move sugar from the blood into other body tissues for use as energy.",
            "amoxicillin": "Amoxicillin is an antibiotic used to treat a wide variety of bacterial infections."
        }
        info = drug_data.get(drug_name.lower(), "Drug information not found.")
        return f"Information about {drug_name}: {info}"

class LabResultInterpreterTool:
    def interpret_results(self, lab_data: dict) -> str:
        # Simulate interpreting various lab results
        interpretations = []
        if "glucose" in lab_data:
            glucose_level = lab_data["glucose"]
            if glucose_level > 125:
                interpretations.append(f"High glucose level ({glucose_level} mg/dL) may indicate diabetes.")
            elif glucose_level < 70:
                interpretations.append(f"Low glucose level ({glucose_level} mg/dL) may indicate hypoglycemia.")
            else:
                interpretations.append(f"Normal glucose level ({glucose_level} mg/dL).")

        if "white blood cells" in lab_data:
            wbc_count = lab_data["white blood cells"]
            if wbc_count > 11000:
                interpretations.append(f"High white blood cell count ({wbc_count}/\u00b5L) may suggest an infection or inflammation.")
            elif wbc_count < 4000:
                interpretations.append(f"Low white blood cell count ({wbc_count}/\u00b5L) may suggest a weakened immune system.")
            else:
                interpretations.append(f"Normal white blood cell count ({wbc_count}/\u00b5L).")

        if not interpretations:
            return "No specific interpretation available for the provided lab data."
        return " ".join(interpretations)

class MedicalCalculatorTool:
    def calculate_bmi(self, weight_kg: float, height_m: float) -> str:
        if height_m <= 0:
            return "Height must be greater than zero."
        bmi = weight_kg / (height_m ** 2)
        category = ""
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 24.9:
            category = "Normal weight"
        elif 25 <= bmi < 29.9:
            category = "Overweight"
        else:
            category = "Obesity"
        return f"Calculated BMI: {bmi:.2f} ({category})."

    def calculate_drug_dosage(self, weight_kg: float, drug_mg_per_kg: float) -> str:
        if weight_kg <= 0 or drug_mg_per_kg <= 0:
            return "Weight and drug dosage per kg must be greater than zero."
        total_dosage = weight_kg * drug_mg_per_kg
        return f"Recommended drug dosage: {total_dosage:.2f} mg."

class LLMRouter:
    def __init__(self):
        self.medical_db = MedicalDatabaseTool()
        self.lab_interpreter = LabResultInterpreterTool()
        self.medical_calculator = MedicalCalculatorTool()

    def route_query(self, query: str) -> str:
        query_lower = query.lower()
        response = []

        # Tool: MedicalDatabaseTool
        if "disease info" in query_lower or "what is" in query_lower and ("disease" in query_lower or "condition" in query_lower):
            if "diabetes" in query_lower:
                response.append(self.medical_db.get_disease_info("diabetes"))
            elif "hypertension" in query_lower:
                response.append(self.medical_db.get_disease_info("hypertension"))
            elif "asthma" in query_lower:
                response.append(self.medical_db.get_disease_info("asthma"))

        if "symptom info" in query_lower or "about symptom" in query_lower:
            if "fever" in query_lower:
                response.append(self.medical_db.get_symptom_info("fever"))
            elif "cough" in query_lower:
                response.append(self.medical_db.get_symptom_info("cough"))

        if "drug info" in query_lower or "about drug" in query_lower or "medication" in query_lower:
            if "paracetamol" in query_lower:
                response.append(self.medical_db.get_drug_info("paracetamol"))
            elif "insulin" in query_lower:
                response.append(self.medical_db.get_drug_info("insulin"))

        # Tool: LabResultInterpreterTool
        # This would ideally parse complex natural language for lab data, but for simulation,
        # we'll use a specific trigger and dummy data.
        if "interpret lab results" in query_lower:
            # Simulate parsing a simple set of lab results from the query or external source
            # In a real system, this would involve more sophisticated NLP or structured input
            if "glucose 150" in query_lower and "wbc 12000" in query_lower:
                lab_data = {"glucose": 150, "white blood cells": 12000}
                response.append(self.lab_interpreter.interpret_results(lab_data))
            elif "glucose 85" in query_lower:
                lab_data = {"glucose": 85}
                response.append(self.lab_interpreter.interpret_results(lab_data))
            else:
                response.append("Please provide lab results in a structured format for interpretation.")

        # Tool: MedicalCalculatorTool
        if "calculate bmi" in query_lower:
            # Example: "calculate bmi for weight 70kg height 1.75m"
            import re
            weight_match = re.search(r"weight (\d+\.?\d*)kg", query_lower)
            height_match = re.search(r"height (\d+\.?\d*)m", query_lower)
            if weight_match and height_match:
                weight = float(weight_match.group(1))
                height = float(height_match.group(1))
                response.append(self.medical_calculator.calculate_bmi(weight, height))
            else:
                response.append("Please provide weight (e.g., '70kg') and height (e.g., '1.75m') to calculate BMI.")

        if "calculate drug dosage" in query_lower:
            # Example: "calculate drug dosage for 60kg patient with 5mg/kg drug"
            import re
            weight_match = re.search(r"patient with (\d+\.?\d*)kg", query_lower)
            dosage_match = re.search(r"(\d+\.?\d*)mg/kg drug", query_lower)
            if weight_match and dosage_match:
                weight = float(weight_match.group(1))
                dosage_per_kg = float(dosage_match.group(1))
                response.append(self.medical_calculator.calculate_drug_dosage(weight, dosage_per_kg))
            else:
                response.append("Please provide patient weight (e.g., '60kg') and drug dosage per kg (e.g., '5mg/kg') to calculate drug dosage.")

        if not response:
            return "I couldn't find a specific tool to address your query. Please rephrase or ask for specific information."
        
        return "\n".join(response)

# --- Example Usage ---
if __name__ == "__main__":
    router = LLMRouter()

    print("\n--- Query 1: Disease Information ---")
    query1 = "What is hypertension disease info?"
    print(f"User: {query1}")
    print(f"Assistant: {router.route_query(query1)}")

    print("\n--- Query 2: Drug Information ---")
    query2 = "Tell me about paracetamol medication."
    print(f"User: {query2}")
    print(f"Assistant: {router.route_query(query2)}")

    print("\n--- Query 3: BMI Calculation ---")
    query3 = "Can you calculate bmi for weight 75kg height 1.80m?"
    print(f"User: {query3}")
    print(f"Assistant: {router.route_query(query3)}")

    print("\n--- Query 4: Lab Results Interpretation (High Glucose & WBC) ---")
    query4 = "Please interpret lab results with glucose 150 and wbc 12000."
    print(f"User: {query4}")
    print(f"Assistant: {router.route_query(query4)}")

    print("\n--- Query 5: Lab Results Interpretation (Normal Glucose) ---")
    query5 = "Interpret lab results for glucose 85."
    print(f"User: {query5}")
    print(f"Assistant: {router.route_query(query5)}")
    
    print("\n--- Query 6: Drug Dosage Calculation ---")
    query6 = "I need to calculate drug dosage for a patient with 65kg and a drug of 10mg/kg."
    print(f"User: {query6}")
    print(f"Assistant: {router.route_query(query6)}")

    print("\n--- Query 7: Unclear Query ---")
    query7 = "Tell me something random."
    print(f"User: {query7}")
    print(f"Assistant: {router.route_query(query7)}")

    print("\n--- Query 8: Symptom Information ---")
    query8 = "What is information about symptom fever?"
    print(f"User: {query8}")
    print(f"Assistant: {router.route_query(query8)}")
