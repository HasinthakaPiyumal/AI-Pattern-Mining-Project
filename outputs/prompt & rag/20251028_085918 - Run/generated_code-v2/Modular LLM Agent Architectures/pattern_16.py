def MedicalKnowledgeBaseAPI(query: str) -> dict:
    knowledge_base = {
        "diabetes": "Diabetes is a chronic condition that affects how your body turns food into energy.",
        "hypertension": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems.",
        "fever": "A fever is a temporary increase in your body temperature, often due to an illness."
    }
    info = knowledge_base.get(query.lower(), "Information not found for your query.")
    return {"tool": "MedicalKnowledgeBaseAPI", "query": query, "result": info}

def MedicalCalculatorAPI(calculation_type: str, parameters: dict) -> dict:
    if calculation_type.lower() == "bmi":
        weight_kg = parameters.get("weight_kg")
        height_m = parameters.get("height_m")
        if weight_kg and height_m and height_m > 0:
            bmi = weight_kg / (height_m ** 2)
            return {"tool": "MedicalCalculatorAPI", "type": "BMI", "result": f"{bmi:.2f}"}
        else:
            return {"tool": "MedicalCalculatorAPI", "type": "BMI", "error": "Invalid parameters for BMI calculation."}
    elif calculation_type.lower() == "dosage":
        drug_mg_per_kg = parameters.get("drug_mg_per_kg")
        patient_weight_kg = parameters.get("patient_weight_kg")
        if drug_mg_per_kg and patient_weight_kg:
            dosage = drug_mg_per_kg * patient_weight_kg
            return {"tool": "MedicalCalculatorAPI", "type": "Dosage", "result": f"{dosage:.2f} mg"}
        else:
            return {"tool": "MedicalCalculatorAPI", "type": "Dosage", "error": "Invalid parameters for dosage calculation."}
    else:
        return {"tool": "MedicalCalculatorAPI", "error": "Unsupported calculation type."}

def SymptomCheckerAPI(symptoms: list[str]) -> dict:
    symptom_data = {
        "headache": {"possible_conditions": ["Migraine", "Tension Headache", "Sinusitis"], "likelihood": "common"},
        "sore throat": {"possible_conditions": ["Common Cold", "Streptococcal Pharyngitis"], "likelihood": "common"},
        "fatigue": {"possible_conditions": ["Anemia", "Chronic Fatigue Syndrome", "Hypothyroidism"], "likelihood": "variable"},
        "chest pain": {"possible_conditions": ["Heart Attack", "Angina", "Acid Reflux"], "likelihood": "serious"}
    }
    
    matching_conditions = {}
    for symptom in symptoms:
        if symptom.lower() in symptom_data:
            data = symptom_data[symptom.lower()]
            for condition in data["possible_conditions"]:
                matching_conditions[condition] = matching_conditions.get(condition, 0) + 1

    if matching_conditions:
        sorted_conditions = sorted(matching_conditions.items(), key=lambda item: item[1], reverse=True)
        top_conditions = [cond for cond, count in sorted_conditions]
        return {"tool": "SymptomCheckerAPI", "symptoms": symptoms, "potential_diagnoses": top_conditions}
    else:
        return {"tool": "SymptomCheckerAPI", "symptoms": symptoms, "potential_diagnoses": []}

def llm_router(user_query: str) -> tuple[str, dict]:
    query_lower = user_query.lower()

    if "what is" in query_lower or "tell me about" in query_lower or "info on" in query_lower or "information about" in query_lower:
        keywords = ["diabetes", "hypertension", "fever"]
        for keyword in keywords:
            if keyword in query_lower:
                return "MedicalKnowledgeBaseAPI", MedicalKnowledgeBaseAPI(keyword)
        return "MedicalKnowledgeBaseAPI", MedicalKnowledgeBaseAPI(query_lower.replace("what is ", "").replace("tell me about ", "").strip())

    elif "calculate bmi" in query_lower or "body mass index" in query_lower:
        try:
            weight_str = user_query.split("weight ")[1].split("kg")[0].strip()
            height_str = user_query.split("height ")[1].split("m")[0].strip()
            weight = float(weight_str)
            height = float(height_str)
            return "MedicalCalculatorAPI", MedicalCalculatorAPI("bmi", {"weight_kg": weight, "height_m": height})
        except (IndexError, ValueError):
            return "MedicalCalculatorAPI", MedicalCalculatorAPI("bmi", {"error": "Could not parse weight or height."})
    
    elif "calculate dosage" in query_lower or "drug dosage" in query_lower:
        try:
            drug_mg_per_kg_str = user_query.split("dosage ")[1].split("mg/kg")[0].strip()
            patient_weight_kg_str = user_query.split("patient weight ")[1].split("kg")[0].strip()
            drug_mg_per_kg = float(drug_mg_per_kg_str)
            patient_weight_kg = float(patient_weight_kg_str)
            return "MedicalCalculatorAPI", MedicalCalculatorAPI("dosage", {"drug_mg_per_kg": drug_mg_per_kg, "patient_weight_kg": patient_weight_kg})
        except (IndexError, ValueError):
            return "MedicalCalculatorAPI", MedicalCalculatorAPI("dosage", {"error": "Could not parse drug mg/kg or patient weight."})

    elif "my symptoms are" in query_lower or "i have these symptoms" in query_lower or "symptom checker" in query_lower:
        symptom_phrases = query_lower.split("symptoms are ")[-1].split(",")
        symptoms = [s.strip() for s in symptom_phrases if s.strip()]
        return "SymptomCheckerAPI", SymptomCheckerAPI(symptoms)

    return "None", {"error": "No relevant tool found for your query."}

def medical_diagnostic_assistant(user_query: str) -> str:
    tool_name, tool_output = llm_router(user_query)

    if tool_name == "MedicalKnowledgeBaseAPI":
        if "result" in tool_output and "Information not found" not in tool_output["result"]:
            return f"Based on medical knowledge, {tool_output['query'].capitalize()}: {tool_output['result']}"
        else:
            return f"I couldn't find information about '{tool_output['query']}'. Please try a different query."
    
    elif tool_name == "MedicalCalculatorAPI":
        if "error" in tool_output:
            return f"I encountered an error during calculation: {tool_output['error']}"
        elif tool_output["type"] == "BMI":
            return f"Your calculated Body Mass Index (BMI) is {tool_output['result']}."
        elif tool_output["type"] == "Dosage":
            return f"The calculated drug dosage is {tool_output['result']}."

    elif tool_name == "SymptomCheckerAPI":
        if tool_output["potential_diagnoses"]:
            conditions = ", ".join([c for c in tool_output['potential_diagnoses'][:3]]) # Show top 3
            return f"Based on your symptoms ({', '.join(tool_output['symptoms'])}), some potential conditions could be: {conditions}. Please consult a medical professional for an accurate diagnosis."
        else:
            return f"Based on the symptoms provided ({', '.join(tool_output['symptoms'])}), I couldn't find specific matches. It's best to consult a doctor."

    elif tool_name == "None":
        return f"I'm sorry, I couldn't understand your request or find a suitable tool to assist you. {tool_output['error']}"
    
    return "An unexpected error occurred."

if __name__ == '__main__':
    print(medical_diagnostic_assistant("What is diabetes?"))
    print(medical_diagnostic_assistant("Tell me about hypertension"))
    print(medical_diagnostic_assistant("Calculate BMI with weight 70kg and height 1.75m"))
    print(medical_diagnostic_assistant("Calculate dosage 10mg/kg with patient weight 60kg"))
    print(medical_diagnostic_assistant("My symptoms are headache, fatigue"))
    print(medical_diagnostic_assistant("I have these symptoms: sore throat, chest pain"))
    print(medical_diagnostic_assistant("What is cancer?"))
    print(medical_diagnostic_assistant("Just a random query"))
    print(medical_diagnostic_assistant("Calculate BMI with weight 70kg and height error"))
    print(medical_diagnostic_assistant("Calculate dosage 10mg/kg with patient weight error"))
    print(medical_diagnostic_assistant("My symptoms are sneezing, runny nose"))