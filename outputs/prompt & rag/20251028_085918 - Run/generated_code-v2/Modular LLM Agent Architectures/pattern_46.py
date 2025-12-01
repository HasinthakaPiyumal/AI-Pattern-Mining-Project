
# Tool Definitions
def get_symptom_diagnosis(symptoms: str) -> str:
    """Simulates a symptom checker API, providing a mock diagnosis."""
    symptoms = symptoms.lower()
    if "fever" in symptoms and "cough" in symptoms:
        return "Possible common cold or flu. Recommend rest and hydration. Consult a doctor if symptoms worsen."
    elif "headache" in symptoms and "nausea" in symptoms:
        return "Could be a migraine or tension headache. Consider pain relievers and rest. Seek medical advice if persistent."
    elif "chest pain" in symptoms:
        return "Chest pain can be serious. Seek immediate medical attention."
    else:
        return f"Based on symptoms: '{symptoms}', further information is needed or symptoms are non-specific. Please consult a healthcare professional."

def check_drug_interactions(drugs: str) -> str:
    """Simulates a drug interaction database, checking for interactions between comma-separated drugs."""
    drug_list = [d.strip().lower() for d in drugs.split(',')]
    if "ibuprofen" in drug_list and "warfarin" in drug_list:
        return "Warning: Ibuprofen and Warfarin can increase bleeding risk. Consult your doctor immediately."
    elif "acetaminophen" in drug_list and "alcohol" in drug_list:
        return "Caution: Excessive alcohol consumption with Acetaminophen can increase liver damage risk."
    elif len(drug_list) < 2:
        return "Please provide at least two drugs to check for interactions."
    else:
        return f"No common severe interactions found for {', '.join(drug_list)} in our mock database. Always consult a pharmacist or doctor."

def perform_medical_calculation(expression: str) -> str:
    """Simulates a medical calculator, evaluating a simple mathematical expression."""
    try:
        # Basic evaluation for common math, but in a real scenario, this would be more robust
        # and domain-specific (e.g., BMI, dosage calculations).
        # Using eval() is risky in production, but for a simplified example, it demonstrates the concept.
        result = eval(expression)
        return f"The result of '{expression}' is: {result}"
    except Exception as e:
        return f"Could not perform calculation: {e}. Please provide a valid mathematical expression."

def search_medical_knowledge_base(query: str) -> str:
    """Simulates a medical knowledge base search, returning mock information."""
    query = query.lower()
    if "diabetes" in query:
        return "Diabetes Mellitus is a chronic condition that affects how your body turns food into energy. It is characterized by high blood glucose levels. Types include Type 1, Type 2, and Gestational Diabetes."
    elif "hypertension" in query:
        return "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease."
    elif "aspirin" in query:
        return "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain (e.g., muscle aches, headaches, arthritis pain). It is also used to reduce the risk of heart attack and stroke in some patients."
    else:
        return f"No specific information found for '{query}' in the mock medical knowledge base. Try a different query."

class MedicalDiagnosticAssistant:
    """
    An LLM-like router for a Medical Diagnostic Assistant, dispatching queries to specialized tools.
    This class simulates the 'LLM router' described in the MRKL system pattern.
    """
    def route_and_respond(self, user_query: str) -> str:
        query_lower = user_query.lower()
        response_parts = []

        # Simple keyword-based routing logic
        if "symptoms" in query_lower or "diagnose" in query_lower or "what's wrong" in query_lower:
            # Attempt to extract symptoms (very basic parsing)
            symptoms_start_idx = query_lower.find("symptoms:")
            if symptoms_start_idx != -1:
                symptoms_text = user_query[symptoms_start_idx + len("symptoms:"):].strip()
                response_parts.append(get_symptom_diagnosis(symptoms_text))
            elif "my symptoms are" in query_lower:
                symptoms_text = user_query[query_lower.find("my symptoms are") + len("my symptoms are"):].strip()
                response_parts.append(get_symptom_diagnosis(symptoms_text))
            else:
                response_parts.append("Please list your symptoms clearly, e.g., 'What is wrong if my symptoms are fever and cough?' or 'Diagnose symptoms: headache, nausea'.")

        if "drug interaction" in query_lower or "medication interaction" in query_lower:
            # Attempt to extract drug names
            drugs_start_idx = query_lower.find("drugs:")
            if drugs_start_idx != -1:
                drugs_text = user_query[drugs_start_idx + len("drugs:"):].strip()
                response_parts.append(check_drug_interactions(drugs_text))
            elif "check interaction between" in query_lower:
                drugs_text = user_query[query_lower.find("check interaction between") + len("check interaction between"):].strip().replace(" and ", ",")
                response_parts.append(check_drug_interactions(drugs_text))
            else:
                response_parts.append("To check drug interactions, please list drugs, e.g., 'Check drug interaction between Ibuprofen and Warfarin' or 'Drugs: Aspirin, Ibuprofen'.")

        if "calculate" in query_lower or ("what is" in query_lower and any(op in query_lower for op in ['+', '-', '*', '/'])):
            # Attempt to extract calculation expression
            calc_start_idx = query_lower.find("calculate")
            if calc_start_idx != -1:
                expression = user_query[calc_start_idx + len("calculate"):].strip()
                response_parts.append(perform_medical_calculation(expression))
            elif "what is" in query_lower and any(op in query_lower for op in ['+', '-', '*', '/']):
                # Simple extraction, assumes expression follows "what is"
                parts = user_query.split("what is", 1)
                if len(parts) > 1:
                    expression = parts[1].strip()
                    response_parts.append(perform_medical_calculation(expression))
            else:
                response_parts.append("Please provide a calculation, e.g., 'Calculate 2+2' or 'What is 10/2'.")

        if "information about" in query_lower or ("what is" in query_lower and not any(op in query_lower for op in ['+', '-', '*', '/']) and not any(kw in query_lower for kw in ["symptoms", "diagnose", "drug interaction", "medication interaction"])):
            # This is a fallback for general knowledge queries
            query_text = user_query.replace("information about", "").replace("what is", "").strip()
            if query_text:
                response_parts.append(search_medical_knowledge_base(query_text))
            else:
                response_parts.append("Please specify what information you are looking for, e.g., 'Information about Diabetes' or 'What is Hypertension?'.")

        if not response_parts:
            return "I am a Medical Diagnostic Assistant. I can help with symptom diagnosis, drug interaction checks, medical calculations, and provide information from a knowledge base. Please ask a specific question."

        return "\n".join(response_parts)

# Main execution block
if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()
    print("Welcome to the Medical Diagnostic Assistant. How can I help you today?")
    print("Examples:")
    print("- 'Diagnose symptoms: fever, cough'")
    print("- 'Check drug interaction between Ibuprofen and Warfarin'")
    print("- 'Calculate 10/2'")
    print("- 'Information about Diabetes'")
    print("- 'What is Hypertension?'")
    print("- 'Exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Assistant: Goodbye!")
            break
        response = assistant.route_and_respond(user_input)
        print(f"Assistant: {response}")
