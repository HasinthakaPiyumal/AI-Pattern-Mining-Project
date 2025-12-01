import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import ChatOpenAI

# --- Simulated External Tools ---

def get_medical_knowledge(query: str) -> str:
    medical_db = {
        "diabetes": "Diabetes is a chronic condition that affects the way the body processes blood sugar (glucose).",
        "hypertension": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
        "paracetamol": "Paracetamol (acetaminophen) is a common pain reliever and a fever reducer.",
        "common cold": "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way.",
        "headache causes": "Headaches can be caused by various factors including stress, dehydration, lack of sleep, eye strain, or more serious underlying conditions."
    }
    return medical_db.get(query.lower(), "No information found for that query in the medical database.")

def check_drug_interaction(drug1: str, drug2: str = None) -> str:
    interactions_db = {
        ("warfarin", "aspirin"): "Increased risk of bleeding.",
        ("ibuprofen", "lisinopril"): "May reduce the effectiveness of lisinopril and increase kidney risk.",
        ("amoxicillin", "methotrexate"): "Amoxicillin may decrease the excretion of methotrexate, potentially increasing methotrexate toxicity.",
        ("paracetamol", "alcohol"): "Increased risk of liver damage if consumed excessively."
    }
    drug1_lower = drug1.lower()
    drug2_lower = drug2.lower() if drug2 else None

    if drug2_lower:
        for (d1, d2), interaction in interactions_db.items():
            if (d1 == drug1_lower and d2 == drug2_lower) or (d1 == drug2_lower and d2 == drug1_lower):
                return f"Interaction between {drug1} and {drug2}: {interaction}"
    else:
        return f"No specific interaction found for {drug1} alone, please specify another drug for interaction check."
    return f"No known interaction found between {drug1} and {drug2} in the database."

def interpret_lab_results(test_name: str, value: float, unit: str) -> str:
    if test_name.lower() == "blood glucose":
        if unit.lower() == "mg/dL":
            if 70 <= value <= 99: return "Blood glucose is within normal fasting range."
            elif 100 <= value <= 125: return "Prediabetes range (impaired fasting glucose)."
            elif value >= 126: return "Diabetes range."
            else: return "Blood glucose is low (hypoglycemia)."
        else:
            return "Unsupported unit for blood glucose. Please use mg/dL."
    elif test_name.lower() == "hemoglobin":
        if unit.lower() == "g/dL":
            if (13.5 <= value <= 17.5) : return "Hemoglobin is within normal range for adult males."
            if (12.0 <= value <= 15.5) : return "Hemoglobin is within normal range for adult females."
            elif value < 12.0: return "Low hemoglobin, possibly indicating anemia."
            else: return "High hemoglobin, may indicate polycythemia or other conditions."
        else:
            return "Unsupported unit for hemoglobin. Please use g/dL."
    return f"Interpretation for {test_name} with value {value} {unit} is not available."

def check_symptoms(symptoms: str) -> str:
    symptom_db = {
        "fever, cough, fatigue": "Possible common cold or flu.",
        "chest pain, shortness of breath": "Could indicate a cardiac issue (e.g., angina, heart attack) or respiratory problem. Seek immediate medical attention.",
        "frequent urination, increased thirst, unexplained weight loss": "Possible diabetes.",
        "sore throat, difficulty swallowing": "Could be strep throat, tonsillitis, or other throat infection."
    }
    symptoms_lower = ', '.join(sorted([s.strip().lower() for s in symptoms.split(',')]))
    for k, v in symptom_db.items():
        if all(s in symptoms_lower for s in k.split(', ')):
            return f"Based on symptoms '{symptoms}': {v}"
    return "No specific condition found for the given symptoms. Please provide more details or consult a doctor."

# --- Langchain Tool Definitions ---

tools = [
    Tool(
        name="MedicalKnowledgeDatabase",
        func=get_medical_knowledge,
        description="Useful for answering general medical questions about diseases, conditions, or treatments. Input should be a specific query term."
    ),
    Tool(
        name="DrugInteractionChecker",
        func=check_drug_interaction,
        description="Useful for checking potential interactions between two drugs or getting information about a single drug. Input should be 'drug1, drug2' or 'drug1'."
    ),
    Tool(
        name="LabResultInterpreter",
        func=interpret_lab_results,
        description="Useful for interpreting specific lab test results. Input should be 'test_name, value, unit' (e.g., 'blood glucose, 105, mg/dL')."
    ),
    Tool(
        name="SymptomChecker",
        func=check_symptoms,
        description="Useful for suggesting potential conditions based on a list of symptoms. Input should be a comma-separated string of symptoms (e.g., 'fever, cough, headache')."
    )
]

# --- LLM and Agent Initialization ---

# Ensure you have your OpenAI API key set as an environment variable (OPENAI_API_KEY)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

llm = ChatOpenAI(temperature=0, model="gpt-4o") # You can choose other models like "gpt-3.5-turbo"

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS, # or AgentType.ZERO_SHOT_REACT_DESCRIPTION
    verbose=True,
    handle_parsing_errors=True
)

# --- Main Application Loop ---

def main():
    print("Welcome to the Medical Diagnostic Assistant! Type 'exit' to quit.")
    while True:
        user_query = input("\nHow can I help you today? ")
        if user_query.lower() == 'exit':
            break
        try:
            response = agent.run(user_query)
            print(f"Assistant: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try rephrasing your query.")

if __name__ == "__main__":
    main()
