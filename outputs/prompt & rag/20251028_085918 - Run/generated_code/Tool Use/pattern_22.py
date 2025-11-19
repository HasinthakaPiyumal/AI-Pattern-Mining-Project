import os
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

# --- Simulated External Tools ---

def retrieve_patient_records(patient_id: str) -> Dict[str, Any]:
    """Accesses and retrieves a patient's medical history, lab results, medications, and demographic information."""
    mock_records = {
        "patient_123": {
            "name": "Alice Smith",
            "age": 45,
            "gender": "Female",
            "medical_history": ["Hypertension", "Type 2 Diabetes"],
            "medications": ["Lisinopril", "Metformin"],
            "allergies": ["Penicillin"],
            "lab_results": {"blood_pressure": "140/90", "glucose": "180 mg/dL"}
        },
        "patient_456": {
            "name": "Bob Johnson",
            "age": 60,
            "gender": "Male",
            "medical_history": ["Coronary Artery Disease", "High Cholesterol"],
            "medications": ["Atorvastatin", "Aspirin"],
            "allergies": [],
            "lab_results": {"cholesterol_ldl": "150 mg/dL"}
        }
    }
    return mock_records.get(patient_id, {"error": "Patient not found"})

def search_medical_literature(query: str) -> str:
    """Searches medical databases for relevant articles, case studies, and guidelines based on a diagnostic query or symptoms."""
    mock_literature = {
        "hypertension treatment": "Recent studies suggest combination therapy for resistant hypertension.",
        "diabetes management": "Glycemic control is crucial for preventing long-term diabetic complications.",
        "coronary artery disease": "Lifestyle modifications and statins are first-line for CAD prevention.",
        "headache causes": "Common causes include tension, migraines, and sinusitis. Less common but serious causes include stroke or tumor."
    }
    for key, value in mock_literature.items():
        if key in query.lower():
            return f"Relevant literature: {value}"
    return "No highly relevant medical literature found for this query."

def check_drug_interactions(drug_list: List[str]) -> str:
    """Checks for potential adverse drug-drug interactions for a given list of medications."""
    if "Lisinopril" in drug_list and "Ibuprofen" in drug_list:
        return "Warning: Potential interaction between Lisinopril and NSAIDs (like Ibuprofen) can reduce antihypertensive effect and increase kidney risk."
    if "Metformin" in drug_list and "Contrast Dye" in drug_list:
        return "Warning: Metformin should be temporarily discontinued before and 48 hours after contrast dye administration due to risk of lactic acidosis."
    return "No significant drug interactions found among the provided medications."

def suggest_lab_tests(symptoms: str, patient_history: str) -> List[str]:
    """Based on symptoms and patient history, suggests relevant diagnostic lab tests."""
    suggested = []
    if "headache" in symptoms.lower() and "vision changes" in symptoms.lower():
        suggested.append("MRI Brain")
    if "fatigue" in symptoms.lower() and "weight loss" in symptoms.lower() and "diabetes" in patient_history.lower():
        suggested.append("HbA1c")
        suggested.append("Thyroid Function Tests")
    if "chest pain" in symptoms.lower() and "coronary artery disease" in patient_history.lower():
        suggested.append("ECG")
        suggested.append("Cardiac Enzymes")
    return suggested if suggested else ["Basic Metabolic Panel", "Complete Blood Count"]

def analyze_patient_data(data: Dict[str, Any]) -> str:
    """Performs basic statistical analysis on patient data (e.g., prevalence, correlation with symptoms)."""
    if "glucose" in data.get("lab_results", {}) and float(data["lab_results"]["glucose"].split()[0]) > 125:
        return "Statistical Insight: Patient's glucose level indicates potential uncontrolled diabetes. Further investigation recommended."
    if "blood_pressure" in data.get("lab_results", {}) and "140/90" in data["lab_results"]["blood_pressure"]:
        return "Statistical Insight: Patient's blood pressure is elevated, suggesting uncontrolled hypertension."
    return "No specific statistical insights generated from the provided data."

# --- Langchain Tool Wrappers ---

tools = [
    Tool(
        name="RetrievePatientRecords",
        func=retrieve_patient_records,
        description="Useful for accessing and retrieving a patient's medical history, lab results, medications, and demographic information based on patient_id."
    ),
    Tool(
        name="SearchMedicalLiterature",
        func=search_medical_literature,
        description="Useful for searching medical databases for relevant articles, case studies, and guidelines based on a diagnostic query or symptoms."
    ),
    Tool(
        name="CheckDrugInteractions",
        func=check_drug_interactions,
        description="Useful for checking potential adverse drug-drug interactions for a given list of medications. Input should be a list of strings."
    ),
    Tool(
        name="SuggestLabTests",
        func=suggest_lab_tests,
        description="Useful for suggesting relevant diagnostic lab tests based on a patient's symptoms and medical history."
    ),
    Tool(
        name="AnalyzePatientData",
        func=analyze_patient_data,
        description="Useful for performing basic statistical analysis on patient data, like prevalence or correlation with symptoms. Input should be a dictionary."
    ),
]

# --- Langchain Agent Setup ---

llm = ChatOpenAI(temperature=0, model="gpt-4") # Ensure OPENAI_API_KEY is set as an environment variable

agent_prompt = PromptTemplate.from_template(
    """You are a Smart Medical Diagnostic Assistant. Your goal is to assist healthcare professionals in diagnosing complex medical conditions by intelligently using the provided tools.
    
    You have access to the following tools:
    
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin! Remember to be thorough and use the tools effectively to gather all necessary information before providing a final answer. If you cannot find a definitive diagnosis, provide a differential diagnosis or suggest further steps. Do not make up information that cannot be supported by tool output.
    
    Question: {input}
    {agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Main Application Logic ---

def run_diagnostic_assistant(doctor_query: str):
    print(f"\n--- Running Smart Medical Diagnostic Assistant ---")
    print(f"Doctor's Query: {doctor_query}")
    try:
        result = agent_executor.invoke({"input": doctor_query})
        print(f"\n--- Diagnostic Assistant's Conclusion ---")
        print(result["output"])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set your OpenAI API key as an environment variable
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

    # Example 1: Patient with known conditions
    run_diagnostic_assistant(
        "Patient ID patient_123 presents with general fatigue and slightly elevated blood pressure. History of hypertension and type 2 diabetes. What could be causing the fatigue and how should their treatment be adjusted?"
    )

    # Example 2: New symptoms, needing literature search and lab suggestions
    run_diagnostic_assistant(
        "A 30-year-old male, no significant past medical history, presents with severe headaches and occasional blurred vision. What are possible diagnoses and what lab tests should be ordered?"
    )

    # Example 3: Drug interaction check
    run_diagnostic_assistant(
        "Patient_123 is currently on Lisinopril and Metformin. They are also taking over-the-counter Ibuprofen for pain. Are there any drug interactions I should be aware of?"
    )

    # Example 4: More complex scenario combining multiple tools
    run_diagnostic_assistant(
        "Patient ID patient_456 has a history of coronary artery disease and high cholesterol. They report occasional mild chest discomfort. Retrieve their records, search for current guidelines on CAD management, and suggest any relevant lab tests or medication adjustments."
    )