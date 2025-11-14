import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import PromptTemplate

# Set your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- Mock External Tools ---

def medical_knowledge_base_tool(query: str) -> str:
    """Simulates searching a medical knowledge base for information."""
    print(f"[Tool Call] Medical Knowledge Base: Querying for '{query}'")
    if "headache" in query.lower() and "fever" in query.lower():
        return "Potential conditions for headache and fever include meningitis, flu, or severe sinus infection. Consider neurological exam and blood tests."
    elif "chest pain" in query.lower():
        return "Chest pain can indicate cardiac issues (e.g., angina, MI), pulmonary problems (e.g., pneumonia, PE), or musculoskeletal pain. Requires immediate assessment."
    return f"Information for '{query}' from medical knowledge base: No specific direct match found, providing general medical context."

def image_analysis_tool(image_id_description: str) -> str:
    """Simulates processing medical images (e.g., X-rays, MRIs)."""
    print(f"[Tool Call] Image Analysis: Analyzing '{image_id_description}'")
    if "chest x-ray patient A" in image_id_description.lower():
        return "Chest X-ray findings for Patient A: Possible infiltrates in the lower left lung lobe, suggestive of pneumonia. No cardiomegaly observed."
    elif "brain mri patient B" in image_id_description.lower():
        return "Brain MRI for Patient B: Small lesion detected in the frontal lobe, requires further investigation. No signs of acute hemorrhage."
    return f"Image analysis for '{image_id_description}': Generic healthy findings. No significant abnormalities detected."

def lab_result_interpretation_tool(lab_data: str) -> str:
    """Simulates analyzing patient lab test results."""
    print(f"[Tool Call] Lab Result Interpretation: Interpreting '{lab_data}'")
    if "wbc 15000" in lab_data.lower() and "crp 50" in lab_data.lower():
        return "Lab results show elevated White Blood Cell count (15,000) and C-reactive protein (50 mg/L), indicating a significant inflammatory or infectious process."
    elif "hba1c 8.5" in lab_data.lower():
        return "HbA1c level of 8.5% indicates poorly controlled diabetes. Suggests reviewing medication and lifestyle interventions."
    return f"Lab result interpretation for '{lab_data}': Within normal limits for most parameters. Consult a specialist for specific abnormal values if any."

def clinical_guidelines_tool(condition_symptoms: str) -> str:
    """Simulates accessing standard treatment protocols and diagnostic criteria."""
    print(f"[Tool Call] Clinical Guidelines: Searching for '{condition_symptoms}'")
    if "pneumonia treatment" in condition_symptoms.lower():
        return "Clinical guidelines for community-acquired pneumonia: Recommend broad-spectrum antibiotics (e.g., Azithromycin + Amoxicillin) for 5-7 days, rest, hydration. Follow up in 48-72 hours."
    elif "hypertension diagnosis" in condition_symptoms.lower():
        return "Clinical guidelines for hypertension diagnosis: Persistent blood pressure readings >= 140/90 mmHg on two separate occasions. Lifestyle modifications are first-line, followed by medication if needed."
    return f"Clinical guidelines for '{condition_symptoms}': No specific guideline found. General diagnostic approach recommended."

def drug_database_tool(drug_patient_info: str) -> str:
    """Simulates checking drug information, interactions, dosages."""
    print(f"[Tool Call] Drug Database: Checking '{drug_patient_info}'")
    if "ibuprofen interactions" in drug_patient_info.lower():
        return "Ibuprofen can interact with anticoagulants (e.g., Warfarin), increasing bleeding risk. Also, use with caution in patients with renal impairment."
    elif "amoxicillin dosage child 50lb" in drug_patient_info.lower():
        return "Amoxicillin dosage for a 50lb child (approx 22.7 kg) is typically 20-40 mg/kg/day divided every 8 hours. Consult pediatric guidelines for precise dosing based on infection severity."
    return f"Drug information for '{drug_patient_info}': Drug not found or general information provided."

# --- LangChain Setup --- 

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview") # You might need to specify model_name if not default

# Create LangChain Tools
tools = [
    Tool(
        name="MedicalKnowledgeBase",
        func=medical_knowledge_base_tool,
        description="Useful for searching a comprehensive medical knowledge base for symptoms, diseases, treatments, and general medical information."
    ),
    Tool(
        name="ImageAnalysis",
        func=image_analysis_tool,
        description="Useful for analyzing medical images like X-rays or MRIs. Input should be a description or identifier of the image and relevant patient info."
    ),
    Tool(
        name="LabResultInterpretation",
        func=lab_result_interpretation_tool,
        description="Useful for interpreting patient lab test results. Input should be the raw lab data or a summary of critical values."
    ),
    Tool(
        name="ClinicalGuidelines",
        func=clinical_guidelines_tool,
        description="Useful for accessing standard clinical guidelines, treatment protocols, and diagnostic criteria for specific conditions or symptoms."
    ),
    Tool(
        name="DrugDatabase",
        func=drug_database_tool,
        description="Useful for checking drug information, potential interactions, dosages, and contraindications. Input should be drug name and optionally patient specifics."
    ),
]

# Define the agent's prompt template
prompt = PromptTemplate.from_template("""You are an Adaptive Medical Diagnostic Agent. Your goal is to diagnose complex and ambiguous patient cases by dynamically using medical tools, refining your understanding, and self-correcting. 

Respond to the user's query by following these steps:
1. Carefully analyze the patient's symptoms and available information.
2. Formulate hypotheses about possible conditions.
3. Use your tools to gather more information, clarify ambiguities, and rule out conditions.
4. Reflect on the tool outputs. If a tool output contradicts your current understanding or reveals new information, adapt your reasoning.
5. If a diagnosis becomes clear, state it confidently and provide supporting evidence. If the case remains ambiguous, explain the remaining uncertainties and suggest further steps or escalation to a human expert.
6. Always aim for a comprehensive and accurate diagnosis. If you make an error or reach a dead end, acknowledge it and try a different approach. 

Use the following tools:

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

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

# Create the LangChain Agent
agent = create_react_agent(llm, tools, prompt)

# Create the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def diagnose_patient_case(patient_case: str) -> Dict[str, Any]:
    """Runs the medical diagnostic agent on a given patient case.""" 
    print(f"\n--- Diagnosing Patient Case: {patient_case} ---")
    try:
        result = agent_executor.invoke({"input": patient_case})
        return result
    except Exception as e:
        print(f"An error occurred during diagnosis: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Example patient cases
    patient_case_1 = "Patient presents with severe headache, high fever (102°F), stiff neck, and light sensitivity for 24 hours. No significant medical history. No allergies."
    patient_case_2 = "Patient reports intermittent chest pain, especially during exertion. History of smoking for 20 years. Lab results show WBC 8000, CRP 10. Chest X-ray ID: chest x-ray patient A."
    patient_case_3 = "A 6-year-old child presents with a persistent cough, shortness of breath, and mild fever. A previous test showed elevated WBC 15000 and CRP 50. The child weighs 50lbs."

    # Run diagnostics for different cases
    diagnosis_1 = diagnose_patient_case(patient_case_1)
    print(f"\nFinal Diagnosis for Case 1:\n{diagnosis_1}\n")

    diagnosis_2 = diagnose_patient_case(patient_case_2)
    print(f"\nFinal Diagnosis for Case 2:\n{diagnosis_2}\n")
    
    diagnosis_3 = diagnose_patient_case(patient_case_3)
    print(f"\nFinal Diagnosis for Case 3:\n{diagnosis_3}\n")

    # Example of self-correction / iterative refinement (less direct in this simple example, but prompt encourages it)
    patient_case_refinement = "Patient B (from previous case) has a new Brain MRI showing a small frontal lobe lesion. The previous chest x-ray was clear. What are the implications and next steps?"
    diagnosis_refinement = diagnose_patient_case(patient_case_refinement)
    print(f"\nFinal Diagnosis for Refinement Case:\n{diagnosis_refinement}\n")
