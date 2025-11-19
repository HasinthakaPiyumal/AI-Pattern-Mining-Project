import streamlit as st
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import FakeListLLM
from loguru import logger

# Configure logger
logger.remove()
logger.add(st.empty().info, format="{message}")

# --- Simulated Medical Tools ---
def medical_kb_tool(query: str) -> str:
    if "diabetes" in query.lower():
        return "Diabetes Mellitus: Chronic condition affecting how your body turns food into energy. Symptoms include increased thirst, frequent urination, hunger, fatigue. Treatment involves insulin, medication, diet, and exercise."
    elif "hypertension" in query.lower():
        return "Hypertension (High Blood Pressure): Common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems. Often asymptomatic. Treatment includes lifestyle changes and medication."
    elif "paracetamol" in query.lower() and "interaction" in query.lower():
        return "Paracetamol (Acetaminophen) can interact with alcohol (increased liver risk) and certain anticoagulants. Consult a doctor for specific interactions."
    return f"No specific information found for '{query}' in the Medical Knowledge Base."

def image_analysis_tool(image_description: str) -> str:
    if "chest X-ray" in image_description.lower() and "pneumonia" in image_description.lower():
        return "Image Analysis: Chest X-ray shows infiltrates in the lower left lung consistent with bacterial pneumonia."
    elif "MRI brain" in image_description.lower() and "tumor" in image_description.lower():
        return "Image Analysis: MRI of the brain indicates a suspicious lesion in the frontal lobe, suggestive of a tumor. Further investigation recommended."
    return f"Mock image analysis for '{image_description}' returned no specific findings."

def ehr_tool(patient_id: str) -> str:
    if patient_id == "P001":
        return "Patient ID: P001, Name: Jane Doe, Age: 45, History: Type 2 Diabetes (diagnosed 5 years ago), Current Meds: Metformin 500mg BID. Lab Results: A1C 7.2% (last month)."
    elif patient_id == "P002":
        return "Patient ID: P002, Name: John Smith, Age: 60, History: Hypertension (diagnosed 10 years ago), Current Meds: Lisinopril 10mg QD. Lab Results: BP 145/90 (recent)."
    return f"No Electronic Health Record found for Patient ID: {patient_id}."

def drug_db_tool(drug_name: str) -> str:
    if "metformin" in drug_name.lower():
        return "Drug: Metformin (Glucophage). Class: Biguanide. Use: Type 2 Diabetes. Dosage: Typically 500-2550mg daily. Side Effects: Nausea, diarrhea, lactic acidosis (rare). Contraindications: Severe renal impairment."
    elif "lisinopril" in drug_name.lower():
        return "Drug: Lisinopril (Prinivil, Zestril). Class: ACE Inhibitor. Use: Hypertension, Heart Failure. Dosage: Typically 10-40mg QD. Side Effects: Cough, dizziness, hyperkalemia. Contraindications: Pregnancy, angioedema history."
    return f"No detailed drug information found for '{drug_name}'."

def symptom_checker_tool(symptoms: str) -> str:
    symptoms_lower = symptoms.lower()
    if "fever" in symptoms_lower and "cough" in symptoms_lower and "shortness of breath" in symptoms_lower:
        return "Differential Diagnosis: Possible pneumonia, bronchitis, or influenza. Consider chest imaging and viral testing."
    elif "abdominal pain" in symptoms_lower and "nausea" in symptoms_lower and "loss of appetite" in symptoms_lower:
        return "Differential Diagnosis: Appendicitis, gastroenteritis, or irritable bowel syndrome. Further clinical assessment recommended."
    return f"Based on symptoms '{symptoms}', a general differential diagnosis could include various common ailments. Consult a physician for accurate diagnosis."

# Create LangChain Tools
tools = [
    Tool(
        name="MedicalKnowledgeBase",
        func=medical_kb_tool,
        description="Useful for answering questions about diseases, symptoms, causes, and basic drug interactions. Input should be a specific medical query."
    ),
    Tool(
        name="MedicalImageAnalysis",
        func=image_analysis_tool,
        description="Useful for analyzing descriptions of medical images (e.g., X-rays, MRIs) and providing findings. Input should be a description of the image and what to look for."
    ),
    Tool(
        name="ElectronicHealthRecord",
        func=ehr_tool,
        description="Useful for retrieving patient history, current medications, lab results, and demographic data using a patient ID. Input should be a patient ID (e.g., P001)."
    ),
    Tool(
        name="DrugDatabase",
        func=drug_db_tool,
        description="Useful for getting detailed information about a specific drug, including dosage, side effects, contraindications, and advanced interactions. Input should be the drug name."
    ),
    Tool(
        name="SymptomChecker",
        func=symptom_checker_tool,
        description="Useful for suggesting potential diagnoses based on a list of patient symptoms. Input should be a comma-separated list of symptoms."
    )
]

# --- LLM Agent Setup ---
# Using FakeListLLM for a runnable prototype without actual API keys
llm = FakeListLLM(responses=[
    "Thought: The user is asking about a patient's medical history and a potential diagnosis based on symptoms. I should first check the EHR for patient P001, then use the symptom checker. \nAction: ElectronicHealthRecord(patient_id='P001')",
    "Thought: I have the patient's EHR. Now I need to check the symptoms. \nAction: SymptomChecker(symptoms='increased thirst, frequent urination, fatigue, hunger')",
    "Thought: I have the patient's EHR and a differential diagnosis suggesting diabetes. The EHR confirms Type 2 Diabetes and high A1C. I should combine this information to provide a diagnosis and treatment recommendation. \nFinal Answer: Patient P001 has Type 2 Diabetes. The EHR shows an A1C of 7.2%, indicating uncontrolled blood sugar. Current medication is Metformin. Based on symptoms (increased thirst, frequent urination, fatigue, hunger), the diagnosis is consistent with poorly managed Type 2 Diabetes. Recommendations: Review Metformin dosage, consider additional medication if A1C remains high, emphasize diet and exercise, and schedule a follow-up with an endocrinologist.",
    "Thought: The user is asking for information about a specific drug. I should use the DrugDatabase tool. \nAction: DrugDatabase(drug_name='Metformin')",
    "Thought: I have retrieved detailed information about Metformin. I can now provide the answer. \nFinal Answer: Metformin (Glucophage) is a biguanide used for Type 2 Diabetes. Typical dosage is 500-2550mg daily. Common side effects include nausea and diarrhea. It is contraindicated in severe renal impairment.",
    "Thought: The user is providing symptoms and asking for a diagnosis. I should use the SymptomChecker tool. \nAction: SymptomChecker(symptoms='fever, cough, shortness of breath')",
    "Thought: The symptom checker suggests pneumonia, bronchitis, or influenza. I should provide this as the differential diagnosis. \nFinal Answer: Based on the symptoms of fever, cough, and shortness of breath, a differential diagnosis includes possible pneumonia, bronchitis, or influenza. Further clinical assessment, including chest imaging and viral testing, is recommended."
])

# Define the agent prompt
prompt = PromptTemplate.from_template(
    """You are a helpful medical assistant AI. You have access to the following tools:

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
Thought:{agent_scratchpad}
"""
)

# Create the agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Medical Diagnosis Assistant", layout="wide")
st.title("🩺 AI-Powered Medical Diagnosis and Treatment Recommendation System")

st.markdown("This system leverages an AI agent to assist healthcare professionals with diagnosis and treatment planning by orchestrating various medical tools.")

st.header("Patient Information & Query")

patient_id = st.text_input("Patient ID (e.g., P001, P002)", "P001")
symptoms = st.text_area("Symptoms (comma-separated)", "increased thirst, frequent urination, fatigue, hunger")
medical_query = st.text_area("Medical Query (e.g., 'Diagnose P001 based on symptoms', 'What are the side effects of Metformin?', 'Analyze chest X-ray for pneumonia')", 
                             "Diagnose P001 based on current symptoms and historical data.")

if st.button("Get Diagnosis/Recommendation"):
    if not medical_query:
        st.warning("Please enter a medical query.")
    else:
        with st.spinner("AI Agent is processing your request..."):
            full_query = f"Patient ID: {patient_id}. Symptoms: {symptoms}. Query: {medical_query}"
            logger.info(f"Processing query: {full_query}")
            try:
                result = agent_executor.invoke({"input": full_query})
                st.success("Processing Complete!")
                st.subheader("AI Agent's Insights and Recommendations:")
                st.write(result["output"])
                logger.info("Displayed AI agent's output.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Error during agent execution: {e}")

st.markdown("""
--- 
**Disclaimer:** This is a prototype AI assistant and should not be used for actual medical diagnosis or treatment. Always consult with a qualified healthcare professional.
""")