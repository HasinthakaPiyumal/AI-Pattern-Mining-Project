import os
from typing import List, Dict, Any, Optional
import json

# Mocking external libraries and environment variables for a self-contained example
# In a real application, you would install and import these.

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
    # Assuming an OpenAI-compatible API for demonstration
    # from langchain_openai import ChatOpenAI
except ImportError:
    print("LangChain components not found. Please install: pip install langchain langchain-core langchain-community")
    # Define mock classes/functions if LangChain is not installed for basic execution
    class MockChatOpenAI:
        def __init__(self, model: str = "gpt-4", temperature: float = 0.7, api_key: str = "mock_key"):
            self.model = model
            self.temperature = temperature
            self.api_key = api_key

        def invoke(self, messages: List[Any]) -> AIMessage:
            # Simple mock response logic
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    if "lab results" in msg.content.lower():
                        return AIMessage(content="Mock response from LabResultInterpreter: Glucose 120 mg/dL (High)")
                    elif "drug dosage" in msg.content.lower():
                        return AIMessage(content="Mock response from DrugDosageCalculator: 500mg twice daily")
                    elif "pneumonia" in msg.content.lower():
                        return AIMessage(content="Mock response from MedicalKnowledgeBase: Pneumonia is an infection that inflames air sacs in one or both lungs...")
                    elif "imaging analysis" in msg.content.lower():
                        return AIMessage(content="Mock response from DiagnosticImagingAnalysis: Possible pneumonia detected in right lung.")
            return AIMessage(content="Mock LLM response: I'm not sure how to respond to that, please try a different query.")

    def tool(*args, **kwargs):
        def decorator(func):
            func.name = func.__name__
            func.description = kwargs.get('description', '')
            return func
        return decorator

    class AIMessage:
        def __init__(self, content: str):
            self.content = content

    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

    ChatOpenAI = MockChatOpenAI
    AgentExecutor = object # Placeholder
    create_tool_calling_agent = object # Placeholder
    ChatPromptTemplate = object # Placeholder


# --- 1. Mock External Tools (Microservices/APIs) ---

@tool("medical_knowledge_base", args_schema=None, description="Accesses a vast medical knowledge base to retrieve information about diseases, treatments, and clinical guidelines.")
def medical_knowledge_base_tool(query: str) -> str:
    """Fetches information from a mock medical knowledge base based on a query."""
    print(f"[Tool Call] Medical Knowledge Base: Querying for '{query}'")
    if "pneumonia" in query.lower():
        return "Pneumonia is an infection that inflames air sacs in one or both lungs. The air sacs may fill with fluid or pus. Symptoms include cough, fever, chills, and difficulty breathing. Treatment often involves antibiotics, antiviral drugs, or antifungal agents, depending on the cause."
    elif "diabetes management" in query.lower():
        return "Diabetes management involves monitoring blood sugar levels, taking prescribed medications (e.g., insulin, metformin), following a healthy diet, and regular exercise. Regular check-ups are crucial to prevent complications."
    else:
        return f"No specific information found for '{query}' in the medical knowledge base."

@tool("diagnostic_imaging_analysis", args_schema=None, description="Analyzes medical images (e.g., X-rays, MRIs) to detect anomalies. Input is a description of the image findings.")
def diagnostic_imaging_analysis_tool(image_description: str) -> str:
    """Simulates analysis of medical imaging based on a description."""
    print(f"[Tool Call] Diagnostic Imaging Analysis: Analyzing '{image_description}'")
    if "chest x-ray with diffuse infiltrates" in image_description.lower():
        return "Possible pneumonia detected in right lower lobe, correlating with diffuse infiltrates. Recommend sputum culture and antibiotic initiation."
    elif "brain mri with focal lesion" in image_description.lower():
        return "MRI indicates a focal lesion in the frontal lobe, suggestive of a tumor. Further investigation with biopsy recommended."
    else:
        return f"No significant findings or specific analysis for '{image_description}'."

@tool("lab_result_interpreter", args_schema=None, description="Interprets numerical lab results, flags abnormalities, and correlates them with patient symptoms.")
def lab_result_interpreter_tool(lab_data_json: str) -> str:
    """Interprets mock lab results provided as a JSON string."""
    print(f"[Tool Call] Lab Result Interpreter: Interpreting '{lab_data_json}'")
    try:
        lab_data = json.loads(lab_data_json)
        interpretations = []
        if "glucose" in lab_data and lab_data["glucose"] > 110:
            interpretations.append(f"Glucose: {lab_data['glucose']} mg/dL (High - consider pre-diabetes/diabetes)")
        elif "glucose" in lab_data and lab_data["glucose"] < 70:
            interpretations.append(f"Glucose: {lab_data['glucose']} mg/dL (Low - consider hypoglycemia)")
        if "creatinine" in lab_data and lab_data["creatinine"] > 1.2:
            interpretations.append(f"Creatinine: {lab_data['creatinine']} mg/dL (High - suggestive of renal impairment)")

        if not interpretations:
            return "Lab results appear within normal limits or no specific interpretations available for provided data."
        else:
            return " | ".join(interpretations)
    except json.JSONDecodeError:
        return "Invalid JSON format for lab data. Please provide a valid JSON string."

@tool("ehr_system_connector", args_schema=None, description="Securely accesses and retrieves patient medical history from Electronic Health Records (EHR).")
def ehr_system_connector_tool(patient_id: str) -> str:
    """Retrieves mock patient history from an EHR system."""
    print(f"[Tool Call] EHR System Connector: Retrieving for patient ID '{patient_id}'")
    if patient_id == "P123":
        return "Patient ID P123: John Doe, 65 years old. History of hypertension, type 2 diabetes. Current medications: Lisinopril, Metformin. Allergies: Penicillin. Last visit: 3 months ago for routine check-up."
    elif patient_id == "P456":
        return "Patient ID P456: Jane Smith, 42 years old. History of asthma. Current medications: Albuterol inhaler. No known allergies. Last visit: 1 month ago for asthma exacerbation."
    else:
        return f"No EHR data found for patient ID '{patient_id}'."

@tool("drug_dosage_calculator", args_schema=None, description="Calculates accurate drug dosages based on patient parameters like weight, age, and renal function.")
def drug_dosage_calculator_tool(drug_name: str, weight_kg: float, age_years: int, renal_function_gfr: Optional[float] = None) -> str:
    """Calculates mock drug dosage based on input parameters."""
    print(f"[Tool Call] Drug Dosage Calculator: Calculating dosage for {drug_name}, {weight_kg}kg, {age_years} years, GFR: {renal_function_gfr}")
    if drug_name.lower() == "amoxicillin":
        base_dose = 25 * weight_kg # mg/kg
        if renal_function_gfr and renal_function_gfr < 30: # Severe renal impairment
            return f"Amoxicillin dosage: {base_dose * 0.5:.0f} mg every 12 hours (adjusted for severe renal impairment)."
        return f"Amoxicillin dosage: {base_dose:.0f} mg every 8 hours."
    elif drug_name.lower() == "metformin":
        return f"Metformin dosage: 500 mg twice daily, increasing to 1000 mg twice daily as tolerated."
    else:
        return f"Dosage information not available for '{drug_name}'. Consult a pharmacist."

# --- 2. Tool Registry ---
class ToolRegistry:
    def __init__(self):
        self._tools = {
            medical_knowledge_base_tool.name: medical_knowledge_base_tool,
            diagnostic_imaging_analysis_tool.name: diagnostic_imaging_analysis_tool,
            lab_result_interpreter_tool.name: lab_result_interpreter_tool,
            ehr_system_connector_tool.name: ehr_system_connector_tool,
            drug_dosage_calculator_tool.name: drug_dosage_calculator_tool,
        }

    def get_tool(self, tool_name: str):
        return self._tools.get(tool_name)

    def get_all_tools(self) -> List[Any]:
        return list(self._tools.values())

# --- 3. MediAgent LLM Controller ---
class MediAgentLLMController:
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.tool_registry = ToolRegistry()
        self.tools = self.tool_registry.get_all_tools()
        self.agent_executor = self._initialize_agent()

        # Placeholder for personalized learning data
        self.personalized_data: Dict[str, Any] = {}

    def _initialize_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are MediAgent, an AI-powered clinical decision support system. Use the available medical tools to answer questions and provide clinical recommendations accurately. Prioritize patient safety and evidence-based medicine. Always state the source of information if you used a tool."),
                HumanMessage(content="{input}"),
                AIMessage(content="{agent_scratchpad}"),
            ]
        )

        # The `create_tool_calling_agent` requires LangChain to be installed.
        # If not installed, this will be a placeholder object, and the invoke method will use mock logic.
        if isinstance(create_tool_calling_agent, object):
            return None # Agent won't be functional without LangChain

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def process_medical_query(self, patient_id: str, query: str) -> str:
        # Simulate fetching personalized data
        user_prefs = self.personalized_data.get(patient_id, {})
        print(f"[MediAgent] Processing query for patient {patient_id}. User preferences: {user_prefs}")

        if self.agent_executor is None: # Fallback for when LangChain isn't installed
            print("[MediAgent] WARNING: LangChain AgentExecutor not initialized. Using mock LLM for response.")
            # Simulate basic tool calling logic
            if "lab results" in query.lower():
                return lab_result_interpreter_tool(json.dumps({"glucose": 150, "creatinine": 1.5}))
            elif "drug dosage" in query.lower():
                return drug_dosage_calculator_tool("amoxicillin", 70, 60, 25.0)
            elif "pneumonia" in query.lower():
                return medical_knowledge_base_tool("pneumonia")
            else:
                return self.llm.invoke([HumanMessage(content=query)]).content

        try:
            # Incorporate patient context into the query for the LLM
            ehr_info = self.tool_registry.get_tool("ehr_system_connector")(patient_id) # Direct call for initial context
            full_query = f"Patient ID: {patient_id}. EHR Info: {ehr_info}. \n\nDoctor's Query: {query}"

            response = self.agent_executor.invoke({"input": full_query, "chat_history": []})
            final_answer = response["output"]

            # --- Simplified Hallucination Detection ---
            if "not sure" in final_answer.lower() or "cannot determine" in final_answer.lower():
                print("[MediAgent] Potential hallucination or uncertainty detected. Further verification needed.")

            return final_answer
        except Exception as e:
            return f"An error occurred while processing the query: {e}"

    def update_personalized_learning(self, patient_id: str, feedback: Dict[str, Any]):
        """Simulates updating user preferences or learning data."""
        self.personalized_data[patient_id] = {**self.personalized_data.get(patient_id, {}), **feedback}
        print(f"[MediAgent] Updated personalized data for {patient_id}: {self.personalized_data[patient_id]}")


# --- Main Demonstration --- 
if __name__ == "__main__":
    print("\n--- Initializing MediAgent ---\n")
    medi_agent = MediAgentLLMController()

    print("\n--- Demonstrating Medical Query Processing ---\n")

    # Query 1: Basic medical information
    query1 = "What is pneumonia and how is it typically treated?"
    print(f"\nDoctor's Query (Patient P123): {query1}")
    response1 = medi_agent.process_medical_query("P123", query1)
    print(f"MediAgent Response: {response1}")

    # Query 2: Lab result interpretation for an existing patient
    query2 = "Patient P123 has a glucose of 150 mg/dL and creatinine of 1.4 mg/dL. Please interpret these lab results and suggest next steps."
    print(f"\nDoctor's Query (Patient P123): {query2}")
    response2 = medi_agent.process_medical_query("P123", query2)
    print(f"MediAgent Response: {response2}")

    # Query 3: Drug dosage calculation with renal impairment for a new patient
    query3 = "I need to prescribe Amoxicillin for a new patient, Patient P456. They weigh 70kg, are 42 years old, and their GFR is 25 mL/min. What is the recommended dosage?"
    print(f"\nDoctor's Query (Patient P456): {query3}")
    response3 = medi_agent.process_medical_query("P456", query3)
    print(f"MediAgent Response: {response3}")

    # Query 4: Imaging analysis and recommendation
    query4 = "Patient P123 has a chest x-ray showing diffuse infiltrates in the right lung. What is the likely diagnosis and immediate recommendation?"
    print(f"\nDoctor's Query (Patient P123): {query4}")
    response4 = medi_agent.process_medical_query("P123", query4)
    print(f"MediAgent Response: {response4}")

    print("\n--- Demonstrating Personalized Learning Update ---\n")
    medi_agent.update_personalized_learning("P123", {"preferred_drug_database": "MedScape", "fast_diagnostic_pathways": ["pneumonia"]})

    print("\n--- End of Demonstration ---\n")
    print("\nNOTE: This is a simplified demonstration. Real-world implementation would involve actual API calls, secure EHR integration, full ML models for imaging/lab analysis, robust hallucination detection, and a user-friendly frontend.\n")
