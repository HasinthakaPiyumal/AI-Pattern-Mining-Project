import os
from dotenv import load_dotenv
from typing import Type
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

# --- Memory Management Module ---
class ClinicalMemory:
    def __init__(self):
        self.patient_records = {}
        self.guidelines = {}
        self.protocols = {}

    def add_patient_history(self, patient_id: str, history: str):
        if patient_id not in self.patient_records:
            self.patient_records[patient_id] = []
        self.patient_records[patient_id].append(history)
        return f"History added for patient {patient_id}."

    def get_patient_history(self, patient_id: str) -> list:
        return self.patient_records.get(patient_id, [])

    def add_guideline(self, guideline_id: str, content: str):
        self.guidelines[guideline_id] = content
        return f"Guideline {guideline_id} added."

    def get_guideline(self, guideline_id: str) -> str:
        return self.guidelines.get(guideline_id, "Guideline not found.")

    def add_protocol(self, protocol_id: str, content: str):
        self.protocols[protocol_id] = content
        return f"Protocol {protocol_id} added."

    def get_protocol(self, protocol_id: str) -> str:
        return self.protocols.get(protocol_id, "Protocol not found.")

# --- Tool Use Module ---
class MedicalSearchInput(BaseModel):
    query: str = Field(description="The medical query to search for (e.g., disease, drug, symptom).")

class MedicalSearchTool(BaseTool):
    name = "medical_search"
    description = "Searches external medical databases for information related to diseases, drugs, symptoms, or research papers."
    args_schema: Type[BaseModel] = MedicalSearchInput

    def _run(self, query: str) -> str:
        if "diabetes" in query.lower():
            return "Found research on Type 2 Diabetes management, including metformin efficacy and lifestyle interventions. Latest guidelines suggest early intensive therapy."
        elif "hypertension" in query.lower():
            return "Information on blood pressure management, including ACE inhibitors, ARBs, and lifestyle modifications. JNC 8 guidelines are widely referenced."
        elif "rare disease X" in query.lower():
            return "Limited information found for Rare Disease X. Suggest genetic screening and specialized diagnostic labs."
        else:
            return f"No specific medical information found for \'{query}\'."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("medical_search does not support async")

class PatientDataAnalysisInput(BaseModel):
    patient_id: str = Field(description="The ID of the patient whose data needs to be analyzed.")
    data_type: str = Field(description="The type of data to analyze (e.g., \'lab_results\', \'vital_signs\', \'genetic_markers\').")

class PatientDataAnalysisTool(BaseTool):
    name = "patient_data_analysis"
    description = "Performs computational analysis on patient-specific data to identify patterns, risks, or predictions."
    args_schema: Type[BaseModel] = PatientDataAnalysisInput

    def _run(self, patient_id: str, data_type: str) -> str:
        if data_type == "lab_results" and patient_id == "P001":
            return "Analysis of P001\'s lab results indicates elevated glucose levels and impaired renal function. High risk for diabetes complications."
        elif data_type == "vital_signs" and patient_id == "P002":
            return "P002\'s vital signs show fluctuating blood pressure readings over the last week. Suggests need for close monitoring and medication review."
        else:
            return f"No specific analysis performed for patient {patient_id} and data type \'{data_type}\'."

    async def _arun(self, patient_id: str, data_type: str) -> str:
        raise NotImplementedError("patient_data_analysis does not support async")

class HospitalAPIInteractionInput(BaseModel):
    action: str = Field(description="The action to perform (e.g., \'schedule_test\', \'order_prescription\', \'get_lab_results\').")
    patient_id: str = Field(description="The ID of the patient for whom the action is performed.")
    details: str = Field(description="Additional details for the action (e.g., test name, drug name, date).", default="")

class HospitalAPIInteractionTool(BaseTool):
    name = "hospital_api_interaction"
    description = "Interacts with hospital systems APIs for tasks like scheduling tests, ordering prescriptions, or retrieving lab results."
    args_schema: Type[BaseModel] = HospitalAPIInteractionInput

    def _run(self, action: str, patient_id: str, details: str = "") -> str:
        if action == "schedule_test":
            return f"Test \'{details}\' scheduled for patient {patient_id}. Confirmation sent to EHR."
        elif action == "order_prescription":
            return f"Prescription for \'{details}\' ordered for patient {patient_id}. Ready for pharmacy pickup."
        elif action == "get_lab_results":
            if patient_id == "P001" and "glucose" in details.lower():
                return "Latest glucose results for P001: 180 mg/dL (fasting)."
            return f"Attempted to retrieve lab results for patient {patient_id} for \'{details}\'. Results might be pending or not found."
        else:
            return f"Unsupported API action: \'{action}\'."

    async def _arun(self, action: str, patient_id: str, details: str = "") -> str:
        raise NotImplementedError("hospital_api_interaction does not support async")

# --- Planning Module ---
class ClinicalPlanner:
    def __init__(self, llm):
        self.llm = llm
        self.planning_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert medical planner. Your task is to break down complex medical queries into a sequence of actionable steps."),
            ("human", "Generate a detailed plan to address the following medical query: {query}. Consider using tools for medical search, patient data analysis, and hospital API interactions. Output the plan as a numbered list of steps.")
        ])
        self.planning_chain = {"query": RunnablePassthrough()} | self.planning_prompt | self.llm | StrOutputParser()

    def generate_plan(self, query: str) -> str:
        return self.planning_chain.invoke({"query": query})

# --- Cognitive Load Management Module ---
def prioritize_information(information_list: list[str], query: str) -> list[str]:
    query_keywords = set(word.lower() for word in query.split() if len(word) > 2)

    def score_item(item: str) -> int:
        item_lower = item.lower()
        score = 0
        for keyword in query_keywords:
            if keyword in item_lower:
                score += 1
        return score

    prioritized = sorted(information_list, key=score_item, reverse=True)
    return prioritized

def summarize_information(information: str, max_length: int = 500) -> str:
    if len(information) <= max_length:
        return information
    else:
        return information[:max_length-3] + "..."

# --- Main Orchestration (ClinicalAssistantAgent) ---
class ClinicalAssistantAgent:
    def __init__(self):
        load_dotenv() # Ensure .env is loaded here

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        self.clinical_memory = ClinicalMemory()
        self.planner = ClinicalPlanner(self.llm)

        self.tools = [
            MedicalSearchTool(),
            PatientDataAnalysisTool(),
            HospitalAPIInteractionTool()
        ]

        self.agent_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="output"
        )

        self.agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Clinical Assistant Agent. Your goal is to assist healthcare professionals by providing accurate information, planning complex tasks, and interacting with medical systems. Use the available tools to achieve this. Always be thorough and consider patient safety."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_openai_tools_agent(self.llm, self.tools, self.agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True, memory=self.agent_memory
        )

    def run_query(self, query: str) -> str:
        print(f"\n--- Processing Query: {query} ---")

        plan = self.planner.generate_plan(query)
        print(f"\n[Planning Module Output]\n{plan}")

        agent_response = self.agent_executor.invoke({"input": query})
        output = agent_response["output"]
        print(f"\n[Agent Execution Output]\n{output}")

        managed_output = prioritize_information([output], query)[0]
        managed_output = summarize_information(managed_output)
        print(f"\n[Cognitive Load Management Output (Prioritized & Summarized)]\n{managed_output}")

        return managed_output

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
        exit(1)

    assistant = ClinicalAssistantAgent()

    # Example 1: Diagnostic Assistance (triggers medical search and planning)
    response1 = assistant.run_query("What are the latest treatment guidelines for Type 2 Diabetes and what are common complications?")
    print(f"\nFinal Assistant Response 1: {response1}")

    # Example 2: Patient-specific inquiry (triggers data analysis and memory interaction)
    assistant.clinical_memory.add_patient_history("P001", "Patient P001 has a history of elevated blood sugar readings and family history of diabetes.")
    response2 = assistant.run_query("Analyze patient P001\'s lab results and tell me the findings. Also, what is P001\'s general medical history?")
    print(f"\nFinal Assistant Response 2: {response2}")

    # Example 3: Administrative Task (triggers hospital API interaction)
    response3 = assistant.run_query("Schedule a follow-up blood test for patient P002 next week and order a prescription for Metformin 500mg for P001.")
    print(f"\nFinal Assistant Response 3: {response3}")

    # Example 4: Complex Multi-step query (demonstrates planning and multiple tool use)
    response4 = assistant.run_query("A patient presents with persistent cough, fever, and fatigue. They also have a history of heart disease. Suggest a diagnostic pathway and potential initial management strategies. What are common treatments for heart disease in general?")
    print(f"\nFinal Assistant Response 4: {response4}")
