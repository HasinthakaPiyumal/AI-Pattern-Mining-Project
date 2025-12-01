from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json

from langchain.tools import tool, Tool
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

class PatientData(BaseModel):
    symptoms: List[str] = Field(..., description="List of reported symptoms.")
    medical_history: List[str] = Field(..., description="Relevant past medical history.")
    medications: List[str] = Field(..., description="Current medications.")
    lab_results: Dict[str, Any] = Field(..., description="Dictionary of lab test results.")

class ReasoningStep(BaseModel):
    tool_name: str
    tool_input: str
    tool_output: str
    llm_rationale_selection: str
    llm_rationale_parameter_extraction: str
    llm_rationale_integration: str

class DiagnosticResult(BaseModel):
    potential_diagnoses: List[str]
    reasoning_trace: List[ReasoningStep]
    overall_explanation: str

@tool
def DrugInteractionChecker(medications_json: str) -> str:
    medications = json.loads(medications_json)
    if "medication_X" in medications and "medication_Y" in medications:
        return "Warning: Potential severe interaction between medication_X and medication_Y. May cause increased bleeding risk."
    return "No significant drug interactions found for the given medications."

@tool
def DiseaseSymptomChecker(symptoms_json: str) -> str:
    symptoms = json.loads(symptoms_json)
    if "fever" in symptoms and "cough" in symptoms and "fatigue" in symptoms:
        return "Possible diagnoses: Influenza, Common Cold, Bronchitis."
    elif "chest_pain" in symptoms and "shortness_of_breath" in symptoms:
        return "Possible diagnoses: Angina, Myocardial Infarction, Anxiety."
    return "Based on symptoms, no specific high-confidence diagnoses identified. Consider further investigation."

@tool
def LabResultInterpreter(lab_results_json: str) -> str:
    lab_results = json.loads(lab_results_json)
    if "CRP" in lab_results and lab_results["CRP"] > 10:
        return f"High CRP ({lab_results['CRP']} mg/L) indicates significant inflammation. Could be bacterial infection or autoimmune flare."
    elif "Hemoglobin" in lab_results and lab_results["Hemoglobin"] < 12:
        return f"Low Hemoglobin ({lab_results['Hemoglobin']} g/dL) indicates anemia. Further investigation needed for cause."
    return "Lab results appear within normal limits or require specific clinical context for interpretation."

medical_tools = [DrugInteractionChecker, DiseaseSymptomChecker, LabResultInterpreter]

class MockLLM(BaseChatModel):
    def invoke(self, messages: List[Any], **kwargs: Any) -> Any:
        user_input = messages[-1].content
        
        simulated_trace_parts = []
        
        if "Medications:" in user_input and '["medication_X", "medication_Y"]' in user_input:
            simulated_trace_parts.append("""
Thought: Patient is on multiple medications. It is critical to check for potential drug interactions as they can influence symptoms or treatment decisions.
Action: DrugInteractionChecker
Action Input: ["medication_X", "medication_Y"]
Observation: Warning: Potential severe interaction between medication_X and medication_Y. May cause increased bleeding risk.
Thought: I extracted 'medication_X' and 'medication_Y' from the provided patient medications list.
Thought: The identified severe drug interaction is a significant finding. This immediately flags a potential cause for adverse effects or complications and must be addressed in the diagnostic and treatment plan. It might explain some of the current symptoms.
""")
        elif "Medications:" in user_input:
             simulated_trace_parts.append("""
Thought: Patient is on medications. Checking for drug interactions is a standard procedure to ensure safety and rule out medication-induced symptoms.
Action: DrugInteractionChecker
Action Input: ["lisinopril", "amoxicillin"]
Observation: No significant drug interactions found for the given medications.
Thought: I parsed the list of medications provided in the patient data.
Thought: The absence of significant drug interactions allows us to rule out medication conflicts as a primary cause for current symptoms and focus on other diagnostic pathways.
""")

        if "Symptoms:" in user_input and '["fever", "cough", "fatigue"]' in user_input:
            simulated_trace_parts.append("""
Thought: The patient presents with symptoms common to several respiratory conditions. Using the DiseaseSymptomChecker will help narrow down the possibilities.
Action: DiseaseSymptomChecker
Action Input: ["fever", "cough", "fatigue"]
Observation: Possible diagnoses: Influenza, Common Cold, Bronchitis.
Thought: I extracted 'fever', 'cough', and 'fatigue' from the patient's reported symptoms list.
Thought: The symptom checker suggests influenza, common cold, and bronchitis. This provides a clear set of initial differential diagnoses to consider and guide further questioning or testing.
""")
        elif "Symptoms:" in user_input:
            simulated_trace_parts.append("""
Thought: The patient has reported various symptoms. Consulting the symptom checker is the logical next step to associate these symptoms with potential medical conditions.
Action: DiseaseSymptomChecker
Action Input: ["headache", "nausea"]
Observation: Based on symptoms, no specific high-confidence diagnoses identified. Consider further investigation.
Thought: I identified 'headache' and 'nausea' as the primary symptoms from the patient's data.
Thought: The symptom checker did not provide a high-confidence diagnosis, indicating that these symptoms might be non-specific or require more context, or that additional tools might be needed.
""")

        if "Lab Results:" in user_input and '"CRP": 15' in user_input:
            simulated_trace_parts.append("""
Thought: Elevated C-reactive protein (CRP) is an important indicator of inflammation. The LabResultInterpreter is necessary to understand its clinical significance.
Action: LabResultInterpreter
Action Input: {"CRP": 15, "Hemoglobin": 13.5}
Observation: High CRP (15 mg/L) indicates significant inflammation. Could be bacterial infection or autoimmune flare.
Thought: I identified the 'CRP' key and its value '15' from the provided lab results dictionary.
Thought: The interpretation of high CRP strongly suggests an inflammatory process, which is crucial for identifying the underlying cause of the patient's condition. This points towards an infection or an autoimmune disease.
""")
        elif "Lab Results:" in user_input:
            simulated_trace_parts.append("""
Thought: The patient has provided lab results, which need to be interpreted to provide objective insights into their physiological state.
Action: LabResultInterpreter
Action Input: {"Glucose": 90, "Cholesterol": 180}
Observation: Lab results appear within normal limits or require specific clinical context for interpretation.
Thought: I extracted 'Glucose' and 'Cholesterol' and their respective values from the lab results.
Thought: The lab results are within normal limits. This rules out common abnormalities in these areas but doesn't provide specific diagnostic clues, so other data sources remain key.
""")

        final_answer_data = {
            "Potential Diagnoses": ["Influenza (tentative)", "Further investigation needed"],
            "Overall Explanation": "Initial assessment indicates a potential viral infection (influenza) based on symptoms. Lab results showed no immediate red flags, but a comprehensive review of all data points to inflammatory process. Further specific tests are recommended."
        }
        final_answer_str = f"Final Answer: {json.dumps(final_answer_data)}"

        return "".join(simulated_trace_parts) + "\n" + final_answer_str

    @property
    def _llm_type(self) -> str:
        return "mock_llm"

class MedicalDiagnosticAgent:
    def __init__(self, llm: BaseChatModel, tools: List[Tool]):
        self.llm = llm
        self.tools = tools

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical diagnostic assistant. Your goal is to help diagnose complex medical conditions by utilizing specialized tools.
            You must be transparent in your reasoning process.
            For each step, explicitly state:
            Thought: Your rationale for selecting a tool.
            Action: The name of the tool you are using.
            Action Input: The input you are providing to the tool (must be a JSON string for list/dict types).
            Observation: The result from the tool.
            Thought: Your rationale for how you extracted parameters for the tool.
            Thought: Your rationale for how the tool's output integrates into your diagnostic process and next steps.

            When you have reached a conclusion or need to provide an interim summary, use the 'Final Answer:' format.
            The 'Final Answer' should contain a JSON object with two keys: 'Potential Diagnoses' (a JSON list of strings) and 'Overall Explanation' (a string).

            Available tools: {tool_names}
            Tool descriptions: {tool_descriptions}
            """),
            ("human", "Patient Data: {patient_data}\nBegin your diagnostic process, explaining each step clearly.")
        ])

    def diagnose(self, patient_data: PatientData) -> DiagnosticResult:
        full_patient_data_str = (
            f"Symptoms: {json.dumps(patient_data.symptoms)}\n"
            f"Medical History: {json.dumps(patient_data.medical_history)}\n"
            f"Medications: {json.dumps(patient_data.medications)}\n"
            f"Lab Results: {json.dumps(patient_data.lab_results)}"
        )
        
        raw_llm_output_string = self.llm.invoke([{"role": "user", "content": self.prompt.format_prompt(tool_names=[tool.name for tool in self.tools], tool_descriptions=[tool.description for tool in self.tools], patient_data=full_patient_data_str).to_messages()[-1].content}])

        reasoning_trace = []
        potential_diagnoses = []
        overall_explanation = ""

        current_step_data = {}
        rationale_buffer = []

        lines = raw_llm_output_string.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                rationale_buffer.append(line[len("Thought:"):].strip())
            elif line.startswith("Action:"):
                current_step_data["tool_name"] = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                current_step_data["tool_input"] = line[len("Action Input:"):].strip()
            elif line.startswith("Observation:"):
                current_step_data["tool_output"] = line[len("Observation:"):].strip()
                if len(rationale_buffer) >= 3:
                    current_step_data["llm_rationale_selection"] = rationale_buffer[0]
                    current_step_data["llm_rationale_parameter_extraction"] = rationale_buffer[1]
                    current_step_data["llm_rationale_integration"] = rationale_buffer[2]
                    
                if all(k in current_step_data for k in ["tool_name", "tool_input", "tool_output", "llm_rationale_selection", "llm_rationale_parameter_extraction", "llm_rationale_integration"]):
                    reasoning_trace.append(ReasoningStep(**current_step_data))
                    current_step_data = {}
                    rationale_buffer = []
            elif line.startswith("Final Answer:"):
                final_answer_json_str = line[len("Final Answer:"):].strip()
                try:
                    final_answer_data = json.loads(final_answer_json_str)
                    potential_diagnoses = final_answer_data.get("Potential Diagnoses", [])
                    overall_explanation = final_answer_data.get("Overall Explanation", "")
                except json.JSONDecodeError:
                    overall_explanation = "Error parsing final diagnosis: " + final_answer_json_str
                    potential_diagnoses = ["Error in Final Answer Parsing"]
                break

        return DiagnosticResult(
            potential_diagnoses=potential_diagnoses,
            reasoning_trace=reasoning_trace,
            overall_explanation=overall_explanation
        )

app = FastAPI()

@app.post("/diagnose", response_model=DiagnosticResult)
async def diagnose_patient(patient_data: PatientData):
    llm = MockLLM()
    agent = MedicalDiagnosticAgent(llm=llm, tools=medical_tools)
    result = agent.diagnose(patient_data)
    return result