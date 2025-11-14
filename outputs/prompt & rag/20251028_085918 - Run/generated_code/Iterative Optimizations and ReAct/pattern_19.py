import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# --- 1. Data Models (Pydantic) ---
class Patient(BaseModel):
    id: str
    name: str
    age: int
    gender: str
    symptoms: List[str]
    medical_history: List[str] = Field(default_factory=list)
    lab_results: Dict[str, Any] = Field(default_factory=dict)
    imaging_reports: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)

class Diagnosis(BaseModel):
    condition: str
    confidence: float
    justification: str
    differential_diagnoses: List[str] = Field(default_factory=list)

class TreatmentPlan(BaseModel):
    diagnosis: Diagnosis
    recommended_actions: List[str]
    medications: List[str] = Field(default_factory=list)
    follow_up_instructions: str
    expected_outcome: str

class AgentState(BaseModel):
    patient: Patient
    current_plan: Optional[str] = None
    last_observation: Optional[str] = None
    reflection: Optional[str] = None
    diagnosis_history: List[Diagnosis] = Field(default_factory=list)
    treatment_history: List[TreatmentPlan] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 5


# --- 2. Mock Core Reasoning LLM ---
class MockLLM:
    """Simulates an LLM for planning, reasoning, and reflection."""
    def __init__(self, model_name: str = "Mock-GPT-4-Medical"):
        self.model_name = model_name

    def invoke(self, prompt: str) -> str:
        # Simple rule-based responses for demonstration
        if "plan" in prompt.lower() and "patient" in prompt.lower():
            return f"Based on patient data, plan: Retrieve EHR, search literature for symptoms, propose initial diagnosis and treatment. {{tool_calls: [\'ehr_retriever\', \'medical_literature_search\']}}"
        elif "reflect" in prompt.lower() and "observation" in prompt.lower():
            return f"Observation: {prompt[prompt.find('Observation:'):]}. Reflection: The plan executed seems partially effective, but more data is needed or a different treatment approach might be better. Considering re-evaluating the diagnosis. {{new_tool_calls: [\'lab_result_interpreter\', \'drug_interaction_checker\']}}"
        elif "diagnosis" in prompt.lower() and "symptoms" in prompt.lower():
            return f"Possible diagnosis: Flu. Confidence: 0.85. Justification: Common symptoms match. Differential: Common cold. {{diagnosis_details: {json.dumps(Diagnosis(condition='Flu', confidence=0.85, justification='Common symptoms match', differential_diagnoses=['Common cold']).dict())}}}"
        elif "treatment" in prompt.lower() and "diagnosis" in prompt.lower():
            return f"Treatment plan for Flu: Rest, fluids, Tamiflu. Follow-up in 3 days. Expected outcome: Recovery. {{treatment_details: {json.dumps(TreatmentPlan(diagnosis=Diagnosis(condition='Flu', confidence=0.85, justification='Common symptoms match', differential_diagnoses=['Common cold']), recommended_actions=['Rest', 'Fluids'], medications=['Tamiflu'], follow_up_instructions='Follow-up in 3 days', expected_outcome='Recovery').dict())}}}"
        else:
            return f"LLM response to: {prompt[:100]}..."


# --- 3. Tool Integration & Management Module (Mock Tools) ---
class ToolManager:
    """Manages and provides access to various medical tools."""
    def __init__(self):
        self.tools = {
            "ehr_retriever": self._ehr_retriever,
            "medical_literature_search": self._medical_literature_search,
            "drug_interaction_checker": self._drug_interaction_checker,
            "lab_result_interpreter": self._lab_result_interpreter,
            "diagnostic_imaging_analysis": self._diagnostic_imaging_analysis
        }

    def _ehr_retriever(self, patient_id: str) -> Dict[str, Any]:
        print(f"Executing EHR Retriever for patient {patient_id}...")
        # Mock data
        if patient_id == "P001":
            return {
                "patient_id": "P001",
                "allergies": ["Penicillin"],
                "past_diagnoses": ["Hypertension"],
                "current_medications": ["Lisinopril"]
            }
        return {"patient_id": patient_id, "error": "Patient not found in EHR"}

    def _medical_literature_search(self, query: str) -> List[str]:
        print(f"Executing Medical Literature Search for query: '{query}'...")
        # Mock data
        if "flu symptoms" in query.lower():
            return [
                "Influenza (flu) is a contagious respiratory illness caused by flu viruses.",
                "Symptoms include fever, cough, sore throat, muscle aches, and fatigue."
            ]
        return [f"No specific literature found for '{query}'."]

    def _drug_interaction_checker(self, drugs: List[str]) -> Dict[str, Any]:
        print(f"Executing Drug Interaction Checker for drugs: {drugs}...")
        # Mock data
        if "Tamiflu" in drugs and "Lisinopril" in drugs:
            return {"interaction_found": False, "details": "No significant interaction between Tamiflu and Lisinopril."}
        return {"interaction_found": False, "details": "No interactions detected."}

    def _lab_result_interpreter(self, lab_results: Dict[str, Any]) -> str:
        print(f"Executing Lab Result Interpreter for results: {lab_results}...")
        # Mock data
        if "WBC" in lab_results and lab_results["WBC"] > 10:
            return "Elevated White Blood Cell count, indicative of infection."
        return "Lab results appear within normal limits."

    def _diagnostic_imaging_analysis(self, image_url: str) -> str:
        print(f"Executing Diagnostic Imaging Analysis for: {image_url}...")
        # Mock data
        if "chest_xray_p001.jpg" in image_url:
            return "Chest X-ray shows mild bronchial inflammation."
        return "Imaging analysis inconclusive."

    def get_tool(self, tool_name: str):
        return self.tools.get(tool_name)


# --- 4. Knowledge & Memory Module (Mock RAG) ---
class KnowledgeBase:
    """Simulates a vector database and RAG pipeline for medical context."""
    def __init__(self):
        self.medical_guidelines = {
            "flu_treatment": "Adult flu treatment typically involves antiviral medications like Tamiflu, rest, and hydration.",
            "hypertension_management": "Hypertension is managed with lifestyle changes and medications like ACE inhibitors."
        }
        self.past_cases = {
            "case_flu_mild": {"symptoms": ["fever", "cough"], "diagnosis": "Flu", "treatment": "Rest and fluids"}
        }

    def retrieve_context(self, query: str) -> List[str]:
        print(f"Retrieving context for query: '{query}' from Knowledge Base...")
        context = []
        for key, value in self.medical_guidelines.items():
            if query.lower() in key.lower() or query.lower() in value.lower():
                context.append(value)
        for key, value in self.past_cases.items():
            if query.lower() in key.lower() or query.lower() in json.dumps(value).lower():
                context.append(json.dumps(value))
        return context


# --- 5. Adaptive Medical Agent ---
class AdaptiveMedicalAgent:
    """An AI agent for medical diagnosis and treatment recommendations, implementing adaptive reasoning.

    The agent follows a Plan -> Execute -> Observe -> Reflect -> Self-Correct loop.
    """

    def __init__(
        self,
        llm: MockLLM,
        tool_manager: ToolManager,
        knowledge_base: KnowledgeBase,
        max_iterations: int = 5,
    ):
        self.llm = llm
        self.tool_manager = tool_manager
        self.knowledge_base = knowledge_base
        self.state = AgentState(patient=Patient(id="", name="", age=0, gender="", symptoms=[]), max_iterations=max_iterations)

    def _update_patient_data(self, new_data: Dict[str, Any]):
        """Updates patient data based on tool outputs or observations."""
        for key, value in new_data.items():
            if hasattr(self.state.patient, key):
                current_value = getattr(self.state.patient, key)
                if isinstance(current_value, list) and isinstance(value, list):
                    setattr(self.state.patient, key, list(set(current_value + value))) # Merge lists
                elif isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value) # Update dictionaries
                else:
                    setattr(self.state.patient, key, value)
        print(f"Patient data updated: {self.state.patient.dict()}")

    def _parse_llm_tool_calls(self, llm_response: str) -> List[str]:
        """Parses tool calls from LLM response (mock implementation)."""
        try:
            start = llm_response.find("{{tool_calls:")
            end = llm_response.find("}}", start)
            if start != -1 and end != -1:
                tool_calls_str = llm_response[start + len("{{tool_calls:") : end].strip()
                return json.loads(tool_calls_str)
            start = llm_response.find("{{new_tool_calls:")
            end = llm_response.find("}}", start)
            if start != -1 and end != -1:
                tool_calls_str = llm_response[start + len("{{new_tool_calls:") : end].strip()
                return json.loads(tool_calls_str)
        except json.JSONDecodeError:
            pass
        return []

    def _plan(self) -> str:
        """The agent generates a plan based on the current state and patient data."""
        context = self.knowledge_base.retrieve_context(f"diagnosis for {self.state.patient.symptoms}")
        prompt = (
            f"Given the patient's data: {self.state.patient.dict()}, "
            f"and relevant medical context: {context}. "
            f"Current task: Diagnose and recommend treatment. "
            f"Previous observation: {self.state.last_observation if self.state.last_observation else 'None'}. "
            f"Previous reflection: {self.state.reflection if self.state.reflection else 'None'}. "
            f"Generate a detailed plan, including necessary tool calls to gather more information or make a diagnosis. "
            f"Format tool calls as a JSON list in a '{{tool_calls: [...]}}' block.\n"
        )
        plan_response = self.llm.invoke(prompt)
        self.state.current_plan = plan_response
        print(f"\n--- Iteration {self.state.iteration} - Plan ---\n{self.state.current_plan}")
        return plan_response

    def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Executes a specific tool and returns its output."""
        tool = self.tool_manager.get_tool(tool_name)
        if not tool:
            print(f"Error: Tool '{tool_name}' not found.")
            return {"error": f"Tool '{tool_name}' not found"}
        print(f"Executing tool: {tool_name} with args: {kwargs}")
        self.state.tools_used.append(tool_name)
        try:
            return tool(**kwargs)
        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}

    def _execute_plan(self, plan_response: str) -> str:
        """Executes the plan by calling identified tools and updating patient data."""
        tool_calls = self._parse_llm_tool_calls(plan_response)
        observation_parts = []

        if not tool_calls and "diagnosis_details" in plan_response:
            # If LLM directly provides diagnosis/treatment without explicit tool calls
            try:
                start_diag = plan_response.find("{{diagnosis_details:")
                end_diag = plan_response.find("}", start_diag)
                if start_diag != -1 and end_diag != -1:
                    diag_json = plan_response[start_diag + len("{{diagnosis_details:") : end_diag + 1].strip()
                    diagnosis_data = json.loads(diag_json)
                    diagnosis = Diagnosis(**diagnosis_data)
                    self.state.diagnosis_history.append(diagnosis)
                    observation_parts.append(f"Diagnosis made: {diagnosis.condition}")

                start_treat = plan_response.find("{{treatment_details:")
                end_treat = plan_response.find("}", start_treat)
                if start_treat != -1 and end_treat != -1:
                    treat_json = plan_response[start_treat + len("{{treatment_details:") : end_treat + 1].strip()
                    treatment_data = json.loads(treat_json)
                    treatment_plan = TreatmentPlan(**treatment_data)
                    self.state.treatment_history.append(treatment_plan)
                    observation_parts.append(f"Treatment planned: {', '.join(treatment_plan.recommended_actions)}")
            except json.JSONDecodeError as e:
                print(f"Error parsing direct LLM output: {e}")

        for tool_name in tool_calls:
            tool_output = None
            if tool_name == "ehr_retriever":
                tool_output = self._execute_tool("ehr_retriever", patient_id=self.state.patient.id)
                if tool_output and "allergies" in tool_output: # Example update
                    self._update_patient_data({"medical_history": tool_output["past_diagnoses"], "current_medications": tool_output["current_medications"]})

            elif tool_name == "medical_literature_search":
                query = self.state.current_plan or "patient symptoms"
                tool_output = self._execute_tool("medical_literature_search", query=query)

            elif tool_name == "drug_interaction_checker":
                tool_output = self._execute_tool("drug_interaction_checker", drugs=self.state.patient.current_medications + self.state.treatment_history[-1].medications if self.state.treatment_history else self.state.patient.current_medications)

            elif tool_name == "lab_result_interpreter":
                if self.state.patient.lab_results:
                    tool_output = self._execute_tool("lab_result_interpreter", lab_results=self.state.patient.lab_results)

            elif tool_name == "diagnostic_imaging_analysis":
                # Mock imaging_report to be present in patient data if relevant
                if self.state.patient.imaging_reports:
                    tool_output = self._execute_tool("diagnostic_imaging_analysis", image_url=self.state.patient.imaging_reports[0])

            if tool_output:
                observation_parts.append(f"Tool '{tool_name}' output: {tool_output}")

        observation = "\n".join(observation_parts) if observation_parts else "No specific observations from tool execution."
        self.state.last_observation = observation
        print(f"\n--- Iteration {self.state.iteration} - Execute & Observe ---\n{observation}")
        return observation

    def _reflect(self) -> str:
        """The agent reflects on the observations and previous actions."""
        prompt = (
            f"Given the patient's current state: {self.state.patient.dict()}, "
            f"the last plan: {self.state.current_plan}, "
            f"and the observations: {self.state.last_observation}. "
            f"Reflect on the effectiveness of the plan and observations. "
            f"Identify any knowledge conflicts, missing information, or potential errors. "
            f"Propose new actions or tool calls if necessary for self-correction. "
            f"Format new tool calls as a JSON list in a '{{new_tool_calls: [...]}}' block.\n"
        )
        reflection_response = self.llm.invoke(prompt)
        self.state.reflection = reflection_response
        print(f"\n--- Iteration {self.state.iteration} - Reflect ---\n{self.state.reflection}")
        return reflection_response

    def _self_correct(self, reflection_response: str) -> bool:
        """Determines if self-correction is needed and updates the state for the next iteration."""
        new_tool_calls = self._parse_llm_tool_calls(reflection_response)
        if new_tool_calls:
            print(f"Self-correction: Agent proposes new tools to call: {new_tool_calls}")
            # We will integrate these new tool calls into the next planning phase implicitly
            # by allowing the LLM to re-plan based on the updated reflection.
            return True  # Indicates more iterations are needed
        
        # Check for termination conditions
        if self.state.diagnosis_history and self.state.treatment_history:
            last_diagnosis = self.state.diagnosis_history[-1]
            if last_diagnosis.confidence > 0.8 and "effective" in self.state.reflection.lower():
                print("Self-correction: High confidence diagnosis and effective treatment reflected. Terminating.")
                return False # No more iterations needed

        print("Self-correction: No explicit new tools proposed, but continuing for potential refinement or more data.")
        return True # Continue by default if no clear termination

    def run_diagnosis_and_treatment(self, patient: Patient) -> TreatmentPlan:
        self.state.patient = patient
        print(f"\n--- Starting AMAI-Assist for Patient: {patient.name} (ID: {patient.id}) ---")

        while self.state.iteration < self.state.max_iterations:
            self.state.iteration += 1
            print(f"\n### Running Iteration {self.state.iteration} ###")

            # Plan
            plan_response = self._plan()

            # Execute & Observe
            observation = self._execute_plan(plan_response)

            # Reflect
            reflection_response = self._reflect()

            # Self-Correct / Check Termination
            if not self._self_correct(reflection_response):
                break # Exit loop if agent decides to terminate

        final_diagnosis = self.state.diagnosis_history[-1] if self.state.diagnosis_history else None
        final_treatment = self.state.treatment_history[-1] if self.state.treatment_history else None

        if final_diagnosis and final_treatment:
            print(f"\n--- AMAI-Assist Finished for Patient: {patient.name} ---")
            print(f"Final Diagnosis: {final_diagnosis.condition} (Confidence: {final_diagnosis.confidence:.2f})")
            print(f"Final Treatment: {', '.join(final_treatment.recommended_actions)}")
            return final_treatment
        else:
            print("\n--- AMAI-Assist could not finalize a diagnosis and treatment plan. --- ")
            return TreatmentPlan(diagnosis=Diagnosis(condition="Undetermined", confidence=0.0, justification="No final plan"), 
                                 recommended_actions=["Consult a specialist"], follow_up_instructions="N/A", expected_outcome="N/A")


# --- Main Execution / Example Usage ---
if __name__ == "__main__":
    # Initialize components
    mock_llm = MockLLM()
    tool_manager = ToolManager()
    knowledge_base = KnowledgeBase()

    # Create the agent
    agent = AdaptiveMedicalAgent(mock_llm, tool_manager, knowledge_base)

    # Define a sample patient
    patient_data = Patient(
        id="P001",
        name="Alice Smith",
        age=45,
        gender="Female",
        symptoms=["fever", "cough", "sore throat", "muscle aches"],
        lab_results={
            "WBC": 12.5, # Elevated
            "CRP": 8.2
        },
        imaging_reports=["chest_xray_p001.jpg"]
    )

    # Run the agent for diagnosis and treatment
    final_plan = agent.run_diagnosis_and_treatment(patient_data)

    print("\nGenerated Final Treatment Plan:")
    print(json.dumps(final_plan.dict(), indent=2))

    print("\n--- Running for a patient with different symptoms ---")
    patient_data_2 = Patient(
        id="P002",
        name="Bob Johnson",
        age=60,
        gender="Male",
        symptoms=["chest pain", "shortness of breath"],
        medical_history=["Type 2 Diabetes"],
        lab_results={
            "Troponin": 0.05, # Normal
            "Glucose": 180 # Elevated
        }
    )
    agent_2 = AdaptiveMedicalAgent(mock_llm, tool_manager, knowledge_base) # New agent for new patient
    final_plan_2 = agent_2.run_diagnosis_and_treatment(patient_data_2)
    print("\nGenerated Final Treatment Plan for Bob Johnson:")
    print(json.dumps(final_plan_2.dict(), indent=2))
