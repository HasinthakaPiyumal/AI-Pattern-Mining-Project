import json
from typing import Dict, Any, List, Callable

# --- 1. Simulated Medical Knowledge Base ---
MEDICAL_KNOWLEDGE_BASE = {
    "flu": {
        "symptoms": ["fever", "cough", "sore throat", "body aches"],
        "common_tests": ["rapid flu test"],
        "treatment": "rest, fluids, antiviral medication (if severe)"
    },
    "strep throat": {
        "symptoms": ["sore throat", "fever", "swollen tonsils", "white patches on throat"],
        "common_tests": ["rapid strep test"],
        "treatment": "antibiotics"
    },
    "common cold": {
        "symptoms": ["runny nose", "sore throat", "cough", "sneezing"],
        "common_tests": [],
        "treatment": "rest, fluids, over-the-counter medication"
    }
    # Add more conditions as needed for a robust system
}

# --- 2. Tool Definitions ---
# These functions simulate external systems or data access

def tool_get_patient_symptoms(patient_record: Dict[str, Any]) -> List[str]:
    """Retrieves reported symptoms from a patient's record."""
    print(f"TOOL_CALL: get_patient_symptoms for patient_id: {patient_record.get('id')}")
    return patient_record.get("symptoms", [])

def tool_query_disease_database(symptoms: List[str], age: int, existing_conditions: List[str]) -> List[str]:
    """Queries a simulated medical database for potential diseases based on symptoms and patient history."""
    print(f"TOOL_CALL: query_disease_database with symptoms: {symptoms}")
    matching_conditions = []
    for disease, data in MEDICAL_KNOWLEDGE_BASE.items():
        # Check if all patient symptoms are present in the disease's known symptoms
        if all(symptom in data["symptoms"] for symptom in symptoms):
            matching_conditions.append(disease)
    return matching_conditions

def tool_recommend_lab_test(suspected_condition: str) -> List[str]:
    """Recommends lab tests for a suspected medical condition."""
    print(f"TOOL_CALL: recommend_lab_test for condition: {suspected_condition}")
    if suspected_condition in MEDICAL_KNOWLEDGE_BASE:
        return MEDICAL_KNOWLEDGE_BASE[suspected_condition].get("common_tests", [])
    return []

def tool_get_lab_test_results(patient_id: str, test_name: str) -> Dict[str, Any]:
    """Simulates fetching lab test results for a patient."""
    print(f"TOOL_CALL: get_lab_test_results for patient_id: {patient_id}, test: {test_name}")
    # In a real system, this would interact with an LIS
    if test_name == "rapid flu test" and patient_id == "PAT001":
        return {"test_name": test_name, "result": "Positive", "interpretation": "Influenza A detected"}
    elif test_name == "rapid strep test" and patient_id == "PAT002":
        return {"test_name": test_name, "result": "Positive", "interpretation": "Streptococcus pyogenes detected"}
    return {"test_name": test_name, "result": "Negative", "interpretation": "No specific finding"}

def tool_get_treatment_guidelines(diagnosis: str) -> str:
    """Provides treatment guidelines for a confirmed diagnosis."""
    print(f"TOOL_CALL: get_treatment_guidelines for diagnosis: {diagnosis}")
    if diagnosis in MEDICAL_KNOWLEDGE_BASE:
        return MEDICAL_KNOWLEDGE_BASE[diagnosis].get("treatment", "No specific guidelines found.")
    return "Diagnosis unknown, consult a specialist."


# Map tool names to functions
TOOL_REGISTRY = {
    "get_patient_symptoms": tool_get_patient_symptoms,
    "query_disease_database": tool_query_disease_database,
    "recommend_lab_test": tool_recommend_lab_test,
    "get_lab_test_results": tool_get_lab_test_results,
    "get_treatment_guidelines": tool_get_treatment_guidelines
}

class MedicalDiagnosticAgent:
    """
    An intelligent medical diagnostic assistant that uses adaptive agentic reasoning,
    tool integration, and self-correction.
    """
    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools
        self.history: List[Dict[str, Any]] = []
        self.current_state: Dict[str, Any] = {}
        print("MedicalDiagnosticAgent initialized.")

    def _simulate_llm_reasoning(self, prompt: str) -> Dict[str, Any]:
        """
        Simulates an LLM's reasoning process.
        In a real application, this would involve an actual LLM API call.
        The output format is crucial for tool calling and decision making.
        """
        print(f"\nLLM_PROMPT: {prompt}\n")
        # Simple rule-based simulation for demonstration
        if "initial assessment" in prompt:
            return {
                "thought": "Patient symptoms need to be retrieved to identify potential conditions.",
                "action": {
                    "tool": "get_patient_symptoms",
                    "args": {"patient_record": self.current_state.get("patient_data")}
                }
            }
        elif "potential conditions" in prompt and "symptoms" in self.current_state:
            return {
                "thought": "Based on symptoms, I will query the disease database.",
                "action": {
                    "tool": "query_disease_database",
                    "args": {
                        "symptoms": self.current_state["symptoms"],
                        "age": self.current_state["patient_data"]["age"],
                        "existing_conditions": self.current_state["patient_data"]["existing_conditions"]
                    }
                }
            }
        elif "recommend lab tests" in prompt and "suspected_condition" in self.current_state:
            return {
                "thought": f"The suspected condition is {self.current_state['suspected_condition']}. I should recommend a lab test for it.",
                "action": {
                    "tool": "recommend_lab_test",
                    "args": {"suspected_condition": self.current_state["suspected_condition"]}
                }
            }
        elif "get test results" in prompt and "recommended_tests" in self.current_state and self.current_state["recommended_tests"]:
            test_to_run = self.current_state["recommended_tests"][0] # Just take the first for simplicity
            return {
                "thought": f"I have recommended tests, now I need to get results for {test_to_run}.",
                "action": {
                    "tool": "get_lab_test_results",
                    "args": {"patient_id": self.current_state["patient_data"]["id"], "test_name": test_to_run}
                }
            }
        elif "final diagnosis" in prompt and "test_results" in self.current_state:
            if "Positive" in self.current_state["test_results"].get("result", ""):
                diagnosis = self.current_state["test_results"]["interpretation"].replace(" detected", "")
                if "Influenza" in diagnosis: diagnosis = "flu"
                if "Streptococcus pyogenes" in diagnosis: diagnosis = "strep throat"
                return {
                    "thought": f"Based on positive test results, the diagnosis is likely {diagnosis}. I should get treatment guidelines.",
                    "action": {
                        "tool": "get_treatment_guidelines",
                        "args": {"diagnosis": diagnosis.lower()}
                    }
                }
            else:
                 # If no specific positive finding, re-evaluate or suggest general treatment for observed symptoms
                return {
                    "thought": "Test results are negative or inconclusive. Re-evaluating based on initial symptoms for a general recommendation, or suggesting follow-up.",
                    "action": {
                        "tool": "get_treatment_guidelines",
                        "args": {"diagnosis": self.current_state.get("suspected_condition", "common cold")} # Fallback
                    }
                }
        elif "self-correction" in prompt:
            # Simple self-correction: if feedback says diagnosis is wrong, try another path or suggest re-assessment
            if "diagnosis incorrect" in prompt:
                return {
                    "thought": "Previous diagnosis was incorrect. I need to re-evaluate symptoms and consider other conditions or ask for more data.",
                    "action": None # No specific tool, but new reasoning path
                }
            return {"thought": "No specific correction needed based on feedback.", "action": None}
        return {"thought": "I am not sure what to do next.", "action": None}

    def _execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Executes a registered tool."""
        tool_func = self.tools.get(tool_name)
        if tool_func:
            return tool_func(**kwargs)
        else:
            raise ValueError(f"Tool '{tool_name}' not found.")

    def _record_step(self, step_type: str, content: Any, observation: Any = None):
        """Records a step in the agent's history."""
        self.history.append({"type": step_type, "content": content, "observation": observation})
        print(f"Recorded {step_type}: {content}")

    def _self_reflect(self, feedback: str) -> None:
        """
        The agent reflects on its performance given external feedback
        and attempts to adjust its internal state or strategy.
        """
        print(f"\nAGENT_REFLECTING: Received feedback: '{feedback}'")
        reflection_prompt = f"Given the task history: {json.dumps(self.history[-3:])} and feedback: '{feedback}', what went wrong and how should I adjust my approach?"
        reflection_result = self._simulate_llm_reasoning(f"Perform self-correction based on feedback. History: {self.history}. Feedback: {feedback}")

        if reflection_result and reflection_result.get("thought"):
            print(f"AGENT_REFLECTION_THOUGHT: {reflection_result['thought']}")
            # In a real system, this thought would influence future actions or update a learned model.
            # For this simulation, we'll just acknowledge the feedback.
            if "re-evaluate symptoms" in reflection_result["thought"]:
                print("ACTION: Agent will re-evaluate symptoms in the next iteration.")
                self.current_state["re_evaluate_symptoms"] = True
            elif "consider other conditions" in reflection_result["thought"]:
                 print("ACTION: Agent will try to consider other conditions in the next iteration.")
                 self.current_state["consider_other_conditions"] = True
        self._record_step("reflection", {"feedback": feedback, "reflection_result": reflection_result})


    def diagnose_patient(self, patient_data: Dict[str, Any], max_iterations: int = 5) -> Dict[str, Any]:
        """
        Orchestrates the diagnostic process with adaptive reasoning,
        tool integration, and a self-correction loop.
        """
        self.history = []
        self.current_state = {"patient_data": patient_data}
        self._record_step("initial_patient_data", patient_data)
        print(f"\nStarting diagnosis for Patient ID: {patient_data['id']}\n")

        for iteration in range(max_iterations):
            print(f"\n--- ITERATION {iteration + 1} ---")
            
            # 1. Reasoning: Agent decides next step (simulated LLM)
            reasoning_prompt = f"Perform initial assessment for patient {patient_data['id']} given current state: {self.current_state}"
            if self.current_state.get("re_evaluate_symptoms"):
                reasoning_prompt = f"Re-evaluating symptoms and conditions for patient {patient_data['id']} after previous feedback. Current state: {self.current_state}"
                self.current_state.pop("re_evaluate_symptoms") # Clear flag
            elif self.current_state.get("consider_other_conditions"):
                 reasoning_prompt = f"Considering other conditions for patient {patient_data['id']} after previous feedback. Current state: {self.current_state}"
                 self.current_state.pop("consider_other_conditions") # Clear flag

            llm_response = self._simulate_llm_reasoning(reasoning_prompt)
            self._record_step("llm_reasoning", llm_response)

            if llm_response.get("action"):
                action = llm_response["action"]
                tool_name = action["tool"]
                tool_args = action["args"]
                
                # 2. Tool Integration: Execute the chosen tool
                try:
                    tool_output = self._execute_tool(tool_name, **tool_args)
                    self._record_step("tool_execution", {"tool": tool_name, "args": tool_args}, tool_output)
                    print(f"TOOL_OUTPUT: {tool_output}")

                    # Update current state based on tool output
                    if tool_name == "get_patient_symptoms":
                        self.current_state["symptoms"] = tool_output
                        self.current_state["last_action"] = "symptoms_retrieved"
                    elif tool_name == "query_disease_database":
                        self.current_state["potential_conditions"] = tool_output
                        if tool_output:
                            self.current_state["suspected_condition"] = tool_output[0] # Take the first as primary suspect
                        self.current_state["last_action"] = "conditions_identified"
                    elif tool_name == "recommend_lab_test":
                        self.current_state["recommended_tests"] = tool_output
                        self.current_state["last_action"] = "tests_recommended"
                    elif tool_name == "get_lab_test_results":
                        self.current_state["test_results"] = tool_output
                        self.current_state["last_action"] = "test_results_obtained"
                    elif tool_name == "get_treatment_guidelines":
                        self.current_state["treatment_guidelines"] = tool_output
                        self.current_state["last_action"] = "treatment_recommended"
                        # This might be a good point to evaluate for termination
                        if self.current_state.get("suspected_condition") and self.current_state.get("treatment_guidelines"):
                             print("\nAGENT_TERMINATING: Diagnosis and treatment guidelines provided.")
                             return {"final_diagnosis": self.current_state.get("suspected_condition"),
                                     "treatment_plan": self.current_state.get("treatment_guidelines"),
                                     "history": self.history}

                except Exception as e:
                    self._record_step("tool_error", {"tool": tool_name, "args": tool_args, "error": str(e)})
                    print(f"ERROR executing tool {tool_name}: {e}")
                    # Simulate LLM reflecting on error
                    self_correction_prompt = f"Error occurred during tool execution: {e}. How should I proceed?"
                    self._self_reflect(f"Tool execution failed: {e}. Re-evaluate strategy.")
                    continue # Try next iteration

            else:
                print("LLM decided no specific action. Potentially stuck or finished reasoning.")
                if self.current_state.get("treatment_guidelines"):
                    print("\nAGENT_TERMINATING: No further actions, and treatment guidelines already provided.")
                    return {"final_diagnosis": self.current_state.get("suspected_condition", "Unknown"),
                            "treatment_plan": self.current_state.get("treatment_guidelines", "Consult a specialist."),
                            "history": self.history}
                break # Exit if no action and no clear resolution

            # 3. Self-Correction (simulated external feedback loop for demo)
            # In a real scenario, this would come from a doctor's input or an evaluation module
            if iteration == 2 and patient_data["id"] == "PAT001": # Simulate feedback for a specific case
                if self.current_state.get("suspected_condition") != "flu":
                    print("\n--- SIMULATING EXTERNAL FEEDBACK ---")
                    feedback = "Diagnosis seems incorrect. Consider flu given the symptoms."
                    self._self_reflect(feedback)

        print("\nAGENT_TERMINATING: Max iterations reached without conclusive outcome or explicit termination.")
        return {"final_diagnosis": self.current_state.get("suspected_condition", "Undetermined"),
                "treatment_plan": self.current_state.get("treatment_guidelines", "Further investigation recommended."),
                "history": self.history}

# --- Main Execution ---
if __name__ == "__main__":
    agent = MedicalDiagnosticAgent(tools=TOOL_REGISTRY)

    # Example Patient 1: Should lead to Flu diagnosis, with a simulated correction
    patient_1_data = {
        "id": "PAT001",
        "age": 35,
        "symptoms": ["fever", "cough", "body aches", "sore throat"],
        "existing_conditions": ["none"]
    }

    print("\n##### Running Diagnosis for Patient PAT001 (Flu-like symptoms) #####")
    final_result_pat1 = agent.diagnose_patient(patient_1_data)
    print("\nFINAL DIAGNOSIS PAT001:", json.dumps(final_result_pat1, indent=2))

    print("\n" + "="*80 + "\n")

    # Example Patient 2: Should lead to Strep Throat diagnosis
    patient_2_data = {
        "id": "PAT002",
        "age": 10,
        "symptoms": ["sore throat", "fever", "swollen tonsils"],
        "existing_conditions": ["asthma"]
    }
    print("\n##### Running Diagnosis for Patient PAT002 (Strep Throat-like symptoms) #####")
    final_result_pat2 = agent.diagnose_patient(patient_2_data)
    print("\nFINAL DIAGNOSIS PAT002:", json.dumps(final_result_pat2, indent=2))