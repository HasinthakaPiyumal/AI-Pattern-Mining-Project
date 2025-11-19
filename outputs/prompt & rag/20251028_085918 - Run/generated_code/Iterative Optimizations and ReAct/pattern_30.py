from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time

# --- Data Models ---

class PatientData(BaseModel):
    patient_id: str
    symptoms: List[str]
    medical_history: List[str]
    lab_results: Dict[str, Any]
    imaging_results: Optional[Dict[str, Any]] = None

class DiagnosisFeedback(BaseModel):
    source: str  # e.g., "expert_review", "new_lab_data", "self_reflection"
    content: str
    is_corrective: bool = False
    severity: str = "low" # low, medium, high

# --- Mock LLM and Tools ---

class MockLLM:
    """A mock Large Language Model for simulating reasoning."""
    def generate_response(self, prompt: str) -> str:
        # Simulate LLM thinking time
        time.sleep(0.5)
        if "initial hypothesis" in prompt:
            return "Initial hypothesis: Autoimmune disorder. Suggested next step: Order specific autoantibody panel."
        elif "autoantibody panel results" in prompt:
            return "Revised hypothesis: Systemic Lupus Erythematosus. Suggested next step: Review latest lupus treatment guidelines and consult rheumatologist."
        elif "expert feedback" in prompt:
            return "Acknowledged expert feedback. Adjusting confidence in SLE diagnosis and considering differential diagnoses like vasculitis. Suggested next step: Further analyze kidney biopsy results for signs of vasculitis."
        elif "self-reflection" in prompt:
            return "Self-reflection complete. Noted discrepancy between initial symptom onset and current lab markers. Re-evaluating timeline for potential missed infection leading to autoimmune trigger. Suggested next step: Search for literature on post-infectious autoimmune conditions."
        return "Further reasoning based on input..."

class MedicalTool:
    """Base class for medical tools."""
    name: str
    description: str

    def execute(self, query: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError

class MedicalLiteratureSearchTool(MedicalTool):
    name: "Medical Literature Search"
    description: "Searches medical databases for relevant articles and guidelines."

    def execute(self, query: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        print(f"[Tool: Literature Search] Searching for: {query}")
        time.sleep(1)
        # Simulate search results
        if "autoantibody panel" in query:
            return {"results": "Found several articles on diagnostic criteria for autoimmune diseases using autoantibody panels.", "success": True}
        elif "post-infectious autoimmune conditions" in query:
            return {"results": "Found recent review on viral triggers for autoimmune diseases, especially those affecting kidneys.", "success": True}
        return {"results": "No direct matches found for the query.", "success": False}

class LabResultInterpretationTool(MedicalTool):
    name: "Lab Result Interpretation"
    description: "Interprets raw lab results in the context of patient data."

    def execute(self, query: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        print(f"[Tool: Lab Interpretation] Interpreting: {query}")
        time.sleep(0.8)
        # Simulate interpretation
        if "autoantibody panel" in query and data and "lab_results" in data:
            mock_panel_results = data["lab_results"].get("autoantibody_panel", {})
            if "ANA" in mock_panel_results and mock_panel_results["ANA"] == "positive":
                return {"interpretation": "Positive ANA (1:640, speckled pattern) strongly suggests an autoimmune process. Further specific antibodies needed.", "success": True}
        return {"interpretation": "Could not provide specific interpretation for the given query/data.", "success": False}

# --- Adaptive Iterative Agent ---

class DiagnosticAgent:
    """An adaptive iterative agent for rare disease diagnosis."""
    def __init__(self, llm: MockLLM, tools: List[MedicalTool], max_iterations: int = 10):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.current_hypothesis: Optional[str] = None
        self.reasoning_history: List[str] = []
        self.feedback_history: List[DiagnosisFeedback] = []
        self.confidence_score: float = 0.5 # Initial confidence

    def _reason(self, prompt: str) -> str:
        """Uses the LLM to generate reasoning and next steps."""
        response = self.llm.generate_response(prompt)
        self.reasoning_history.append(f"LLM Reasoning: {response}")
        return response

    def _execute_tool(self, tool_name: str, query: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executes a specified tool and returns its output."""
        if tool_name in self.tools:
            print(f"Executing tool: {tool_name} with query: {query}")
            return self.tools[tool_name].execute(query, data)
        else:
            print(f"Error: Tool '{tool_name}' not found.")
            return {"success": False, "error": f"Tool '{tool_name}' not found."}

    def _process_feedback(self, feedback: DiagnosisFeedback):
        """Integrates feedback into the agent's state and triggers self-correction if needed."""
        self.feedback_history.append(feedback)
        print(f"[Feedback Received] Source: {feedback.source}, Content: {feedback.content}")
        if feedback.is_corrective:
            self._reflect_and_correct(feedback.content, feedback.severity)
        # Adjust confidence based on feedback (simplified)
        if "correct" in feedback.content.lower():
            self.confidence_score = min(1.0, self.confidence_score + 0.1)
        elif "incorrect" in feedback.content.lower() or feedback.is_corrective:
            self.confidence_score = max(0.1, self.confidence_score - 0.2)

    def _reflect_and_correct(self, correction_hint: str, severity: str):
        """Simulates the agent's self-correction mechanism based on feedback."""
        print(f"[Self-Correction Triggered] Severity: {severity}. Hint: {correction_hint}")
        reflection_prompt = f"Reflect on the current hypothesis '{self.current_hypothesis}' given the corrective feedback: '{correction_hint}'. Consider how to adjust future reasoning and actions."
        reflection_output = self._reason(f"Self-reflection based on feedback: {correction_hint}")
        # In a real system, this would lead to refining internal world models, updating beliefs, or adjusting planning parameters.
        self.reasoning_history.append(f"Self-Correction Output: {reflection_output}")
        print(f"Agent reflects and considers new approach: {reflection_output}")

    def _suggest_new_tool(self, current_state: str) -> Optional[str]:
        """Simulates the agent's ability to suggest new diagnostic tools or research avenues."""
        # This is a highly simplified placeholder. A real LLM would reason based on gaps.
        if "need more specific genetic markers" in current_state.lower() and "genetic sequencing tool" not in self.tools:
            print("Agent suggests: A 'Genetic Sequencing Analysis Tool' might be needed for more specific markers.")
            return "Genetic Sequencing Analysis Tool"
        if "unexplained inflammation" in current_state.lower() and "advanced inflammatory marker analysis" not in self.tools:
            print("Agent suggests: An 'Advanced Inflammatory Marker Analysis Tool' could provide deeper insights.")
            return "Advanced Inflammatory Marker Analysis Tool"
        return None

    def diagnose(self, patient_data: PatientData) -> str:
        """Main iterative diagnostic loop."""
        print(f"\n--- Starting Diagnosis for Patient ID: {patient_data.patient_id} ---")
        self.current_hypothesis = "Initial broad assessment based on symptoms."
        iteration = 0

        while iteration < self.max_iterations and self.confidence_score < 0.95: # Termination condition
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # 1. Agent Reasoning: Formulate next steps based on current state and history
            reasoning_prompt = f"Patient ID: {patient_data.patient_id}. Current hypothesis: {self.current_hypothesis}. Symptoms: {', '.join(patient_data.symptoms)}. Medical History: {', '.join(patient_data.medical_history)}. Lab Results: {patient_data.lab_results}. Previous reasoning: {self.reasoning_history[-1] if self.reasoning_history else 'None'}. Given this, what is the next best diagnostic step (e.g., tool to use, specific query, or refined hypothesis)?"
            llm_response = self._reason(reasoning_prompt)
            print(f"Agent's current thought: {llm_response}")

            # Update hypothesis based on reasoning
            if "hypothesis:" in llm_response.lower():
                self.current_hypothesis = llm_response.split("hypothesis:")[-1].split(".")[0].strip()
                print(f"Updated Hypothesis: {self.current_hypothesis}")

            # 2. Tool Manipulation (simulated based on LLM response)
            tool_used = False
            if "order specific autoantibody panel" in llm_response.lower() or "specific antibodies needed" in llm_response.lower():
                tool_output = self._execute_tool("Lab Result Interpretation", "autoantibody panel", data=patient_data.dict())
                if tool_output["success"]:
                    patient_data.lab_results["autoantibody_panel_interpretation"] = tool_output["interpretation"]
                    # Provide feedback to the agent from the tool output
                    self._process_feedback(DiagnosisFeedback(source="Lab_Tool", content=f"Lab interpretation: {tool_output['interpretation']}"))
                tool_used = True
            elif "review latest lupus treatment guidelines" in llm_response.lower() or "search for literature" in llm_response.lower():
                search_query = "Systemic Lupus Erythematosus treatment guidelines" if "lupus" in llm_response.lower() else "post-infectious autoimmune conditions"
                tool_output = self._execute_tool("Medical Literature Search", search_query)
                if tool_output["success"]:
                    # Provide feedback to the agent from the tool output
                    self._process_feedback(DiagnosisFeedback(source="Literature_Tool", content=f"Literature search results: {tool_output['results']}"))
                tool_used = True

            if not tool_used:
                print("No specific tool identified for execution in this iteration, proceeding with self-reflection.")

            # 3. Self-reflection and Feedback Integration (simulated)
            # In a real system, this could come from user input, new test results, or a dedicated self-reflection module.
            if iteration == 3:
                # Simulate expert feedback
                expert_feedback = DiagnosisFeedback(
                    source="expert_review",
                    content="Consider vasculitis as a strong differential given the kidney involvement and lack of full SLE criteria. Review kidney biopsy more carefully.",
                    is_corrective=True,
                    severity="high"
                )
                self._process_feedback(expert_feedback)
            elif iteration == 5:
                # Simulate new lab data (e.g., specific genetic marker test returns negative, contradicting a hypothesis)
                self._process_feedback(DiagnosisFeedback(source="new_lab_data", content="Genetic markers for common autoimmune conditions were negative. Re-evaluate primary diagnosis direction.", is_corrective=True, severity="medium"))

            # 4. Meta-cognitive ability: Suggest new tools if needed
            suggested_tool = self._suggest_new_tool(self.current_hypothesis + " " + llm_response)
            if suggested_tool:
                print(f"Agent considered suggesting a new tool: {suggested_tool}")
                # In a real system, this would prompt creation/integration of the new tool

            print(f"Current Confidence Score: {self.confidence_score:.2f}")
            if self.confidence_score >= 0.95:
                print("Agent confident enough to terminate diagnosis.")
                break

        final_diagnosis = self.current_hypothesis if self.confidence_score >= 0.7 else "Unable to reach a confident diagnosis. Further expert consultation recommended."
        print(f"\n--- Final Diagnosis for Patient ID {patient_data.patient_id}: {final_diagnosis} ---")
        return final_diagnosis

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize Mock LLM and Tools
    mock_llm = MockLLM()
    medical_tools = [
        MedicalLiteratureSearchTool(),
        LabResultInterpretationTool()
    ]

    # 2. Create the Diagnostic Agent
    agent = DiagnosticAgent(llm=mock_llm, tools=medical_tools, max_iterations=7)

    # 3. Prepare Patient Data
    patient_case_1 = PatientData(
        patient_id="P12345",
        symptoms=["fatigue", "joint pain", "skin rash", "kidney issues"],
        medical_history=["recurrent infections", "family history of autoimmune conditions"],
        lab_results={
            "ANA": "positive",
            "ESR": "elevated",
            "CRP": "elevated",
            "creatinine": "high",
            "autoantibody_panel": {"ANA": "positive", "Anti-dsDNA": "pending"}
        }
    )

    # 4. Run the Diagnostic Process
    agent.diagnose(patient_case_1)

    print("\n--- Agent's Reasoning History ---")
    for entry in agent.reasoning_history:
        print(entry)

    print("\n--- Agent's Feedback History ---")
    for feedback in agent.feedback_history:
        print(f"[{feedback.source}] {feedback.content} (Corrective: {feedback.is_corrective}, Severity: {feedback.severity})")
