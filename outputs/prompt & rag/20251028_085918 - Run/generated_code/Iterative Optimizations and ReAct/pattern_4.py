import json
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

# 1. Data Models (Pydantic)
class Symptom(BaseModel):
    name: str
    severity: str = "mild"
    duration_days: int = 1

class PatientData(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    symptoms: List[Symptom] = Field(default_factory=list)
    past_medical_history: List[str] = Field(default_factory=list)
    lab_results: Dict[str, Any] = Field(default_factory=dict)
    imaging_results: Optional[Dict[str, Any]] = None

class Diagnosis(BaseModel):
    disease_name: str
    icd_code: str
    confidence: float  # 0.0 to 1.0
    explanation: str

class TreatmentPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: "TP" + str(hash(json.dumps(dict())) % 10000))
    diagnosis: Diagnosis
    recommendations: List[str]
    medications: List[Dict[str, str]] = Field(default_factory=list)
    referrals: List[str] = Field(default_factory=list)
    prognosis: str
    notes: str

class ToolOutput(BaseModel):
    tool_name: str
    output_data: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None

class AgentAction(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any]
    thought: str

class AgentObservation(BaseModel):
    action: AgentAction
    tool_output: ToolOutput

class FeedbackType(str, Enum):
    CORRECT = "correct"
    INCORRECT_DIAGNOSIS = "incorrect_diagnosis"
    INCORRECT_TREATMENT = "incorrect_treatment"
    MISSING_INFORMATION = "missing_information"
    ADVERSE_EVENT = "adverse_event"
    EXPERT_OVERRIDE = "expert_override"

class Feedback(BaseModel):
    feedback_type: FeedbackType
    details: str
    suggested_correction: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: "2024-01-01T00:00:00Z") # Placeholder

# 2. Enum for Tool Names
class MedicalToolName(str, Enum):
    KNOWLEDGE_BASE = "MedicalKnowledgeBase"
    IMAGING_ANALYSIS = "DiagnosticImagingAnalysis"
    EHR_INTERFACE = "EHRInterface"
    LITERATURE_SEARCH = "MedicalLiteratureSearch"
    SYMPTOM_CHECKER = "SymptomChecker"

# 3. Abstract Tool Interface (Conceptual)
class BaseMedicalTool:
    name: MedicalToolName
    description: str

    def run(self, tool_input: Dict[str, Any]) -> ToolOutput:
        raise NotImplementedError

    def __str__(self):
        return self.name.value

# 4. Concrete Tool Implementations (Placeholders - simulating external services)
class MedicalKnowledgeBaseTool(BaseMedicalTool):
    name = MedicalToolName.KNOWLEDGE_BASE
    description = "Provides information on diseases, symptoms, treatments, and drug interactions."

    def run(self, tool_input: Dict[str, Any]) -> ToolOutput:
        query = tool_input.get("query", "")
        if "pneumonia" in query.lower():
            return ToolOutput(tool_name=self.name, output_data={
                "disease": "Pneumonia",
                "symptoms": ["cough", "fever", "shortness of breath", "chest pain"],
                "treatment": ["antibiotics", "rest", "fluids"],
                "icd_code": "J18.9"
            })
        elif "diabetes" in query.lower():
            return ToolOutput(tool_name=self.name, output_data={
                "disease": "Diabetes Mellitus Type 2",
                "symptoms": ["frequent urination", "increased thirst", "fatigue", "blurred vision"],
                "treatment": ["dietary changes", "exercise", "medication (metformin)"],
                "icd_code": "E11.9"
            })
        return ToolOutput(tool_name=self.name, output_data={"message": f"No specific information found for '{query}'"}, success=False)


class DiagnosticImagingAnalysisTool(BaseMedicalTool):
    name = MedicalToolName.IMAGING_ANALYSIS
    description = "Analyzes medical images (X-rays, CT scans) for abnormalities."

    def run(self, tool_input: Dict[str, Any]) -> ToolOutput:
        image_id = tool_input.get("image_id")
        image_type = tool_input.get("image_type")

        if image_type == "X-ray" and "lung_opacity" in str(image_id).lower():
            return ToolOutput(tool_name=self.name, output_data={
                "finding": "Bilateral lung opacities, consistent with pneumonia",
                "severity": "moderate",
                "confidence": 0.85
            })
        elif image_type == "CT" and "brain_lesion" in str(image_id).lower():
             return ToolOutput(tool_name=self.name, output_data={
                "finding": "Small lesion in left temporal lobe",
                "severity": "mild",
                "confidence": 0.70
            })
        return ToolOutput(tool_name=self.name, output_data={"message": "No significant findings or image not processed."}, success=False)


class EHRInterfaceTool(BaseMedicalTool):
    name = MedicalToolName.EHR_INTERFACE
    description = "Accesses patient's Electronic Health Records, including history and lab results."

    def run(self, tool_input: Dict[str, Any]) -> ToolOutput:
        patient_id = tool_input.get("patient_id")
        if patient_id == "P001":
            return ToolOutput(tool_name=self.name, output_data={
                "patient_id": "P001",
                "past_medical_history": ["Asthma", "Hypertension"],
                "medications": ["Albuterol", "Lisinopril"],
                "allergies": ["Penicillin"],
                "last_lab_results": {"WBC": 15.2, "CRP": 85}
            })
        return ToolOutput(tool_name=self.name, output_data={"message": f"EHR for {patient_id} not found."}, success=False)


class MedicalLiteratureSearchTool(BaseMedicalTool):
    name = MedicalToolName.LITERATURE_SEARCH
    description = "Searches recent medical literature and clinical guidelines."

    def run(self, tool_input: Dict[str, Any]) -> ToolOutput:
        topic = tool_input.get("topic")
        if "antibiotics pneumonia" in topic.lower():
            return ToolOutput(tool_name=self.name, output_data={
                "articles": [
                    {"title": "Guidelines for Community-Acquired Pneumonia", "link": "pubmed.gov/article1"},
                    {"title": "Efficacy of Azithromycin in CAP", "link": "pubmed.gov/article2"}
                ],
                "summary": "Latest guidelines recommend macrolides or doxycycline for CAP."
            })
        return ToolOutput(tool_name=self.name, output_data={"message": f"No literature found for '{topic}'"}, success=False)


class SymptomCheckerTool(BaseMedicalTool):
    name = MedicalToolName.SYMPTOM_CHECKER
    description = "Suggests potential diagnoses based on a list of symptoms."

    def run(self, tool_input: Dict[str, Any]) -> ToolOutput:
        symptoms = tool_input.get("symptoms", [])
        if "cough" in symptoms and "fever" in symptoms and "shortness of breath" in symptoms:
            return ToolOutput(tool_name=self.name, output_data={
                "possible_diagnoses": [
                    {"name": "Pneumonia", "likelihood": 0.7},
                    {"name": "Bronchitis", "likelihood": 0.2}
                ]
            })
        elif "fatigue" in symptoms and "thirst" in symptoms:
            return ToolOutput(tool_name=self.name, output_data={
                "possible_diagnoses": [
                    {"name": "Diabetes", "likelihood": 0.6},
                    {"name": "Dehydration", "likelihood": 0.3}
                ]
            })
        return ToolOutput(tool_name=self.name, output_data={"possible_diagnoses": []})


# 5. LLMAgent Class (Simulated Reasoning Core)
class LLMAgent:
    def __init__(self, tools: List[BaseMedicalTool]):
        self.tools = {tool.name.value: tool for tool in tools}
        self.reasoning_history: List[AgentObservation] = []
        self.current_strategy: str = "Initial assessment and differential diagnosis."
        self.reflection_count = 0
        print("\nLLMAgent initialized with tools:", [tool.name.value for tool in tools])

    def _simulate_llm_reasoning(self, patient_data: PatientData, history: List[AgentObservation], current_strategy: str) -> AgentAction:
        """Simulates the LLM's thought process and tool selection."""
        thought = f"Current strategy: {current_strategy}. Analyzing patient data for {patient_data.chief_complaint}."

        # Simple logic to simulate tool selection based on keywords or history
        if not history and patient_data.symptoms:
            thought += " Starting with symptom checking."
            return AgentAction(tool_name=MedicalToolName.SYMPTOM_CHECKER,
                               tool_input={"symptoms": [s.name for s in patient_data.symptoms]},
                               thought=thought)
        
        # After symptom checking, maybe query knowledge base for top diagnosis
        if MedicalToolName.SYMPTOM_CHECKER.value in [obs.action.tool_name for obs in history] and not any(isinstance(obs.tool_output.output_data.get("disease"), str) for obs in history):
            symptom_checker_output = next((obs.tool_output for obs in history if obs.action.tool_name == MedicalToolName.SYMPTOM_CHECKER.value), None)
            if symptom_checker_output and symptom_checker_output.output_data.get("possible_diagnoses"):
                top_diagnosis = symptom_checker_output.output_data["possible_diagnoses"][0]["name"]
                thought += f" Symptom checker suggests {top_diagnosis}. Querying knowledge base for details."
                return AgentAction(tool_name=MedicalToolName.KNOWLEDGE_BASE,
                                   tool_input={"query": top_diagnosis},
                                   thought=thought)

        # If imaging results are present but not yet analyzed
        if patient_data.imaging_results and not any(obs.action.tool_name == MedicalToolName.IMAGING_ANALYSIS.value for obs in history):
            thought += " Imaging results available. Initiating diagnostic imaging analysis."
            return AgentAction(tool_name=MedicalToolName.IMAGING_ANALYSIS,
                               tool_input={"image_id": patient_data.imaging_results.get("id"),
                                             "image_type": patient_data.imaging_results.get("type")},
                               thought=thought)

        # After a diagnosis is formed, look for literature on treatment
        if any(isinstance(obs.tool_output.output_data.get("disease"), str) for obs in history) and not any(obs.action.tool_name == MedicalToolName.LITERATURE_SEARCH.value for obs in history):
            diagnosis_output = next((obs.tool_output for obs in history if isinstance(obs.tool_output.output_data.get("disease"), str)), None)
            if diagnosis_output:
                disease = diagnosis_output.output_data["disease"]
                thought += f" Diagnosis for {disease} formed. Searching medical literature for treatment guidelines."
                return AgentAction(tool_name=MedicalToolName.LITERATURE_SEARCH,
                                   tool_input={"topic": f"treatment {disease}"},
                                   thought=thought)
        
        # If EHR data hasn't been fetched and patient_id is available
        if patient_data.patient_id and not any(obs.action.tool_name == MedicalToolName.EHR_INTERFACE.value for obs in history):
            thought += " Checking EHR for patient history and lab results."
            return AgentAction(tool_name=MedicalToolName.EHR_INTERFACE,
                               tool_input={"patient_id": patient_data.patient_id},
                               thought=thought)

        thought += " No further tools deemed immediately necessary based on current state."
        return AgentAction(tool_name="FinalDecision", tool_input={}, thought=thought)


    def _reflect_and_correct(self, feedback: Feedback, current_state: Dict[str, Any]) -> str:
        """Simulates the LLM reflecting on feedback and adjusting its strategy."""
        self.reflection_count += 1
        print(f"\n--- Agent Reflecting (Reflection #{self.reflection_count}) ---")
        print(f"Received feedback: {feedback.feedback_type} - {feedback.details}")

        new_strategy = self.current_strategy
        if feedback.feedback_type == FeedbackType.INCORRECT_DIAGNOSIS:
            new_strategy = "Re-evaluate symptoms and consider a broader differential diagnosis. Prioritize deeper knowledge base queries."
            if feedback.suggested_correction:
                print(f"Suggested correction: {feedback.suggested_correction.get('diagnosis_name')}")
                # In a real LLM, this would influence prompt for next turn
        elif feedback.feedback_type == FeedbackType.INCORRECT_TREATMENT:
            new_strategy = "Review literature for alternative treatment protocols. Cross-reference drug interactions carefully."
        elif feedback.feedback_type == FeedbackType.MISSING_INFORMATION:
            new_strategy = "Identify specific missing data points and plan tool calls to acquire them (e.g., EHR, more detailed symptom queries)."
        elif feedback.feedback_type == FeedbackType.ADVERSE_EVENT:
            new_strategy = "Immediately reassess treatment and patient risk factors. Search for contraindications."
        elif feedback.feedback_type == FeedbackType.EXPERT_OVERRIDE:
            new_strategy = "Analyze expert's reasoning to identify gaps in agent's knowledge or decision logic. Learn from demonstration."
        
        print(f"Adjusting strategy to: {new_strategy}")
        self.current_strategy = new_strategy
        return new_strategy

    def _evaluate_termination_condition(self) -> bool:
        """Simulates the agent evaluating if a confident diagnosis and treatment plan are achieved."""
        # Very simplified: if we have a diagnosis and some treatment recommendations from a tool, we'll consider it done.
        has_diagnosis = any(isinstance(obs.tool_output.output_data.get("disease"), str) for obs in self.reasoning_history)
        has_treatment = any("recommendations" in obs.tool_output.output_data for obs in self.reasoning_history)
        return has_diagnosis and has_treatment

    def process_patient_case(self, patient_data: PatientData) -> Optional[TreatmentPlan]:
        print(f"\n--- Processing Patient: {patient_data.patient_id} - Chief Complaint: {patient_data.chief_complaint} ---")
        self.reasoning_history = []
        self.current_strategy = "Initial assessment and differential diagnosis."
        max_iterations = 5  # Limit iterations for this simulation
        iteration = 0
        
        while iteration < max_iterations and not self._evaluate_termination_condition():
            print(f"\nIteration {iteration + 1}:")
            action = self._simulate_llm_reasoning(patient_data, self.reasoning_history, self.current_strategy)
            print(f"Agent Thought: {action.thought}")
            print(f"Agent Action: Calling {action.tool_name} with input: {action.tool_input}")

            tool = self.tools.get(action.tool_name)
            if tool:
                tool_output = tool.run(action.tool_input)
                self.reasoning_history.append(AgentObservation(action=action, tool_output=tool_output))
                print(f"Tool Output ({tool.name}): {tool_output.output_data}")
            elif action.tool_name == "FinalDecision":
                print("Agent decided to make a final decision.")
                break # Exit loop if agent decides it's done
            else:
                print(f"Error: Unknown tool {action.tool_name}.")
                break
            iteration += 1
        
        final_diagnosis = None
        final_recommendations = []
        
        # Extract final diagnosis and treatment from history
        for obs in self.reasoning_history:
            if obs.action.tool_name == MedicalToolName.KNOWLEDGE_BASE.value and obs.tool_output.success:
                if obs.tool_output.output_data.get("disease"): # Check if it's a disease output
                    final_diagnosis = Diagnosis(
                        disease_name=obs.tool_output.output_data["disease"],
                        icd_code=obs.tool_output.output_data.get("icd_code", "N/A"),
                        confidence=0.9, # Simulated confidence
                        explanation=f"Based on knowledge base query and symptoms: {obs.tool_output.output_data.get('symptoms', [])}"
                    )
            if obs.action.tool_name == MedicalToolName.LITERATURE_SEARCH.value and obs.tool_output.success:
                if obs.tool_output.output_data.get("summary"): # Check if it's a summary output
                    final_recommendations.append(f"Literature suggests: {obs.tool_output.output_data['summary']}")
            if obs.action.tool_name == MedicalToolName.EHR_INTERFACE.value and obs.tool_output.success:
                 final_recommendations.append(f"Patient's history: {obs.tool_output.output_data.get('past_medical_history')}")
            if obs.action.tool_name == MedicalToolName.IMAGING_ANALYSIS.value and obs.tool_output.success:
                 final_recommendations.append(f"Imaging finding: {obs.tool_output.output_data.get('finding')}")
        
        if final_diagnosis:
            # Add general treatment based on diagnosis
            general_treatment = self.tools[MedicalToolName.KNOWLEDGE_BASE].run({"query": final_diagnosis.disease_name}).output_data
            if general_treatment.get("treatment"): 
                final_recommendations.extend(general_treatment["treatment"])
            
            treatment_plan = TreatmentPlan(
                diagnosis=final_diagnosis,
                recommendations=list(set(final_recommendations)), # Remove duplicates
                medications=[{"name": "ExampleDrug", "dosage": "500mg"}] if final_diagnosis.disease_name == "Pneumonia" else [],
                prognosis="Good with timely intervention",
                notes="Generated by AI Agent. Review by human clinician required."
            )
            print("\n--- Final Proposed Treatment Plan ---")
            print(treatment_plan.model_dump_json(indent=2))
            return treatment_plan
        else:
            print("\nAgent could not form a confident diagnosis and treatment plan.")
            return None

    def apply_feedback_loop(self, patient_data: PatientData, initial_plan: TreatmentPlan, feedback: Feedback) -> Optional[TreatmentPlan]:
        print("\n--- Applying Feedback Loop ---")
        new_strategy = self._reflect_and_correct(feedback, {"patient_data": patient_data, "initial_plan": initial_plan})
        print("\n--- Re-processing with Adjusted Strategy ---")
        # Reset history for re-processing with new strategy
        self.reasoning_history = [] 
        # In a real system, the LLM would be re-prompted with the new strategy and context
        return self.process_patient_case(patient_data) # Re-run the process with adjusted strategy


# 6. Main Execution Block (Simulation)
if __name__ == "__main__":
    # Instantiate Tools
    medical_tools: List[BaseMedicalTool] = [
        MedicalKnowledgeBaseTool(),
        DiagnosticImagingAnalysisTool(),
        EHRInterfaceTool(),
        MedicalLiteratureSearchTool(),
        SymptomCheckerTool()
    ]

    # Instantiate LLM Agent
    agent = LLMAgent(medical_tools)

    # --- Scenario 1: Basic Pneumonia Case ---
    patient_1_data = PatientData(
        patient_id="P001",
        age=45,
        gender="Male",
        chief_complaint="Severe cough and fever",
        symptoms=[
            Symptom(name="cough", severity="severe", duration_days=3),
            Symptom(name="fever", severity="moderate", duration_days=2),
            Symptom(name="shortness of breath", severity="mild", duration_days=1)
        ],
        imaging_results={"id": "xray_lung_opacity_001", "type": "X-ray", "url": "simulated_url"}
    )

    # Agent processes the case
    initial_plan_1 = agent.process_patient_case(patient_1_data)

    # Simulate Human Review and Feedback for Patient 1
    if initial_plan_1:
        human_feedback_1 = Feedback(
            feedback_type=FeedbackType.INCORRECT_TREATMENT,
            details="Treatment plan did not consider penicillin allergy from EHR. Also, no specific antibiotic was recommended.",
            suggested_correction={"medication": "Azithromycin", "reason": "penicillin allergy"}
        )
        corrected_plan_1 = agent.apply_feedback_loop(patient_1_data, initial_plan_1, human_feedback_1)
        if corrected_plan_1:
            print("\n--- Corrected Treatment Plan after Feedback (Patient 1) ---")
            print(corrected_plan_1.model_dump_json(indent=2))

    print("\n" + "="*80 + "\n")

    # --- Scenario 2: Diabetes with missing lab data ---
    patient_2_data = PatientData(
        patient_id="P002",
        age=60,
        gender="Female",
        chief_complaint="Increased thirst and fatigue",
        symptoms=[
            Symptom(name="increased thirst", severity="moderate", duration_days=7),
            Symptom(name="fatigue", severity="moderate", duration_days=10),
            Symptom(name="frequent urination", severity="mild", duration_days=5)
        ],
        past_medical_history=["High Cholesterol"]
    )

    initial_plan_2 = agent.process_patient_case(patient_2_data)

    if initial_plan_2:
        human_feedback_2 = Feedback(
            feedback_type=FeedbackType.MISSING_INFORMATION,
            details="System did not request or integrate fasting blood glucose or HbA1c lab results which are crucial for diabetes diagnosis.",
            suggested_correction={"missing_labs": ["Fasting Glucose", "HbA1c"]}
        )
        corrected_plan_2 = agent.apply_feedback_loop(patient_2_data, initial_plan_2, human_feedback_2)
        if corrected_plan_2:
            print("\n--- Corrected Treatment Plan after Feedback (Patient 2) ---")
            print(corrected_plan_2.model_dump_json(indent=2))
