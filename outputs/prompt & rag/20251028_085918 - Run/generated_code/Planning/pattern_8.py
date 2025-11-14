from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Mock LangChain components and other libraries
class MockLLM:
    def invoke(self, prompt: str) -> str:
        # Simple mock response based on prompt keywords
        if "symptoms" in prompt.lower() and "analyze" in prompt.lower():
            return "Identified symptoms: fever, cough, fatigue."
        elif "differential diagnosis" in prompt.lower() and "based on" in prompt.lower():
            return "Potential diagnoses: Flu, Common Cold, Pneumonia."
        elif "recommend tests" in prompt.lower():
            return "Recommended tests: Chest X-ray, Blood panel, Viral swab."
        elif "assess risk" in prompt.lower():
            return "Risk assessment: Low risk for Flu, Moderate risk for Pneumonia if untreated."
        elif "query medical knowledge base" in prompt.lower():
            if "flu" in prompt.lower():
                return "Influenza is a viral infection that attacks the respiratory system."
            return "No specific information found for that query."
        else:
            return f"Mock LLM response to: {prompt[:50]}..."

class StructuredTool:
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class AgentExecutor:
    def __init__(self, agent_llm, tools, agent_name="GenericAgent"):
        self.agent_llm = agent_llm
        self.tools = tools
        self.agent_name = agent_name

    def invoke(self, input_data: Dict[str, Any]) -> str:
        print(f"[{self.agent_name}] Invoked with input: {input_data}")
        # In a real LangChain agent, this would involve parsing the LLM's thought process
        # and tool calls. For this mock, we'll directly call a tool if indicated,
        # or use the LLM to generate a response.
        prompt = input_data.get("input", "")
        for tool in self.tools:
            if tool.name.lower().replace('_', ' ') in prompt.lower():
                print(f"[{self.agent_name}] Calling tool: {tool.name}")
                # This is a highly simplified mock of tool invocation
                if tool.name == "analyze_symptoms":
                    symptoms = input_data.get("symptoms", "")
                    return tool.run(symptoms)
                elif tool.name == "query_medical_knowledge_base":
                    query = input_data.get("query", "")
                    return tool.run(query)
                elif tool.name == "assess_differential_diagnosis":
                    # Requires more complex input for a real scenario
                    return tool.run(patient_data=input_data.get("patient_data"), diagnosis=input_data.get("diagnosis"))
                elif tool.name == "recommend_tests":
                    return tool.run(patient_data=input_data.get("patient_data"), potential_diagnoses=input_data.get("potential_diagnoses"))
                elif tool.name == "evaluate_risk":
                    return tool.run(diagnosis=input_data.get("diagnosis"), patient_history=input_data.get("patient_history"))

        # Fallback to LLM if no tool is explicitly mocked/called
        return self.agent_llm.invoke(prompt)

# 1. Patient Data Model
class PatientData(BaseModel):
    patient_id: str
    symptoms: str
    lab_results: Dict[str, Any] = Field(default_factory=dict)
    medical_history: List[str] = Field(default_factory=list)
    imaging_reports: List[str] = Field(default_factory=list)
    potential_diagnoses: List[str] = Field(default_factory=list)
    recommended_tests: List[str] = Field(default_factory=list)

# 2. Mock LLM and Simulated Medical Knowledge Base
mock_llm = MockLLM()

simulated_medical_knowledge_base = {
    "Flu": "Influenza is a common viral infection that can be deadly, especially in high-risk groups. Symptoms include fever, cough, sore throat, muscle aches, and fatigue. Diagnostic tests include viral swabs.",
    "Common Cold": "A viral infection of the nose and throat. Symptoms include runny nose, sneezing, and sore throat. Less severe than flu.",
    "Pneumonia": "An infection that inflames air sacs in one or both lungs, which may fill with fluid. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. Diagnosis often involves chest X-ray.",
    "Appendicitis": "Inflammation of the appendix, a finger-shaped pouch that projects from your colon. Symptoms include sudden pain that begins on the right side of the lower abdomen, nausea, vomiting, and fever. Requires immediate medical attention."
}

# 3. Tools (Mocked Functions and LangChain StructuredTool wrappers)
def analyze_symptoms_func(symptoms: str) -> List[str]:
    # In a real scenario, this would use NLP to extract entities
    print(f"Analyzing symptoms: {symptoms}")
    extracted_symptoms = []
    if "fever" in symptoms.lower(): extracted_symptoms.append("fever")
    if "cough" in symptoms.lower(): extracted_symptoms.append("cough")
    if "fatigue" in symptoms.lower(): extracted_symptoms.append("fatigue")
    if "abdominal pain" in symptoms.lower(): extracted_symptoms.append("abdominal pain")
    if "nausea" in symptoms.lower(): extracted_symptoms.append("nausea")
    return extracted_symptoms

def query_medical_knowledge_base_func(query: str) -> str:
    print(f"Querying medical knowledge base for: {query}")
    for condition, info in simulated_medical_knowledge_base.items():
        if condition.lower() in query.lower():
            return info
    return "No information found for the query."

def assess_differential_diagnosis_func(patient_data: PatientData, diagnosis: str) -> bool:
    print(f"Assessing differential diagnosis: {diagnosis} for patient {patient_data.patient_id}")
    # Simplified logic: check if key symptoms of diagnosis are present in patient data
    if diagnosis == "Flu":
        return all(s in patient_data.symptoms.lower() for s in ["fever", "cough"])
    if diagnosis == "Pneumonia":
        return all(s in patient_data.symptoms.lower() for s in ["cough", "fever"]) and any(r in patient_data.imaging_reports for r in ["chest x-ray shows consolidation"])
    if diagnosis == "Appendicitis":
        return all(s in patient_data.symptoms.lower() for s in ["abdominal pain", "nausea"])
    return False

def recommend_tests_func(patient_data: PatientData, potential_diagnoses: List[str]) -> List[str]:
    print(f"Recommending tests for patient {patient_data.patient_id} with potential diagnoses: {potential_diagnoses}")
    tests = []
    if "Flu" in potential_diagnoses: tests.append("Viral swab")
    if "Pneumonia" in potential_diagnoses: tests.append("Chest X-ray")
    if "Appendicitis" in potential_diagnoses: tests.append("Abdominal Ultrasound")
    return list(set(tests)) # Return unique tests

def evaluate_risk_func(diagnosis: str, patient_history: str) -> str:
    print(f"Evaluating risk for diagnosis: {diagnosis} with history: {patient_history}")
    risk = "Low"
    if "elderly" in patient_history.lower() or "immunocompromised" in patient_history.lower():
        if diagnosis in ["Flu", "Pneumonia"]: risk = "High"
    elif "chronic heart disease" in patient_history.lower() and diagnosis == "Pneumonia":
        risk = "High"
    return f"Risk for {diagnosis}: {risk}"


analyze_symptoms_tool = StructuredTool(
    name="analyze_symptoms",
    description="Analyzes raw patient symptoms to extract key medical entities and concepts.",
    func=analyze_symptoms_func,
)

query_medical_knowledge_base_tool = StructuredTool(
    name="query_medical_knowledge_base",
    description="Retrieves relevant information from the simulated medical knowledge base based on a query.",
    func=query_medical_knowledge_base_func,
)

assess_differential_diagnosis_tool = StructuredTool(
    name="assess_differential_diagnosis",
    description="Evaluates the plausibility of a given diagnosis against patient data and known medical rules.",
    func=assess_differential_diagnosis_func,
)

recommend_tests_tool = StructuredTool(
    name="recommend_tests",
    description="Suggests relevant diagnostic tests based on patient data and potential diagnoses.",
    func=recommend_tests_func,
)

evaluate_risk_tool = StructuredTool(
    name="evaluate_risk",
    description="Assesses potential risks associated with a diagnosis given patient history.",
    func=evaluate_risk_func,
)

# 4. Agents (Mocked LangChain AgentExecutor)
symptom_analyzer_agent = AgentExecutor(
    agent_llm=mock_llm,
    tools=[analyze_symptoms_tool],
    agent_name="SymptomAnalyzerAgent"
)

differential_diagnosis_agent = AgentExecutor(
    agent_llm=mock_llm,
    tools=[query_medical_knowledge_base_tool, assess_differential_diagnosis_tool],
    agent_name="DifferentialDiagnosisAgent"
)

test_recommendation_agent = AgentExecutor(
    agent_llm=mock_llm,
    tools=[recommend_tests_tool],
    agent_name="TestRecommendationAgent"
)

risk_assessment_agent = AgentExecutor(
    agent_llm=mock_llm,
    tools=[evaluate_risk_tool, query_medical_knowledge_base_tool],
    agent_name="RiskAssessmentAgent"
)

# 5. DiagnosticCoordinatorAgent (Orchestrator Logic)
class DiagnosticCoordinatorAgent:
    def __init__(self, llm, agents: Dict[str, AgentExecutor]):
        self.llm = llm
        self.agents = agents

    def run_diagnosis(self, patient_data: PatientData) -> Dict[str, Any]:
        print("\n--- Starting Diagnostic Process ---")
        current_patient_data = patient_data.copy()
        diagnostic_plan = {"steps": []}
        final_diagnosis = "Undetermined"
        recommendations = []
        risk_assessments = {}

        # Step 1: Analyze Symptoms
        print("\n[Coordinator] Step 1: Analyzing symptoms...")
        analyzed_symptoms_output = self.agents["symptom_analyzer"].invoke({
            "input": f"Analyze the following symptoms: {current_patient_data.symptoms}",
            "symptoms": current_patient_data.symptoms
        })
        print(f"[Coordinator] Analyzed symptoms output: {analyzed_symptoms_output}")
        # Mock parsing: assuming analyzed_symptoms_output is a string that can be processed
        if "Identified symptoms:" in analyzed_symptoms_output:
            extracted_symptoms_str = analyzed_symptoms_output.split(": ", 1)[1]
            current_patient_data.medical_history.append(f"Analyzed symptoms: {extracted_symptoms_str}")
            diagnostic_plan["steps"].append({"step": 1, "action": "Symptom Analysis", "result": extracted_symptoms_str})

        # Step 2: Generate Differential Diagnoses
        print("\n[Coordinator] Step 2: Generating differential diagnoses...")
        differential_diagnosis_output = self.agents["differential_diagnosis"].invoke({
            "input": f"Generate differential diagnosis based on patient symptoms: {current_patient_data.symptoms} and history: {current_patient_data.medical_history}",
            "patient_data": current_patient_data # Passing patient_data for context within the mock tool
        })
        print(f"[Coordinator] Differential diagnosis output: {differential_diagnosis_output}")
        # Mock parsing: assuming it returns a string with potential diagnoses
        if "Potential diagnoses:" in differential_diagnosis_output:
            potential_diagnoses_str = differential_diagnosis_output.split(": ", 1)[1]
            current_patient_data.potential_diagnoses = [d.strip() for d in potential_diagnoses_str.split(',')]
            diagnostic_plan["steps"].append({"step": 2, "action": "Differential Diagnosis", "result": current_patient_data.potential_diagnoses})

        # Step 3: Recommend Tests (if diagnoses are ambiguous)
        if len(current_patient_data.potential_diagnoses) > 1:
            print("\n[Coordinator] Step 3: Diagnoses are ambiguous, recommending tests...")
            recommended_tests_output = self.agents["test_recommendation"].invoke({
                "input": f"Recommend tests for patient with potential diagnoses: {current_patient_data.potential_diagnoses}",
                "patient_data": current_patient_data,
                "potential_diagnoses": current_patient_data.potential_diagnoses
            })
            print(f"[Coordinator] Recommended tests output: {recommended_tests_output}")
            if "Recommended tests:" in recommended_tests_output:
                tests_str = recommended_tests_output.split(": ", 1)[1]
                current_patient_data.recommended_tests = [t.strip() for t in tests_str.split(',')]
                diagnostic_plan["steps"].append({"step": 3, "action": "Test Recommendation", "result": current_patient_data.recommended_tests})
                recommendations.extend(current_patient_data.recommended_tests)

        # Simulate receiving new lab results after tests (dynamic adjustment)
        if current_patient_data.recommended_tests:
            print("\n[Coordinator] Simulating new lab results after recommended tests...")
            current_patient_data.lab_results["viral_swab"] = "negative"
            current_patient_data.imaging_reports.append("chest x-ray shows no consolidation") # Mock an inconclusive X-ray initially

            # Re-evaluate diagnoses with new data
            print("\n[Coordinator] Re-evaluating diagnoses with new lab results...")
            # Let's say, after tests, we narrow down or change our primary suspect
            new_potential_diagnoses = []
            for diag in current_patient_data.potential_diagnoses:
                if self.agents["differential_diagnosis"].invoke({
                    "input": f"Assess if {diag} is plausible given updated patient data and history, including negative viral swab and chest x-ray showing no consolidation.",
                    "patient_data": current_patient_data,
                    "diagnosis": diag
                }):
                    new_potential_diagnoses.append(diag)
            current_patient_data.potential_diagnoses = new_potential_diagnoses if new_potential_diagnoses else current_patient_data.potential_diagnoses
            print(f"[Coordinator] Updated potential diagnoses: {current_patient_data.potential_diagnoses}")
            diagnostic_plan["steps"].append({"step": 3.5, "action": "Re-evaluation with new data", "result": current_patient_data.potential_diagnoses})


        # Step 4: Assess Risks for final or leading diagnoses
        if current_patient_data.potential_diagnoses:
            print("\n[Coordinator] Step 4: Assessing risks for potential diagnoses...")
            for diagnosis in current_patient_data.potential_diagnoses:
                risk_output = self.agents["risk_assessment"].invoke({
                    "input": f"Assess risk for {diagnosis} given patient history: {current_patient_data.medical_history}",
                    "diagnosis": diagnosis,
                    "patient_history": " ".join(current_patient_data.medical_history) # Pass relevant history
                })
                print(f"[Coordinator] {risk_output}")
                risk_assessments[diagnosis] = risk_output
            diagnostic_plan["steps"].append({"step": 4, "action": "Risk Assessment", "result": risk_assessments})

        # Final determination (simplified)
        if current_patient_data.potential_diagnoses:
            final_diagnosis = current_patient_data.potential_diagnoses[0] # Take the first as the most plausible after steps
            diagnostic_plan["final_diagnosis"] = final_diagnosis

        recommendations.append(f"Consider {final_diagnosis} as the primary diagnosis based on current assessment.")
        diagnostic_plan["recommendations"] = recommendations

        print("\n--- Diagnostic Process Complete ---")
        return {
            "patient_id": patient_data.patient_id,
            "final_assessment": final_diagnosis,
            "diagnostic_plan": diagnostic_plan,
            "risk_assessments": risk_assessments,
            "recommendations": recommendations
        }

# Main execution block
if __name__ == "__main__":
    # Example Patient Data
    patient_case_1 = PatientData(
        patient_id="P001",
        symptoms="Patient presents with sudden onset of high fever, persistent cough, and severe fatigue for 3 days.",
        medical_history=["No significant past medical history."]
    )

    patient_case_2 = PatientData(
        patient_id="P002",
        symptoms="Patient complains of progressively worsening lower right abdominal pain, nausea, and loss of appetite for 12 hours. Mild fever also noted.",
        medical_history=["Known to have irritable bowel syndrome."]
    )

    # Initialize Coordinator with all agents
    coordinator = DiagnosticCoordinatorAgent(
        llm=mock_llm,
        agents={
            "symptom_analyzer": symptom_analyzer_agent,
            "differential_diagnosis": differential_diagnosis_agent,
            "test_recommendation": test_recommendation_agent,
            "risk_assessment": risk_assessment_agent,
        }
    )

    # Run diagnosis for patient 1
    print("\n========================================")
    print("Diagnosing Patient P001")
    print("========================================")
    result_p001 = coordinator.run_diagnosis(patient_case_1)
    import json
    print("\n--- Final Result for P001 ---")
    print(json.dumps(result_p001, indent=2))

    print("\n\n========================================")
    print("Diagnosing Patient P002")
    print("========================================")
    result_p002 = coordinator.run_diagnosis(patient_case_2)
    print("\n--- Final Result for P002 ---")
    print(json.dumps(result_p002, indent=2))

    # Demonstrate dynamic adjustment / backtracking (simplified)
    print("\n\n========================================")
    print("Demonstrating Dynamic Adjustment / Backtracking (P001 revised)")
    print("========================================")
    # Imagine P001's initial tests were inconclusive for Flu, now new info suggests a more serious condition
    patient_case_1_revised = PatientData(
        patient_id="P001_Revised",
        symptoms="Patient P001 still has fever and cough, but now also reports severe shortness of breath. Initial viral swab negative, but chest X-ray shows diffuse infiltrates.",
        medical_history=["No significant past medical history."],
        lab_results={"viral_swab": "negative"},
        imaging_reports=["chest x-ray shows diffuse infiltrates"]
    )
    result_p001_revised = coordinator.run_diagnosis(patient_case_1_revised)
    print("\n--- Final Result for P001 (Revised) ---")
    print(json.dumps(result_p001_revised, indent=2))


