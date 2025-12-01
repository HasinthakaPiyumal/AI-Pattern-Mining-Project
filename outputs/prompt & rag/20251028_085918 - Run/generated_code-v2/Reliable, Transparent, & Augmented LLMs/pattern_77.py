import json
from typing import Dict, List, Union
from pydantic import BaseModel, Field
import gradio as gr


class ReasoningLog:
    def __init__(self):
        self.log = []

    def add_entry(self, log_type: str, details: Dict):
        self.log.append({"type": log_type, **details})

    def get_full_log(self) -> List[Dict]:
        return self.log

    def generate_explanation(self) -> str:
        explanation = []
        for entry in self.log:
            if entry["type"] == "tool_selection":
                explanation.append(f"**Tool Selection:** Chosen '{entry['tool_name']}' because: {entry['rationale']}")
            elif entry["type"] == "parameter_extraction":
                explanation.append(f"**Parameter Extraction:** Extracted parameters {entry['parameters']} for '{entry['tool_name']}'.")
            elif entry["type"] == "tool_execution":
                explanation.append(f"**Tool Execution:** '{entry['tool_name']}' returned: {json.dumps(entry['output'], indent=2)}")
            elif entry["type"] == "output_integration":
                explanation.append(f"**Output Integration:** Integrated results from '{entry['tool_name']}'. Key findings: {entry['key_findings']}. Impact on reasoning: {entry['impact_on_reasoning']}")
            elif entry["type"] == "diagnosis_formulation":
                explanation.append(f"**Diagnosis Formulation:** Arrived at diagnosis '{entry['diagnosis']}' based on synthesized evidence.")
        return "\n\n".join(explanation)


# Pydantic Models for Tool Inputs/Outputs
class LabTestInput(BaseModel):
    test_name: str = Field(description="Name of the lab test")
    values: Dict[str, Union[float, str]] = Field(description="Dictionary of test parameters and their values")


class LabTestOutput(BaseModel):
    interpretation: str = Field(description="Interpretation of the lab test results")
    key_findings: List[str] = Field(description="List of key findings")


class ImagingInput(BaseModel):
    modality: str = Field(description="Imaging modality (e.g., X-ray, MRI, CT scan)")
    findings_description: str = Field(description="Description of findings from the imaging report")


class ImagingOutput(BaseModel):
    interpretation: str = Field(description="Interpretation of the imaging findings")
    potential_conditions: List[str] = Field(description="List of potential conditions suggested by imaging")


class PatientHistoryInput(BaseModel):
    patient_id: str = Field(description="Unique patient identifier")
    query: str = Field(description="Specific query about patient history")


class PatientHistoryOutput(BaseModel):
    history_data: str = Field(description="Relevant patient history data")


class KnowledgeBaseInput(BaseModel):
    topic: str = Field(description="Topic to query in the medical knowledge base")


class KnowledgeBaseOutput(BaseModel):
    info: str = Field(description="Information retrieved from the knowledge base")


# Simulated Medical Tools
class BaseTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, input_data: BaseModel) -> BaseModel:
        raise NotImplementedError


class LabTestAnalysisTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="LabTestAnalysisTool",
            description="Analyzes simulated lab test results (e.g., blood panel, urinalysis)."
        )

    def run(self, input_data: LabTestInput) -> LabTestOutput:
        interpretation = f"Analyzing {input_data.test_name} with values: {input_data.values}."
        key_findings = []

        if "glucose" in input_data.values and input_data.values["glucose"] > 120:
            key_findings.append("Elevated glucose levels")
            interpretation += " Suggests potential hyperglycemia."
        if "white_blood_cells" in input_data.values and input_data.values["white_blood_cells"] > 10:
            key_findings.append("High white blood cell count")
            interpretation += " Indicates inflammation or infection."
        if not key_findings:
            key_findings.append("No significant abnormalities detected.")

        return LabTestOutput(interpretation=interpretation, key_findings=key_findings)


class ImagingInterpretationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="ImagingInterpretationTool",
            description="Interprets findings from medical imaging reports (e.g., X-ray, MRI, CT scan)."
        )

    def run(self, input_data: ImagingInput) -> ImagingOutput:
        interpretation = f"Interpreting {input_data.modality} findings: {input_data.findings_description}."
        potential_conditions = []

        if "fracture" in input_data.findings_description.lower():
            potential_conditions.append("Bone Fracture")
            interpretation += " Suggests a bone fracture."
        if "pneumonia" in input_data.findings_description.lower() or "consolidation" in input_data.findings_description.lower():
            potential_conditions.append("Pneumonia")
            interpretation += " Consistent with pneumonia."
        if not potential_conditions:
            potential_conditions.append("No specific conditions immediately apparent.")

        return ImagingOutput(interpretation=interpretation, potential_conditions=potential_conditions)


class PatientHistoryDBTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="PatientHistoryDBTool",
            description="Retrieves relevant patient history information from a simulated database."
        )

    def run(self, input_data: PatientHistoryInput) -> PatientHistoryOutput:
        # Mock patient history data
        mock_history = {
            "patient_123": "Known history of diabetes and hypertension. Allergic to penicillin.",
            "patient_456": "No significant past medical history. Presents with recent fever."
        }
        history_data = mock_history.get(input_data.patient_id, "No specific history found for this patient ID.")
        return PatientHistoryOutput(history_data=history_data)


class MedicalKnowledgeBaseTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="MedicalKnowledgeBaseTool",
            description="Queries a simulated medical knowledge base for information on diseases, symptoms, treatments."
        )

    def run(self, input_data: KnowledgeBaseInput) -> KnowledgeBaseOutput:
        # Mock knowledge base data
        mock_kb = {
            "diabetes": "Chronic metabolic disease characterized by high blood sugar levels. Symptoms include increased thirst, frequent urination, and unexplained weight loss.",
            "pneumonia": "Infection that inflames air sacs in one or both lungs, which may fill with fluid. Symptoms include cough with phlegm, fever, chills, and difficulty breathing.",
            "hypertension": "A common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease."
        }
        info = mock_kb.get(input_data.topic.lower(), f"Information on '{input_data.topic}' not found in knowledge base.")
        return KnowledgeBaseOutput(info=info)


class MedicalDiagnosticAgent:
    def __init__(self):
        self.tools = {
            "LabTestAnalysisTool": LabTestAnalysisTool(),
            "ImagingInterpretationTool": ImagingInterpretationTool(),
            "PatientHistoryDBTool": PatientHistoryDBTool(),
            "MedicalKnowledgeBaseTool": MedicalKnowledgeBaseTool(),
        }
        self.reasoning_log = ReasoningLog()

    def diagnose(self, patient_symptoms: str, patient_id: str = None) -> Dict:
        self.reasoning_log = ReasoningLog() # Reset log for new diagnosis

        diagnosis = "Undetermined"
        explanation = []

        self.reasoning_log.add_entry(
            "initial_input",
            {"symptoms": patient_symptoms, "patient_id": patient_id, "rationale": "Starting diagnostic process with patient symptoms."}
        )

        # Step 1: Check patient history if ID is provided
        if patient_id:
            self.reasoning_log.add_entry(
                "tool_selection",
                {"tool_name": "PatientHistoryDBTool", "rationale": "Patient ID provided, checking for relevant medical history to inform diagnosis."}
            )
            history_input = PatientHistoryInput(patient_id=patient_id, query="general medical history")
            self.reasoning_log.add_entry(
                "parameter_extraction",
                {"tool_name": "PatientHistoryDBTool", "parameters": history_input.model_dump_json()}
            )
            history_output = self.tools["PatientHistoryDBTool"].run(history_input)
            self.reasoning_log.add_entry(
                "tool_execution",
                {"tool_name": "PatientHistoryDBTool", "output": history_output.model_dump()}
            )
            self.reasoning_log.add_entry(
                "output_integration",
                {
                    "tool_name": "PatientHistoryDBTool",
                    "key_findings": [history_output.history_data],
                    "impact_on_reasoning": f"Identified relevant patient history: {history_output.history_data}."
                }
            )
            explanation.append(f"**Patient History:** {history_output.history_data}")

        # Step 2: Simulate initial LLM thought process based on symptoms
        if "fever" in patient_symptoms.lower() and "cough" in patient_symptoms.lower():
            diagnosis_hypothesis = "respiratory infection"
            self.reasoning_log.add_entry(
                "llm_thought",
                {
                    "thought": "Fever and cough often indicate a respiratory infection. Will consider using LabTestAnalysisTool or ImagingInterpretationTool to confirm.",
                    "hypothesis": diagnosis_hypothesis
                }
            )

            # Step 3: Use Lab Test Tool
            self.reasoning_log.add_entry(
                "tool_selection",
                {"tool_name": "LabTestAnalysisTool", "rationale": f"Symptoms '{patient_symptoms}' suggest {diagnosis_hypothesis}, requiring lab confirmation. Checking for inflammation markers."}
            )
            lab_test_input = LabTestInput(test_name="CBC", values={"white_blood_cells": 12.5, "glucose": 90})
            self.reasoning_log.add_entry(
                "parameter_extraction",
                {"tool_name": "LabTestAnalysisTool", "parameters": lab_test_input.model_dump_json()}
            )
            lab_test_output = self.tools["LabTestAnalysisTool"].run(lab_test_input)
            self.reasoning_log.add_entry(
                "tool_execution",
                {"tool_name": "LabTestAnalysisTool", "output": lab_test_output.model_dump()}
            )
            self.reasoning_log.add_entry(
                "output_integration",
                {
                    "tool_name": "LabTestAnalysisTool",
                    "key_findings": lab_test_output.key_findings,
                    "impact_on_reasoning": f"Lab tests show {', '.join(lab_test_output.key_findings)}. This supports/refines the {diagnosis_hypothesis} hypothesis."
                }
            )
            explanation.append(f"**Lab Test Results:** {lab_test_output.interpretation} (Key findings: {', '.join(lab_test_output.key_findings)}) ")

            if "High white blood cell count" in lab_test_output.key_findings:
                diagnosis = "Bacterial Respiratory Infection (e.g., Pneumonia)"
                # Step 4: Use Imaging Tool for further confirmation
                self.reasoning_log.add_entry(
                    "tool_selection",
                    {"tool_name": "ImagingInterpretationTool", "rationale": "High WBC count and respiratory symptoms strongly suggest pneumonia, using imaging to confirm pulmonary consolidation."}
                )
                imaging_input = ImagingInput(modality="Chest X-ray", findings_description="Left lower lobe consolidation with air bronchograms.")
                self.reasoning_log.add_entry(
                    "parameter_extraction",
                    {"tool_name": "ImagingInterpretationTool", "parameters": imaging_input.model_dump_json()}
                )
                imaging_output = self.tools["ImagingInterpretationTool"].run(imaging_input)
                self.reasoning_log.add_entry(
                    "tool_execution",
                    {"tool_name": "ImagingInterpretationTool", "output": imaging_output.model_dump()}
                )
                self.reasoning_log.add_entry(
                    "output_integration",
                    {
                        "tool_name": "ImagingInterpretationTool",
                        "key_findings": imaging_output.potential_conditions,
                        "impact_on_reasoning": f"Imaging confirms {', '.join(imaging_output.potential_conditions)}, further supporting the pneumonia diagnosis."
                    }
                )
                explanation.append(f"**Imaging Results:** {imaging_output.interpretation} (Potential conditions: {', '.join(imaging_output.potential_conditions)}) ")
                diagnosis = f"Confirmed Pneumonia (likely bacterial)"

                # Step 5: Consult Knowledge Base for Pneumonia
                self.reasoning_log.add_entry(
                    "tool_selection",
                    {"tool_name": "MedicalKnowledgeBaseTool", "rationale": "After confirming pneumonia, consulting knowledge base for general information on the condition."}
                )
                kb_input = KnowledgeBaseInput(topic="pneumonia")
                self.reasoning_log.add_entry(
                    "parameter_extraction",
                    {"tool_name": "MedicalKnowledgeBaseTool", "parameters": kb_input.model_dump_json()}
                )
                kb_output = self.tools["MedicalKnowledgeBaseTool"].run(kb_input)
                self.reasoning_log.add_entry(
                    "tool_execution",
                    {"tool_name": "MedicalKnowledgeBaseTool", "output": kb_output.model_dump()}
                )
                self.reasoning_log.add_entry(
                    "output_integration",
                    {
                        "tool_name": "MedicalKnowledgeBaseTool",
                        "key_findings": [kb_output.info],
                        "impact_on_reasoning": "Integrated general information about pneumonia from the knowledge base."
                    }
                )
                explanation.append(f"**Medical Knowledge Base:** {kb_output.info}")

        elif "headache" in patient_symptoms.lower() and "nausea" in patient_symptoms.lower():
            diagnosis = "Migraine (likely)"
            self.reasoning_log.add_entry(
                "llm_thought",
                {
                    "thought": "Headache and nausea are common migraine symptoms. No specific tests immediately required based on this alone for initial diagnosis.",
                    "hypothesis": diagnosis
                }
            )
            self.reasoning_log.add_entry(
                "diagnosis_formulation",
                {"diagnosis": diagnosis, "rationale": "Symptoms highly suggestive of migraine."}
            )
            explanation.append(f"**Symptom Analysis:** Headache and nausea are classic symptoms of migraine. No specific lab or imaging tests are typically performed for an initial migraine diagnosis based solely on these symptoms.")

        else:
            diagnosis = "Further investigation needed"
            self.reasoning_log.add_entry(
                "llm_thought",
                {
                    "thought": "Symptoms are not clear-cut for a specific condition with available tools. Further investigation or more detailed symptom input required.",
                    "hypothesis": diagnosis
                }
            )
            self.reasoning_log.add_entry(
                "diagnosis_formulation",
                {"diagnosis": diagnosis, "rationale": "Insufficient information to provide a precise diagnosis at this time."}
            )
            explanation.append(f"**Symptom Analysis:** The provided symptoms are non-specific, and more information or tests would be required to narrow down a diagnosis.")

        final_explanation = self.reasoning_log.generate_explanation()
        return {"diagnosis": diagnosis, "detailed_reasoning": final_explanation}


def predict_diagnosis(symptoms, patient_id):
    agent = MedicalDiagnosticAgent()
    result = agent.diagnose(symptoms, patient_id)
    return result["diagnosis"], result["detailed_reasoning"]


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# Transparent Medical Diagnostic AI Assistant")
        gr.Markdown(
            "This AI assistant provides a diagnosis along with its step-by-step reasoning, "
            "including tool selection, parameter extraction, and output integration. "
            "Note: This is a simulated environment for demonstration purposes."
        )

        with gr.Row():
            symptoms_input = gr.Textbox(label="Patient Symptoms", placeholder="e.g., fever, cough, shortness of breath")
            patient_id_input = gr.Textbox(label="Patient ID (Optional)", placeholder="e.g., patient_123")

        diagnose_button = gr.Button("Get Diagnosis and Reasoning")

        diagnosis_output = gr.Textbox(label="Diagnosis", interactive=False)
        reasoning_output = gr.Markdown(label="Detailed Reasoning", interactive=False)

        diagnose_button.click(
            fn=predict_diagnosis,
            inputs=[symptoms_input, patient_id_input],
            outputs=[diagnosis_output, reasoning_output],
        )

    demo.launch()
