import logging
from typing import List, Dict, Any
from agent import MedicalDiagnosticAgent
from tools import PatientRecordTool, MedicalLiteratureTool, LabResultTool
from utils import MockLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_diagnostic_session(patient_id: str, initial_symptoms: List[str]):
    logging.info(f"\n--- Starting Diagnostic Session for Patient ID: {patient_id} ---")

    # Initialize mock LLM and tools
    llm = MockLLM()
    tools = [
        PatientRecordTool(db={'patient123': {'age': 45, 'gender': 'male', 'history': 'hypertension', 'medications': 'lisinopril'}}),
        MedicalLiteratureTool(kb={'hypertension': 'common disease affecting blood pressure', 'headache': 'symptom with many causes'}),
        LabResultTool(results={'patient123': {'blood_pressure': '140/90', 'cholesterol': 'high'}})
    ]

    agent = MedicalDiagnosticAgent(llm=llm, tools=tools)

    current_diagnosis = ""
    diagnostic_confidence = 0.0
    iterations = 0
    max_iterations = 5

    # Initial reasoning step
    logging.info(f"Initial symptoms: {', '.join(initial_symptoms)}")
    agent.reason(context={'symptoms': initial_symptoms})

    while diagnostic_confidence < 0.9 and iterations < max_iterations:
        iterations += 1
        logging.info(f"\n--- Iteration {iterations} ---")

        # Agent uses tools based on current reasoning
        action_plan = agent.reason_for_tools(context={'diagnosis': current_diagnosis, 'symptoms': initial_symptoms})
        for action in action_plan:
            tool_name = action.get("tool_name")
            tool_input = action.get("tool_input")
            if tool_name and tool_input:
                tool_output = agent.use_tool(tool_name, tool_input)
                logging.info(f"Tool '{tool_name}' output: {tool_output}")
                agent.update_context({'tool_output': {tool_name: tool_output}})

        # Agent refines diagnosis and self-corrects
        new_diagnosis, confidence_change = agent.self_correct_and_diagnose(initial_symptoms)
        current_diagnosis = new_diagnosis
        diagnostic_confidence += confidence_change # Simulate confidence update
        logging.info(f"Current Diagnosis: {current_diagnosis}, Confidence: {diagnostic_confidence:.2f}")

        # Simulate real-time data or expert feedback
        if iterations == 2:
            feedback = {'conflicting_info': 'patient reports new symptom: severe dizziness'}
            logging.info(f"Simulating expert/real-time feedback: {feedback}")
            agent.update_context({'feedback': feedback})
            agent.resolve_conflict(feedback)

        # Evaluate termination conditions
        if agent.evaluate_diagnosis(diagnostic_confidence):
            logging.info("Diagnosis considered stable or confident enough.")
            break

    logging.info(f"\n--- Diagnostic Session Completed for Patient ID: {patient_id} ---")
    logging.info(f"Final Diagnosis: {current_diagnosis}")
    logging.info(f"Final Confidence: {diagnostic_confidence:.2f}")
    if diagnostic_confidence < 0.9:
        logging.warning("Diagnosis not fully confident, consider escalation to human specialist.")

if __name__ == "__main__":
    run_diagnostic_session("patient123", ["headache", "fatigue"])
