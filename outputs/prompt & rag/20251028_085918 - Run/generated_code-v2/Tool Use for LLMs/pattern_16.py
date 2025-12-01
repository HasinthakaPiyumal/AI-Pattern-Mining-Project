import json
import re

class SimulatedMedicalToolInterface:
    def perform_lab_test(self, test_name):
        if test_name == "blood_count":
            return {"tool_name": "blood_count", "result": {"WBC": 7.5, "RBC": 4.8, "Platelets": 250}}
        elif test_name == "glucose":
            return {"tool_name": "glucose", "result": {"fasting_glucose": 110}}
        return {"tool_name": test_name, "result": "Normal"}

    def order_imaging(self, imaging_type, body_part):
        if imaging_type == "MRI" and body_part == "brain":
            return {"tool_name": "MRI_brain", "result": "MRI of brain shows no significant abnormalities."}
        return {"tool_name": imaging_type, "result": f"{imaging_type} of {body_part} shows no significant abnormalities."}

    def prescribe_medication(self, medication_name, dosage):
        return {"tool_name": "medication_prescription", "result": f"Prescribed {medication_name} {dosage}."}

class PromptEngineeringModule:
    def __init__(self):
        self.system_instructions = "You are a highly intelligent AI assistant specializing in clinical decision-making. Your task is to generate realistic and structured clinical trajectories, including rationales for actions and simulated tool usage. The output must strictly follow the specified interleaved format."
        self.format_instructions = """
Please respond in the following interleaved format:
RATIONALE: <Natural language explanation for the next step>
PROGRAM: <tool_name>(<param1>=<value1>, <param2>=<value2>, ...)
TOOL_OUTPUT: <Output from the simulated tool call>
NEXT_RATIONALE: <Natural language explanation for the subsequent step, or conclusion if the trajectory ends>
If no tool is needed, omit the PROGRAM and TOOL_OUTPUT lines.
"""
        self.few_shot_examples = [
            """
RATIONALE: The patient presents with symptoms consistent with a bacterial infection. A complete blood count is necessary to assess inflammation markers.
PROGRAM: perform_lab_test(test_name="blood_count")
TOOL_OUTPUT: {"tool_name": "blood_count", "result": {"WBC": 15.2, "RBC": 4.5, "Platelets": 280}}
NEXT_RATIONALE: The elevated white blood cell count confirms an inflammatory process. Given the symptoms, we should consider a broad-spectrum antibiotic.
PROGRAM: prescribe_medication(medication_name="Amoxicillin", dosage="500mg TDS")
TOOL_OUTPUT: {"tool_name": "medication_prescription", "result": "Prescribed Amoxicillin 500mg TDS."}
NEXT_RATIONALE: Patient advised to take medication and follow up in 3 days.
            """,
            """
RATIONALE: Patient reports persistent headache and visual disturbances. An MRI of the brain is warranted to rule out neurological causes.
PROGRAM: order_imaging(imaging_type="MRI", body_part="brain")
TOOL_OUTPUT: {"tool_name": "MRI_brain", "result": "MRI of brain shows no significant abnormalities."}
NEXT_RATIONALE: Given the normal MRI, consider ophthalmological consultation for visual symptoms and symptomatic treatment for headache.
            """
        ]

    def construct_prompt(self, patient_history, full_interaction_history_so_far=""):
        prompt_parts = [
            self.system_instructions,
            self.format_instructions,
            "Here are a few examples to guide your generation:",
        ]
        prompt_parts.extend(self.few_shot_examples)
        prompt_parts.append("\n" + "="*50 + "\n")
        prompt_parts.append(f"Patient History: {patient_history}\n")
        if full_interaction_history_so_far:
            prompt_parts.append(f"CURRENT INTERACTION TRAJECTORY (continue from here):\n{full_interaction_history_so_far}\n")
        prompt_parts.append("Please generate the next steps based on the patient's history and the ongoing interaction trajectory, following the specified format.")
        return "\n".join(prompt_parts)

class LLMIntegration:
    def __init__(self):
        self._mock_responses = [
            """
RATIONALE: The patient presents with symptoms consistent with a suspected infection. Initial investigation with a complete blood count is appropriate.
PROGRAM: perform_lab_test(test_name="blood_count")
NEXT_RATIONALE: Await lab results to inform further management.
            """,
            """
RATIONALE: The blood count indicates an elevated white blood cell count, suggesting an active infection. A broad-spectrum antibiotic is indicated.
PROGRAM: prescribe_medication(medication_name="Amoxicillin", dosage="500mg TDS")
NEXT_RATIONALE: Instruct patient on medication and schedule a follow-up.
            """,
            """
RATIONALE: Patient reports persistent dizziness and fatigue. We should rule out common metabolic causes.
PROGRAM: perform_lab_test(test_name="glucose")
NEXT_RATIONALE: Evaluate glucose levels for potential diabetes or hypoglycemia.
            """,
            """
RATIONALE: Fasting glucose is within normal limits. Consider neurological causes for dizziness if symptoms persist.
PROGRAM: order_imaging(imaging_type="CT", body_part="head")
NEXT_RATIONALE: Await CT results.
            """,
            """
RATIONALE: CT head is normal. Recommend patient to follow up with a neurologist for further evaluation.
NEXT_RATIONALE: End of simulation.
            """
        ]
        self._response_idx = 0

    def generate_response(self, prompt):
        if self._response_idx < len(self._mock_responses):
            response = self._mock_responses[self._response_idx]
            self._response_idx += 1
            return response
        else:
            return "RATIONALE: Trajectory concluded. NEXT_RATIONALE: End of simulation."

    def reset_responses(self):
        self._response_idx = 0

class TrajectoryGenerationEngine:
    def __init__(self, llm_integration, prompt_engineer, tool_interface):
        self.llm = llm_integration
        self.prompt_engineer = prompt_engineer
        self.tool_interface = tool_interface
        self.trajectory_log = []

    def parse_llm_output(self, llm_output):
        rationale_match = re.search(r"RATIONALE: (.*?)\n", llm_output, re.DOTALL)
        program_match = re.search(r"PROGRAM: (.*?)\n", llm_output)
        next_rationale_match = re.search(r"NEXT_RATIONALE: (.*)", llm_output, re.DOTALL)

        rationale = rationale_match.group(1).strip() if rationale_match else None
        program_str = program_match.group(1).strip() if program_match else None
        next_rationale = next_rationale_match.group(1).strip() if next_rationale_match else None

        return rationale, program_str, next_rationale

    def execute_program(self, program_str):
        if not program_str:
            return None

        try:
            match = re.match(r"(\w+)\((.*)\)", program_str)
            if match:
                tool_name = match.group(1)
                args_str = match.group(2)
                args = {}
                for arg_pair in args_str.split(','):
                    if '=' in arg_pair:
                        key, value = arg_pair.split('=', 1)
                        try:
                            value = json.loads(value.strip())
                        except json.JSONDecodeError:
                            value = value.strip().strip('"')
                        args[key.strip()] = value

                tool_method = getattr(self.tool_interface, tool_name, None)
                if tool_method and callable(tool_method):
                    return tool_method(**args)
                else:
                    return {"error": f"Tool '{tool_name}' not found or not callable."}
            else:
                return {"error": "Invalid program format."}
        except Exception as e:
            return {"error": f"Error executing program: {e}"}

    def generate_trajectory(self, patient_history, max_steps=5):
        self.trajectory_log = []
        self.llm.reset_responses()
        full_interaction_history = ""

        for step in range(max_steps):
            prompt = self.prompt_engineer.construct_prompt(patient_history, full_interaction_history)
            llm_output = self.llm.generate_response(prompt)
            
            rationale, program_str, next_rationale = self.parse_llm_output(llm_output)

            step_log = {
                "step": step + 1,
                "rationale": rationale,
                "program": program_str,
                "tool_output": None,
                "next_rationale": next_rationale
            }
            
            current_step_segment = f"RATIONALE: {rationale}\n"
            if program_str:
                current_step_segment += f"PROGRAM: {program_str}\n"
                tool_result = self.execute_program(program_str)
                step_log["tool_output"] = tool_result
                current_step_segment += f"TOOL_OUTPUT: {json.dumps(tool_result)}\n"
            
            if next_rationale:
                current_step_segment += f"NEXT_RATIONALE: {next_rationale}\n"
            
            full_interaction_history += current_step_segment
            self.trajectory_log.append(step_log)

            if "End of simulation" in (next_rationale if next_rationale else "") or not next_rationale:
                break
            
        return self.trajectory_log