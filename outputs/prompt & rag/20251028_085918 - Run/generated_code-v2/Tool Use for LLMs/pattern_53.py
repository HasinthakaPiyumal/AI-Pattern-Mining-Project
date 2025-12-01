import json
import re
import time

class PromptGenerator:
    def __init__(self):
        self.instructions = (
            "You are a medical diagnostic assistant. Your task is to generate structured diagnostic pathways "
            "for patients based on their symptoms. The process involves interleaved steps of natural language "
            "rationale, calling diagnostic tools (PROGRAM), and interpreting tool outputs (TOOL_OUTPUT). "
            "Formulate a final diagnosis and a preliminary treatment plan."
            "\n\n" + \
            "Follow this interleaved format:\n"
            "RATIONALE: [Your reasoning for the next step]\n"
            "PROGRAM: [tool_name(param1='value1', param2='value2')]\n"
            "TOOL_OUTPUT: [Output from the tool]\n"
            "RATIONALE: [Further reasoning]\n"
            "... until a final diagnosis.\n\n"
        )
        self.few_shot_examples = [
            {
                "patient_symptoms": "Patient presents with persistent cough, fever, and shortness of breath for 3 days.",
                "trajectory": [
                    "RATIONALE: The patient's symptoms suggest a respiratory infection. A complete blood count (CBC) and a chest X-ray are initial diagnostic steps to assess inflammation and lung involvement.",
                    "PROGRAM: order_blood_test(test_type='CBC')",
                    "TOOL_OUTPUT: {'CBC_results': {'WBC': '15.2 x 10^9/L (High)', 'Neutrophils': '80% (High)'}}",
                    "RATIONALE: The elevated white blood cell count and neutrophils indicate a bacterial infection. A chest X-ray is crucial to evaluate for pneumonia.",
                    "PROGRAM: interpret_xray(body_part='chest', focus='lungs')",
                    "TOOL_OUTPUT: {'Xray_findings': 'Right lower lobe infiltrate consistent with pneumonia.'}",
                    "RATIONALE: The X-ray confirms pneumonia. The next step is to initiate antibiotic treatment and monitor the patient's response.",
                    "FINAL_DIAGNOSIS: Bacterial Pneumonia (Right Lower Lobe)",
                    "TREATMENT_PLAN: Administer broad-spectrum antibiotics (e.g., Azithromycin + Ceftriaxone), provide supportive care (oxygen if needed, fever reducers), and monitor respiratory status."
                ]
            },
            {
                "patient_symptoms": "Patient reports severe abdominal pain localized to the right lower quadrant, nausea, and loss of appetite for 12 hours.",
                "trajectory": [
                    "RATIONALE: The acute onset of right lower quadrant pain, nausea, and anorexia are highly suggestive of appendicitis. A physical examination and a CT scan of the abdomen are warranted.",
                    "PROGRAM: order_imaging_scan(scan_type='CT scan', body_part='abdomen', contrast='true')",
                    "TOOL_OUTPUT: {'CT_findings': 'Dilated appendix with wall thickening and periappendiceal fat stranding, consistent with acute appendicitis.'}",
                    "RATIONALE: The CT scan findings strongly confirm acute appendicitis. Urgent surgical consultation is necessary.",
                    "PROGRAM: consult_specialist(specialty='general surgery')",
                    "TOOL_OUTPUT: {'Consultation_summary': 'General surgeon confirms the diagnosis and recommends immediate appendectomy.'}",
                    "FINAL_DIAGNOSIS: Acute Appendicitis",
                    "TREATMENT_PLAN: Urgent appendectomy, pre-operative antibiotics, pain management, and post-operative care."
                ]
            }
        ]

    def generate_prompt(self, patient_symptoms):
        prompt = self.instructions
        for example in self.few_shot_examples:
            prompt += f"\n### Example Patient Symptoms:\n{example['patient_symptoms']}\n"
            for step in example['trajectory']:
                prompt += f"{step}\n"

        prompt += f"\n### New Patient Symptoms:\n{patient_symptoms}\n"
        prompt += "RATIONALE: " # Start the LLM's generation
        return prompt

class LLMClient:
    def generate_response(self, prompt):
        # This is a simulated LLM response for demonstration purposes.
        # In a real application, this would call an actual LLM API (e.g., OpenAI, Gemini).
        # The simulation will provide a hardcoded interleaved response.

        # Simulate a delay for LLM processing
        time.sleep(1)

        if "persistent cough, fever, and shortness of breath" in prompt:
            return (
                "RATIONALE: The patient's symptoms are highly suggestive of a respiratory infection, possibly viral or bacterial. "
                "Given the severity and progression, a CBC and a basic metabolic panel (BMP) are good initial steps to assess for systemic inflammation and kidney function, respectively. "
                "We should also consider a rapid flu test given the season.\n"
                "PROGRAM: order_blood_test(test_type='CBC', additional_tests=['BMP', 'Rapid Flu Test'])\n"
                "TOOL_OUTPUT: {'CBC_results': {'WBC': '12.5 x 10^9/L (High)', 'Lymphocytes': '15% (Low)', 'Neutrophils': '75% (High)'}, 'BMP_results': {'Creatinine': '0.9 mg/dL (Normal)'}, 'Rapid_Flu_Test': 'Negative'}\n"
                "RATIONALE: The elevated WBC and neutrophilia still point towards a bacterial component, even with a negative flu test. Lymphopenia could be seen in viral infections, but the overall picture leans bacterial. A chest X-ray is crucial to rule out pneumonia given the shortness of breath and cough.\n"
                "PROGRAM: interpret_xray(body_part='chest', focus='lungs')\n"
                "TOOL_OUTPUT: {'Xray_findings': 'Diffuse interstitial infiltrates in both lungs, suggestive of atypical pneumonia or viral pneumonitis.'}\n"
                "RATIONALE: The X-ray findings suggest an atypical pneumonia or viral pneumonitis rather than classic bacterial pneumonia, which often presents with lobar consolidation. Given this, broad-spectrum antibiotics for atypical organisms (e.g., macrolides) and supportive care are indicated. Further viral panel might be considered if initial treatment fails or symptoms worsen.\n"
                "FINAL_DIAGNOSIS: Atypical Pneumonia / Viral Pneumonitis\n"
                "TREATMENT_PLAN: Initiate Azithromycin or Doxycycline, provide supportive care (oxygen if O2 sat < 92%, antipyretics), monitor respiratory status and consider viral panel if no improvement in 48-72 hours."
            )
        elif "severe abdominal pain localized to the right lower quadrant" in prompt:
            return (
                "RATIONALE: The patient's presentation is classic for acute appendicitis. While a physical exam is primary, imaging is often used to confirm the diagnosis and rule out other causes. A bedside ultrasound or a CT scan would be appropriate.\n"
                "PROGRAM: order_imaging_scan(scan_type='Ultrasound', body_part='abdomen', focus='right lower quadrant')\n"
                "TOOL_OUTPUT: {'Ultrasound_findings': 'Non-compressible, dilated appendix measuring 9mm in diameter with surrounding fluid. Suggestive of appendicitis.'}\n"
                "RATIONALE: The ultrasound findings are highly consistent with acute appendicitis. Immediate surgical consultation is required.\n"
                "PROGRAM: consult_specialist(specialty='general surgery')\n"
                "TOOL_OUTPUT: {'Consultation_summary': 'General surgeon evaluated patient, agrees with ultrasound findings, and plans for emergent appendectomy.'}\n"
                "FINAL_DIAGNOSIS: Acute Appendicitis\n"
                "TREATMENT_PLAN: Emergent appendectomy, pre-operative intravenous antibiotics, pain control, and fluid management."
            )
        else:
            return (
                "RATIONALE: The patient's symptoms are vague and require further investigation. Let's start with a comprehensive metabolic panel (CMP) and a physical examination.\n"
                "PROGRAM: order_blood_test(test_type='CMP')\n"
                "TOOL_OUTPUT: {'CMP_results': {'Glucose': '95 mg/dL (Normal)', 'Sodium': '140 mEq/L (Normal)', 'Potassium': '4.0 mEq/L (Normal)'}}\n"
                "RATIONALE: The CMP results are normal, which doesn't narrow down the diagnosis. We need more information. Let's consider a detailed symptom history and a physical examination.\n"
                "FINAL_DIAGNOSIS: Undetermined - Further Investigation Needed\n"
                "TREATMENT_PLAN: Detailed history taking, comprehensive physical examination, and consider specialist consultation based on new findings."
            )

class ToolSimulator:
    def order_blood_test(self, test_type, **kwargs):
        print(f"Simulating blood test: {test_type} with additional params {kwargs}")
        if test_type == 'CBC':
            return {'CBC_results': {'WBC': '9.0 x 10^9/L (Normal)', 'RBC': '5.0 x 10^12/L (Normal)'}}
        elif test_type == 'CMP':
            return {'CMP_results': {'Glucose': '100 mg/dL (Normal)', 'Sodium': '138 mEq/L (Normal)'}}
        elif test_type == 'Thyroid Panel':
            return {'Thyroid_results': {'TSH': '2.5 mIU/L (Normal)'}}
        else:
            return {'Blood_Test_Results': f'Results for {test_type}: Normal'}

    def interpret_xray(self, body_part, focus):
        print(f"Simulating X-ray interpretation: {body_part}, focusing on {focus}")
        if body_part == 'chest' and focus == 'lungs':
            return {'Xray_findings': 'No acute cardiopulmonary pathology.'}
        else:
            return {'Xray_findings': 'Normal findings for specified area.'}

    def order_imaging_scan(self, scan_type, body_part, contrast='false', **kwargs):
        print(f"Simulating imaging scan: {scan_type} of {body_part}, contrast={contrast}, additional params {kwargs}")
        if scan_type == 'CT scan' and body_part == 'head':
            return {'CT_findings': 'No acute intracranial abnormality.'}
        elif scan_type == 'Ultrasound' and body_part == 'abdomen':
            return {'Ultrasound_findings': 'No significant abnormalities detected.'}
        else:
            return {'Imaging_Scan_Results': f'{scan_type} of {body_part}: Normal'}

    def consult_specialist(self, specialty):
        print(f"Simulating consultation with {specialty} specialist")
        return {'Consultation_summary': f'{specialty} specialist reviewed the case and provided recommendations.'}

class MedicalDiagnosticAssistant:
    def __init__(self):
        self.prompt_generator = PromptGenerator()
        self.llm_client = LLMClient()
        self.tool_simulator = ToolSimulator()

    def generate_diagnostic_trajectory(self, patient_symptoms, max_steps=10):
        trajectory = []
        current_prompt_context = patient_symptoms # Initial context for LLM

        for step_num in range(max_steps):
            full_prompt = self.prompt_generator.generate_prompt(current_prompt_context)
            # In a real scenario, we'd append previous trajectory steps to current_prompt_context
            # to maintain conversational history for the LLM. For this simulation, we'll keep it simple.

            llm_response_text = self.llm_client.generate_response(full_prompt)
            trajectory.extend(llm_response_text.strip().split('\n'))

            # Check for final diagnosis to stop early
            if "FINAL_DIAGNOSIS:" in llm_response_text:
                print("Final diagnosis reached, stopping trajectory generation.")
                break

            # Parse LLM response for PROGRAM calls
            program_match = re.search(r"PROGRAM: (\w+)\((.*?)\)", llm_response_text)
            if program_match:
                tool_name = program_match.group(1)
                args_str = program_match.group(2)
                kwargs = {}
                for arg_pair in args_str.split(','):
                    if '=' in arg_pair:
                        key, val = arg_pair.split('=', 1)
                        kwargs[key.strip()] = val.strip().strip("'\"")

                print(f"Detected PROGRAM call: {tool_name} with args {kwargs}")
                tool_output = None
                try:
                    tool_func = getattr(self.tool_simulator, tool_name)
                    tool_output = tool_func(**kwargs)
                except AttributeError:
                    tool_output = {"error": f"Tool '{tool_name}' not found in simulator."}
                except Exception as e:
                    tool_output = {"error": f"Error executing tool '{tool_name}': {str(e)}"}

                trajectory.append(f"TOOL_OUTPUT: {tool_output}")
                # Update context for the next LLM call based on tool output
                current_prompt_context = f"Patient symptoms: {patient_symptoms}\nPrevious Trajectory:\n{'\n'.join(trajectory)}"
            else:
                print("No PROGRAM call detected in LLM response. Continuing or stopping...")
                # If LLM doesn't call a tool but also doesn't provide a final diagnosis, it might be stuck or done.
                # For this simulation, if no program is called and no final diagnosis, we stop.
                if "FINAL_DIAGNOSIS:" not in llm_response_text:
                    break

        # Clean up trajectory for final output
        cleaned_trajectory = []
        llm_output_lines = []
        for line in trajectory:
            if line.startswith("RATIONALE:") or line.startswith("PROGRAM:") or line.startswith("TOOL_OUTPUT:") or line.startswith("FINAL_DIAGNOSIS:") or line.startswith("TREATMENT_PLAN:"):
                cleaned_trajectory.append(line)
            elif line.strip() and not llm_output_lines and not any(tag in line for tag in ["RATIONALE", "PROGRAM", "TOOL_OUTPUT", "FINAL_DIAGNOSIS", "TREATMENT_PLAN"]):
                # This handles the initial LLM response that might not perfectly start with RATIONALE
                # but is part of the first LLM thought process. This is a simplification.
                cleaned_trajectory.append(line)
            else:
                # Capture intermediate LLM thoughts before a structured tag
                llm_output_lines.append(line)
        if llm_output_lines:
            # If there are accumulated LLM output lines not starting with a tag, prepend them as a rationale if appropriate
            if not cleaned_trajectory or not cleaned_trajectory[-1].startswith("RATIONALE:"):
                cleaned_trajectory.insert(0, f"RATIONALE: {' '.join(llm_output_lines).strip()}")

        # Further refine: remove duplicate tool outputs that might come from LLM and simulator both
        final_trajectory = []
        for i, line in enumerate(cleaned_trajectory):
            if line.startswith("TOOL_OUTPUT:") and i > 0 and cleaned_trajectory[i-1].startswith("PROGRAM:"):
                # If a TOOL_OUTPUT directly follows a PROGRAM, it's the simulator's. Skip LLM's generated one if present.
                # This logic assumes the LLM's 'TOOL_OUTPUT' is just a placeholder or might be slightly off.
                # For this simulation, we trust the simulator's output.
                pass # The simulator's TOOL_OUTPUT will be appended explicitly.
            else:
                final_trajectory.append(line)
        
        # A more robust approach for parsing and integration:
        processed_trajectory = []
        raw_response_parts = llm_response_text.strip().split('\n')
        for part in raw_response_parts:
            processed_trajectory.append(part.strip())
        
        return processed_trajectory

def save_trajectory_to_json(trajectory, filename):
    with open(filename, 'w') as f:
        json.dump(trajectory, f, indent=4)
    print(f"Trajectory saved to {filename}")

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    # Sample Patient Case 1
    patient_symptoms_1 = "Patient presents with persistent cough, fever, and shortness of breath for 3 days."
    print(f"\n--- Generating trajectory for Patient 1: {patient_symptoms_1} ---")
    trajectory_1 = assistant.generate_diagnostic_trajectory(patient_symptoms_1)
    print("\nGenerated Trajectory 1:")
    for step in trajectory_1:
        print(step)
    save_trajectory_to_json(trajectory_1, "diagnostic_trajectory_1.json")

    # Sample Patient Case 2
    patient_symptoms_2 = "Patient reports severe abdominal pain localized to the right lower quadrant, nausea, and loss of appetite for 12 hours."
    print(f"\n--- Generating trajectory for Patient 2: {patient_symptoms_2} ---")
    trajectory_2 = assistant.generate_diagnostic_trajectory(patient_symptoms_2)
    print("\nGenerated Trajectory 2:")
    for step in trajectory_2:
        print(step)
    save_trajectory_to_json(trajectory_2, "diagnostic_trajectory_2.json")

    # Sample Patient Case 3 (vague symptoms to test general handling)
    patient_symptoms_3 = "Patient complains of general malaise and fatigue for several weeks."
    print(f"\n--- Generating trajectory for Patient 3: {patient_symptoms_3} ---")
    trajectory_3 = assistant.generate_diagnostic_trajectory(patient_symptoms_3)
    print("\nGenerated Trajectory 3:")
    for step in trajectory_3:
        print(step)
    save_trajectory_to_json(trajectory_3, "diagnostic_trajectory_3.json")
