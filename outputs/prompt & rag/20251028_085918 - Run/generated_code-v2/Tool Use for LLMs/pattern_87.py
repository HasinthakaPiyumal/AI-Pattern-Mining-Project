import json

# --- Simulated Medical Tools ---
class MockDrugDatabase:
    """Simulates a drug database lookup."""
    def lookup_drug(self, drug_name: str) -> str:
        if "paracetamol" in drug_name.lower():
            return "Paracetamol: Analgesic, antipyretic. Common side effects: mild GI upset. Max dose: 4g/day."
        elif "amoxicillin" in drug_name.lower():
            return "Amoxicillin: Antibiotic (penicillin class). Common side effects: nausea, diarrhea, rash. Contraindicated in penicillin allergy."
        return f"Drug '{drug_name}' not found or information limited."

class MockLabResultInterpreter:
    """Simulates interpreting lab results."""
    def interpret_blood_test(self, test_results: dict) -> str:
        if "hemoglobin" in test_results and test_results["hemoglobin"] < 12:
            return "Hemoglobin is low, suggesting anemia."
        if "glucose" in test_results and test_results["glucose"] > 120:
            return "Elevated glucose, consistent with hyperglycemia. Further investigation needed for diabetes."
        return "Lab results appear within normal limits for specified parameters."

class MockDiagnosticAlgorithm:
    """Simulates running a diagnostic algorithm based on symptoms and history."""
    def run_diagnosis(self, symptoms: list, medical_history: list) -> str:
        symptoms_lower = [s.lower() for s in symptoms]
        history_lower = [h.lower() for h in medical_history]

        if "fever" in symptoms_lower and "cough" in symptoms_lower and "fatigue" in symptoms_lower:
            return "Possible Viral Infection (e.g., Influenza or common cold). Recommend rest, hydration, symptomatic relief."
        if "chest pain" in symptoms_lower and "shortness of breath" in symptoms_lower and "history of heart disease" in history_lower:
            return "High suspicion for cardiac event. Immediate medical attention required."
        if "abdominal pain" in symptoms_lower and "nausea" in symptoms_lower:
            return "Gastrointestinal distress. Consider food poisoning or gastroenteritis. Recommend supportive care and monitoring."
        return "Insufficient information or unclear pattern for a definitive diagnosis."

# --- Prompt Engineering Module ---
class PromptGenerator:
    """Generates initial prompts for the LLM based on patient cases and few-shot examples."""
    def generate_initial_prompt(self, patient_case: dict, few_shot_examples: list) -> str:
        prompt = f"You are a medical diagnostic AI assistant. Your goal is to use provided tools to diagnose patients and recommend next steps.\n\n"
        prompt += "Here are some examples of how to interact with the tools:\n"
        for i, example in enumerate(few_shot_examples):
            prompt += f"Example {i+1}:\nPatient: {example['patient']}\nRationale: {example['rationale']}\nProgram: {example['program']}\nOutput: {example['output']}\n\n"

        prompt += f"Now, analyze the following patient case:\n"
        prompt += f"Patient ID: {patient_case['id']}\n"
        prompt += f"Symptoms: {', '.join(patient_case['symptoms'])}\n"
        prompt += f"Medical History: {', '.join(patient_case['medical_history'])}\n"
        prompt += f"Initial Query: {patient_case['initial_query']}\n\n"
        prompt += "Rationale: "
        return prompt

# --- LLM Interaction Module (Simulated) ---
class LLMTrajectorySynthesizer:
    """Simulates an LLM interacting with medical tools to generate diagnostic trajectories."""
    def __init__(self, drug_db: MockDrugDatabase, lab_interpreter: MockLabResultInterpreter, diagnostic_algo: MockDiagnosticAlgorithm):
        self.drug_db = drug_db
        self.lab_interpreter = lab_interpreter
        self.diagnostic_algo = diagnostic_algo
        self.available_tools = {
            "drug_database_lookup": self.drug_db.lookup_drug,
            "lab_result_interpreter_interpret_blood_test": self.lab_interpreter.interpret_blood_test,
            "diagnostic_algorithm_run_diagnosis": self.diagnostic_algo.run_diagnosis,
        }

    def _execute_tool_program(self, program_str: str) -> str:
        """Executes a simulated tool program string and returns its output."""
        try:
            # This is a simplified parser for demonstration. In reality, a robust parser
            # would handle various function calls and arguments.
            if program_str.startswith("drug_database_lookup("):
                drug_name = program_str.split("(")[1].split(")")[0].strip("'").strip("\"")
                return self.drug_db.lookup_drug(drug_name)
            elif program_str.startswith("lab_result_interpreter_interpret_blood_test("):
                # This expects a simple dict string, which is fragile. For demo purposes.
                args_str = program_str.split("(")[1].split(")")[0]
                # A safer approach would be to use ast.literal_eval if inputs are trusted
                # or a more robust custom parser.
                test_results_dict = eval(args_str.replace("test_results=", "")) # DANGER: eval() can be dangerous with untrusted input
                return self.lab_interpreter.interpret_blood_test(test_results_dict)
            elif program_str.startswith("diagnostic_algorithm_run_diagnosis("):
                args_str = program_str.split("(")[1].split(")")[0]
                # Example: symptoms=['fever'], medical_history=[]
                symptoms_str = args_str.split("symptoms=")[1].split(", medical_history=")[0]
                history_str = args_str.split("medical_history=")[1]
                symptoms_list = eval(symptoms_str) # DANGER: eval()
                history_list = eval(history_str) # DANGER: eval()
                return self.diagnostic_algo.run_diagnosis(symptoms_list, history_list)
            elif program_str.startswith("FINAL_ANSWER:"):
                return program_str.replace("FINAL_ANSWER: ", "").strip()
            else:
                return f"Tool execution error: Unrecognized program format '{program_str}'"
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    def generate_trajectory(self, initial_prompt: str, max_steps: int = 5) -> dict:
        """Simulates the LLM's step-by-step diagnostic process."""
        trajectory_steps = []
        current_prompt_context = initial_prompt
        print(f"\n--- Simulating LLM for prompt ---\n{initial_prompt[:200]}...")

        # This simulates a simplified greedy decoding strategy.
        # In a real scenario, this would involve actual LLM API calls and parsing.
        for step_num in range(max_steps):
            # Simulate LLM generating rationale and program based on current context
            # This logic is hardcoded for demonstration; a real LLM would generate this dynamically.
            rationale = ""
            program = ""
            output = ""

            # Simplified logic to determine next step
            if "chest pain" in current_prompt_context.lower() and "heart disease" in current_prompt_context.lower() and not any("cardiac event" in s.get("output", "").lower() for s in trajectory_steps):
                rationale = "Patient reports chest pain and has a history of heart disease. High suspicion for cardiac event. Must use the diagnostic algorithm immediately."
                program = "diagnostic_algorithm_run_diagnosis(symptoms=['chest pain', 'shortness of breath'], medical_history=['history of heart disease'])"
            elif "fever" in current_prompt_context.lower() and "cough" in current_prompt_context.lower() and not any("viral infection" in s.get("output", "").lower() for s in trajectory_steps):
                rationale = "Patient presents with fever and cough. Considering a viral infection. Using the diagnostic algorithm."
                program = "diagnostic_algorithm_run_diagnosis(symptoms=['fever', 'cough', 'fatigue'], medical_history=['none'])"
            elif "abdominal pain" in current_prompt_context.lower() and "nausea" in current_prompt_context.lower() and not any("gastrointestinal distress" in s.get("output", "").lower() for s in trajectory_steps):
                 rationale = "Patient has abdominal pain and nausea. Investigating potential gastrointestinal issues with the diagnostic algorithm."
                 program = "diagnostic_algorithm_run_diagnosis(symptoms=['abdominal pain', 'nausea'], medical_history=['none'])"
            elif "hemoglobin low" in current_prompt_context.lower() or ("low hemoglobin" in current_prompt_context.lower() and any("viral infection" in s.get("output", "").lower() for s in trajectory_steps)):
                rationale = "Previous diagnosis suggests viral infection. Now focusing on low hemoglobin. Interpreting blood test results."
                program = "lab_result_interpreter_interpret_blood_test(test_results={'hemoglobin': 10})}"
            elif step_num == max_steps - 1 or not program: # Final step or if no specific tool use logic triggered
                rationale = "Based on the information gathered so far, I will provide a final diagnosis and recommendation."
                # Attempt to extract a diagnosis from previous steps or default
                final_diagnosis_candidate = "Unclear diagnosis. Recommend further consultation."
                for s in trajectory_steps:
                    if "diagnostic_algorithm_run_diagnosis" in s["program"] and "Output" in s: # Check for the 'Output' key
                        diag_output = s["output"]
                        if "viral infection" in diag_output.lower():
                            final_diagnosis_candidate = "Possible Viral Infection. Symptomatic treatment."
                            break
                        elif "cardiac event" in diag_output.lower():
                            final_diagnosis_candidate = "High suspicion for cardiac event. Immediate emergency medical attention."
                            break
                        elif "gastrointestinal distress" in diag_output.lower():
                            final_diagnosis_candidate = "Gastrointestinal distress. Supportive care."
                            break


                program = f"FINAL_ANSWER: {final_diagnosis_candidate}"
            else:
                # Default fallback if no specific rule matches for a tool call
                rationale = "Continuing analysis. Using general diagnostic algorithm for initial assessment."
                program = "diagnostic_algorithm_run_diagnosis(symptoms=['general malaise'], medical_history=[])"

            output = self._execute_tool_program(program)

            step_data = {"rationale": rationale, "program": program, "output": output}
            trajectory_steps.append(step_data)

            # Update context for next iteration (simplified)
            current_prompt_context += f"\nRationale: {rationale}\nProgram: {program}\nOutput: {output}\n"
            
            if program.startswith("FINAL_ANSWER:"):
                break # End trajectory generation if final answer is reached

        return {"trajectory": trajectory_steps}

# --- Trajectory Curation and Filtering Module ---
class TrajectoryCurationModule:
    """Filters and evaluates generated trajectories for correctness and errors."""
    def filter_trajectory(self, trajectory: dict, patient_case: dict) -> dict:
        is_correct = False
        has_tool_error = False
        final_diagnosis = "N/A"

        # Iterate through trajectory steps to find final answer and check for errors
        for step in trajectory["trajectory"]:
            if "FINAL_ANSWER:" in step["program"]:
                final_diagnosis = step["program"].replace("FINAL_ANSWER: ", "").strip()
                # Simplified correctness check based on keywords in final diagnosis and patient symptoms
                if ("viral infection" in final_diagnosis.lower() and "fever" in [s.lower() for s in patient_case["symptoms"]]) or \
                   ("cardiac event" in final_diagnosis.lower() and "chest pain" in [s.lower() for s in patient_case["symptoms"]]) or \
                   ("gastrointestinal distress" in final_diagnosis.lower() and "abdominal pain" in [s.lower() for s in patient_case["symptoms"]]):
                    is_correct = True

            # Check for generic tool execution errors
            if "Tool execution error:" in step["output"]:
                has_tool_error = True
                break
            # Check for specific 'not found' or 'insufficient' messages as potential soft errors/limitations
            if "not found" in step["output"].lower() or "insufficient information" in step["output"].lower():
                # This indicates a limitation of the tool/data, not necessarily a 'tool-use error' by the LLM.
                # For strict 'tool-use error' detection, look for malformed program syntax or uncallable tools.
                pass

        return {
            "trajectory_steps": trajectory["trajectory"],
            "is_correct": is_correct,
            "has_tool_error": has_tool_error,
            "final_diagnosis": final_diagnosis
        }

# --- Main Application Logic ---
def main():
    # 1. Initialize Simulated Medical Tools
    drug_db = MockDrugDatabase()
    lab_interpreter = MockLabResultInterpreter()
    diagnostic_algo = MockDiagnosticAlgorithm()

    # 2. Initialize Core Modules
    prompt_generator = PromptGenerator()
    llm_synthesizer = LLMTrajectorySynthesizer(drug_db, lab_interpreter, diagnostic_algo)
    curation_module = TrajectoryCurationModule()

    # 3. Define Few-shot Examples for Prompting (demonstrating interleaved format)
    few_shot_examples = [
        {
            "patient": "A 45-year-old male with persistent cough and fatigue.",
            "rationale": "Initial assessment points to respiratory issues. Running a general diagnosis.",
            "program": "diagnostic_algorithm_run_diagnosis(symptoms=['cough', 'fatigue'], medical_history=[])",
            "output": diagnostic_algo.run_diagnosis(symptoms=['cough', 'fatigue'], medical_history=[])
        },
        {
            "patient": "A 60-year-old female with sudden chest pain and history of hypertension.",
            "rationale": "Urgent case. High suspicion for cardiac event due to symptoms and history. Directly using the diagnostic algorithm for critical conditions.",
            "program": "diagnostic_algorithm_run_diagnosis(symptoms=['chest pain', 'shortness of breath'], medical_history=['hypertension', 'history of heart disease'])",
            "output": diagnostic_algo.run_diagnosis(symptoms=['chest pain', 'shortness of breath'], medical_history=['hypertension', 'history of heart disease'])
        }
    ]

    # 4. Define Patient Cases for Trajectory Generation
    patient_cases = [
        {
            "id": "P001",
            "symptoms": ["fever", "cough", "fatigue"],
            "medical_history": ["none"],
            "initial_query": "What is the likely diagnosis and recommended next steps?"
        },
        {
            "id": "P002",
            "symptoms": ["chest pain", "shortness of breath"],
            "medical_history": ["history of heart disease", "hypertension"],
            "initial_query": "Please provide an urgent diagnostic assessment."
        },
        {
            "id": "P003",
            "symptoms": ["headache", "dizziness"],
            "medical_history": ["migraines"],
            "initial_query": "What could be causing these symptoms and what is the treatment?"
        },
        {
            "id": "P004",
            "symptoms": ["abdominal pain", "nausea"],
            "medical_history": ["none"],
            "initial_query": "Investigate potential causes."
        },
        {
            "id": "P005",
            "symptoms": ["fever", "low hemoglobin"], # Introduce a symptom that might trigger lab interpreter
            "medical_history": ["none"],
            "initial_query": "Patient has fever and recent blood test showed low hemoglobin. Diagnose and recommend."
        }
    ]

    # 5. Generate and Curate Trajectories for each patient case
    curated_dataset = []
    print("\n--- Starting Trajectory Generation and Curation ---")
    for i, case in enumerate(patient_cases):
        print(f"\nProcessing Patient Case {case['id']} ({i+1}/{len(patient_cases)})...")
        initial_prompt = prompt_generator.generate_initial_prompt(case, few_shot_examples)
        generated_trajectory = llm_synthesizer.generate_trajectory(initial_prompt)
        filtered_trajectory = curation_module.filter_trajectory(generated_trajectory, case)

        curated_dataset.append({
            "patient_case": case,
            "curated_trajectory": filtered_trajectory
        })
        print(f"Finished processing Patient Case {case['id']}. Correct: {filtered_trajectory['is_correct']}, Errors: {filtered_trajectory['has_tool_error']}")

    # 6. Output Results: Save the curated dataset to a JSONL file
    output_filename = "medisim_tutor_dataset.jsonl"
    with open(output_filename, "w") as f:
        for item in curated_dataset:
            f.write(json.dumps(item) + "\n")

    print(f"\n--- Synthetic Dataset Generation Complete ---")
    print(f"Generated {len(curated_dataset)} trajectories and saved to '{output_filename}'")
    
    # Display an example of the generated data
    if curated_dataset:
        print("\nExample Curated Data (First Entry):")
        print(json.dumps(curated_dataset[0], indent=2))

if __name__ == "__main__":
    main()