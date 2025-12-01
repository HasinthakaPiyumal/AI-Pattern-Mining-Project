import json
import re
from prompt_builder import build_diagnostic_prompt
from medical_tools import check_symptoms, order_lab_tests, consult_drug_database

# A simple dictionary to simulate expected outcomes for filtering purposes
# In a real scenario, this would be a more robust evaluation system or human annotation.
simulated_ground_truths = {
    "Patient reports severe abdominal pain, fever, and vomiting for 24 hours.": {"diagnosis": "Appendicitis (requires immediate attention)", "keywords": ["abdominal pain", "fever", "vomiting", "appendicitis"]},
    "Patient has persistent cough, mild fever, and fatigue.": {"diagnosis": "Common Cold/Flu", "keywords": ["cough", "fever", "fatigue", "cold", "flu"]},
    "Patient is on Warfarin and presents with a new prescription for Ibuprofen.": {"diagnosis": "Drug Interaction", "keywords": ["warfarin", "ibuprofen", "interaction", "bleeding risk"]},
    "Patient reports severe headache and nausea.": {"diagnosis": "Migraine", "keywords": ["headache", "nausea", "migraine"]},
    "Patient has fever and persistent cough.": {"diagnosis": "Common Cold/Flu", "keywords": ["fever", "cough", "cold", "flu"]},
    "Patient reports chest pain and shortness of breath.": {"diagnosis": "Cardiac Issue", "keywords": ["chest pain", "shortness of breath", "cardiac"]},
}

def simulate_llm_response(prompt: str) -> str:
    """Simulates an LLM generating a diagnostic trajectory based on a prompt."""
    # This is a highly simplified simulation. A real LLM would generate this dynamically.
    # For demonstration, we'll try to match some common patterns and produce a plausible output.

    # Example 1: Abdominal pain scenario
    if "severe abdominal pain, fever, and vomiting" in prompt:
        return """
Rationale: The patient's symptoms strongly suggest an acute abdominal condition that requires urgent investigation. I should first use the symptom checker to get an initial assessment and then consider lab tests.
Tool Call: print(check_symptoms("severe abdominal pain, fever, vomiting"))
Tool Output: {'tool_name': 'check_symptoms', 'input': 'severe abdominal pain, fever, vomiting', 'output': ['Appendicitis (requires immediate attention)']}
Rationale: Given the high-priority condition indicated by the symptom checker, further investigation is critical. I'll recommend immediate medical consultation.
Diagnosis/Recommendation: Immediate medical consultation is required due to suspected Appendicitis.
"""
    # Example 2: Common cold/flu scenario
    elif "persistent cough, mild fever, and fatigue" in prompt:
        return """
Rationale: The patient's symptoms are consistent with a common respiratory infection. I will use the symptom checker to narrow down potential conditions.
Tool Call: print(check_symptoms("persistent cough, mild fever, fatigue"))
Tool Output: {'tool_name': 'check_symptoms', 'input': 'persistent cough, mild fever, fatigue', 'output': ['Common Cold/Flu']}
Rationale: The symptom checker confirms a common cold or flu. Rest and hydration are typically recommended.
Diagnosis/Recommendation: Based on the symptoms, a common cold or flu is likely. Recommend rest and hydration.
"""
    # Example 3: Drug interaction scenario
    elif "Warfarin and presents with a new prescription for Ibuprofen" in prompt:
        return """
Rationale: It's crucial to check for potential drug interactions between Warfarin and Ibuprofen, as they can have significant effects.
Tool Call: print(consult_drug_database(["Warfarin", "Ibuprofen"]))
Tool Output: {'tool_name': 'consult_drug_database', 'input': ['Warfarin', 'Ibuprofen'], 'output': ['Increased bleeding risk with Ibuprofen and Warfarin.']}
Rationale: The drug database indicates a significant interaction. This information needs to be relayed to the patient and prescribing physician.
Diagnosis/Recommendation: There is an increased bleeding risk. Advise patient and doctor about the interaction and suggest an alternative pain reliever if possible.
"""
    # Default / fallback response
    return """
Rationale: I will begin by checking the patient's symptoms to identify potential conditions.
Tool Call: print(check_symptoms("general symptoms"))
Tool Output: {'tool_name': 'check_symptoms', 'input': 'general symptoms', 'output': ['Observation needed, no clear immediate condition.']}
Diagnosis/Recommendation: Further observation and potentially more detailed information are needed.
"""

def execute_tool_call(tool_call_str: str) -> dict:
    """Parses and executes a tool call string, returning its output."""
    try:
        # Extract function name and arguments using regex
        match = re.match(r"print\((\w+)\((\[.*?\]|\".*?\"|.*?)\)\)", tool_call_str)
        if not match:
            raise ValueError("Invalid tool call format.")

        tool_name = match.group(1)
        args_str = match.group(2)

        # Safely evaluate arguments if they are lists or strings
        if args_str.startswith('[') and args_str.endswith(']'):
            args = json.loads(args_str.replace('\'', '"')) # Convert single quotes to double for json.loads
        elif args_str.startswith('"') and args_str.endswith('"'):
            args = args_str.strip('"')
        else:
            args = args_str # Assume simple string for now, could be more complex

        # Map tool name to actual function
        tool_functions = {
            "check_symptoms": check_symptoms,
            "order_lab_tests": order_lab_tests,
            "consult_drug_database": consult_drug_database,
        }

        if tool_name not in tool_functions:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Execute the tool function
        if isinstance(args, list):
            tool_output = tool_functions[tool_name](args)
        else:
            tool_output = tool_functions[tool_name](args)

        return tool_output
    except Exception as e:
        return {"tool_name": "ERROR", "input": tool_call_str, "output": f"Error during tool execution: {e}"}

def filter_trajectory(trajectory: dict, patient_case: str) -> bool:
    """Filters a generated trajectory based on correctness and absence of tool errors."""
    # Check for tool execution errors
    for step in trajectory.get("steps", []):
        if "ERROR" in step.get("tool_output", {}).get("tool_name", ""):
            print(f"  Filtering: Tool error found in trajectory for {patient_case}")
            return False

    # Check for presence of final output
    if "final_output" not in trajectory or not trajectory["final_output"].strip():
        print(f"  Filtering: Missing final output in trajectory for {patient_case}")
        return False

    # Simplified correctness check (heuristic based on keywords)
    ground_truth = simulated_ground_truths.get(patient_case)
    if ground_truth:
        final_output_lower = trajectory["final_output"].lower()
        if not any(keyword in final_output_lower for keyword in ground_truth["keywords"]):
            print(f"  Filtering: Final output for {patient_case} does not contain expected keywords.")
            return False
    else:
        # If no ground truth, we can't verify correctness automatically, so pass (or require human review)
        pass

    return True

def curate_medical_trajectories(patient_cases: list, few_shot_examples: list) -> list:
    """Generates and curates interactive tool-use trajectories for medical diagnostic problems."""
    curated_dataset = []

    for i, patient_case in enumerate(patient_cases):
        print(f"Processing patient case {i+1}/{len(patient_cases)}: {patient_case[:50]}...")
        prompt = build_diagnostic_prompt(patient_case, few_shot_examples)
        llm_response = simulate_llm_response(prompt)

        # Parse LLM's response into a trajectory
        trajectory = {"patient_case": patient_case, "steps": [], "final_output": ""}
        lines = llm_response.split('\n')
        current_step = {}
        for line in lines:
            if line.startswith("Rationale:"):
                if current_step: # Save previous step if exists
                    trajectory["steps"].append(current_step)
                current_step = {"rationale": line[len("Rationale:"):].strip()}
            elif line.startswith("Tool Call:"):
                current_step["tool_call"] = line[len("Tool Call:"):].strip()
                # Execute the tool call
                tool_output = execute_tool_call(current_step["tool_call"])
                current_step["tool_output"] = tool_output
            elif line.startswith("Tool Output:"):
                # This line is usually redundant if we execute tools, but capture for consistency
                try:
                    current_step["tool_output"] = json.loads(line[len("Tool Output:"):].strip().replace('\'', '"'))
                except json.JSONDecodeError:
                    pass # Will be overwritten by actual execution output if present
            elif line.startswith("Diagnosis/Recommendation:"):
                trajectory["final_output"] = line[len("Diagnosis/Recommendation:"):].strip()

        if current_step:
            trajectory["steps"].append(current_step)

        # Filter the generated trajectory
        if filter_trajectory(trajectory, patient_case):
            curated_dataset.append(trajectory)
            print(f"  Curated: Trajectory for {patient_case[:50]}... added to dataset.")
        else:
            print(f"  Rejected: Trajectory for {patient_case[:50]}... due to filtering criteria.")

    return curated_dataset

if __name__ == "__main__":
    # Sample patient cases
    sample_patient_cases = [
        "Patient reports severe abdominal pain, fever, and vomiting for 24 hours.",
        "Patient has persistent cough, mild fever, and fatigue.",
        "Patient is on Warfarin and presents with a new prescription for Ibuprofen.",
        "Patient reports chest pain and shortness of breath.",
        "Patient has a rash and mild itching."
    ]

    # Few-shot examples (re-using from prompt_builder for consistency)
    few_shot_examples_for_curation = [
        {
            "patient_case": "Patient presents with persistent cough, mild fever, and fatigue.",
            "rationale": "The symptoms suggest a possible respiratory infection. I will use the symptom checker to get initial potential conditions.",
            "tool_call": "print(check_symptoms(\"persistent cough, mild fever, fatigue\"))",
            "tool_output": "{'tool_name': 'check_symptoms', 'input': 'persistent cough, mild fever, fatigue', 'output': ['Common Cold/Flu']}",
            "final_output": "Based on the symptoms, a common cold or flu is likely. Recommend rest and hydration."
        },
        {
            "patient_case": "Patient is on Warfarin and presents with a new prescription for Ibuprofen.",
            "rationale": "I need to check for potential drug interactions between Warfarin and Ibuprofen.",
            "tool_call": "print(consult_drug_database([\"Warfarin\", \"Ibuprofen\"]))",
            "tool_output": "{'tool_name': 'consult_drug_database', 'input': ['Warfarin', 'Ibuprofen'], 'output': ['Increased bleeding risk with Ibuprofen and Warfarin.']}",
            "final_output": "There is an increased bleeding risk. Advise patient and doctor about the interaction and suggest an alternative pain reliever if possible."
        }
    ]

    print("--- Starting Trajectory Curation ---")
    curated_data = curate_medical_trajectories(sample_patient_cases, few_shot_examples_for_curation)

    output_filename = "toracorpus_medical_diagnostic.json"
    with open(output_filename, "w") as f:
        json.dump(curated_data, f, indent=2)

    print(f"\n--- Curation Complete ---")
    print(f"Generated {len(curated_data)} high-quality trajectories. Saved to {output_filename}")