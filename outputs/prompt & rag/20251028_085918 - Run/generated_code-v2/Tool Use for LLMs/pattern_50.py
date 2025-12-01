import openai
import json
import random
import re
import os
from datetime import datetime

# --- 1. Core LLM Integration ---
class LLMClient:
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical AI assistant capable of reasoning and using provided tools to assist with diagnosis and treatment planning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return ""

# --- 2. Prompt Engineering Module ---
class PromptEngineer:
    def __init__(self):
        self.few_shot_examples = [
            {
                "problem": "A 45-year-old male presents with sudden onset severe headache, stiff neck, and photophobia.",
                "trajectory": """
Rationale: The symptoms suggest a potential neurological emergency, possibly meningitis or subarachnoid hemorrhage. I need to determine the most likely cause by evaluating the symptoms and considering diagnostic steps.
Tool Use: diagnose_symptoms(symptoms=["severe headache", "stiff neck", "photophobia"])
Tool Output: ["Meningitis", "Subarachnoid Hemorrhage", "Migraine"]
Rationale: Given the severity and combination of symptoms, Meningitis or Subarachnoid Hemorrhage are critical considerations. I should recommend immediate medical evaluation and specific diagnostic tests like a lumbar puncture.
Final Answer: Immediate medical evaluation for suspected meningitis or subarachnoid hemorrhage, including a lumbar puncture and CT scan.
"""
            },
            {
                "problem": "A 60-year-old patient with hypertension is prescribed Warfarin and is about to start a course of Ciprofloxacin for a UTI.",
                "trajectory": """
Rationale: Warfarin and Ciprofloxacin can have significant drug interactions. It's crucial to check for potential adverse effects.
Tool Use: check_drug_interactions(drug1="Warfarin", drug2="Ciprofloxacin")
Tool Output: "Severe interaction: Ciprofloxacin can increase the anticoagulant effect of Warfarin, leading to increased bleeding risk. Monitor INR closely and consider alternative antibiotics or dose adjustment of Warfarin."
Rationale: The interaction is severe and increases bleeding risk. The patient's physician must be informed to adjust medication or choose an alternative.
Final Answer: Inform the prescribing physician immediately about the severe drug interaction between Warfarin and Ciprofloxacin, recommending close INR monitoring or an alternative antibiotic for the UTI.
"""
            }
        ]

    def create_prompt(self, medical_problem: str) -> str:
        examples_str = "\n\n".join([ex["trajectory"] for ex in self.few_shot_examples])

        prompt = f"""
Given a medical problem, generate an interactive tool-use trajectory. This trajectory should include a rationale, any tool uses (with parameters), and the simulated tool outputs, leading to a final answer.

Here are a few-shot examples:
{examples_str}

Medical Problem: {medical_problem}

Generate the trajectory:
"""
        return prompt

# --- 3. Tool Emulation Module ---
class MedicalToolEmulator:
    def diagnose_symptoms(self, symptoms: list) -> list:
        if not isinstance(symptoms, list) or not symptoms:
            return ["Error: Invalid symptoms list provided."]
        
        symptoms_lower = [s.lower() for s in symptoms]

        if "severe headache" in symptoms_lower and "stiff neck" in symptoms_lower:
            return ["Meningitis", "Subarachnoid Hemorrhage", "Tension Headache"]
        elif "chest pain" in symptoms_lower and "shortness of breath" in symptoms_lower:
            return ["Myocardial Infarction", "Pulmonary Embolism", "Anxiety Attack"]
        elif "fever" in symptoms_lower and "cough" in symptoms_lower:
            return ["Pneumonia", "Bronchitis", "Common Cold"]
        else:
            return ["General malaise", "Undetermined"]

    def check_drug_interactions(self, drug1: str, drug2: str) -> str:
        if not drug1 or not drug2:
            return "Error: Both drug names must be provided."

        drug1 = drug1.lower()
        drug2 = drug2.lower()

        if ("warfarin" in drug1 and "ciprofloxacin" in drug2) or ("ciprofloxacin" in drug1 and "warfarin" in drug2):
            return "Severe interaction: Ciprofloxacin can increase the anticoagulant effect of Warfarin, leading to increased bleeding risk. Monitor INR closely and consider alternative antibiotics or dose adjustment of Warfarin."
        elif ("ibuprofen" in drug1 and "aspirin" in drug2) or ("aspirin" in drug1 and "ibuprofen" in drug2):
            return "Moderate interaction: Increased risk of gastrointestinal bleeding."
        elif ("metformin" in drug1 and "iodinated contrast" in drug2) or ("iodinated contrast" in drug1 and "metformin" in drug2):
            return "Moderate interaction: Risk of lactic acidosis in patients with renal impairment. Metformin should be temporarily discontinued."
        else:
            return "No significant interaction found for the given drugs."

    def interpret_lab_results(self, lab_data: dict) -> str:
        if not isinstance(lab_data, dict) or not lab_data:
            return "Error: Invalid lab data provided."

        interpretations = []
        if "WBC" in lab_data and lab_data["WBC"] > 11.0:
            interpretations.append("Elevated WBC: Suggests infection or inflammation.")
        elif "WBC" in lab_data and lab_data["WBC"] < 4.0:
            interpretations.append("Low WBC: Suggests bone marrow issues, autoimmune disease, or severe infection.")

        if "Hemoglobin" in lab_data and lab_data["Hemoglobin"] < 12.0:
            interpretations.append("Low Hemoglobin: Suggests anemia.")
        elif "Hemoglobin" in lab_data and lab_data["Hemoglobin"] > 17.0:
            interpretations.append("Elevated Hemoglobin: Suggests polycythemia or dehydration.")

        if "Creatinine" in lab_data and lab_data["Creatinine"] > 1.2:
            interpretations.append("Elevated Creatinine: Suggests kidney dysfunction.")

        if interpretations:
            return "; ".join(interpretations)
        else:
            return "Lab results within normal limits or no specific interpretations available for provided data."

# --- 4. Trajectory Generation Orchestrator ---
class TrajectoryGenerator:
    def __init__(self, llm_client: LLMClient, prompt_engineer: PromptEngineer, tool_emulator: MedicalToolEmulator):
        self.llm_client = llm_client
        self.prompt_engineer = prompt_engineer
        self.tool_emulator = tool_emulator

    def generate_single_trajectory(self, problem: str) -> dict:
        prompt = self.prompt_engineer.create_prompt(problem)
        llm_output = self.llm_client.generate_response(prompt)
        
        trajectory = {
            "problem": problem,
            "full_llm_output": llm_output,
            "steps": []
        }

        # Parse LLM output into rationale, tool_use, tool_output
        segments = re.split(r"(Tool Use:|Tool Output:|Rationale:|Final Answer:)", llm_output)
        segments = [s.strip() for s in segments if s.strip()]

        current_step = {}
        i = 0
        while i < len(segments):
            segment_type = segments[i]
            segment_content = segments[i+1] if i+1 < len(segments) else ""
            
            if segment_type == "Rationale:":
                if "rationale" in current_step: # If a new rationale starts, save previous step
                    trajectory["steps"].append(current_step)
                    current_step = {}
                current_step["rationale"] = segment_content
            elif segment_type == "Tool Use:":
                current_step["tool_use"] = segment_content
                
                # Execute the tool call using eval for simplicity (CAUTION in real systems!)
                tool_call_str = segment_content.replace("diagnose_symptoms", "self.tool_emulator.diagnose_symptoms") \
                                 .replace("check_drug_interactions", "self.tool_emulator.check_drug_interactions") \
                                 .replace("interpret_lab_results", "self.tool_emulator.interpret_lab_results")
                
                simulated_output = "Error: Tool execution failed."
                try:
                    # Safely evaluate tool calls (still risky, but better than direct eval for demonstration)
                    if "self.tool_emulator." in tool_call_str:
                        simulated_output = eval(tool_call_str)
                    else:
                        simulated_output = "Error: Malformed tool call in LLM output."
                except Exception as e:
                    simulated_output = f"Error during tool execution: {e}"
                
                current_step["simulated_tool_output"] = simulated_output
            elif segment_type == "Tool Output:":
                # We use simulated_tool_output, so this LLM generated one is just for context
                current_step["llm_generated_tool_output"] = segment_content
            elif segment_type == "Final Answer:":
                if current_step: # Save any preceding step before final answer
                    trajectory["steps"].append(current_step)
                trajectory["final_answer"] = segment_content
                current_step = {}
            i += 2
        
        if current_step: # Add the last step if it exists
            trajectory["steps"].append(current_step)

        return trajectory

# --- 5. Validation and Filtering Module ---
class TrajectoryValidator:
    def is_medically_correct(self, trajectory: dict) -> bool:
        # Basic rule-based check: Does the final answer mention key terms from the problem or a plausible medical term?
        problem_lower = trajectory["problem"].lower()
        final_answer_lower = trajectory.get("final_answer", "").lower()

        # Simple keyword matching for correctness demonstration
        if "meningitis" in problem_lower and "meningitis" not in final_answer_lower and "subarachnoid hemorrhage" not in final_answer_lower:
            return False
        if "warfarin" in problem_lower and "bleeding risk" not in final_answer_lower:
            return False
        
        # More robust validation would involve NLP, medical ontologies, or expert review.
        # For this example, we'll assume basic coherence based on tool output presence
        has_tool_output_error = self.detect_tool_errors(trajectory)
        if has_tool_output_error:
            return False # If tool errors, it's not correct

        if not final_answer_lower: # Must have a final answer
            return False
        
        # Check if rationale is present for each step
        for step in trajectory.get("steps", []):
            if "rationale" not in step or not step["rationale"].strip():
                return False

        return True

    def detect_tool_errors(self, trajectory: dict) -> bool:
        for step in trajectory.get("steps", []) or []:
            if "simulated_tool_output" in step:
                # Check for explicit error strings from our emulator
                if isinstance(step["simulated_tool_output"], str) and "Error:" in step["simulated_tool_output"]:
                    return True
                # If it's a list, check if any element is an error
                if isinstance(step["simulated_tool_output"], list) and any("Error:" in str(item) for item in step["simulated_tool_output"]):
                    return True
        return False

# --- 6. Data Storage ---
def save_trajectory(filename: str, trajectory_data: dict):
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(trajectory_data) + "\n")
    except Exception as e:
        print(f"Error saving trajectory to {filename}: {e}")

# --- Main Execution --- 
def main():
    # Load OpenAI API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it.")

    llm_client = LLMClient(api_key=api_key)
    prompt_engineer = PromptEngineer()
    tool_emulator = MedicalToolEmulator()
    trajectory_generator = TrajectoryGenerator(llm_client, prompt_engineer, tool_emulator)
    trajectory_validator = TrajectoryValidator()

    output_filename = f"medassist_trajectories_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl"

    medical_problems = [
        "A 72-year-old female presents with sudden onset left-sided weakness and difficulty speaking.",
        "A 30-year-old male with no significant medical history has persistent cough, fatigue, and night sweats for 3 weeks. Lab results: WBC 13.5, Hemoglobin 11.0, Creatinine 0.8.",
        "A patient is on Metformin for diabetes and needs to undergo a CT scan with iodinated contrast. What should be considered?",
        "A 55-year-old patient complains of severe abdominal pain radiating to the back, nausea, and vomiting. Lab results: Amylase 500 U/L, Lipase 450 U/L.",
        "A 25-year-old pregnant woman has symptoms of a urinary tract infection. She is allergic to penicillin. Which antibiotic is safe and effective?", # This problem requires external knowledge or more advanced tools, likely to fail with current simple tools.
        "A patient with a history of heart failure is experiencing sudden weight gain and swollen ankles."
    ]

    generated_count = 0
    successful_count = 0
    MAX_TRAJECTORIES_TO_GENERATE = 5 # Limit for demonstration

    print(f"Starting trajectory generation. Output will be saved to {output_filename}")

    for i, problem in enumerate(medical_problems):
        if generated_count >= MAX_TRAJECTORIES_TO_GENERATE:
            break

        print(f"\n--- Processing Problem {i+1}/{len(medical_problems)} ---")
        print(f"Problem: {problem}")
        
        trajectory = trajectory_generator.generate_single_trajectory(problem)
        generated_count += 1

        is_valid = trajectory_validator.is_medically_correct(trajectory)
        has_errors = trajectory_validator.detect_tool_errors(trajectory)

        print(f"LLM Raw Output:\n{trajectory.get('full_llm_output', 'N/A')}")
        print(f"Trajectory valid: {is_valid}, Tool errors detected: {has_errors}")

        if is_valid and not has_errors:
            successful_count += 1
            save_trajectory(output_filename, trajectory)
            print("Trajectory saved successfully.")
        else:
            print("Trajectory failed validation or contained tool errors, not saved.")

    print(f"\n--- Generation Summary ---")
    print(f"Total problems processed: {len(medical_problems)}")
    print(f"Total trajectories attempted: {generated_count}")
    print(f"Total successful and valid trajectories: {successful_count}")

if __name__ == "__main__":
    main()