import json

class PromptEngineer:
    def __init__(self):
        self.interleaved_format_instructions = (
            "Generate a structured diagnostic protocol following this interleaved format:\n"
            "Rationale: [natural language rationale for the step]\n"
            "Program: [medical_tool_function_call(args)]\n"
            "Tool Output: [simulated or actual output from the tool]\n"
            "Next Rationale: [natural language rationale for the next step or conclusion]\n"
            "Ensure each step is clearly delineated and logically follows the previous one."
        )

    def _format_tool_definitions(self, tool_definitions):
        formatted_tools = "Available Medical Tools:\n"
        for tool_name, description in tool_definitions.items():
            formatted_tools += f"- {tool_name}: {description}\n"
        return formatted_tools

    def _format_few_shot_examples(self, few_shot_examples):
        formatted_examples = "\nFew-Shot Examples:\n"
        for i, example in enumerate(few_shot_examples):
            formatted_examples += f"\nExample {i+1}:\n"
            formatted_examples += f"Rationale: {example['rationale']}\n"
            formatted_examples += f"Program: {example['program']}\n"
            formatted_examples += f"Tool Output: {example['tool_output']}\n"
            formatted_examples += f"Next Rationale: {example['next_rationale']}\n"
        return formatted_examples

    def construct_prompt(self, patient_case_description, tool_definitions, few_shot_examples):
        prompt_parts = []
        prompt_parts.append("### Patient Case Description ###\n")
        prompt_parts.append(patient_case_description)
        prompt_parts.append("\n\n### Instructions ###\n")
        prompt_parts.append(self.interleaved_format_instructions)
        prompt_parts.append("\n\n")
        prompt_parts.append(self._format_tool_definitions(tool_definitions))
        prompt_parts.append(self._format_few_shot_examples(few_shot_examples))
        prompt_parts.append("\n\n### Generate Diagnostic Protocol ###\n")
        prompt_parts.append("Given the patient's case and the available tools, generate a structured diagnostic protocol:\n")
        return "".join(prompt_parts)

class LLMProtocolGenerator:
    def __init__(self):
        # In a real application, this would be an API client for an LLM (e.g., OpenAI, Gemini)
        pass

    def generate_protocol(self, prompt):
        # This is a mock implementation of an LLM call.
        # In a real scenario, the prompt would be sent to an LLM, and its response would be parsed.
        print(f"\n--- Mock LLM Input Prompt ---\n{prompt}\n---\n")
        
        # Simulate a structured output from the LLM
        mock_llm_output = {
            "protocol_steps": [
                {
                    "step": 1,
                    "rationale": "Patient presents with severe abdominal pain and fever, suggesting an acute inflammatory process. Initial assessment points to potential appendicitis.",
                    "program": "order_lab_tests(blood_count=True, crp=True)",
                    "tool_output": "Blood count: WBC 18.5 (high), CRP: 120 mg/L (high).",
                    "next_rationale": "Elevated inflammatory markers confirm acute inflammation. Further imaging is required to pinpoint the source and confirm appendicitis."
                },
                {
                    "step": 2,
                    "rationale": "To visualize abdominal organs and confirm the diagnosis of appendicitis, an abdominal CT scan is necessary.",
                    "program": "order_imaging(type=\"CT scan\", area=\"abdomen\")",
                    "tool_output": "CT scan report: Pericecal fat stranding, dilated appendix (10mm diameter), presence of appendicolith. No free fluid.",
                    "next_rationale": "CT findings are highly suggestive of acute appendicitis. Surgical consultation is recommended for appendectomy."
                }
            ]
        }
        return mock_llm_output


# Example Usage:
if __name__ == "__main__":
    prompt_engineer = PromptEngineer()
    llm_generator = LLMProtocolGenerator()

    patient_case = (
        "A 45-year-old male presents to the emergency department with acute onset right lower quadrant abdominal pain "
        "for the past 12 hours. The pain started periumbilically and migrated to the right lower quadrant. "
        "He reports nausea, anorexia, and a low-grade fever (38.1 C). On physical examination, "
        "there is tenderness and guarding in the right lower quadrant with positive rebound tenderness."
    )

    available_tools = {
        "order_lab_tests": "Orders various laboratory tests (e.g., blood count, CRP, electrolytes). Args: blood_count: bool, crp: bool, electrolytes: bool",
        "order_imaging": "Orders imaging studies like X-ray, Ultrasound, CT scan, MRI. Args: type: str (e.g., \"CT scan\"), area: str (e.g., \"abdomen\"), contrast: bool",
        "consult_specialist": "Refers the patient to a specialist. Args: specialist: str (e.g., \"surgeon\", \"cardiologist\")",
        "prescribe_medication": "Prescribes medication. Args: medication_name: str, dosage: str, frequency: str"
    }

    few_shot_examples = [
        {
            "rationale": "Patient with shortness of breath and chest discomfort. Suspect acute cardiac event.",
            "program": "perform_ecg()",
            "tool_output": "ECG shows ST elevation in leads II, III, aVF.",
            "next_rationale": "Consistent with inferior wall myocardial infarction. Activate cath lab and administer aspirin."
        },
        {
            "rationale": "Child presents with high fever and rash. Rule out common childhood infections.",
            "program": "order_lab_tests(viral_panel=True, bacterial_culture=True)",
            "tool_output": "Viral panel negative. Bacterial culture positive for Group A Streptococcus.",
            "next_rationale": "Diagnosis of Strep throat. Prescribe oral antibiotics."
        }
    ]

    # Construct the prompt
    full_prompt = prompt_engineer.construct_prompt(patient_case, available_tools, few_shot_examples)
    
    # Generate the protocol (mock LLM call)
    diagnostic_protocol = llm_generator.generate_protocol(full_prompt)

    print("\n--- Generated Diagnostic Protocol (from Mock LLM) ---\n")
    for step in diagnostic_protocol['protocol_steps']:
        print(f"Step {step['step']}:")
        print(f"  Rationale: {step['rationale']}")
        print(f"  Program: {step['program']}")
        print(f"  Tool Output: {step['tool_output']}")
        print(f"  Next Rationale: {step['next_rationale']}")
        print()
