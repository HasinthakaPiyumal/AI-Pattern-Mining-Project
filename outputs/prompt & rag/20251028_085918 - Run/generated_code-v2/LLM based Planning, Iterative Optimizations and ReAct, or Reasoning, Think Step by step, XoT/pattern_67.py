import os
from collections import Counter
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Ensure you have your OpenAI API key set as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

class SelfConsistencyDiagnosticAssistant:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7, n_reasoning_paths=5):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.n_reasoning_paths = n_reasoning_paths
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent medical diagnostic assistant. Analyze the patient's symptoms and medical history to provide a probable diagnosis and a brief reasoning path."),
            ("user", "Patient Symptoms: {symptoms}\nMedical History: {medical_history}\n\nProvide a probable diagnosis and the reasoning steps."
            )
        ])

    def _generate_diagnosis_path(self, symptoms, medical_history):
        chain = self.prompt_template | self.llm
        response = chain.invoke({"symptoms": symptoms, "medical_history": medical_history})
        # Simple parsing to extract diagnosis, assumes diagnosis is at the start or easily identifiable
        # In a real-world scenario, more robust parsing or structured output from LLM would be needed
        content = response.content.strip()
        # Try to find a line that looks like a diagnosis
        lines = content.split('\n')
        diagnosis_line = ""
        for line in lines:
            if "diagnosis:" in line.lower() or "probable diagnosis:" in line.lower():
                diagnosis_line = line.strip()
                break
        if not diagnosis_line and lines:
            # Fallback to the first non-empty line if no specific diagnosis line is found
            diagnosis_line = lines[0].strip()

        return {"full_response": content, "diagnosis": diagnosis_line if diagnosis_line else "No clear diagnosis found.", "reasoning": content}

    def get_self_consistent_diagnosis(self, symptoms, medical_history):
        all_diagnoses = []
        all_reasoning_paths = []

        for _ in range(self.n_reasoning_paths):
            path = self._generate_diagnosis_path(symptoms, medical_history)
            all_diagnoses.append(path["diagnosis"])
            all_reasoning_paths.append(path["full_response"])

        # Perform majority voting on the extracted diagnoses
        diagnosis_counts = Counter(all_diagnoses)
        most_common_diagnosis = diagnosis_counts.most_common(1)[0][0] if diagnosis_counts else "No diagnosis could be determined."
        
        # Find reasoning paths associated with the most common diagnosis
        consistent_reasoning = [path for diag, path in zip(all_diagnoses, all_reasoning_paths) if diag == most_common_diagnosis]

        return most_common_diagnosis, consistent_reasoning, all_reasoning_paths

# Gradio Interface
def diagnose_patient(symptoms, medical_history):
    if not os.environ.get("OPENAI_API_KEY"):
        return "Error: OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.", [], []

    assistant = SelfConsistencyDiagnosticAssistant()
    final_diagnosis, consistent_reasoning, all_paths = assistant.get_self_consistent_diagnosis(symptoms, medical_history)

    detailed_output = f"### Final Self-Consistent Diagnosis:\n{final_diagnosis}\n\n"
    detailed_output += "### Reasoning Paths Supporting the Final Diagnosis:\n"
    if consistent_reasoning:
        for i, reason in enumerate(consistent_reasoning):
            detailed_output += f"\n--- Reasoning Path {i+1} ---\n{reason}\n"
    else:
        detailed_output += "No specific consistent reasoning paths identified for the final diagnosis.\n"
    
    detailed_output += "\n### All Generated Reasoning Paths (for diversity review):\n"
    for i, path in enumerate(all_paths):
        detailed_output += f"\n--- Path {i+1} ---\n{path}\n"

    return detailed_output

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set. The application might not function correctly.")
        print("Please set it using: export OPENAI_API_KEY='your_api_key_here'")

    iface = gr.Interface(
        fn=diagnose_patient,
        inputs=[
            gr.Textbox(lines=5, label="Patient Symptoms (e.g., fever, cough, fatigue)"),
            gr.Textbox(lines=5, label="Medical History (e.g., allergies, chronic conditions, medications)")
        ],
        outputs=gr.Markdown(label="Diagnostic Report"),
        title="Intelligent Medical Diagnostic Assistant with Self-Consistency",
        description="Enter patient symptoms and medical history to get a self-consistent preliminary diagnosis. The system leverages an LLM and multiple reasoning paths for robustness."
    )
    iface.launch(share=False)