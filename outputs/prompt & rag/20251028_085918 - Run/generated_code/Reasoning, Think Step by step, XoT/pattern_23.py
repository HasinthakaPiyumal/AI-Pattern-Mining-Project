import gradio as gr

# Placeholder for LLM interaction
# In a real scenario, this would use an actual LLM API (e.g., OpenAI, Anthropic)
class MockLLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    def generate(self, prompt, temperature=0.7):
        # Simulate LLM response for diagnostic reasoning
        if "patient symptoms" in prompt:
            if "fever" in prompt and "cough" in prompt:
                return f"Thinking step 1: Patient has fever and cough. These are common symptoms for respiratory infections.\n" \
                       f"Thinking step 2: Consider influenza, common cold, or pneumonia.\n" \
                       f"Thinking step 3: Without more data (e.g., test results), narrowing down is hard.\n" \
                       f"Proposed Diagnosis: Possible Viral Respiratory Infection (e.g., Flu or Common Cold)."
            elif "headache" in prompt and "stiff neck" in prompt:
                return f"Thinking step 1: Patient has headache and stiff neck. This is a red flag for meningitis.\n" \
                       f"Thinking step 2: Rule out other causes like tension headache, but prioritize meningitis due to severity.\n" \
                       f"Proposed Diagnosis: Suspected Meningitis. Immediate medical attention advised."
            else:
                return f"Thinking step 1: Analyzing symptoms: {prompt.split('Symptoms: ')[1].split('History:')[0].strip()}.\n" \
                       f"Thinking step 2: Considering common conditions based on these symptoms.\n" \
                       f"Proposed Diagnosis: General Illness (requires further investigation)."
        if "Critically evaluate" in prompt:
            if "meningitis" in prompt and "stiff neck" in prompt:
                return "The reasoning for meningitis given headache and stiff neck is medically sound. No inconsistencies found."
            elif "General Illness" in prompt:
                return "The proposed diagnosis of 'General Illness' is vague but consistent with the limited symptoms provided. Suggests need for more data."
            else:
                return "The reasoning seems plausible, but always recommend further clinical review. No obvious hallucinations found based on provided data."
        return "Simulated LLM response."

class ReasoningEngine:
    def __init__(self, llm):
        self.llm = llm

    def generate_initial_reasoning(self, patient_data):
        prompt = f"Given the patient's data, generate a step-by-step diagnostic reasoning process and a proposed diagnosis.\n\n" \
                 f"Patient Symptoms: {patient_data['symptoms']}\n" \
                 f"Patient History: {patient_data['history']}\n" \
                 f"Test Results: {patient_data['test_results'] if 'test_results' in patient_data else 'None'}\n\n" \
                 f"Think step-by-step:"
        response = self.llm.generate(prompt)
        reasoning_steps = [line for line in response.split('\n') if line.startswith('Thinking step')]
        diagnosis = response.split('Proposed Diagnosis: ')[-1].strip() if 'Proposed Diagnosis:' in response else 'Undetermined'
        return {"reasoning_steps": reasoning_steps, "diagnosis": diagnosis}

    def explore_multiple_paths(self, patient_data, num_paths=2):
        # Simulate Tree-of-Thoughts by generating multiple independent reasonings
        all_paths = []
        for i in range(num_paths):
            # print(f"Generating reasoning path {i+1}...") # Removed for cleaner output in Gradio
            path_output = self.generate_initial_reasoning(patient_data)
            all_paths.append(path_output)
        return all_paths

class VerificationEngine:
    def __init__(self, verifier_llm=None, medical_knowledge_base=None):
        self.verifier_llm = verifier_llm if verifier_llm else MockLLM()
        # In a real system, medical_knowledge_base would be a robust RAG system or a database
        self.medical_knowledge_base = medical_knowledge_base if medical_knowledge_base else self._load_mock_knowledge_base()

    def _load_mock_knowledge_base(self):
        # Simplified mock knowledge base
        return {
            "fever and cough": "Common symptoms of respiratory infections like influenza, common cold, bronchitis, pneumonia.",
            "headache and stiff neck": "Potential indicators of meningitis, but also severe tension headaches or migraines. Requires urgent medical review.",
            "influenza": "Viral respiratory infection, symptoms include fever, cough, sore throat, body aches.",
            "meningitis": "Inflammation of the membranes surrounding the brain and spinal cord, often bacterial or viral. Requires urgent medical attention and lumbar puncture for diagnosis."
        }

    def verify_reasoning_and_diagnosis(self, reasoning_output, patient_data):
        reasoning_steps = reasoning_output["reasoning_steps"]
        diagnosis = reasoning_output["diagnosis"]
        verification_feedback = []
        is_verified = True

        # 1. LLM-based verification
        verifier_prompt = f"Critically evaluate the following diagnostic reasoning and proposed diagnosis for accuracy and medical soundness, given the patient data. Identify any inconsistencies or potential hallucinations.\n\n" \
                          f"Patient Symptoms: {patient_data['symptoms']}\n" \
                          f"Patient History: {patient_data['history']}\n" \
                          f"Test Results: {patient_data['test_results'] if 'test_results' in patient_data else 'None'}\n" \
                          f"Proposed Reasoning: {' '.join(reasoning_steps)}\n" \
                          f"Proposed Diagnosis: {diagnosis}\n\n" \
                          f"Verification Report:"
        llm_verification = self.verifier_llm.generate(verifier_prompt, temperature=0.3)
        verification_feedback.append(f"LLM Verifier Report: {llm_verification}")
        if "inconsistent" in llm_verification.lower() or "hallucination" in llm_verification.lower():
            is_verified = False

        # 2. Knowledge-base cross-referencing (simplified)
        kb_check_status = "Pass"
        kb_feedback = []
        combined_text = ' '.join(reasoning_steps).lower() + ' ' + diagnosis.lower() + ' ' + patient_data['symptoms'].lower()

        for key, value in self.medical_knowledge_base.items():
            if key in combined_text:
                kb_feedback.append(f"Cross-referenced '{key}': {value}")
        
        if not kb_feedback: # If nothing found in KB directly related to key terms
             kb_check_status = "Warning"
             kb_feedback.append("Warning: No direct knowledge base entry found for primary key terms. Further review recommended.")

        if kb_check_status == "Warning":
            is_verified = False # Consider it not fully verified if KB couldn't confirm
        verification_feedback.append(f"Knowledge Base Check: {kb_check_status}")
        verification_feedback.extend(kb_feedback)

        return {"is_verified": is_verified, "feedback": verification_feedback}

class MedicalDiagnosisAssistant:
    def __init__(self):
        self.reasoning_llm = MockLLM(model_name="diagnostic-llm")
        self.verifier_llm = MockLLM(model_name="verifier-llm")
        self.reasoning_engine = ReasoningEngine(self.reasoning_llm)
        self.verification_engine = VerificationEngine(verifier_llm=self.verifier_llm)

    def diagnose_patient(self, symptoms, history, test_results="None"):
        patient_data = {
            "symptoms": symptoms,
            "history": history,
            "test_results": test_results
        }

        # print("\n--- Step 1: Generating Initial Reasoning (Chain-of-Thought / Decomposed Prompting) ---")
        # initial_reasoning_output = self.reasoning_engine.generate_initial_reasoning(patient_data)
        # print("Initial Reasoning Steps:")
        # for step in initial_reasoning_output["reasoning_steps"]:
        #     print(f"- {step}")
        # print(f"Proposed Diagnosis: {initial_reasoning_output['diagnosis']}")

        # print("\n--- Step 2: Exploring Multiple Reasoning Paths (Tree-of-Thoughts Simulation) ---")
        all_reasoning_paths = self.reasoning_engine.explore_multiple_paths(patient_data, num_paths=3)
        # print(f"Generated {len(all_reasoning_paths)} reasoning paths.")

        # print("\n--- Step 3: Self-Consistency Check & External Verification ---")
        verified_diagnoses = []
        full_verification_reports = []

        # Self-Consistency (simplified: compare diagnoses from multiple paths)
        diagnoses_from_paths = [path['diagnosis'] for path in all_reasoning_paths]
        unique_diagnoses = set(diagnoses_from_paths)
        
        consistency_status = ""
        if len(unique_diagnoses) > 1:
            consistency_status = f"Inconsistent diagnoses across paths. Diagnoses found: {', '.join(unique_diagnoses)}."
            # print(f"Self-Consistency Warning: {consistency_status}") # Removed for cleaner output in Gradio
        else:
            consistency_status = f"Consistent diagnoses across all paths. Diagnosis: {list(unique_diagnoses)[0]}."
            # print(f"Self-Consistency Check: {consistency_status}") # Removed for cleaner output in Gradio

        # External Verification for each path
        for i, path_output in enumerate(all_reasoning_paths):
            # print(f"\n--- Verifying Path {i+1} ---") # Removed for cleaner output in Gradio
            verification_report = self.verification_engine.verify_reasoning_and_diagnosis(path_output, patient_data)
            full_verification_reports.append(verification_report)

            if verification_report["is_verified"]:
                verified_diagnoses.append(path_output['diagnosis'])
                # print(f"Path {i+1} Verified. Diagnosis: {path_output['diagnosis']}") # Removed for cleaner output in Gradio
            # else:
                # print(f"Path {i+1} NOT fully verified. Diagnosis: {path_output['diagnosis']}") # Removed for cleaner output in Gradio
            # for fb in verification_report["feedback"]:
                # print(f"  - {fb}") # Removed for cleaner output in Gradio

        final_diagnosis = "Undetermined (requires further specialist review)"
        if verified_diagnoses:
            # Simple majority vote or just take the first verified
            final_diagnosis = max(set(verified_diagnoses), key=verified_diagnoses.count)
            # print(f"\n--- Final Consensus Diagnosis (from verified paths): {final_diagnosis} ---") # Removed for cleaner output in Gradio
        else:
            # print("\n--- No fully verified diagnosis could be established. ---") # Removed for cleaner output in Gradio
            pass # Keep final_diagnosis as undetermined

        return {
            "final_diagnosis": final_diagnosis,
            "consistency_status": consistency_status,
            "all_reasoning_paths": all_reasoning_paths,
            "full_verification_reports": full_verification_reports
        }

def diagnosis_interface(symptoms, history, test_results):
    assistant = MedicalDiagnosisAssistant()
    result = assistant.diagnose_patient(symptoms, history, test_results)

    output_text = [
        f"**Final Consensus Diagnosis:** {result['final_diagnosis']}",
        f"**Self-Consistency Check:** {result['consistency_status']}",
        "\n---\n**Detailed Reasoning Paths & Verification Reports**\n---"
    ]

    for i, path in enumerate(result['all_reasoning_paths']):
        output_text.append(f"\n### Reasoning Path {i+1}:")
        output_text.append("**Proposed Reasoning Steps:**")
        for step in path['reasoning_steps']:
            output_text.append(f"- {step}")
        output_text.append(f"**Proposed Diagnosis:** {path['diagnosis']}")

        verification_report = result['full_verification_reports'][i]
        output_text.append(f"**Verification Status:** {'✅ Verified' if verification_report['is_verified'] else '❌ NOT Fully Verified'}")
        output_text.append("**Verification Feedback:**")
        for fb in verification_report['feedback']:
            output_text.append(f"  - {fb}")

    return "\n".join(output_text)

# Gradio Interface
iface = gr.Interface(
    fn=diagnosis_interface,
    inputs=[
        gr.Textbox(label="Patient Symptoms (e.g., 'fever, cough, sore throat')", lines=3),
        gr.Textbox(label="Patient History (e.g., 'no significant medical history, non-smoker')", lines=3),
        gr.Textbox(label="Test Results (e.g., 'Negative for Strep, WBC normal')", lines=2)
    ],
    outputs=gr.Markdown(),
    title="Medical Diagnosis Assistant with Verified Reasoning (SVR Pattern)",
    description="This assistant demonstrates the **Structured and Verified Reasoning (SVR)** pattern for complex medical diagnosis. It generates multiple reasoning paths, checks for consistency, and verifies findings against a mock medical knowledge base and a verifier LLM to enhance accuracy and transparency."
)

if __name__ == "__main__":
    iface.launch()