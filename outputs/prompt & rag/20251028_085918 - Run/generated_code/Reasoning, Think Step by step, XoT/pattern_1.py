from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

class MedicalDiagnosticAssistant:
    def __init__(self, model_name="gpt-3.5-turbo", openai_api_key: str = None):
        # Initialize the LLM. For production, consider robust, HIPAA-compliant LLM services.
        # This uses ChatOpenAI as a placeholder. Ensure openai_api_key is provided or set as an environment variable.
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7, openai_api_key=openai_api_key)
        
        # Define Chain-of-Thought (CoT) prompt template
        self.cot_template = ChatPromptTemplate.from_messages([
            ("system", "You are a highly intelligent medical diagnostic AI assistant. Analyze the patient's information carefully and think step-by-step to arrive at the most probable diagnosis and potential treatment recommendations. Provide your reasoning clearly and concisely."),
            ("user", "Patient Information:\n{patient_data}\n\nThink step-by-step to formulate a diagnostic hypothesis and treatment plan:")
        ])
        
        # Define self-correction/verification prompt template
        self.verify_template = ChatPromptTemplate.from_messages([
            ("system", "You are a medical expert tasked with critically reviewing a diagnostic hypothesis and reasoning. Given the original patient information and an initial diagnostic reasoning, evaluate the initial reasoning for accuracy, completeness, and potential biases. Suggest alternative diagnoses if valid, highlight any inconsistencies, and refine the treatment recommendations to ensure they are robust and safe. Your goal is to enhance the faithfulness, accuracy, and reliability of the diagnosis."),
            ("user", "Patient Information:\n{patient_data}\n\nInitial Diagnostic Reasoning:\n{initial_reasoning}\n\nCritically review, verify, and refine this diagnosis and treatment plan:")
        ])
        self.output_parser = StrOutputParser()

    def diagnose(self, patient_data: str) -> dict:
        print("\n--- Stage 1: Initial Chain-of-Thought Reasoning ---")
        # Step 1: Initial Chain-of-Thought Reasoning
        # The LLM breaks down the problem and generates a step-by-step thought process.
        cot_chain = self.cot_template | self.llm | self.output_parser
        initial_reasoning = cot_chain.invoke({"patient_data": patient_data})
        print(initial_reasoning)

        print("\n--- Stage 2: Self-Correction and Verification ---")
        # Step 2: Self-Correction and Verification
        # The LLM reviews its own initial reasoning for consistency, accuracy, and completeness,
        # potentially incorporating implicit external knowledge or logical checks.
        verify_chain = self.verify_template | self.llm | self.output_parser
        refined_output = verify_chain.invoke({
            "patient_data": patient_data,
            "initial_reasoning": initial_reasoning
        })
        print(refined_output)

        # In a real-world application, these outputs would be parsed into structured data
        # (e.g., JSON) for further processing or display to a healthcare professional.
        return {
            "initial_diagnostic_reasoning": initial_reasoning,
            "refined_diagnostic_output": refined_output
        }

# Example Usage (demonstrates how to use the class):
# if __name__ == "__main__":
#     # IMPORTANT: Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
#     # or ensure it's set as an environment variable (OPENAI_API_KEY).
#     # For local testing without an actual key, you can comment out the API key parameter
#     # if your LLM setup doesn't require it, or use a local LLM if configured.
#     # assistant = MedicalDiagnosticAssistant(openai_api_key="YOUR_OPENAI_API_KEY")
#     assistant = MedicalDiagnosticAssistant() # Assumes OPENAI_API_KEY is in environment variables

#     patient_case_1 = """
#     Patient Name: Alice Smith
#     Age: 32
#     Symptoms:
#     - Severe headache for 2 days, throbbing on one side
#     - Sensitivity to light and sound
#     - Nausea and occasional vomiting
#     - Visual aura (flashing lights) before headache onset
#     Medical History: No significant medical history, non-smoker.
#     Lab Results: All routine blood tests normal.
#     """

#     patient_case_2 = """
#     Patient Name: Robert Johnson
#     Age: 68
#     Symptoms:
#     - Persistent cough for 2 months, often worse at night
#     - Shortness of breath, especially when climbing stairs
#     - Unexplained weight loss (10 lbs in 3 months)
#     - Fatigue and general weakness
#     Medical History: Long-term smoker (40 pack-years), history of hypertension.
#     Lab Results: Chest X-ray shows suspicious nodule in upper right lung.
#     """

#     print("\n--- Diagnosing Patient Case 1 ---")
#     diagnosis_result_1 = assistant.diagnose(patient_case_1)
#     # print("\nFinal Output for Patient 1:\n", diagnosis_result_1["refined_diagnostic_output"])

#     print("\n--- Diagnosing Patient Case 2 ---")
#     diagnosis_result_2 = assistant.diagnose(patient_case_2)
#     # print("\nFinal Output for Patient 2:\n", diagnosis_result_2["refined_diagnostic_output"])
