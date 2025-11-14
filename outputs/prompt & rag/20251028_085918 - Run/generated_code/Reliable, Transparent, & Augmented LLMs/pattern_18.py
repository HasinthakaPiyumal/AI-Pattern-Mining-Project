import os
import json
from typing import List, Dict, Any

import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Mocking ChromaDB and sentence-transformers for a self-contained example
# In a real application, these would involve actual installations and data loading
class MockChromaDB:
    def __init__(self):
        self.documents = {
            "fever and cough": "Common cold, Flu, Bronchitis. Consult a doctor for diagnosis.",
            "severe headache and stiff neck": "Meningitis. Seek emergency medical attention.",
            "chest pain and shortness of breath": "Heart attack, Angina, Anxiety. Seek emergency medical attention.",
            "abdominal pain and nausea": "Gastroenteritis, Appendicitis. Consult a doctor."
        }
    
    def query(self, query_text: str) -> List[str]:
        results = []
        for doc_key, doc_val in self.documents.items():
            if query_text.lower() in doc_key.lower() or query_text.lower() in doc_val.lower():
                results.append(doc_val)
        return results

mock_chroma_db = MockChromaDB()

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY") # Replace with your actual key or set env var
CONFIDENCE_THRESHOLD = 0.75 # If LLM confidence is below this, system abstains

# --- 1. Define Tools for Langchain Agent ---

class MedicalKnowledgeBaseTool(BaseTool):
    name = "MedicalKnowledgeBase"
    description = "Useful for querying a medical knowledge base for information related to symptoms, conditions, and treatments."

    def _run(self, query: str) -> str:
        # In a real scenario, this would query ChromaDB with embeddings
        # For this demo, we use a simple string matching on our mock DB
        results = mock_chroma_db.query(query)
        if results:
            return "\n".join(results)
        return "No relevant information found in the medical knowledge base."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented for this demo tool")

class LabSystemQueryTool(BaseTool):
    name = "LabSystemQuery"
    description = "Useful for simulating a query to an external lab system to retrieve mock lab results for specific tests."

    def _run(self, test_name: str) -> str:
        mock_lab_results = {
            "CBC": "White Blood Cell Count: 8.5 (Normal), Hemoglobin: 14.2 (Normal)",
            "CMP": "Glucose: 95 (Normal), Creatinine: 0.9 (Normal)",
            "CRP": "C-Reactive Protein: 12 mg/L (Elevated)"
        }
        return mock_lab_results.get(test_name, f"No mock results available for {test_name}.")

    async def _arun(self, test_name: str) -> str:
        raise NotImplementedError("Async not implemented for this demo tool")

class SymptomCheckerTool(BaseTool):
    name = "SymptomChecker"
    description = "Useful for checking common symptoms and associating them with potential conditions. Input should be a comma-separated list of symptoms."

    def _run(self, symptoms: str) -> str:
        symptoms_list = [s.strip().lower() for s in symptoms.split(',')]
        if "fever" in symptoms_list and "cough" in symptoms_list:
            return "Potential conditions: Common Cold, Flu, Bronchitis."
        elif "chest pain" in symptoms_list and "shortness of breath" in symptoms_list:
            return "Potential conditions: Heart Attack, Angina, Anxiety. Seek emergency medical attention."
        elif "abdominal pain" in symptoms_list and "nausea" in symptoms_list:
            return "Potential conditions: Gastroenteritis, Appendicitis."
        else:
            return "Could not find common conditions for the given symptoms. Consider using MedicalKnowledgeBase."

    async def _arun(self, symptoms: str) -> str:
        raise NotImplementedError("Async not implemented for this demo tool")

# Initialize Tools
medical_kb_tool = MedicalKnowledgeBaseTool()
lab_system_tool = LabSystemQueryTool()
symptom_checker_tool = SymptomCheckerTool()

tools = [medical_kb_tool, lab_system_tool, symptom_checker_tool]

# --- 2. LLM and Agent Setup ---

llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Define output schema for structured responses
response_schemas = [
    ResponseSchema(name="diagnosis", description="The most likely medical diagnosis."),
    ResponseSchema(name="confidence", description="A confidence score for the diagnosis, ranging from 0.0 to 1.0.", type="float"),
    ResponseSchema(name="reasoning", description="The step-by-step reasoning that led to the diagnosis."),
    ResponseSchema(name="alternative_diagnoses", description="A list of other possible diagnoses, if any.", type="list")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

system_message_template = """
    You are an AI-powered medical diagnostic assistant. Your goal is to assist medical professionals by providing diagnostic suggestions, reasoning, and confidence estimations based on patient data. You have access to various medical tools.
    When providing a diagnosis, always explain your reasoning clearly and estimate your confidence level (between 0.0 and 1.0).
    If your confidence is low (e.g., below 0.7), indicate uncertainty and suggest further investigation or consultation.
    Always consider multiple possibilities and list alternative diagnoses.
    
    {format_instructions}
"""

# Initialize the Langchain agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS, # Uses OpenAI's function calling capabilities
    verbose=True,
    agent_kwargs={
        "system_message": SystemMessage(content=system_message_template.format(format_instructions=format_instructions))
    }
)

# --- 3. Core Diagnostic Function ---

def get_diagnosis_and_explanation(
    symptoms: str,
    medical_history: str,
    lab_results: str
) -> Dict[str, Any]:
    
    user_input = f"Patient Symptoms: {symptoms}\nMedical History: {medical_history}\nLab Results: {lab_results}"

    try:
        # Run the agent with the user's input
        agent_response_raw = agent.run(user_input)
        
        # Parse the structured output
        parsed_output = output_parser.parse(agent_response_raw)

        diagnosis = parsed_output.get("diagnosis", "N/A")
        confidence = parsed_output.get("confidence", 0.0)
        reasoning = parsed_output.get("reasoning", "No reasoning provided.")
        alternative_diagnoses = parsed_output.get("alternative_diagnoses", [])
        
        abstention_message = ""
        if confidence < CONFIDENCE_THRESHOLD:
            abstention_message = (
                "The system's confidence is low. Further investigation or consultation "
                "with a human medical professional is strongly recommended before making any decisions."
            )

        return {
            "diagnosis": diagnosis,
            "confidence": f"{confidence:.2f}",
            "reasoning": reasoning,
            "alternative_diagnoses": ", ".join(alternative_diagnoses) if alternative_diagnoses else "None",
            "abstention_message": abstention_message
        }
    except Exception as e:
        return {
            "diagnosis": "Error",
            "confidence": "0.00",
            "reasoning": f"An error occurred: {str(e)}",
            "alternative_diagnoses": "None",
            "abstention_message": "Please try again or contact support."
        }

# --- 4. Feedback Logging (Simple) ---

feedback_log = []

def log_feedback(input_symptoms, input_history, input_lab_results, ai_diagnosis, user_feedback_type, user_feedback_text):
    feedback_entry = {
        "timestamp": gr.processing_utils.get_current_time_ms(),
        "input_symptoms": input_symptoms,
        "input_history": input_history,
        "input_lab_results": input_lab_results,
        "ai_diagnosis": ai_diagnosis,
        "user_feedback_type": user_feedback_type,
        "user_feedback_text": user_feedback_text
    }
    feedback_log.append(feedback_entry)
    print("--- Feedback Logged ---")
    print(json.dumps(feedback_entry, indent=2))
    print("-----------------------")
    return "Thank you for your feedback!"

# --- 5. Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown("# AI-powered Medical Diagnostic Assistant")
    gr.Markdown("This assistant helps medical professionals with diagnostic suggestions, reasoning, and confidence estimations.")
    
    with gr.Row():
        with gr.Column():
            symptoms_input = gr.Textbox(label="Patient Symptoms", placeholder="e.g., Fever, persistent cough, fatigue")
            history_input = gr.Textbox(label="Medical History", placeholder="e.g., Diabetes, hypertension, recent travel")
            lab_results_input = gr.Textbox(label="Lab Results", placeholder="e.g., CBC results, X-ray findings, CRP levels")
            diagnose_button = gr.Button("Get Diagnosis")
        
        with gr.Column():
            diagnosis_output = gr.Textbox(label="Primary Diagnosis", interactive=False)
            confidence_output = gr.Textbox(label="Confidence Score (0.0-1.0)", interactive=False)
            reasoning_output = gr.Textbox(label="Reasoning Path", interactive=False, lines=5)
            alternatives_output = gr.Textbox(label="Alternative Diagnoses", interactive=False)
            abstention_message_output = gr.Textbox(label="Abstention/Warning", interactive=False, type="str")

    diagnose_button.click(
        get_diagnosis_and_explanation,
        inputs=[symptoms_input, history_input, lab_results_input],
        outputs=[diagnosis_output, confidence_output, reasoning_output, alternatives_output, abstention_message_output]
    )

    gr.Markdown("## Provide Feedback")
    with gr.Row():
        feedback_type = gr.Radio(["Correct", "Incorrect", "Needs Improvement"], label="Is the diagnosis useful?")
        feedback_text = gr.Textbox(label="Additional Comments (Optional)", placeholder="e.g., 'The alternative diagnoses were helpful, but the primary diagnosis was not quite right.'")
        feedback_button = gr.Button("Submit Feedback")

    feedback_output = gr.Textbox(label="Feedback Status", interactive=False)
    
    feedback_button.click(
        log_feedback,
        inputs=[
            symptoms_input,
            history_input,
            lab_results_input,
            diagnosis_output, # Use the AI's displayed diagnosis for context
            feedback_type,
            feedback_text
        ],
        outputs=feedback_output
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()