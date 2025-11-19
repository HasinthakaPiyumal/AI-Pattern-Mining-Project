import gradio as gr
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from typing import Type

# Mock API keys for demonstration (replace with actual keys in a real application)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# --- 3. External Tools Integration ---

class MedicalDatabaseTool(BaseTool):
    name = "MedicalDatabase"
    description = "Useful for querying a vast medical knowledge base for information on diseases, symptoms, and conditions."

    def _run(self, query: str) -> str:
        if "diabetes" in query.lower():
            return "Diabetes Mellitus is a metabolic disease that causes high blood sugar. Symptoms include frequent urination, increased thirst, and unexplained weight loss. Long-term complications include heart disease, stroke, kidney disease, and nerve damage."
        elif "hypertension" in query.lower():
            return "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Often has no symptoms."
        elif "common cold" in query.lower():
            return "The common cold is a viral infection of your nose and throat. Symptoms include runny nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, and low-grade fever."
        return f"No specific information found in the medical database for '{query}'."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async execution")

class TreatmentGuidelineTool(BaseTool):
    name = "TreatmentGuideline"
    description = "Useful for finding standard treatment protocols for diagnosed medical conditions."

    def _run(self, disease: str) -> str:
        if "diabetes" in disease.lower():
            return "Treatment for Diabetes typically involves diet and exercise, medication (e.g., metformin, insulin), and regular blood sugar monitoring."
        elif "hypertension" in disease.lower():
            return "Treatment for Hypertension often includes lifestyle changes (diet, exercise), and medications like ACE inhibitors, ARBs, diuretics, or beta-blockers."
        elif "common cold" in disease.lower():
            return "Treatment for the Common Cold focuses on symptom relief: rest, fluids, over-the-counter pain relievers, and decongestants."
        return f"No specific treatment guidelines found for '{disease}'."

    async def _arun(self, disease: str) -> str:
        raise NotImplementedError("This tool does not support async execution")

class DiagnosticCriteriaTool(BaseTool):
    name = "DiagnosticCriteria"
    description = "Useful for retrieving established diagnostic criteria for various diseases to confirm a potential diagnosis."

    def _run(self, disease: str) -> str:
        if "diabetes" in disease.lower():
            return "Diagnostic Criteria for Diabetes (ADA): Fasting plasma glucose >= 126 mg/dL, or 2-hour plasma glucose >= 200 mg/dL during OGTT, or HbA1c >= 6.5%, or random plasma glucose >= 200 mg/dL in a patient with classic symptoms."
        elif "hypertension" in disease.lower():
            return "Diagnostic Criteria for Hypertension (ACC/AHA): Blood pressure >= 130/80 mmHg on two or more occasions."
        return f"No specific diagnostic criteria found for '{disease}'."

    async def _arun(self, disease: str) -> str:
        raise NotImplementedError("This tool does not support async execution")

# --- 2. Agentic Orchestration Layer ---

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

tools = [
    MedicalDatabaseTool(),
    TreatmentGuidelineTool(),
    DiagnosticCriteriaTool(),
]

# Custom prompt to encourage reasoning paths and confidence scores
CUSTOM_PROMPT = """You are an AI-powered diagnostic assistant for healthcare professionals. Your goal is to provide accurate and trustworthy diagnostic assistance, treatment recommendations, and explanations. When responding, always include: 
1. Your step-by-step reasoning process.
2. A self-rated confidence score (e.g., 'Confidence: 85%') for your diagnosis and recommendations.
3. The final diagnosis and/or treatment recommendation.

Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. This must include your reasoning path, a confidence score, and the diagnosis/recommendation.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(CUSTOM_PROMPT)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# --- 4. Trustworthiness & Transparency Module ---

def parse_agent_response(response: str):
    reasoning_path = "No explicit reasoning path found."
    confidence_score = "N/A"
    diagnosis_recommendation = "No specific diagnosis or recommendation found."

    # Extract reasoning path
    if "Thought:" in response:
        reasoning_start = response.find("Thought:")
        reasoning_end = response.rfind("Final Answer:")
        if reasoning_start != -1 and reasoning_end != -1:
            reasoning_path = response[reasoning_start:reasoning_end].strip()
        elif reasoning_start != -1:
            reasoning_path = response[reasoning_start:].strip()

    # Extract confidence score
    if "Confidence:" in response:
        conf_start = response.find("Confidence:")
        conf_end = response.find("\n", conf_start)
        confidence_score = response[conf_start:conf_end if conf_end != -1 else len(response)].strip()

    # Extract final diagnosis/recommendation (after Final Answer)
    if "Final Answer:" in response:
        final_answer_start = response.find("Final Answer:") + len("Final Answer:")
        diagnosis_recommendation = response[final_answer_start:].strip()

    return reasoning_path, confidence_score, diagnosis_recommendation

# --- 5. Evaluation & Quality Control Module (Conceptual/Placeholder) ---

def validate_medical_output(output: str) -> str:
    if "hallucination" in output.lower() or "incorrect" in output.lower():
        return "Warning: Potential factual inconsistency or hallucination detected."
    return "Output appears medically plausible."

def detect_hallucination(output: str) -> bool:
    if "I am a medical professional" in output or "I cannot provide medical advice" in output:
        return True
    return False

# --- 1. User Interface (UI) with Gradio ---

def diagnose_patient(symptoms: str, medical_history: str):
    full_query = f"Patient symptoms: {symptoms}. Medical history: {medical_history}. Provide a potential diagnosis and treatment plan, including your reasoning and confidence score."
    
    try:
        response = agent_executor.invoke({"input": full_query})
        agent_output = response["output"]
        
        # Parse for trustworthiness and transparency
        reasoning, confidence, final_output = parse_agent_response(agent_output)

        # Perform basic quality control checks
        validation_message = validate_medical_output(final_output)
        is_hallucinating = detect_hallucination(final_output)

        qc_info = f"\n--- Quality Control ---\nValidation: {validation_message}"
        if is_hallucinating:
            qc_info += "\nHallucination Detection: Possible hallucination detected (generic medical disclaimer)."
        else:
            qc_info += "\nHallucination Detection: No obvious hallucination detected."

        return f"**Reasoning Path:**\n{reasoning}\n\n**Confidence Score:**\n{confidence}\n\n**Diagnosis & Recommendation:**\n{final_output}{qc_info}"
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again or refine your query. Make sure your OPENAI_API_KEY is set correctly."

iface = gr.Interface(
    fn=diagnose_patient,
    inputs=[
        gr.Textbox(label="Patient Symptoms", placeholder="e.g., fever, cough, fatigue"),
        gr.Textbox(label="Medical History (Optional)", placeholder="e.g., diabetic, high blood pressure"),
    ],
    outputs=gr.Markdown(label="AI Diagnostic Assistant Response"),
    title="AI-Powered Diagnostic Assistant",
    description="Enter patient symptoms and medical history to receive potential diagnoses, treatment recommendations, reasoning paths, and confidence scores."
)

if __name__ == "__main__":
    iface.launch()