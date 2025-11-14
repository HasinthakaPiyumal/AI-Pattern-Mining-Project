
import gradio as gr
import pandas as pd
import numpy as np
import os

# LangChain, OpenAI, and VectorDB Imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set your OpenAI API key as an environment variable or replace with your actual key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- 1. Knowledge Retrieval System (RAG) ---
# Initialize embedding model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a dummy Chroma vector store with some medical documents
# In a real application, this would be populated with a vast amount of medical literature.
medical_docs = [
    "Symptoms of influenza include fever, cough, sore throat, and muscle aches. Treatment often involves rest and antivirals.",
    "Diabetes mellitus is a chronic metabolic disease characterized by high blood glucose levels. Types include Type 1 and Type 2.",
    "Hypertension, or high blood pressure, increases the risk of heart disease and stroke. Lifestyle changes and medication are common treatments.",
    "Appendicitis is an inflammation of the appendix, typically presenting with abdominal pain, nausea, and fever. Surgical removal is usually required.",
    "COVID-19 symptoms range from mild (fever, cough, fatigue) to severe (difficulty breathing). Vaccination is crucial for prevention.",
]
vectorstore = Chroma.from_texts(medical_docs, embeddings, collection_name="medical_knowledge")
retriever = vectorstore.as_retriever()

def retrieve_medical_context(query: str) -> str:
    """Retrieves relevant medical context from the knowledge base."""
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

# --- 2. Tool Integration Layer ---

@tool
def medical_database_search(query: str) -> str:
    """Searches a simulated medical database for information related to the query.
    In a real application, this would integrate with external APIs like PubMed or clinical guidelines.
    """
    print(f"[Tool Call] Searching medical database for: {query}")
    # Simulate API call latency and response
    if "influenza treatment" in query.lower():
        return "Influenza treatment often includes antiviral medications like oseltamivir or zanamivir, along with supportive care."
    elif "diabetes management" in query.lower():
        return "Diabetes management involves blood sugar monitoring, diet, exercise, and sometimes insulin or other medications."
    elif "appendicitis symptoms" in query.lower():
        return "Common symptoms of appendicitis include sudden pain that begins on the right side of the lower abdomen, nausea, vomiting, and loss of appetite."
    return f"No specific information found for '{query}'. Please refine your query or consult a comprehensive medical text."

@tool
def lab_result_analyzer(lab_results_csv: str) -> str:
    """Analyzes simulated lab results provided as a CSV string.
    Example input: 'Test,Value,Unit,ReferenceRange\nCBC_WBC,12.5,10^9/L,4.0-10.0\nCRP,15.2,mg/L,<5.0'
    """
    print(f"[Tool Call] Analyzing lab results: {lab_results_csv}")
    try:
        # Using pandas to parse and interpret lab results
        df = pd.read_csv(pd.io.common.StringIO(lab_results_csv))
        analysis = []
        for _, row in df.iterrows():
            test = row["Test"]
            value = row["Value"]
            unit = row["Unit"]
            ref_range = row["ReferenceRange"]
            
            # Simple interpretation logic
            if "WBC" in test and value > 10.0: # Assuming typical high range for WBC
                analysis.append(f"{test}: {value}{unit} (High - suggestive of infection/inflammation)")
            elif "CRP" in test and value > 5.0: # Assuming typical high range for CRP
                analysis.append(f"{test}: {value}{unit} (Elevated - indicative of inflammation)")
            elif "Glucose" in test and value > 120: # Assuming high glucose
                analysis.append(f"{test}: {value}{unit} (High - potentially indicative of diabetes or insulin resistance)")
            else:
                analysis.append(f"{test}: {value}{unit} (Within or near reference range for general context)")
        return "\n".join(analysis)
    except Exception as e:
        return f"Error analyzing lab results: {e}. Please ensure CSV format is correct."

@tool
def medical_imaging_analysis(imaging_report_summary: str) -> str:
    """Provides a summary interpretation of a medical imaging report.
    This is a placeholder for integration with an actual image analysis service.
    """
    print(f"[Tool Call] Analyzing imaging report: {imaging_report_summary}")
    # In a real scenario, this would call an AI model (e.g., a CNN for X-ray analysis)
    # or an external API that provides structured imaging interpretations.
    if "pneumonia" in imaging_report_summary.lower():
        return "Imaging report suggests findings consistent with pneumonia (e.g., consolidations, infiltrates). Correlation with clinical symptoms and lab findings is recommended."
    elif "fracture" in imaging_report_summary.lower():
        return "Imaging report indicates a suspected fracture. Further orthopedic evaluation is advised."
    return f"Based on the imaging report summary: '{imaging_report_summary}'. Interpretation requires expert review and context."

# List of all tools available to the agent
tools = [
    medical_database_search,
    lab_result_analyzer,
    medical_imaging_analysis,
]

# --- 3. AI Agent Orchestrator ---

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define the agent prompt
# This prompt guides the LLM to think, use tools, and provide a diagnosis.
agent_prompt_template = PromptTemplate.from_template(
    """You are an AI-powered medical diagnostic assistant. Your goal is to provide a differential diagnosis and recommended next steps for complex medical cases.
    You have access to several medical tools to assist you. Always think step-by-step and explain your reasoning.

    Here is the patient's information:
    Patient Summary: {patient_summary}
    Lab Results (CSV format):\n{lab_results_str}
    Imaging Report Summary: {imaging_report_str}
    
    Initial Medical Context from RAG: {rag_context}

    User Feedback so far: {user_feedback}

    Use the following format for your responses:

    Thought: You should always think about what to do.
    Tool: the tool to call, should be one of [{tool_names}]
    Tool Input: the input to the tool
    Observation: the result of the tool
    ... (this Thought/Tool/Tool Input/Observation can repeat multiple times)
    Thought: I have gathered enough information and can now provide a diagnosis.
    Final Diagnosis: [Your differential diagnosis and reasoning]
    Recommended Next Steps: [Further tests, specialist consults, treatment suggestions]
    Self-Correction Note: [Reflect on previous steps and how the diagnosis was refined or corrected based on new information/observations, especially considering user feedback.]

    Begin! Remember to be thorough and consider all available information.
    If no specific lab results or imaging reports are provided, state that they are unavailable.
    Always try to provide a 'Final Diagnosis' and 'Recommended Next Steps' even if uncertain, indicating the level of certainty.
    """
)

# Create the LangChain agent
agent = create_react_agent(llm, tools, agent_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 4. Feedback and Learning Module (Simplified) ---
# In a real system, Langsmith/Wandb would capture traces, evaluations, and feedback.
# Here, we'll just print a placeholder message.
def record_feedback(patient_id, diagnosis, feedback):
    print(f"[Feedback Recorded] Patient ID: {patient_id}, Diagnosis: {diagnosis}, Feedback: {feedback}")
    # Here you would integrate with Langsmith, Wandb, or a custom logging/database solution.
    # For example: wandb.log({"patient_id": patient_id, "diagnosis": diagnosis, "feedback": feedback})
    # Or send to a Langsmith feedback endpoint.

# --- 5. Data Management & Preprocessing (Implicit in tool inputs) ---
# For this example, data preprocessing is handled implicitly by expecting specific string formats
# for lab results (CSV) and summaries for imaging/patient data.

# --- 6. Gradio User Interface ---
def diagnose_patient(
    patient_summary: str,
    lab_results_str: str,
    imaging_report_str: str,
    expert_feedback: str = ""
) -> str:
    """Orchestrates the AI diagnostic process based on patient data and expert feedback."""
    
    full_query = f"Patient summary: {patient_summary}"
    if lab_results_str: full_query += f" Lab results: {lab_results_str}"
    if imaging_report_str: full_query += f" Imaging report: {imaging_report_str}"
    
    rag_context = retrieve_medical_context(full_query)
    print(f"[RAG Context Retrieved]: {rag_context[:200]}...")

    # Prepare the agent input with all available information
    agent_input = {
        "patient_summary": patient_summary,
        "lab_results_str": lab_results_str if lab_results_str else "N/A",
        "imaging_report_str": imaging_report_str if imaging_report_str else "N/A",
        "rag_context": rag_context if rag_context else "No specific context found.",
        "user_feedback": expert_feedback if expert_feedback else "No feedback yet."
    }

    try:
        # Invoke the agent
        result = agent_executor.invoke(agent_input)
        diagnosis_output = result.get("output", "Could not generate a full diagnosis.")
        
        # Simulate recording feedback
        record_feedback(patient_id="patient_xyz", diagnosis=diagnosis_output, feedback=expert_feedback)
        
        return diagnosis_output
    except Exception as e:
        return f"An error occurred during diagnosis: {e}\n\nAgent input was: {agent_input}"

# Gradio Interface setup
iface = gr.Interface(
    fn=diagnose_patient,
    inputs=[
        gr.Textbox(label="Patient Summary (Symptoms, History)", lines=5, placeholder="e.g., 65-year-old male, sudden onset right lower quadrant pain, nausea, mild fever, no appetite for 24 hours..."),
        gr.Textbox(label="Lab Results (CSV format)", lines=5, placeholder="Test,Value,Unit,ReferenceRange\nCBC_WBC,12.5,10^9/L,4.0-10.0\nCRP,15.2,mg/L,<5.0\nGlucose,95,mg/dL,70-100"),
        gr.Textbox(label="Medical Imaging Report Summary", lines=5, placeholder="e.g., CT abdomen: Periappendiceal fat stranding, dilated appendix measuring 10mm with appendicolith. No free fluid."),
        gr.Textbox(label="Expert Feedback (Optional - for self-correction)", lines=2, placeholder="e.g., 'Considered diverticulitis initially, but patient's age and specific pain location point more towards appendicitis.'")
    ],
    outputs=gr.Textbox(label="AI Diagnostic Assistant Output"),
    title="Adaptive AI Medical Diagnostic Assistant",
    description="Enter patient data to get a differential diagnosis and recommended next steps from an AI agent that integrates medical tools and self-corrects."
)

# Launch the Gradio app
if __name__ == "__main__":
    print("Starting Gradio interface...")
    print("Make sure to set your OPENAI_API_KEY environment variable or replace 'YOUR_OPENAI_API_KEY' in the script.")
    iface.launch(share=True) # Set share=True to get a public link for easy sharing
