from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI # Requires OPENAI_API_KEY
import gradio as gr
import json

# --- 1. Tool Definitions ---

@tool
def medical_database_search(query: str) -> str:
    """
    Searches a simulated medical database for information related to symptoms, conditions, or treatments.
    Returns relevant medical facts.
    """
    if "pneumonia symptoms" in query.lower():
        return "Pneumonia symptoms include cough, fever, shortness of breath, chest pain, and fatigue. Common causes are bacterial or viral infections."
    elif "diabetes treatment" in query.lower():
        return "Diabetes treatment often involves lifestyle changes (diet, exercise), oral medications, or insulin injections, depending on the type and severity."
    elif "chest pain causes" in query.lower():
        return "Chest pain can be caused by cardiac issues (e.g., heart attack, angina), lung problems (e.g., pneumonia, pleurisy), gastrointestinal issues (e.g., GERD), or musculoskeletal problems."
    else:
        return f"No specific information found for '{query}'. Please try a different query."

@tool
def medical_image_analysis(image_description: str) -> str:
    """
    Simulates the analysis of a medical image (e.g., X-ray, MRI report).
    Returns a simulated finding or a potential diagnosis based on the description.
    """
    image_description_lower = image_description.lower()
    if "chest x-ray" in image_description_lower and "bilateral infiltrates" in image_description_lower:
        return "Simulated finding: Chest X-ray shows bilateral infiltrates, highly suggestive of pneumonia or acute respiratory distress syndrome (ARDS)."
    elif "mri brain" in image_description_lower and "lesion" in image_description_lower:
        return "Simulated finding: MRI brain indicates a focal lesion in the temporal lobe, requiring further investigation for potential tumor or inflammatory process."
    elif "no significant findings" in image_description_lower:
        return "Simulated finding: Medical image analysis indicates no significant pathological findings."
    else:
        return "Simulated finding: Image analysis inconclusive based on description. More detailed image data or specific context needed."

# --- 2. LLM and Agent Setup ---

# Initialize LLM (requires an OpenAI API key set as an environment variable or passed directly)
# For local development, you might use Ollama with `ChatOllama` or similar.
llm = ChatOpenAI(model="gpt-4o", temperature=0) # Or "gpt-3.5-turbo"

tools = [medical_database_search, medical_image_analysis]

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an AI-powered medical diagnostic assistant. Your goal is to help healthcare professionals by analyzing patient data, using specialized tools, and providing transparent diagnostic suggestions.\n        \n        Follow these steps:\n        1. Analyze the patient's symptoms and any provided image descriptions.\n        2. Use the 'medical_database_search' tool to gather relevant information about potential conditions or causes.\n        3. Use the 'medical_image_analysis' tool if an image description is provided to get simulated findings.\n        4. Synthesize all information to propose a likely diagnosis or a list of differential diagnoses.\n        5. Clearly state your reasoning, referencing the information you gathered from the tools.\n        6. Provide a confidence score (0-100%) for your primary diagnosis.\n        7. List your sources (which tools you used and key information from them).\n        8. If you are highly uncertain (e.g., confidence below 30%), you MUST state that you abstain from a definitive diagnosis and explain why, recommending further human evaluation.\n        9. Format your final output as a JSON object with keys: "diagnosis", "reasoning", "confidence", "sources", "abstain".\n        Example:\n        ```json\n        {{\n            "diagnosis": "Pneumonia",\n            "reasoning": "Based on symptoms (cough, fever, shortness of breath) and chest X-ray showing bilateral infiltrates. Medical database search confirmed these are consistent with pneumonia.",\n            "confidence": 90,\n            "sources": ["medical_database_search: pneumonia symptoms", "medical_image_analysis: chest X-ray findings"],\n            "abstain": false\n        }}\n        ```\n        """),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. Diagnostic Function ---

def get_diagnostic_report(symptoms: str, image_description: str) -> str:
    """
    Generates a diagnostic report using the AI agent based on patient symptoms and image description.
    """
    input_text = f"Patient Symptoms: {symptoms}\n"
    if image_description:
        input_text += f"Medical Image Description: {image_description}\n"
    
    try:
        response = agent_executor.invoke({"input": input_text})
        raw_output = response["output"]

        # Attempt to parse the JSON output from the LLM
        try:
            report_data = json.loads(raw_output)
            diagnosis = report_data.get("diagnosis", "N/A")
            reasoning = report_data.get("reasoning", "No detailed reasoning provided.")
            confidence = report_data.get("confidence", "N/A")
            sources = ", ".join(report_data.get("sources", [])) if report_data.get("sources") else "None"
            abstain = report_data.get("abstain", False)

            if abstain:
                return (f"**AI Diagnostic Assistant Report (ABSTAINED)**\n\n"
                        f"The AI is highly uncertain and abstains from providing a definitive diagnosis at this time. Further human evaluation is strongly recommended.\n\n"
                        f"**Reasoning for Abstention:** {reasoning}\n\n"
                        f"**Confidence:** {confidence}% (Below threshold)\n\n"
                        f"**Sources Consulted:** {sources}")
            else:
                return (f"**AI Diagnostic Assistant Report**\n\n"
                        f"**Primary Diagnosis:** {diagnosis}\n\n"
                        f"**Reasoning:** {reasoning}\n\n"
                        f"**Confidence:** {confidence}%\n\n"
                        f"**Sources Consulted:** {sources}")
        except json.JSONDecodeError:
            return (f"**AI Diagnostic Assistant Report (Parsing Error)**\n\n"
                    f"The AI agent's output could not be parsed as valid JSON. Raw output:\n\n"
                    f"```\n{raw_output}\n```\n\n"
                    f"Please review the input or agent prompt for formatting issues.")

    except Exception as e:
        return f"An error occurred during diagnosis: {str(e)}"

# --- 4. Gradio Interface ---

def provide_feedback(symptoms: str, image_description: str, diagnostic_report: str, feedback: str):
    """Simulates storing human feedback."""
    print(f"--- Human Feedback Received ---")
    print(f"Symptoms: {symptoms}")
    print(f"Image Description: {image_description}")
    print(f"Report: {diagnostic_report}")
    print(f"Feedback: {feedback}")
    print(f"-----------------------------")
    return "Feedback submitted! Thank you."

with gr.Blocks() as demo:
    gr.Markdown("# 🩺 AI-Powered Medical Diagnostic Assistant")
    gr.Markdown("Enter patient symptoms and any relevant medical image descriptions to receive a diagnostic report.")

    with gr.Row():
        symptoms_input = gr.Textbox(label="Patient Symptoms (e.g., 'cough, fever, shortness of breath')", lines=5)
        image_desc_input = gr.Textbox(label="Medical Image Description (e.g., 'Chest X-ray shows bilateral infiltrates')", lines=5)
    
    diagnose_button = gr.Button("Get Diagnostic Report")
    
    output_report = gr.Markdown(label="Diagnostic Report")

    diagnose_button.click(
        get_diagnostic_report,
        inputs=[symptoms_input, image_desc_input],
        outputs=output_report
    )

    gr.Markdown("---")
    gr.Markdown("### Human Feedback & Correction")
    gr.Markdown("Help us improve the AI by providing feedback on the generated diagnosis.")

    with gr.Row():
        feedback_input = gr.Radio(["Accurate", "Partially Accurate", "Inaccurate", "Missing Information", "Confusing"], label="Is the diagnosis helpful?")
        feedback_text = gr.Textbox(label="Optional: Provide detailed feedback or correction", lines=3)
    
    submit_feedback_button = gr.Button("Submit Feedback")
    feedback_output = gr.Textbox(label="Feedback Status")

    submit_feedback_button.click(
        provide_feedback,
        inputs=[symptoms_input, image_desc_input, output_report, feedback_text], # Pass report content for logging
        outputs=feedback_output
    )

# To run the Gradio app:
# demo.launch()