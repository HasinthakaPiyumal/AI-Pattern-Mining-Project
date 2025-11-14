import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
import os

# --- 1. Simulated Tools ---

@tool
def ehr_api_tool(patient_id: str) -> str:
    """Fetches electronic health records for a given patient ID. Provides patient history, demographics, and past diagnoses."""
    st.info(f"Simulating fetching EHR for Patient ID: {patient_id}")
    if patient_id == "P001":
        return "Patient P001: 55-year-old male. History of hypertension, type 2 diabetes. Last visit: 3 months ago for routine check-up. Medications: Metformin, Lisinopril. Allergies: Penicillin."
    elif patient_id == "P002":
        return "Patient P002: 30-year-old female. No significant medical history. Recent complaints: persistent headache, fatigue for 2 weeks. No known allergies. No regular medications."
    else:
        return f"No EHR found for Patient ID: {patient_id}."

@tool
def medical_database_search_tool(query: str) -> str:
    """Searches a medical knowledge base for information on symptoms, conditions, treatment protocols, or drug information."""
    st.info(f"Simulating medical database search for query: '{query}'")
    if "headache" in query.lower() and "fatigue" in query.lower():
        return "Common causes of headache and fatigue include viral infections, stress, dehydration, anemia, and sleep deprivation. Less common but serious causes include neurological conditions or chronic diseases. Further investigation needed."
    elif "hypertension treatment" in query.lower():
        return "Treatment for hypertension typically involves lifestyle modifications (diet, exercise) and medications such as ACE inhibitors (e.g., Lisinopril), ARBs, diuretics, or beta-blockers. Regular monitoring is crucial."
    elif "diabetes management" in query.lower():
        return "Type 2 diabetes management includes diet control, regular exercise, and medications like Metformin, sulfonylureas, or insulin. Blood glucose monitoring is essential."
    else:
        return f"Medical database search for '{query}' returned no specific results. Try a more general query."

@tool
def drug_interaction_checker_tool(drug1: str, drug2: str) -> str:
    """Checks for potential adverse interactions between two specified drugs."""
    st.info(f"Simulating drug interaction check for {drug1} and {drug2}")
    drug1_lower = drug1.lower()
    drug2_lower = drug2.lower()
    if ("lisinopril" in drug1_lower and "ibuprofen" in drug2_lower) or ("ibuprofen" in drug1_lower and "lisinopril" in drug2_lower):
        return "Potential interaction: NSAIDs (like Ibuprofen) can reduce the effectiveness of ACE inhibitors (like Lisinopril) and may increase the risk of kidney problems, especially in elderly or dehydrated patients. Monitor kidney function."
    elif ("metformin" in drug1_lower and "contrast dye" in drug2_lower) or ("contrast dye" in drug1_lower and "metformin" in drug2_lower):
        return "Potential interaction: Metformin should be temporarily discontinued before and for 48 hours after administration of iodinated contrast material due to increased risk of lactic acidosis."
    else:
        return f"No significant interaction found between {drug1} and {drug2}."

@tool
def image_analysis_service_tool(image_report_description: str) -> str:
    """Interprets a textual description of a medical image report (e.g., X-ray, MRI, CT scan) to extract key findings."""
    st.info(f"Simulating image analysis for: '{image_report_description}'")
    if "chest x-ray" in image_report_description.lower() and "infiltrates" in image_report_description.lower():
        return "Chest X-ray findings suggest possibility of pneumonia or other inflammatory process in the lungs. Recommend further clinical correlation and possibly follow-up imaging."
    elif "mri brain" in image_report_description.lower() and "lesion" in image_report_description.lower():
        return "MRI Brain report indicates a focal lesion in the temporal lobe. Further characterization with contrast and neurological consultation recommended to rule out tumor or other pathology."
    else:
        return f"Analysis of image report description '{image_report_description}' yields no critical findings or requires more specific details."

# List of all tools
tools = [
    ehr_api_tool,
    medical_database_search_tool,
    drug_interaction_checker_tool,
    image_analysis_service_tool,
]

# --- 2. Initialize LLM and Agent ---

# Ensure OPENAI_API_KEY is set as an environment variable
if "OPENAI_API_KEY" not in os.environ:
    st.error("OPENAI_API_KEY environment variable not set. Please set it to use the OpenAI LLM.")
    st.stop()

llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

# Create the Langchain agent
agent = create_tool_calling_agent(llm, tools, st.session_state.prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. Streamlit UI ---
st.set_page_config(page_title="AI Medical Assistant", layout="wide")
st.title("🩺 AI-Powered Medical Assistant")
st.write("This assistant helps healthcare professionals with diagnostics and treatment planning by orchestrating specialized medical tools.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.prompt = "You are a helpful AI medical assistant. Your goal is to assist healthcare professionals by answering their questions, providing diagnostic insights, and suggesting treatment plans. You have access to specialized medical tools. Always try to use the most relevant tool(s) to answer the user's query. If you need more information, ask clarifying questions. Explain your reasoning and sources (tools used) clearly."

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the agent with the current conversation history
                # Langchain expects a list of messages for history
                # We will construct this from st.session_state.messages
                chat_history_for_agent = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        chat_history_for_agent.append(("human", msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history_for_agent.append(("ai", msg["content"]))

                response = agent_executor.invoke({"input": prompt, "chat_history": chat_history_for_agent})
                assistant_response = response["output"]
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"An error occurred: {e}. Please check your OpenAI API key and try again.")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})

st.sidebar.header("Configuration")
st.sidebar.markdown("This application requires an OpenAI API key set as an environment variable `OPENAI_API_KEY`.")
st.sidebar.text_area("Agent Initial Prompt (modify with caution)", st.session_state.prompt, height=200, key="prompt_editor")
if st.sidebar.button("Update Prompt"):
    st.session_state.prompt = st.session_state.prompt_editor
    st.sidebar.success("Prompt updated! Restart the conversation for changes to take full effect.")
    # Re-initialize agent with new prompt if needed, though for tool_calling_agent, prompt is generally fixed after creation
    # For this simple example, we'll just update the session state variable. A more robust solution might re-create the agent.
