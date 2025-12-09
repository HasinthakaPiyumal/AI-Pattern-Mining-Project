import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import os

# --- Mock External Knowledge Sources (Simulated APIs/Databases) ---

@tool
def get_patient_ehr(patient_id: str) -> str:
    """Retrieve summarized Electronic Health Record (EHR) data for a given patient ID. """
    if patient_id == "P12345":
        return "Patient P12345 has a history of type 2 diabetes, hypertension, and allergies to penicillin. Current medications: Metformin, Lisinopril. Recent lab results: HbA1c 7.5%, BP 140/90." 
    elif patient_id == "P67890":
        return "Patient P67890 has no significant medical history, no known allergies. Presenting with flu-like symptoms. Vaccinations up-to-date." 
    else:
        return "Patient ID not found in EHR system."

@tool
def get_drug_interactions(drug_name: str, patient_conditions: str = "") -> str:
    """Retrieve known drug interactions and contraindications for a specified drug, optionally considering patient conditions. """
    if drug_name.lower() == "metformin":
        response = "Known interactions: Cimetidine, contrast dyes (risk of lactic acidosis). Contraindicated in severe renal impairment, metabolic acidosis. "
        if "renal impairment" in patient_conditions.lower():
            response += "Caution advised if patient has renal impairment." 
        return response
    elif drug_name.lower() == "lisinopril":
        response = "Known interactions: NSAIDs, potassium-sparing diuretics. Contraindicated in angioedema. "
        if "hypertension" in patient_conditions.lower():
            response += "Commonly used for hypertension, but monitor blood pressure closely." 
        return response
    else:
        return f"No specific interaction data found for {drug_name}. Consult a full pharmacology reference."

@tool
def search_medical_literature(query: str) -> str:
    """Search a simulated medical literature database (e.g., PubMed) for articles relevant to the query. """
    if "new treatment for diabetes" in query.lower():
        return "Recent studies show promising results for SGLT2 inhibitors in reducing cardiovascular events in diabetic patients. (Simulated search result from PubMed)"
    elif "covid-19 vaccine efficacy" in query.lower():
        return "Large-scale studies confirm high efficacy of mRNA vaccines against severe COVID-19. (Simulated search result from CDC/WHO data)"
    else:
        return f"No specific literature found for '{query}'. Please refine your query or consult specialized databases."

# --- LLM and Agent Setup ---

def setup_agent():
    # Ensure OPENAI_API_KEY is set in environment variables
    if "OPENAI_API_KEY" not in os.environ:
        st.error("OPENAI_API_KEY not found in environment variables. Please set it to use the LLM.")
        st.stop()

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [get_patient_ehr, get_drug_interactions, search_medical_literature]
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful Medical Query Assistant for healthcare professionals. Use the provided tools to get accurate and up-to-date medical information. If patient-specific information is needed, ask for the patient ID. Always aim to provide evidence-based answers."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# --- Streamlit UI ---
st.set_page_config(page_title="Medical Query Assistant", layout="wide")
st.title("🩺 Medical Query Assistant for Healthcare Professionals")
st.markdown("Ask medical questions and get augmented answers using simulated external knowledge bases.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent_executor = setup_agent()

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"): 
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"): 
            st.markdown(message.content)

# User input
user_query = st.chat_input("Ask your medical question here...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"): 
        st.markdown(user_query)
    
    with st.chat_message("assistant"): 
        with st.spinner("Thinking..."):
            try:
                # Pass the chat history to the agent
                response = agent_executor.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
                assistant_response = response["output"]
                st.markdown(assistant_response)
                st.session_state.chat_history.append(AIMessage(content=assistant_response))
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Please ensure your OPENAI_API_KEY is correctly set in your environment variables.")

st.sidebar.header("How to Use")
st.sidebar.markdown("This assistant integrates an LLM with simulated external medical knowledge sources. You can ask questions like:")
st.sidebar.markdown("- `What is the EHR for patient P12345?`")
st.sidebar.markdown("- `What are the contraindications for Metformin?`")
st.sidebar.markdown("- `What are the drug interactions for Lisinopril if the patient has hypertension?`")
st.sidebar.markdown("- `Are there any new treatments for diabetes?`")
st.sidebar.markdown("--- --- --- --- --- --- --- --- ")
st.sidebar.markdown("**Note:** This is a demonstration with mock data. In a real application, these tools would connect to actual databases and APIs.")

