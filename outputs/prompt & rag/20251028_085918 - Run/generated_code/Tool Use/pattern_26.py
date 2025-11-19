import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List

load_dotenv()

# --- Simulated Data for Knowledge Base ---
MEDICAL_DOCUMENTS = [
    "Diabetes Mellitus: A chronic condition characterized by high blood sugar levels. Type 1 diabetes is an autoimmune disease, while Type 2 is often linked to lifestyle factors. Treatment involves insulin, medication, diet, and exercise.",
    "Hypertension (High Blood Pressure): A common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Management includes lifestyle changes, diet, exercise, and medications.",
    "Asthma: A condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out and shortness of breath. Treatment often involves inhalers and avoiding triggers.",
    "Common cold: A viral infection of your nose and throat. It's usually harmless, although it might not feel that way. Symptoms include a runny nose, sore throat, cough, congestion, and sometimes body aches or headache. Treatment is supportive, focusing on symptom relief.",
    "Pain Relievers: Over-the-counter options include ibuprofen, acetaminophen, and naproxen. Prescription pain relievers include opioids, which should be used with caution due to addiction risk.",
    "Cardiologist: A doctor who specializes in the study and treatment of heart diseases and heart conditions.",
    "Endocrinologist: A doctor who specializes in glands and hormones, and the diseases that affect them, such as diabetes and thyroid conditions."
]

# --- Embedding Model and Vector Store for RAG ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.create_documents(MEDICAL_DOCUMENTS)

# Using a Sentence-Transformer model via HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings_model)
retriever = vectorstore.as_retriever()

# --- Simulated Tools ---

@tool
def medical_knowledge_search(query: str) -> str:
    """Searches a medical knowledge base for information related to a given query. Use this tool to find definitions, treatments, or general information about medical conditions, drugs, or specialists."""
    results = retriever.invoke(query)
    if results:
        return "\n".join([doc.page_content for doc in results])
    return "No relevant medical information found."

@tool
def clinical_guidelines_api(condition: str) -> str:
    """Retrieves standardized clinical treatment guidelines for a specific medical condition. Provide the exact condition name."""
    guidelines = {
        "Diabetes Mellitus": "Standard guidelines for Type 2 Diabetes include regular blood sugar monitoring, metformin as a first-line medication, dietary changes, and increased physical activity. Annual eye exams and foot checks are also recommended.",
        "Hypertension": "Clinical guidelines for hypertension recommend lifestyle modifications (DASH diet, reduced sodium, exercise) and, if necessary, medications like ACE inhibitors, ARBs, calcium channel blockers, or diuretics, depending on patient specifics."
    }
    return guidelines.get(condition, f"No specific clinical guidelines found for {condition}.")

@tool
def drug_interaction_checker_api(medications: List[str]) -> str:
    """Checks for potential drug-to-drug interactions between a list of specified medications. Provide medications as a list of strings."""
    med_set = set([m.lower() for m in medications])
    if "ibuprofen" in med_set and "warfarin" in med_set:
        return "WARNING: Increased risk of bleeding when combining Ibuprofen and Warfarin. Consult a doctor immediately."
    if "metformin" in med_set and "contrast dye" in med_set:
        return "CAUTION: Metformin and IV contrast dye can lead to kidney issues. Discuss with your physician."
    return f"No significant interactions found for {', '.join(medications)}. Always consult a healthcare professional."

@tool
def appointment_scheduler(specialty: str, date: str = "any", time: str = "any") -> str:
    """Simulates scheduling an appointment with a healthcare specialist. Specify the desired specialty, and optionally a preferred date and time. Returns a confirmation message or availability."""
    if specialty.lower() == "endocrinologist" and date == "any":
        return f"Appointment with an Endocrinologist confirmed for November 15th, 2023, at 10:00 AM. A confirmation email has been sent."
    if specialty.lower() == "cardiologist":
        return f"Appointments for Cardiologist are available next week. Please specify a preferred day."
    return f"Could not schedule an appointment for {specialty}. Please try again with different criteria."

@tool
def insurance_billing_info(query: str) -> str:
    """Retrieves information regarding insurance coverage, co-pays, deductibles, or billing procedures. Provide specific questions about insurance or billing."""
    if "deductible" in query.lower():
        return "Your annual deductible for in-network services is $1000. It resets every January 1st."
    if "co-pay" in query.lower():
        return "Your co-pay for specialist visits is $40, and for primary care visits is $20."
    return "Please provide a more specific query regarding insurance or billing."

@tool
def medical_terminology_explainer(term: str) -> str:
    """Explains complex medical terminology in simple, understandable language. Uses the medical knowledge base for definitions."""
    explanation = medical_knowledge_search(f"Explain {term} simply.")
    if "No relevant medical information found" in explanation:
        explanation = medical_knowledge_search(term) # Broader search if direct explanation fails
    return f"Here's a simplified explanation of '{term}': {explanation}"

@tool
def personalized_health_record_access(patient_id: str, query: str) -> str:
    """(SIMULATED - In a real scenario, this requires stringent security) Accesses relevant anonymized or consented patient health data for personalized recommendations. Provide a patient ID and a specific query. Returns a dummy health summary."""
    if patient_id == "PAT123":
        if "allergies" in query.lower():
            return "Patient PAT123 has reported allergies to Penicillin."
        if "current medications" in query.lower():
            return "Patient PAT123 is currently taking Metformin (for Type 2 Diabetes) and Lisinopril (for Hypertension)."
        return "Patient PAT123: No specific health record information found for that query in this simulation."
    return "Patient ID not recognized or unauthorized access attempt."

all_tools = [
    medical_knowledge_search,
    clinical_guidelines_api,
    drug_interaction_checker_api,
    appointment_scheduler,
    insurance_billing_info,
    medical_terminology_explainer,
    personalized_health_record_access
]

# --- LLM and Agent Setup ---

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Memory for conversation history
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Healthcare Navigator AI. Assist patients and providers with medical information, appointments, and billing. Use the available tools when necessary."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the agent
agent = create_tool_calling_agent(llm, all_tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=all_tools, memory=memory, verbose=True, handle_parsing_errors=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Healthcare Navigator AI", layout="centered")
st.title("🏥 Healthcare Navigator AI")
st.markdown("I can assist you with medical information, appointments, drug interactions, and billing queries.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def get_response(query):
    with st.spinner("Thinking..."):
        try:
            # The agent_executor uses the internal memory directly
            response = agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            return f"An error occurred: {e}"

if prompt := st.chat_input("Ask me a question about healthcare..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_content = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    with st.chat_message("assistant"):
        st.markdown(response_content)
