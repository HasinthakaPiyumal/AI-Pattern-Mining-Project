
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.tools import GoogleSerperAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# 1. Environment Setup
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()

# 5. Tool Integration Module (Simulated/Mocked and Real)

@tool
def drug_interaction_checker(drugs: str) -> str:
    """Checks for potential interactions between a comma-separated list of drugs. Returns interaction details."""
    drug_list = [d.strip().lower() for d in drugs.split(',')]
    interactions = {
        ("ibuprofen", "warfarin"): "Severe: Increased risk of bleeding.",
        ("metformin", "lisinopril"): "Moderate: Potential for increased risk of hypoglycemia and renal dysfunction. Monitor patient closely.",
        ("omeprazole", "clopidogrel"): "Moderate: Omeprazole can reduce the antiplatelet effect of clopidogrel."
    }
    
    found_interactions = []
    for i in range(len(drug_list)):
        for j in range(i + 1, len(drug_list)):
            pair1 = (drug_list[i], drug_list[j])
            pair2 = (drug_list[j], drug_list[i])
            if pair1 in interactions:
                found_interactions.append(f"Interaction between {drug_list[i].capitalize()} and {drug_list[j].capitalize()}: {interactions[pair1]}")
            elif pair2 in interactions:
                found_interactions.append(f"Interaction between {drug_list[j].capitalize()} and {drug_list[i].capitalize()}: {interactions[pair2]}")

    if not found_interactions:
        return f"No significant interactions found for: {', '.join(drug_list)}. (Note: This is a simulated checker and may not be exhaustive.)"
    else:
        return "\n".join(found_interactions) + " (Note: This is a simulated checker and may not be exhaustive.)"

@tool
def ehr_data_query(patient_id: str, fields: str) -> str:
    """Queries a simulated Electronic Health Record (EHR) system for patient information. 
    Provide the patient_id and a comma-separated list of fields (e.g., 'diagnosis', 'medications', 'allergies', 'age').
    Returns the requested de-identified patient data."""
    simulated_ehr_data = {
        "P123": {
            "diagnosis": "Type 2 Diabetes, Hypertension",
            "medications": "Metformin (500mg BID), Lisinopril (10mg QD)",
            "allergies": "Penicillin",
            "age": "62",
            "last_visit": "2023-10-26"
        },
        "P456": {
            "diagnosis": "Asthma, Seasonal Allergies",
            "medications": "Albuterol (PRN), Loratadine (10mg QD)",
            "allergies": "None reported",
            "age": "35",
            "last_visit": "2024-01-15"
        }
    }

    patient_data = simulated_ehr_data.get(patient_id.upper())
    if not patient_data:
        return f"Patient ID {patient_id} not found in the simulated EHR system."

    requested_fields = [f.strip().lower() for f in fields.split(',')]
    result = []
    for field in requested_fields:
        if field in patient_data:
            result.append(f"{field.replace('_', ' ').capitalize()}: {patient_data[field]}")
        else:
            result.append(f"Field '{field}' not found for patient {patient_id}.")
    
    return "\n".join(result) + " (Data is simulated and de-identified.)"

# Google Serper API Tool
# Note: This requires a SERPER_API_KEY in your .env file
class CustomGoogleSerperAPIWrapper(GoogleSerperAPIWrapper):
    def _get_api_key(self) -> str:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("SERPER_API_KEY not found. Please set it in your .env file.")
        return api_key

google_search_tool = CustomGoogleSerperAPIWrapper()
google_search_tool.name = "google_search"
google_search_tool.description = "A search engine for general web queries, useful for finding up-to-date information, latest guidelines, and recent research in medicine."

# Combine tools
tools = [drug_interaction_checker, ehr_data_query, google_search_tool]


# 4. Knowledge Augmentation (RAG) System Setup

# Simulated Medical Data for PoC
medical_data_raw = [
    "Type 2 Diabetes Mellitus (T2DM) is a chronic metabolic disorder characterized by high blood glucose levels. Management typically involves lifestyle modifications, oral medications, and sometimes insulin. Recent guidelines emphasize early and aggressive management to prevent complications.",
    "Hypertension, or high blood pressure, significantly increases the risk of heart disease and stroke. Treatment often includes diet, exercise, and antihypertensive medications like ACE inhibitors, ARBs, calcium channel blockers, and diuretics.",
    "Aspirin is commonly used as an antiplatelet agent to prevent cardiovascular events, but it carries a risk of bleeding, especially in combination with other anticoagulants.",
    "Metformin is a first-line medication for Type 2 Diabetes, working by decreasing glucose production by the liver and improving insulin sensitivity. Common side effects include gastrointestinal issues.",
    "Lisinopril is an ACE inhibitor used to treat hypertension and heart failure. It can cause a dry cough and, in rare cases, angioedema. Regular monitoring of kidney function and potassium levels is recommended.",
    "Penicillin is a common antibiotic, but many patients have reported allergic reactions, ranging from mild rashes to severe anaphylaxis. It is crucial to document and verify penicillin allergies.",
    "The latest research on Alzheimer's disease focuses on amyloid-beta plaques and tau tangles, with new therapeutic approaches targeting these pathologies. Early diagnosis and intervention are key.",
    "CRISPR-Cas9 technology allows for precise gene editing and holds immense promise for treating genetic diseases. Recent advancements include base editing and prime editing for even greater precision."
]

# Initialize embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents(medical_data_raw)

# Create ChromaDB vector store
vectorstore = Chroma.from_documents(documents=texts, embedding=embedding_model)
retriever = vectorstore.as_retriever()

# 3. Large Language Model (LLM) Module
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)


# 2. Backend / Orchestration Layer

# Setup memory for conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt template for the agent
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful and highly knowledgeable medical assistant for clinicians. Your goal is to provide accurate, up-to-date, and evidence-based medical information, patient insights, and drug interaction checks. Always prioritize patient safety and suggest consulting a medical professional for definitive diagnoses and treatment plans. When answering, if you use information from external sources (search, EHR, drug checker), mention it. Be concise and precise."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

# Optional: Conversational Retrieval Chain for direct RAG if agent decides it's best
# For simplicity, we'll let the agent handle everything, including deciding when to 'retrieve' via its general knowledge or by using search tool if the data is not in the vectorstore.
# A more complex setup might have a dedicated RAG chain the agent can explicitly call.


# 1. User Interface (UI) Layer - Streamlit
st.set_page_config(page_title="Dynamic Medical Assistant for Clinicians", layout="wide")
st.title("🩺 Dynamic Medical Information Assistant")
st.markdown("--- This AI assistant helps clinicians by providing up-to-date medical information, personalized patient insights, and drug interaction checks. --- ")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a medical question or check patient/drug info..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # The agent will decide whether to use RAG (implicitly via its LLM knowledge or explicitly via search tool) or other tools.
                response = agent_executor.invoke({"input": prompt})
                assistant_response = response["output"]
            except Exception as e:
                assistant_response = f"An error occurred: {e}. Please try again or rephrase your question."
            
            st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

st.sidebar.header("Instructions")
st.sidebar.info(
    "**Examples of questions you can ask:**\n\n"
    "- *'What are the latest guidelines for managing Type 2 Diabetes?'* (Uses RAG/Search)\n"
    "- *'Check interactions between Ibuprofen, Warfarin, and Omeprazole.'* (Uses Drug Interaction Tool)\n"
    "- *'What are the current medications and allergies for patient P123?'* (Uses EHR Data Query Tool)\n"
    "- *'What is CRISPR-Cas9 technology?'* (Uses RAG/Search)\n"
    "- *'Summarize the key points about hypertension treatment.'* (Uses RAG)"
)

st.sidebar.markdown("--- This is a demonstration. Always verify information with official medical sources. ---")
