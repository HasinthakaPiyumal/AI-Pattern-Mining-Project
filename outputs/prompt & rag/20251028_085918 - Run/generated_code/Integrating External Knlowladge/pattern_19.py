
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub
import os

# --- Environment Setup (Mock for demonstration) ---
# In a real application, you'd load these from environment variables
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# --- 1. Knowledge Base Setup (RAG - Medical Literature) ---
def setup_medical_literature_rag():
    # Simulate loading medical literature
    # In a real scenario, this would come from actual medical texts, journals, etc.
    medical_docs_content = [
        "Recent studies indicate that a new antiviral drug, Remdesivir, shows promise in treating severe cases of viral pneumonia.",
        "Hypertension management often involves lifestyle modifications such as diet and exercise, alongside medications like ACE inhibitors or diuretics.",
        "Diabetes Mellitus Type 2 is characterized by insulin resistance and relative insulin deficiency. Treatment pathways include metformin, SGLT2 inhibitors, and GLP-1 receptor agonists.",
        "The symptoms of myocardial infarction include chest pain radiating to the left arm, shortness of breath, and sweating. Immediate medical attention is crucial."
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(medical_docs_content)

    # Use a dummy embedding model if OpenAI API key is not set, for local testing
    if os.getenv("OPENAI_API_KEY"):
        embeddings = OpenAIEmbeddings()
    else:
        # Placeholder for local embeddings or a message to set API key
        st.warning("OPENAI_API_KEY not set. Using a dummy embedding placeholder. RAG functionality will be limited.")
        class DummyEmbeddings:
            def embed_query(self, text): return [0.0] * 1536 # Dummy vector
            def embed_documents(self, texts): return [[0.0] * 1536 for _ in texts]
        embeddings = DummyEmbeddings()

    # Initialize ChromaDB as our vector store for medical literature
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db_medical_literature")
    return vectorstore

# --- 2. Tool Definitions (Simulated External Resources) ---
def medical_literature_search(query: str) -> str:
    """Searches the curated medical literature knowledge base for relevant information."""
    vectorstore = st.session_state.get("medical_literature_vectorstore")
    if not vectorstore:
        return "Medical literature search is not initialized. Please set OPENAI_API_KEY if you want real embeddings."
    
    results = vectorstore.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in results]) if results else "No relevant medical literature found."

def ehr_lookup(patient_id: str) -> str:
    """Looks up patient-specific data from the Electronic Health Records (EHR) system."""
    # This is a mock function. In a real system, it would query an actual EHR database.
    if patient_id == "P12345":
        return "Patient P12345: Age 65, Male, Diagnosed with Type 2 Diabetes (last visit 2023-01-15), Medications: Metformin 500mg BID. Allergies: Penicillin."
    elif patient_id == "P67890":
        return "Patient P67890: Age 42, Female, Diagnosed with Hypertension (last visit 2023-03-20), Medications: Lisinopril 10mg QD. No known allergies."
    else:
        return f"No EHR data found for patient ID: {patient_id}."

def drug_interaction_check(drug1: str, drug2: str) -> str:
    """Checks for potential interactions between two specified drugs using a medical knowledge graph."""
    # This is a mock function. In a real system, it would query a comprehensive drug-drug interaction database.
    drug1 = drug1.lower()
    drug2 = drug2.lower()
    if "metformin" in [drug1, drug2] and "contrast dye" in [drug1, drug2]:
        return "WARNING: Potential risk of lactic acidosis when Metformin is used with iodinated contrast media. Metformin should be temporarily discontinued."
    elif "lisinopril" in [drug1, drug2] and "potassium supplements" in [drug1, drug2]:
        return "CAUTION: Concomitant use of Lisinopril and potassium supplements may lead to hyperkalemia. Monitor potassium levels closely."
    else:
        return f"No significant interaction found between {drug1} and {drug2} in the knowledge graph."

# Define the list of tools available to the LLM agent
tools = [
    Tool(
        name="MedicalLiteratureSearch",
        func=medical_literature_search,
        description="Useful for searching general medical literature and guidelines for a given medical condition or topic."
    ),
    Tool(
        name="EHR_Lookup",
        func=ehr_lookup,
        description="Useful for retrieving patient-specific information from Electronic Health Records using a patient ID. Input should be a patient ID (e.g., 'P12345')."
    ),
    Tool(
        name="DrugInteractionCheck",
        func=drug_interaction_check,
        description="Useful for checking potential drug-drug interactions between two medications. Input should be two drug names separated by a comma (e.g., 'Metformin, Contrast Dye')."
    ),
]

# --- 3. LLM Agent Setup ---
@st.cache_resource
def initialize_agent_executor():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set the OPENAI_API_KEY environment variable to use the LLM agent.")
        return None

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Get the prompt to use - you can modify this or use a custom one
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Dynamic Medical Assistant", layout="wide")
st.title("🩺 Dynamic Medical Assistant for Clinicians")
st.markdown("---<br>")

st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY"))
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.sidebar.warning("Please enter your OpenAI API Key.")

# Initialize RAG and Agent on first run or when API key is available
if "medical_literature_vectorstore" not in st.session_state:
    st.session_state["medical_literature_vectorstore"] = setup_medical_literature_rag()

if "agent_executor" not in st.session_state and os.getenv("OPENAI_API_KEY"):
    st.session_state["agent_executor"] = initialize_agent_executor()

st.markdown("This AI assistant provides dynamic, knowledge-augmented insights for clinicians by integrating with medical literature, EHRs, and drug interaction databases.")
st.markdown("\n---<br>")

user_query = st.text_area("Enter your medical query or patient scenario:", height=150)

if st.button("Get Assistance"):
    if not os.getenv("OPENAI_API_KEY") or st.session_state.get("agent_executor") is None:
        st.error("Please configure your OpenAI API Key and ensure the assistant is initialized.")
    elif user_query:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state["agent_executor"].invoke({"input": user_query})
                st.subheader("Assistant's Response:")
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Please check the console for detailed error messages or try again with a different query.")
    else:
        st.warning("Please enter a query to get assistance.")

st.markdown("\n---<br>")
st.caption("Disclaimer: This is a prototype and should not be used for actual medical diagnosis or treatment. Always consult with qualified healthcare professionals.")
