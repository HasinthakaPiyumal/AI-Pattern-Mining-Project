
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub
import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
from contextlib import redirect_stdout


# --- 0. Environment Setup and Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Dynamic Knowledge Base Layer ---
class KnowledgeBase:
    def __init__(self):
        self.vectorstore = None
        # Initialize SentenceTransformerEmbeddings only once
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def initialize_vectorstore(self, documents=None):
        if documents:
            st.info("Creating vector store from provided documents...")
            texts = self.text_splitter.split_documents(documents)
            self.vectorstore = Chroma.from_documents(texts, self.embedding_model, persist_directory="./chroma_db")
            st.success("Vector store created/updated.")
        else:
            # Try to load existing vectorstore
            try:
                self.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=self.embedding_model)
                st.info("Loaded existing vector store.")
            except Exception as e:
                st.warning(f"No existing vector store found or error loading: {e}. Please add documents.")
                self.vectorstore = None

    def add_documents_to_kb(self, documents):
        if self.vectorstore:
            st.info(f"Adding {len(documents)} new documents to the knowledge base...")
            texts = self.text_splitter.split_documents(documents)
            self.vectorstore.add_documents(texts)
            self.vectorstore.persist()
            st.success("Documents added and vector store persisted.")
        else:
            self.initialize_vectorstore(documents)

# --- 2. Knowledge Processing & Refinement (Simplified/Mocked) ---
class KnowledgeProcessor:
    def __init__(self):
        # self.nlp = spacy.load("en_core_web_sm") # Mock spacy loading
        logging.info("KnowledgeProcessor initialized (spaCy mocked).")

    def extract_entities(self, text):
        # Mocking Named Entity Recognition
        mock_entities = {
            "disease": ["diabetes", "hypertension", "flu"],
            "drug": ["metformin", "lisinopril", "paracetamol"],
            "symptom": ["fever", "cough", "headache"]
        }
        extracted = {}
        for entity_type, keywords in mock_entities.items():
            found = [kw for kw in keywords if kw in text.lower()]
            if found:
                extracted[entity_type] = found
        return extracted

    def structure_data(self, data_list):
        # Mocking data structuring with pandas
        if not data_list:
            return pd.DataFrame()
        return pd.DataFrame(data_list)

# --- 3. Tool Definitions ---

# Mock API for Drug Interaction Checker
def check_drug_interactions(drugs: str) -> str:
    """Checks for potential drug-drug interactions between a comma-separated list of drugs."""
    logging.info(f"Checking drug interactions for: {drugs}")
    drugs_list = [d.strip().lower() for d in drugs.split(',')]
    interactions = []
    if "metformin" in drugs_list and "contrast dye" in drugs_list:
        interactions.append("Metformin and contrast dye: Risk of lactic acidosis.")
    if "warfarin" in drugs_list and "ibuprofen" in drugs_list:
        interactions.append("Warfarin and Ibuprofen: Increased risk of bleeding.")
    if not interactions:
        return f"No significant interactions found for {drugs} based on mock data."
    return "Potential interactions: " + "; ".join(interactions)

# Mock API for EHR Access
def get_patient_ehr(patient_id: str, data_type: str) -> str:
    """Retrieves specific data (e.g., 'medical_history', 'allergies', 'medications', 'lab_results') for a given patient ID."""
    logging.info(f"Accessing EHR for patient {patient_id}, data type: {data_type}")
    mock_ehr_data = {
        "P1001": {
            "medical_history": "Type 2 Diabetes, Hypertension",
            "allergies": "Penicillin",
            "medications": "Metformin 500mg BID, Lisinopril 10mg OD",
            "lab_results": "HbA1c: 7.2%, Creatinine: 1.1 mg/dL"
        },
        "P1002": {
            "medical_history": "Asthma",
            "allergies": "Dust mites",
            "medications": "Salbutamol inhaler PRN",
            "lab_results": "FEV1: 85% predicted"
        }
    }
    if patient_id in mock_ehr_data and data_type in mock_ehr_data[patient_id]:
        return f"Patient {patient_id} {data_type}: {mock_ehr_data[patient_id][data_type]}"
    return f"No {data_type} found for patient {patient_id} or patient not found in mock EHR."

# Mock API for Diagnostic Imaging Analysis (placeholder)
def analyze_imaging_report(report_text: str) -> str:
    """Analyzes a diagnostic imaging report text for key findings and potential abnormalities."""
    logging.info(f"Analyzing imaging report: {report_text[:50]}...")
    if "nodule" in report_text.lower() or "mass" in report_text.lower():
        return "Potential abnormality detected: Further investigation recommended for suspected nodule/mass."
    return "No significant abnormalities noted in the report based on mock analysis."

# Simplified Controlled Web Browsing
def controlled_web_browse(url: str) -> str:
    """Safely browses a whitelisted URL to extract main content. Only allows trusted medical sites."""
    trusted_domains = ["pubmed.ncbi.nlm.nih.gov", "clinicaltrials.gov", "www.who.int", "www.cdc.gov"]
    parsed_url = requests.utils.urlparse(url)
    if parsed_url.netloc not in trusted_domains:
        return f"Error: {url} is not a trusted medical domain for browsing."

    logging.info(f"Browsing trusted URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract main content, e.g., paragraphs
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return "\n".join(paragraphs[:5]) # Return first 5 paragraphs for brevity
    except requests.exceptions.RequestException as e:
        return f"Error accessing {url}: {e}"

# --- 4. Application Logic & Orchestration Layer (LangChain Agent) ---
class ICDSSAgent:
    def __init__(self, knowledge_base: KnowledgeBase):
        # Ensure OPENAI_API_KEY is set before initializing ChatOpenAI
        if "OPENAI_API_KEY" not in os.environ:
            st.error("OPENAI_API_KEY not found. Please set it as an environment variable.")
            st.stop()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0) # Use a powerful LLM
        self.knowledge_base = knowledge_base

        # Retriever for RAG
        self.retriever = self.knowledge_base.vectorstore.as_retriever() if self.knowledge_base.vectorstore else None
        if self.retriever:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
        else:
            self.qa_chain = None
            st.warning("RAG chain not initialized: No vector store available.")

        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()

    def _setup_tools(self):
        tools = [
            Tool(
                name="DrugInteractionChecker",
                func=check_drug_interactions,
                description="Use this tool to check for potential drug-drug interactions. Input should be a comma-separated string of drug names."
            ),
            Tool(
                name="EHR_Access",
                func=lambda x: get_patient_ehr(*x.split(',')),
                description="Use this tool to retrieve specific patient data from the Electronic Health Record. Input format: 'patient_id,data_type' (e.g., 'P1001,medical_history'). Supported data_types: medical_history, allergies, medications, lab_results."
            ),
            Tool(
                name="DiagnosticImagingAnalysis",
                func=analyze_imaging_report,
                description="Use this tool to analyze text from diagnostic imaging reports for key findings and potential abnormalities."
            ),
            Tool(
                name="ControlledWebBrowsing",
                func=controlled_web_browse,
                description="Use this tool to safely browse pre-approved, trusted medical websites for emerging information. Input should be a valid URL from a trusted domain (e.g., pubmed.ncbi.nlm.nih.gov)." +
                            " Example: https://pubmed.ncbi.nlm.nih.gov/32345678/"
            )
        ]
        # Conditionally add RAG tool if vectorstore is initialized
        if self.qa_chain:
             tools.append(
                Tool(
                    name="MedicalKnowledgeRetriever",
                    func=lambda query: self.qa_chain({"query": query})["result"],
                    description="Use this tool to retrieve relevant medical knowledge (research papers, guidelines, case studies) from the internal knowledge base. Input should be a clear medical query."
                )
            )
        return tools

    def _setup_agent(self):
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    def process_query(self, query: str):
        st.subheader("Agent Thought Process (Check your console for full logs):")
        with st.expander("Show detailed thought process"):
            # Redirect stdout to capture agent's verbose output
            f = io.StringIO()
            with redirect_stdout(f):
                response = self.agent_executor.invoke({"input": query})
            st.code(f.getvalue())
        return response['output']


# --- 5. Streamlit User Interface ---
def main():
    st.set_page_config(page_title="Intelligent Clinical Decision Support System (ICDSS)", layout="wide")
    st.title("👨‍⚕️ Intelligent Clinical Decision Support System (ICDSS)")
    st.markdown("""
        This AI-powered platform assists healthcare professionals with clinical decision-making by leveraging a
        Dynamic Knowledge-Augmented LLM. It integrates external knowledge, patient data, and specialized tools.
        **Note:** This is a prototype; all external API calls are mocked or simplified.
    """)

    # Initialize Knowledge Base
    kb = KnowledgeBase()
    # Initialize Processor (mostly mocked for this single file)
    kp = KnowledgeProcessor() # Not directly used in the Streamlit UI, but shows architectural component

    # Sidebar for Knowledge Base Management
    st.sidebar.header("Knowledge Base Management")
    st.sidebar.markdown("Upload PDFs or provide URLs to augment the medical knowledge base.")

    # Ensure temp_docs directory exists for PDF uploads
    os.makedirs("./temp_docs", exist_ok=True)

    uploaded_files = st.sidebar.file_uploader("Upload PDF Medical Documents", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join("./temp_docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        if documents:
            kb.add_documents_to_kb(documents)
        else:
            st.sidebar.warning("No documents loaded from uploaded PDFs.")

    url_to_load = st.sidebar.text_input("Load document from URL (e.g., PubMed article):", key="url_input")
    if st.sidebar.button("Add URL to KB") and url_to_load:
        try:
            loader = WebBaseLoader(url_to_load)
            documents = loader.load()
            if documents:
                kb.add_documents_to_kb(documents)
            else:
                st.sidebar.warning(f"No content found at the provided URL: {url_to_load}")
        except Exception as e:
            st.sidebar.error(f"Error loading URL: {e}")

    if st.sidebar.button("Initialize/Load Existing KB or Create New"):
        kb.initialize_vectorstore()

    # Main Chat Interface
    st.header("Ask the ICDSS for Clinical Support")
    user_query = st.text_area("Enter your clinical question or scenario:", height=150, key="user_query_input")

    if "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] == "":
        st.warning("Please set your OPENAI_API_KEY as an environment variable or in a .env file to use the LLM.")
        st.info("You can get an API key from https://platform.openai.com/account/api-keys")
        st.stop()

    if user_query and st.button("Get ICDSS Recommendation"):
        if kb.vectorstore is None:
            st.error("Please initialize or add documents to the Knowledge Base first in the sidebar before asking a question.")
        else:
            with st.spinner("Processing your query..."):
                try:
                    icdss_agent = ICDSSAgent(knowledge_base=kb)
                    response = icdss_agent.process_query(user_query)
                    st.success("Query processed!")
                    st.subheader("ICDSS Recommendation:")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred during query processing: {e}")
                    st.exception(e)

    st.markdown("---")
    st.subheader("Ethical & Privacy Considerations (Prototype Acknowledgment)")
    st.info("""
        *   **Data Privacy:** All patient data used in this prototype is mocked and anonymized. In a real system,
            strict HIPAA compliance, data encryption, and access controls would be paramount.
        *   **Human-in-the-Loop:** This system is designed to **assist**, not replace, healthcare professionals.
            All recommendations must be reviewed and validated by a qualified human expert.
        *   **Bias Mitigation:** Continuous monitoring and evaluation would be required to identify and mitigate
            biases in the LLM's responses or the underlying knowledge base.
        *   **Transparency:** The agent's thought process (enabled by `verbose=True` in LangChain) helps in understanding
            how it arrived at a recommendation.
    """)


if __name__ == "__main__":
    main()
