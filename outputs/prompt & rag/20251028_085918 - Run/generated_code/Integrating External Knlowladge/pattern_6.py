"""
# Dynamic Medical Assistant for Clinicians

This project implements a Dynamic Medical Assistant for Clinicians, an AI application designed to provide real-time, evidence-based medical information. It integrates a Large Language Model (LLM) with various external knowledge sources and tools to enhance factual accuracy, mitigate hallucinations, and support clinical decision-making.

### Pattern Implemented: Dynamic Knowledge-Augmented LLMs

This assistant embodies the "Dynamic Knowledge-Augmented LLMs" pattern by:
-   **Integrating Diverse External Resources**: Connecting to simulated Electronic Health Records (EHR), medical research databases, drug information APIs, and clinical guidelines.
-   **Utilizing RAG Systems**: Storing and retrieving medical knowledge as vectors in a vector database (ChromaDB) for context-aware responses.
-   **Enabling Tool Use**: Allowing the LLM to interact with external functions (tools) to fetch specific, up-to-date information.
-   **Enhancing Factual Accuracy**: By prioritizing information from integrated sources over the LLM's static pretraining knowledge.

### Architecture Highlights

The core architecture consists of:
1.  **LLM Core**: A central large language model (e.g., OpenAI's GPT models) that processes clinician queries.
2.  **Knowledge Retrieval (RAG)**: A Retrieval-Augmented Generation system using a vector store (ChromaDB) and embeddings (e.g., Sentence Transformers or OpenAIEmbeddings) to fetch relevant medical documents and snippets.
3.  **External Tooling**: A set of specialized functions (tools) that the LLM can invoke to access real-time or structured data from:
    *   **Simulated EHR**: `get_patient_record`
    *   **Simulated Drug Database**: `get_drug_information`
    *   **Simulated Medical Research Database**: `search_medical_research`
    *   **Simulated Clinical Guidelines Database**: `get_clinical_guidelines`
4.  **Agent Orchestration**: A Langchain `AgentExecutor` that decides when to use the knowledge base (RAG) and when to invoke specific tools based on the clinician's query.

### How to Run

1.  **Install Dependencies**: Make sure you have the following Python libraries installed. You can install them using pip:
    ```bash
    pip install langchain langchain-community langchain-openai chromadb sentence-transformers python-dotenv
    ```
2.  **Set up OpenAI API Key**: If you intend to use `ChatOpenAI`, you need an OpenAI API key. Create a `.env` file in the same directory as this script and add your API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Alternatively, ensure `OPENAI_API_KEY` is set in your environment variables.
3.  **Run the Script**: Execute the Python script from your terminal:
    ```bash
    python medical_assistant_app.py
    ```

You can then interact with the medical assistant by typing your queries.

### Example Queries:

*   "What is the patient record for P001?"
*   "Tell me about Metformin."
*   "Search for recent research on SGLT2 inhibitors and heart failure."
*   "What are the clinical guidelines for hypertension management?"
*   "Can you provide information on Bob Johnson's condition?"

"""

import os
import json
import datetime
from typing import List, Dict, Any

# --- Load environment variables (for API keys) ---
from dotenv import load_dotenv
load_dotenv()

# --- Langchain Core Imports ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Langchain LLM and Embeddings Imports ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings # For local embeddings

# --- Langchain Vectorstore Imports ---
from langchain_community.vectorstores import Chroma

# --- Langchain Agent Imports ---
from langchain.agents import AgentExecutor, create_react_agent, Tool


# --- Simulated External Resources (APIs, EHR) ---
# In a real-world scenario, these would be actual API calls to external systems.

_MOCK_PATIENT_RECORDS = {
    "P001": {
        "id": "P001",
        "name": "Alice Smith",
        "age": 62,
        "condition": "Type 2 Diabetes",
        "medications": ["Metformin", "Insulin"],
        "allergies": ["Penicillin"],
        "last_visit": "2023-10-26",
        "notes": "Patient reports good glycemic control with current regimen."
    },
    "P002": {
        "id": "P002",
        "name": "Bob Johnson",
        "age": 55,
        "condition": "Hypertension",
        "medications": ["Lisinopril"],
        "allergies": [],
        "last_visit": "2023-11-15",
        "notes": "Blood pressure well-controlled on Lisinopril."
    }
}

_MOCK_DRUG_DATABASE = {
    "Metformin": {
        "name": "Metformin",
        "class": "Biguanide",
        "indications": "Type 2 Diabetes",
        "dosage": "500-2000 mg/day",
        "side_effects": "Nausea, diarrhea, lactic acidosis (rare)",
        "interactions": ["Alcohol", "Iodinated contrast media"],
        "contraindications": "Severe renal impairment"
    },
    "Lisinopril": {
        "name": "Lisinopril",
        "class": "ACE Inhibitor",
        "indications": "Hypertension, Heart Failure, Post-MI",
        "dosage": "5-40 mg/day",
        "side_effects": "Cough, dizziness, hyperkalemia, angioedema (rare)",
        "interactions": ["Potassium-sparing diuretics", "NSAIDs"]
    }
}

_MOCK_MEDICAL_RESEARCH_ARTICLES = [
    "A meta-analysis on the efficacy of SGLT2 inhibitors in heart failure patients with preserved ejection fraction. Published in Circulation, 2023. Key finding: Significant reduction in hospitalizations.",
    "New guidelines for the management of chronic kidney disease: KDIGO 2024 updates. Key recommendations include early screening and multidisciplinary care.",
    "Comparative study of different insulin regimens for glycemic control in hospitalized diabetic patients. Findings suggest basal-bolus regimen superiority.",
    "Impact of dietary interventions on gut microbiome composition and metabolic health in obese individuals. A randomized controlled trial published in Nature Medicine, 2022."
]

_MOCK_CLINICAL_GUIDELINES = [
    "**AHA/ACC Guideline for the Management of Hypertension (2017)**: Recommends blood pressure target <130/80 mmHg for most adults. Lifestyle modifications are foundational.",
    "**ADA Standards of Medical Care in Diabetes (2023)**: Emphasizes individualized glycemic targets, metformin as first-line for most, and consideration of GLP-1 RAs or SGLT2 inhibitors for cardiovascular/renal benefits.",
    "**NICE Guideline for Chronic Heart Failure (2018)**: Recommends ACE inhibitors, beta-blockers, and MRAs for symptomatic patients with reduced ejection fraction. Device therapy considered for selected patients."
]

def get_patient_record(patient_id: str) -> str:
    """Retrieves patient medical records from a simulated EHR system. Input: patient ID (e.g., 'P001')."""
    print(f"\n[TOOL CALL] Retrieving patient record for ID: {patient_id}")
    record = _MOCK_PATIENT_RECORDS.get(patient_id)
    if record:
        return json.dumps(record, indent=2)
    return json.dumps({"error": f"Patient with ID {patient_id} not found."})

def get_drug_information(drug_name: str) -> str:
    """Retrieves detailed drug information (dosage, side effects, interactions) from a simulated drug database API. Input: drug name (e.g., 'Metformin')."""
    print(f"\n[TOOL CALL] Retrieving drug information for: {drug_name}")
    info = _MOCK_DRUG_DATABASE.get(drug_name.capitalize()) # Basic capitalization for mock
    if info:
        return json.dumps(info, indent=2)
    return json.dumps({"error": f"Drug {drug_name} not found."})

def search_medical_research(query: str) -> str:
    """Searches a simulated medical research database for relevant articles. Input: concise search query (e.g., 'SGLT2 inhibitors heart failure')."""
    print(f"\n[TOOL CALL] Searching medical research for query: {query}")
    results = [
        article for article in _MOCK_MEDICAL_RESEARCH_ARTICLES
        if query.lower() in article.lower()
    ]
    if results:
        return json.dumps(results, indent=2)
    return json.dumps({"message": f"No specific research found for '{query}'. Consider broadening your search or using related terms."})

def get_clinical_guidelines(topic: str) -> str:
    """Retrieves relevant clinical guidelines from a simulated database. Input: medical topic (e.g., 'hypertension management')."""
    print(f"\n[TOOL CALL] Retrieving clinical guidelines for topic: {topic}")
    results = [
        guideline for guideline in _MOCK_CLINICAL_GUIDELINES
        if topic.lower() in guideline.lower()
    ]
    if results:
        return json.dumps(results, indent=2)
    return json.dumps({"message": f"No specific guidelines found for '{topic}'. Here are general guidelines: " + ', '.join(_MOCK_CLINICAL_GUIDELINES[:1])})


# --- Embeddings and Vector Store Setup (ChromaDB) ---

# Choose embeddings model:
# Option 1: OpenAIEmbeddings (requires OPENAI_API_KEY)
# embeddings = OpenAIEmbeddings()
# Option 2: Sentence-Transformers (local, no API key needed, but model download on first run)
#    model_name can be 'all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5', etc.
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB persistent client
CHROMA_DB_PATH = "./chroma_db"
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

def initialize_vectorstore_with_data(vectorstore: Chroma):
    """Populates the vector store with initial medical knowledge if not already populated."""
    print("Initializing vector store with medical data...")
    existing_docs = vectorstore.get()
    if existing_docs['ids']:
        print("Vector store already contains data. Skipping initial population.")
        return

    documents_to_add = []
    for record_id, record_data in _MOCK_PATIENT_RECORDS.items():
        documents_to_add.append(Document(page_content=json.dumps(record_data), metadata={"source": "EHR", "id": record_id}))
    for drug_name, drug_data in _MOCK_DRUG_DATABASE.items():
        documents_to_add.append(Document(page_content=json.dumps(drug_data), metadata={"source": "DrugDB", "name": drug_name}))
    for i, article in enumerate(_MOCK_MEDICAL_RESEARCH_ARTICLES):
        documents_to_add.append(Document(page_content=article, metadata={"source": "Research", "id": f"ART{i+1}"}))
    for i, guideline in enumerate(_MOCK_CLINICAL_GUIDELINES):
        documents_to_add.append(Document(page_content=guideline, metadata={"source": "Guidelines", "id": f"GL{i+1}"}))

    vectorstore.add_documents(documents_to_add)
    vectorstore.persist()
    print("Vector store populated and persisted.")

initialize_vectorstore_with_data(vectorstore)

# Create a retriever for RAG
retriever = vectorstore.as_retriever()


# --- LLM Setup ---
# For a real application, replace with a robust LLM provider.
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini") # Using gpt-4o-mini for cost-effectiveness and good performance


# --- Define Tools for the LLM Agent ---
# These tools allow the LLM to interact with external systems/data sources.

tools = [
    Tool(
        name="get_patient_record",
        func=get_patient_record,
        description="Useful for retrieving a patient's comprehensive medical record using their patient ID. Input should be a patient ID (e.g., 'P001')."
    ),
    Tool(
        name="get_drug_information",
        func=get_drug_information,
        description="Useful for finding detailed information about a specific drug, including dosage, side effects, interactions, and contraindications. Input should be a drug name (e.g., 'Metformin')."
    ),
    Tool(
        name="search_medical_research",
        func=search_medical_research,
        description="Useful for searching up-to-date medical research articles and studies in scientific databases. Input should be a concise search query (e.g., 'SGLT2 inhibitors heart failure outcomes')."
    ),
    Tool(
        name="get_clinical_guidelines",
        func=get_clinical_guidelines,
        description="Useful for retrieving established clinical guidelines from authoritative bodies for various medical conditions or topics. Input should be a medical topic (e.g., 'diabetes management guidelines')."
    ),
]


# --- Agent Prompt and Executor Setup ---

# Define the agent prompt template
# This prompt guides the LLM on its role, tools, and how to respond.
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a highly knowledgeable and accurate Dynamic Medical Assistant for Clinicians. "
         "Your primary goal is to provide real-time, evidence-based medical information to support diagnosis, "
         "treatment planning, and patient education. Always strive for factual accuracy and mitigate hallucinations "
         "by extensively using the tools and retrieved context provided. "
         "If a user asks for information that can be found using a tool, prioritize using the tool. "
         "Also, integrate relevant information from the knowledge base when appropriate, especially for general medical context or facts not directly covered by specific tools."
         "When presenting information, cite your sources (e.g., 'from patient record P001', 'according to Metformin drug info', 'as per Circulation 2023 study', 'AHA/ACC Guideline')."
         f"Current Date: {datetime.date.today().isoformat()}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the ReAct agent
# The agent combines the LLM, tools, and a prompt to reason and act.
agent = create_react_agent(llm, tools, agent_prompt)

# Create the AgentExecutor to run the agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Set to True to see the agent's thought process
    handle_parsing_errors=True,
    max_iterations=10 # Limit iterations to prevent infinite loops
)

# --- RAG Chain (for when direct tool use isn't obvious, but context is needed) ---
# This chain is used to retrieve context from the vector store and augment the LLM's response.

rag_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful medical assistant. Use the following retrieved context to answer the question accurately. "
     "If the context does not contain enough information, state that you cannot provide a definitive answer "
     "based on the provided context, but do not make up information. "
     "When answering, cite the source of the context if possible. "
     "Retrieved Context:\n{context}"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

rag_chain = (
    RunnablePassthrough.assign(context=retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])))
    | rag_prompt_template
    | llm
    | StrOutputParser()
)


# --- Main Application Loop ---
if __name__ == "__main__":
    print("Welcome to the Dynamic Medical Assistant for Clinicians!")
    print("Type 'exit' to quit. Type 'help' for example queries.")

    chat_history = []

    while True:
        query = input("\nClinician Query: ")
        if query.lower() == 'exit':
            break
        elif query.lower() == 'help':
            print("\nExample Queries:\n")
            print("  - What is the patient record for P001?")
            print("  - Tell me about Lisinopril.")
            print("  - Search for recent research on chronic kidney disease management.")
            print("  - What are the clinical guidelines for diabetes?")
            print("  - Summarize Alice Smith's condition based on her record.")
            print("  - What are the side effects of Metformin?")
            continue

        try:
            # Determine if a tool is explicitly requested or if general RAG is sufficient
            # This is a simple heuristic; a more advanced agent can decide this implicitly
            if any(tool_name in query.lower() for tool_name in [t.name.replace('_', ' ') for t in tools]):
                print("\n--- Agent Deciding (Tool Use Likely) ---")
                response = agent_executor.invoke({"input": query, "chat_history": chat_history})
                final_response = response["output"]
            else:
                print("\n--- RAG Chain (Context Retrieval) ---")
                # For queries not explicitly targeting a tool, use the RAG chain
                response = rag_chain.invoke({"input": query, "chat_history": chat_history})
                final_response = response

            print(f"\nMedical Assistant: {final_response}")
            chat_history.append(("human", query))
            chat_history.append(("ai", final_response))

        except Exception as e:
            print(f"\nMedical Assistant: An error occurred: {e}")
            print("Please try again or rephrase your query.")
            # Optionally, log the full traceback for debugging
            # import traceback
            # traceback.print_exc()
