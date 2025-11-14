
import os
from dotenv import load_dotenv
import streamlit as st
from fastapi import FastAPI
import uvicorn
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from loguru import logger
import json
import requests # For FastAPI to Streamlit communication
import threading # For running FastAPI in a separate thread
import time # For checking FastAPI readiness

# --- 0. Environment Setup ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it in a .env file or as an environment variable.")
    st.stop()

logger.add("medical_assistant.log", rotation="500 MB")
logger.info("Medical Assistant application started.")

# --- Pydantic Models for Data Validation ---
class DiagnosisRecommendation(BaseModel):
    diagnosis: str = Field(..., description="The most likely medical diagnosis.")
    recommendations: List[str] = Field(..., description="List of recommended treatments or next steps.")
    potential_risks: List[str] = Field(default=[], description="Potential risks associated with the recommendations.")
    confidence_score: Optional[float] = Field(None, description="Confidence score (0-1) for the diagnosis.")
    sources: List[str] = Field(default=[], description="List of knowledge sources used for this recommendation.")

class PatientInput(BaseModel):
    symptoms: str = Field(..., description="Patient's reported symptoms.")
    medical_history: Optional[str] = Field(None, description="Relevant medical history of the patient.")
    lab_results: Optional[str] = Field(None, description="Relevant lab results.")
    additional_questions: Optional[str] = Field(None, description="Any specific questions for the assistant.")

class ChatInput(BaseModel):
    message: str = Field(..., description="User's chat message.")
    history: List[Dict[str, str]] = Field(default=[], description="Conversation history.")

# --- 1. RAG System Initialization ---
@st.cache_resource
def initialize_rag_system():
    logger.info("Initializing RAG system...")
    # Mock Medical Documents
    medical_docs = [
        "Influenza (Flu) is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness, and at times can lead to death. Symptoms include fever, cough, sore throat, runny or stuffy nose, muscle or body aches, headaches, and fatigue. Antiviral drugs can be used to treat flu. Vaccination is recommended annually.",
        "Common cold is a viral infectious disease of the upper respiratory tract that primarily affects the nose. Symptoms include coughing, sore throat, runny nose, sneezing, and fever. There is no cure for the common cold, but symptoms can be treated with over-the-counter medications like pain relievers and decongestants. Rest and fluids are important.",
        "Diabetes Mellitus is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Symptoms include frequent urination, increased thirst, increased hunger, weight loss, fatigue, and blurred vision. Treatment involves insulin therapy, oral medications, diet, and exercise.",
        "Hypertension (High Blood Pressure) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Symptoms are often not present, hence it's called a 'silent killer'. Treatment includes lifestyle changes (diet, exercise) and medications (e.g., ACE inhibitors, diuretics).",
        "Allergies are a number of conditions caused by hypersensitivity of the immune system to typically harmless substances in the environment. These diseases include hay fever, food allergies, atopic dermatitis, allergic asthma, and anaphylaxis. Symptoms vary widely depending on the allergen. Treatment involves avoiding allergens, antihistamines, corticosteroids, and immunotherapy."
    ]

    # Initialize embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize ChromaDB
    # Using a persistent client for demonstration, adjust for production
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_or_create_collection(name="medical_knowledge")
        if collection.count() == 0:
            logger.info("Adding documents to ChromaDB...")
            collection.add(documents=medical_docs, ids=[f"doc_{i}" for i in range(len(medical_docs))])
            logger.info(f"Added {len(medical_docs)} documents to ChromaDB.")
        else:
            logger.info(f"ChromaDB already contains {collection.count()} documents.")
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        st.error(f"Error initializing ChromaDB: {e}. Please ensure ChromaDB can write to ./chroma_db")
        st.stop()

    vectorstore = Chroma(client=client, collection_name="medical_knowledge", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    logger.info("RAG system initialized successfully.")
    return retriever

rag_retriever = initialize_rag_system()

# --- 2. Mock Internal Patient Database ---
patient_data = pd.DataFrame({
    "patient_id": ["P001", "P002", "P003"],
    "age": [45, 30, 60],
    "gender": ["Male", "Female", "Male"],
    "symptoms": [
        "persistent cough, fatigue, mild fever",
        "runny nose, sore throat, sneezing",
        "frequent urination, increased thirst, weight loss"
    ],
    "diagnosis": [
        "Influenza",
        "Common Cold",
        "Type 2 Diabetes"
    ],
    "treatment": [
        "Antiviral medication, rest, fluids",
        "Over-the-counter cold medicine, rest, fluids",
        "Insulin therapy, diet control"
    ],
    "medications": [
        "Oseltamivir",
        "Paracetamol",
        "Metformin, Insulin Glargine"
    ]
})

# --- 3. Custom LangChain Tools ---
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

@tool
def search_medical_database(query: str) -> str:
    """Searches the internal patient database for relevant information based on keywords.
    Input should be a keyword or a simple phrase to search in symptoms, diagnosis, or medications.
    Returns a string summary of matching patient records."""
    logger.info(f"Searching internal medical database for: {query}")
    results = patient_data[patient_data.apply(lambda row: query.lower() in str(row).lower(), axis=1)]
    if not results.empty:
        return results.to_json(orient="records", indent=2)
    return "No matching patient records found in the internal database."

@tool
def check_drug_interaction(drug1: str, drug2: str) -> str:
    """Simulates checking for drug interactions between two specified drugs.
    Returns a string indicating potential interactions or 'No known interactions'."""
    logger.info(f"Checking drug interaction between {drug1} and {drug2}")
    # This is a highly simplified simulation
    if "warfarin" in drug1.lower() and "aspirin" in drug2.lower():
        return f"**WARNING**: High risk of bleeding when {drug1} and {drug2} are taken together. Consult a doctor immediately."
    elif "metformin" in drug1.lower() and "iodinated contrast" in drug2.lower():
        return f"**WARNING**: Potential for lactic acidosis when {drug1} and {drug2} are combined, especially in renal impairment. Monitor kidney function."
    elif "amoxicillin" in drug1.lower() and "methotrexate" in drug2.lower():
        return f"**WARNING**: Amoxicillin may reduce methotrexate clearance, increasing toxicity. Monitor methotrexate levels."
    else:
        return f"No significant known interactions reported between {drug1} and {drug2} in this simulated check. Always refer to official drug information."

@tool
def web_search_medical(query: str) -> str:
    """Performs a web search for up-to-date medical information, guidelines, or research findings.
    Input should be a clear medical question or topic."""
    logger.info(f"Performing web search for: {query}")
    search_tool = DuckDuckGoSearchRun()
    try:
        return search_tool.run(query)
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        return f"Could not perform web search for '{query}'. Error: {e}"

# --- 4. LangChain Agent Setup ---

# Tool for RAG system (specifically for querying the vector store)
@tool
def knowledge_base_query(query: str) -> str:
    """Queries the specialized medical knowledge base (vector database) for relevant documents.
    Use this tool when you need general medical information, disease descriptions, or treatment guidelines.
    Input should be a concise question or keywords related to the medical topic."""
    logger.info(f"Querying knowledge base for: {query}")
    docs = rag_retriever.get_relevant_documents(query)
    if docs:
        return "\n---\n".join([doc.page_content for doc in docs])
    return "No relevant information found in the knowledge base."

# Define the tools available to the agent
tools = [
    search_medical_database,
    check_drug_interaction,
    web_search_medical,
    knowledge_base_query
]

# Agent Prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly intelligent and helpful medical assistant. Your primary goal is to assist medical professionals with diagnoses and treatment recommendations based on the provided patient information, your internal knowledge, and external tools. Always strive for accuracy, provide explanations, and cite sources when possible. If you need to check for drug interactions, ask for both drug names. If a diagnosis or recommendation is not clear, state the uncertainty. After providing a diagnosis and recommendations, always ask if the user has further questions."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 5. FastAPI Backend ---
app = FastAPI()

@app.post("/diagnose", response_model=DiagnosisRecommendation)
async def diagnose_patient(patient_input: PatientInput):
    logger.info(f"Received diagnosis request for symptoms: {patient_input.symptoms}")
    query = f"Patient symptoms: {patient_input.symptoms}. " \
            f"Medical history: {patient_input.medical_history if patient_input.medical_history else 'None'}. " \
            f"Lab results: {patient_input.lab_results if patient_input.lab_results else 'None'}. " \
            f"Additional questions: {patient_input.additional_questions if patient_input.additional_questions else 'None'}. " \
            f"Please provide a diagnosis, treatment recommendations, potential risks, and cite any sources used."
    
    try:
        response = agent_executor.invoke({"input": query, "chat_history": []})
        agent_output = response["output"]
        logger.info(f"Agent output for diagnosis: {agent_output}")
        
        # Attempt to parse the agent's output into the Pydantic model
        # This requires the LLM to output in a structured format.
        # For demonstration, we'll try a regex or direct parse, but for robust apps,
        # a structured output prompt or output parser would be better.
        
        # Simple parsing logic (can be improved with LLM outputting JSON directly)
        diagnosis = "No clear diagnosis generated." # Default
        recommendations = []
        potential_risks = []
        sources = []
        confidence = None

        if "Diagnosis:" in agent_output:
            diagnosis_start = agent_output.find("Diagnosis:") + len("Diagnosis:")
            recommendations_start = agent_output.find("Recommendations:")
            diagnosis = agent_output[diagnosis_start:recommendations_start].strip() if recommendations_start != -1 else agent_output[diagnosis_start:].strip()
        
        if "Recommendations:" in agent_output:
            rec_start = agent_output.find("Recommendations:") + len("Recommendations:")
            risks_start = agent_output.find("Potential Risks:")
            recs_text = agent_output[rec_start:risks_start].strip() if risks_start != -1 else agent_output[rec_start:].strip()
            recommendations = [item.strip() for item in recs_text.split('\n') if item.strip() and not item.strip().startswith('---')]
        
        if "Potential Risks:" in agent_output:
            risks_start = agent_output.find("Potential Risks:") + len("Potential Risks:")
            sources_start = agent_output.find("Sources:")
            risks_text = agent_output[risks_start:sources_start].strip() if sources_start != -1 else agent_output[risks_start:].strip()
            potential_risks = [item.strip() for item in risks_text.split('\n') if item.strip() and not item.strip().startswith('---')]

        if "Sources:" in agent_output:
            src_start = agent_output.find("Sources:") + len("Sources:")
            sources_text = agent_output[src_start:].strip()
            sources = [item.strip() for item in sources_text.split('\n') if item.strip()]

        if "Confidence Score:" in agent_output:
            conf_start = agent_output.find("Confidence Score:") + len("Confidence Score:")
            conf_end = agent_output.find("\n", conf_start)
            try:
                confidence = float(agent_output[conf_start:conf_end].strip())
            except ValueError:
                confidence = None # Handle parsing error

        return DiagnosisRecommendation(
            diagnosis=diagnosis,
            recommendations=recommendations if recommendations else [agent_output], # Fallback if parsing fails
            potential_risks=potential_risks,
            confidence_score=confidence,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error during diagnosis: {e}")
        return DiagnosisRecommendation(
            diagnosis="An error occurred during diagnosis.",
            recommendations=[f"Please try again or refine your input. Error: {e}"],
            sources=[]
        )

@app.post("/chat")
async def chat_with_assistant(chat_input: ChatInput):
    logger.info(f"Received chat message: {chat_input.message}")
    try:
        # LangChain agent expects chat_history in a specific format
        formatted_history = []
        for msg in chat_input.history:
            if msg["role"] == "human":
                formatted_history.append(("human", msg["content"]))
            elif msg["role"] == "ai":
                formatted_history.append(("ai", msg["content"]))
        
        response = agent_executor.invoke({"input": chat_input.message, "chat_history": formatted_history})
        return {"reply": response["output"]}
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        return {"reply": f"An error occurred during chat: {e}"}

# --- FastAPI Server Control ---
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start FastAPI in a separate thread
fastapi_thread = threading.Thread(target=run_fastapi)
fastapi_thread.daemon = True  # Allows the main program to exit even if the thread is running
fastapi_thread.start()

# Wait for FastAPI to be ready
def wait_for_fastapi(host="0.0.0.0", port=8000, timeout=30):
    url = f"http://{host}:{port}/docs"
    start_time = time.time()
    while True:
        try:
            requests.get(url, timeout=1)
            logger.info("FastAPI is ready.")
            return True
        except requests.ConnectionError:
            time.sleep(0.5)
            if time.time() - start_time > timeout:
                logger.error("FastAPI did not become ready in time.")
                return False
        except Exception as e:
            logger.error(f"Error checking FastAPI readiness: {e}")
            return False

if not wait_for_fastapi():
    st.error("Could not start FastAPI backend. Please check logs for details.")
    st.stop()

# --- 6. Streamlit Frontend ---
st.set_page_config(layout="wide", page_title="Intelligent Medical Assistant")
st.title("🧠 Intelligent Medical Assistant")

BASE_URL = "http://localhost:8000"

# Sidebar for information
st.sidebar.header("About")
st.sidebar.info(
    "This Intelligent Medical Assistant leverages a Large Language Model augmented with a dynamic knowledge base "
    "(RAG system), internal patient data, and external tools (simulated drug interaction, web search) to provide "
    "diagnoses and treatment recommendations. It's designed to assist medical professionals."
)
st.sidebar.header("How to Use")
st.sidebar.markdown(
    "1. **Diagnosis**: Enter patient symptoms and other details to get a diagnosis and recommendations."
    "2. **Chat**: Ask general medical questions or follow-up questions related to a diagnosis."
)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Main Tabs ---
tab1, tab2 = st.tabs(["🔍 Patient Diagnosis", "💬 Medical Chat"]) 

with tab1:
    st.header("Patient Diagnosis and Treatment Recommendations")
    st.write("Provide patient details to receive a potential diagnosis and treatment plan.")

    with st.form(key='diagnosis_form'):
        symptoms = st.text_area("Patient Symptoms (e.g., 'Persistent cough, fever for 3 days, body aches'):", height=100)
        medical_history = st.text_area("Relevant Medical History (optional):", height=70)
        lab_results = st.text_area("Recent Lab Results (optional):", height=70)
        additional_questions = st.text_area("Additional Questions/Concerns (optional):", height=70)

        submit_button = st.form_submit_button(label='Get Diagnosis')

        if submit_button:
            if not symptoms:
                st.warning("Please enter patient symptoms to get a diagnosis.")
            else:
                st.info("Generating diagnosis and recommendations...")
                patient_data_input = PatientInput(
                    symptoms=symptoms,
                    medical_history=medical_history,
                    lab_results=lab_results,
                    additional_questions=additional_questions
                )
                try:
                    response = requests.post(f"{BASE_URL}/diagnose", json=patient_data_input.model_dump())
                    response.raise_for_status() # Raise an exception for HTTP errors
                    diagnosis_output = DiagnosisRecommendation(**response.json())
                    
                    st.subheader("Diagnosis")
                    st.success(diagnosis_output.diagnosis)

                    if diagnosis_output.confidence_score is not None:
                        st.metric(label="Confidence Score", value=f"{diagnosis_output.confidence_score:.2f}")

                    st.subheader("Treatment Recommendations")
                    for rec in diagnosis_output.recommendations:
                        st.markdown(f"- {rec}")
                    
                    if diagnosis_output.potential_risks:
                        st.subheader("Potential Risks")
                        for risk in diagnosis_output.potential_risks:
                            st.warning(f"- {risk}")

                    if diagnosis_output.sources:
                        st.subheader("Sources")
                        for source in diagnosis_output.sources:
                            st.caption(f"- {source}")

                    st.success("Diagnosis complete. Feel free to ask follow-up questions in the 'Medical Chat' tab.")

                    # Clear previous chat history if new diagnosis is made
                    st.session_state.chat_history = [] 

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend. Is FastAPI running?")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error from backend: {e}. Details: {response.text if response else 'No response'}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

with tab2:
    st.header("Medical Chat")
    st.write("Ask general medical questions or follow up on a diagnosis.")

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask me anything medical..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                # Prepare history for the backend (only content and role)
                chat_history_for_api = []
                for msg in st.session_state.chat_history:
                    chat_history_for_api.append({"role": msg["role"], "content": msg["content"]})

                chat_input_data = ChatInput(message=prompt, history=chat_history_for_api[:-1]) # Exclude current prompt from history sent to API
                response = requests.post(f"{BASE_URL}/chat", json=chat_input_data.model_dump())
                response.raise_for_status()
                assistant_response = response.json()["reply"]
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Is FastAPI running?")
            except requests.exceptions.RequestException as e:
                st.error(f"Error from backend: {e}. Details: {response.text if response else 'No response'}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

