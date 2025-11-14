import os
import uvicorn
import threading
import time
from typing import List, Dict

# Try to import necessary libraries, handle potential ImportError for demonstration
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    # Use HuggingFacePipeline for LLM as specified in architecture, fallback to MockLLM
    from langchain_community.llms import HuggingFacePipeline 
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    import streamlit as st
    import requests
    import json
    import logging
    import shutil # For cleaning ChromaDB
except ImportError as e:
    print(f"A required library is not installed: {e}. Please install them using pip:")
    print("pip install fastapi uvicorn pydantic langchain-community langchain-core langchain transformers sentence-transformers chromadb streamlit requests")
    print("Exiting...")
    exit(1) # Exit if essential libraries are missing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Set to True for a very basic mock LLM if you don't want to download a model
# For a real application, you'd load a suitable LLM.
USE_MOCK_LLM = False # Set to True to force MockLLM, False to try HuggingFacePipeline

# For the HuggingFacePipeline, a small model is chosen for demonstration purposes.
# In a real application, you'd use a more powerful medical-specific LLM.
# Be aware: gpt2 is a general-purpose model, not medical-specific.
# Loading this model requires internet access and some download time on first run.
HF_MODEL_NAME = "gpt2"
HF_TOKENIZER_NAME = "gpt2"
HF_MAX_NEW_TOKENS = 150 # Limit for faster generation and to avoid excessive output

# FastAPI server details
FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000
FASTAPI_BASE_URL = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"

# ChromaDB directory (for persistent storage)
CHROMA_DB_DIR = "./chroma_medical_db"

# --- 1. Knowledge Base Management and Retrieval ---

def setup_knowledge_base():
    """
    Sets up the Chroma vector store with dummy medical data.
    Cleans up existing ChromaDB directory to ensure a fresh start for the demo.
    """
    logger.info("Setting up knowledge base...")

    # Clean up existing ChromaDB directory
    if os.path.exists(CHROMA_DB_DIR):
        logger.info(f"Cleaning up existing ChromaDB directory: {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)

    # Create dummy medical documents
    medical_docs_content = [
        "Patient presents with fever, cough, and shortness of breath. Possible pneumonia or severe acute respiratory syndrome (SARS). Recommend chest X-ray and PCR test for viral pathogens.",
        "Differential diagnosis for headache includes tension headache, migraine, cluster headache, and sinusitis. Consider patient history for triggers and associated symptoms like aura or nasal congestion.",
        "Type 2 Diabetes Mellitus is characterized by insulin resistance and hyperglycemia. Management involves diet, exercise, and medications such as metformin. Regular blood glucose monitoring is crucial.",
        "Hypertension, or high blood pressure, increases the risk of heart disease and stroke. Lifestyle modifications, including reduced sodium intake and regular physical activity, are often the first line of treatment. Antihypertensive drugs may be prescribed.",
        "Symptoms of a heart attack include chest pain, shortness of breath, pain radiating to the arm, and dizziness. Immediate medical attention is vital. ECG and blood tests for cardiac enzymes are diagnostic.",
        "Anaphylaxis is a severe, life-threatening allergic reaction. Symptoms include difficulty breathing, swelling of the face/throat, hives, and a drop in blood pressure. Epinephrine is the primary treatment.",
        "Diagnosis of appendicitis often involves physical examination, blood tests (elevated white blood cell count), and imaging like ultrasound or CT scan. Surgical removal of the appendix is the standard treatment.",
        "Depression is a mood disorder causing persistent sadness and loss of interest. Treatment options include psychotherapy, antidepressant medications, or a combination. Lifestyle changes can also help.",
        "Common cold symptoms typically include runny nose, sore throat, and sneezing. It's a viral infection and usually resolves on its own. Rest and fluids are recommended.",
        "Influenza, or the flu, presents with fever, body aches, fatigue, and respiratory symptoms. Antiviral medications can be prescribed, especially for high-risk individuals. Annual vaccination is recommended.",
        "Osteoarthritis is a degenerative joint disease. Symptoms include joint pain, stiffness, and reduced flexibility. Management includes pain relief, physical therapy, and in severe cases, joint replacement surgery.",
        "Asthma is a chronic respiratory condition causing inflammation and narrowing of the airways. Symptoms include wheezing, coughing, chest tightness, and shortness of breath. Inhalers are commonly used for symptom control and prevention."
    ]

    # Simulate saving to files for TextLoader
    temp_data_dir = "./temp_medical_data"
    if not os.path.exists(temp_data_dir):
        os.makedirs(temp_data_dir)
    file_paths = []
    for i, doc_content in enumerate(medical_docs_content):
        file_path = os.path.join(temp_data_dir, f"doc_{i}.txt")
        with open(file_path, "w") as f:
            f.write(doc_content)
        file_paths.append(file_path)

    # Load documents
    documents = []
    for file_path in file_paths:
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
    
    # Clean up temporary data files
    if os.path.exists(temp_data_dir):
        shutil.rmtree(temp_data_dir)

    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    if not docs:
        logger.error("No documents were loaded or split. Check data ingestion.")
        raise ValueError("No documents available for vector store.")

    # Initialize embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Chroma vector store
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=CHROMA_DB_DIR, collection_name="medical_knowledge"
    )
    db.persist()
    logger.info(f"Knowledge base set up with {len(docs)} documents in {CHROMA_DB_DIR}.")
    return db

# --- 2. Unified Retrieval and Reasoning LLM ---

class MockLLM:
    """A simple mock LLM for demonstration purposes."""
    def invoke(self, prompt: str) -> str:
        logger.info(f"MockLLM received prompt (first 100 chars): {prompt[:100]}...")
        # Simulate some reasoning based on keywords
        prompt_lower = prompt.lower()
        if "pneumonia" in prompt_lower or "fever" in prompt_lower and "cough" in prompt_lower:
            return "Based on the symptoms (fever, cough) and retrieved context, possible diagnosis includes pneumonia or SARS. Further tests like chest X-ray and PCR for viral pathogens are recommended to confirm."
        elif "headache" in prompt_lower:
            return "Considering the headache symptoms, differential diagnoses include migraine, tension headache, or sinusitis. Patient history regarding triggers, aura, and associated symptoms like nasal congestion is crucial for a precise diagnosis."
        elif "diabetes" in prompt_lower:
            return "The information points to Type 2 Diabetes Mellitus, characterized by insulin resistance and hyperglycemia. Management strategies typically involve diet, exercise, and medications such as metformin. Regular blood glucose monitoring is essential."
        elif "hypertension" in prompt_lower:
            return "Hypertension (high blood pressure) is indicated. Recommend lifestyle modifications, particularly reduced sodium intake and increased physical activity. Antihypertensive drugs may also be prescribed depending on severity."
        elif "heart attack" in prompt_lower:
            return "Symptoms (chest pain, shortness of breath, radiating arm pain) suggest a potential heart attack. Immediate medical attention is critical. Diagnostic tests include ECG and blood tests for cardiac enzymes."
        elif "anaphylaxis" in prompt_lower:
            return "This clinical picture is highly suggestive of anaphylaxis, a severe, life-threatening allergic reaction. Immediate administration of epinephrine is the primary treatment, along with supportive care."
        elif "appendicitis" in prompt_lower:
            return "Based on the query, appendicitis is a strong consideration. Diagnosis often involves physical examination, blood tests (elevated white blood cell count), and imaging like ultrasound or CT scan. Surgical removal of the appendix is the standard treatment."
        elif "depression" in prompt_lower:
            return "Symptoms align with depression, a mood disorder characterized by persistent sadness and loss of interest. Treatment options typically include psychotherapy, antidepressant medications, or a combination of both. Lifestyle changes can also be beneficial."
        elif "cold" in prompt_lower:
            return "Symptoms are consistent with a common cold, a viral infection that usually resolves on its own. Rest and fluids are recommended for symptom relief."
        elif "flu" in prompt_lower:
            return "Symptoms indicate influenza (flu), presenting with fever, body aches, fatigue, and respiratory symptoms. Antiviral medications can be prescribed, especially for high-risk individuals. Annual vaccination is highly recommended for prevention."
        elif "osteoarthritis" in prompt_lower:
            return "The findings suggest osteoarthritis, a degenerative joint disease. Management typically includes pain relief (e.g., NSAIDs), physical therapy, and in severe cases, joint replacement surgery to improve quality of life."
        elif "asthma" in prompt_lower:
            return "The symptoms are indicative of asthma, a chronic respiratory condition. Inhalers are commonly used for symptom control and prevention of exacerbations. Identification and avoidance of triggers are also important aspects of management."
        else:
            return "Based on the provided information, I can offer general diagnostic insights. For a more focused diagnosis, please provide more specific details about the patient's symptoms and medical history."


def initialize_llm_chain(vector_store):
    """
    Initializes the LangChain retrieval and reasoning chain.
    """
    logger.info("Initializing LLM chain...")

    retriever = vector_store.as_retriever()

    global USE_MOCK_LLM
    if USE_MOCK_LLM:
        llm = MockLLM()
        logger.warning("Using MockLLM. For real inference, set USE_MOCK_LLM = False in config.")
    else:
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME)
            model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
            
            # Create a Hugging Face pipeline for text generation
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=HF_MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id # Important for some models to avoid errors
            )
            llm = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"Using HuggingFace LLM: {HF_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {HF_MODEL_NAME}: {e}")
            logger.warning("Falling back to MockLLM due to HuggingFace model loading error.")
            llm = MockLLM()
            USE_MOCK_LLM = True # Ensure mock is used if HF fails


    template = """You are a highly intelligent Healthcare Diagnostic Assistant.
    Your task is to analyze patient symptoms and medical context to suggest potential diagnoses, recommend further tests, and explain your reasoning.
    Synthesize information from the provided medical context and the user's query to give a comprehensive answer.

    Medical Context:
    {context}

    Patient Symptoms/Query: {question}

    Your Diagnosis and Recommendations (keep it concise but informative):
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Construct the RAG chain
    # The 'context' RunnablePassthrough converts the list of Document objects from the retriever
    # into a single string by joining their page_content.
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: "\n".join([doc.page_content for doc in x["context"]])))
        | prompt
        | llm
        | StrOutputParser()
    )

    # The main RAG chain that first retrieves context, then passes it to the LLM.
    # 'question' is passed through directly from the input.
    # 'context' is retrieved using the retriever.
    # 'answer' is generated by rag_chain_from_docs, which consumes the 'question' and 'context'.
    rag_chain = RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    ).assign(answer=rag_chain_from_docs) # The final output will contain 'context', 'question', and 'answer'
    
    logger.info("LLM chain initialized successfully.")
    return rag_chain

# --- FastAPI Application ---
app = FastAPI(
    title="Healthcare Diagnostic Assistant API",
    description="API for a unified retrieval and reasoning system to assist medical professionals.",
    version="1.0.0",
)
rag_chain_instance = None # Will be initialized on startup

class DiagnosticQuery(BaseModel):
    query: str

class DiagnosticResponse(BaseModel):
    diagnosis: str
    sources: List[str]

@app.on_event("startup")
async def startup_event():
    """Initializes the knowledge base and LLM chain on application startup."""
    global rag_chain_instance
    logger.info("FastAPI startup event triggered.")
    try:
        vector_store = setup_knowledge_base()
        rag_chain_instance = initialize_llm_chain(vector_store)
        logger.info("FastAPI: Knowledge base and LLM chain loaded.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # In a production environment, you might want a more graceful shutdown or error page
        raise HTTPException(status_code=500, detail=f"Failed to initialize diagnostic system: {str(e)}")

@app.post("/diagnose", response_model=DiagnosticResponse)
async def diagnose(query_data: DiagnosticQuery):
    """
    Endpoint to receive a diagnostic query and return a diagnosis,
    recommendations, and the sources used.
    """
    logger.info(f"Received diagnostic query: {query_data.query[:100]}...")
    if rag_chain_instance is None:
        raise HTTPException(status_code=500, detail="Diagnostic system not initialized. Please check server logs.")

    try:
        # LangChain's .invoke() returns a dictionary when RunnableParallel is used
        # The input to invoke() for the 'rag_chain' (RunnableParallel) is just the question string.
        # This question string is then passed to the 'question' key, and also used by the 'context' retriever.
        result = rag_chain_instance.invoke(query_data.query)
        
        # Extract the answer from the 'answer' key in the result
        generated_diagnosis = result.get('answer', "Could not generate a specific diagnosis.")
        
        # Extract sources from the 'context' key (which contains Document objects)
        retrieved_docs = result.get('context', [])
        # For sources, let's include metadata if available (e.g., filename), otherwise just content
        sources = []
        for doc in retrieved_docs:
            source_info = doc.page_content
            if doc.metadata and 'source' in doc.metadata:
                source_info = f"Source: {doc.metadata['source']} - {doc.page_content}"
            sources.append(source_info)

        logger.info("Diagnosis generated successfully.")
        return DiagnosticResponse(diagnosis=generated_diagnosis, sources=sources)
    except Exception as e:
        logger.error(f"Error during diagnosis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during diagnosis: {str(e)}")


# --- Streamlit UI (Client) ---

def run_streamlit_app():
    """
    Streamlit client application to interact with the FastAPI backend.
    """
    st.set_page_config(page_title="Healthcare Diagnostic Assistant", layout="wide")
    st.title("👨‍⚕️ Healthcare Diagnostic Assistant")
    st.markdown("---")

    st.write(
        """
        This intelligent assistant helps medical professionals by integrating
        medical knowledge and leveraging Large Language Models to suggest diagnoses
        and recommend further tests.
        """
    )

    query_input = st.text_area(
        "Enter patient symptoms or a diagnostic question:",
        height=150,
        placeholder="e.g., Patient has severe chest pain radiating to the left arm, shortness of breath, and dizziness. What could be the diagnosis and what tests are recommended?",
        key="diagnostic_query"
    )

    if st.button("Get Diagnosis", use_container_width=True):
        if query_input:
            with st.spinner("Analyzing symptoms and generating diagnosis..."):
                try:
                    # Make a request to the FastAPI backend
                    response = requests.post(
                        f"{FASTAPI_BASE_URL}/diagnose",
                        json={"query": query_input}
                    )
                    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                    
                    data = response.json()
                    st.success("Diagnosis Generated!")
                    st.subheader("Potential Diagnosis and Recommendations:")
                    st.markdown(f"**Diagnosis:** {data.get('diagnosis', 'No diagnosis could be generated.')}")

                    st.subheader("Information Used (Sources):")
                    if data.get("sources"):
                        for i, source in enumerate(data.get("sources")):
                            st.markdown(f"- `{source[:200]}...`") # Show first 200 chars
                    else:
                        st.info("No specific sources retrieved for this query (may happen with Mock LLM or general queries).")
                        
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to the FastAPI server at {FASTAPI_BASE_URL}. Please ensure the server is running in a separate terminal.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"API Error: {e.response.status_code} - {e.response.text}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON response from the server. Check server logs for errors.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Please enter some symptoms or a question to get a diagnosis.")

    st.markdown("---")
    st.info(
        "**Disclaimer:** This is a demonstration system and should *not* be used for actual medical diagnosis or treatment. Always consult with a qualified medical professional for healthcare decisions."
    )


# --- Main execution block ---
if __name__ == "__main__":
    # Determine if running as Streamlit or directly
    # Streamlit sets an environment variable
    if os.getenv("STREAMLIT_SERVER_CMDLINE"):
        run_streamlit_app()
    else:
        # This block runs when the script is executed directly (e.g., `python your_script.py`)
        # or via `uvicorn your_script_name:app`.
        logger.info("Attempting to start FastAPI server.")
        
        # Provide instructions for running both components
        print("\n" + "="*80)
        print("                 Healthcare Diagnostic Assistant Setup")
        print("="*80)
        print("\nTo run this application, you need to execute two separate commands in two different terminal windows:\n")
        
        print("1. **Start the FastAPI Backend Server:**")
        print(f"   Open a terminal and run: `uvicorn {os.path.basename(__file__).replace('.py', '')}:app --host {FASTAPI_HOST} --port {FASTAPI_PORT} --reload`")
        print(f"   (The `--reload` flag is optional but useful for development.)")
        print(f"   Wait until you see messages like 'Uvicorn running on {FASTAPI_BASE_URL}'")
        
        print("\n2. **Start the Streamlit Frontend Client:**")
        print(f"   Open a *second* terminal and run: `streamlit run {os.path.basename(__file__)}`")
        print("   This will open the Streamlit application in your web browser.")
        
        print("\n" + "="*80)
        print("The FastAPI server will attempt to load the LLM and knowledge base on startup.")
        print("If using a HuggingFace model, this might take some time on the first run to download the model.")
        print("If there are errors, ensure all required libraries are installed.")
        print("="*80 + "\n")

        # To avoid the program exiting immediately if not run by uvicorn, we just pass.
        # The instructions above guide the user on how to correctly run the server and client.
        pass