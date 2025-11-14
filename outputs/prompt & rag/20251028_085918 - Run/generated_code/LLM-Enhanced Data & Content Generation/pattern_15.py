
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from typing import List, Dict, Any, Optional

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Streamlit Imports (commented out for combined FastAPI/Streamlit execution, instructions provided) ---
# import streamlit as st

# --- LLM/RAG Framework Imports ---
# Mock imports and classes for demonstration purposes as actual LLM models and vector stores cannot be run here.
# In a real application, you would replace these mocks with actual implementations.

# Mock for transformers pipeline
class MockPipeline:
    def __init__(self, task: str, model: str):
        self.task = task
        self.model = model
        logger.info(f"MockPipeline initialized for task: {task} with model: {model}")

    def __call__(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        logger.info(f"MockPipeline called with text: {text}")
        if self.task == "text-generation":
            return [{"generated_text": f"LLM response to '{text}': This is a simulated generation based on your input."}]
        elif self.task == "sentiment-analysis": # Example of another task
            return [{"label": "POSITIVE", "score": 0.99}]
        return [{"output": f"Mocked output for {self.task} based on '{text}'"}]

# Mock for SentenceTransformer
class MockSentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"MockSentenceTransformer initialized with model: {model_name}")
        self._dimension = 768 # Standard dimension for many models

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        logger.info(f"MockSentenceTransformer encoding {len(sentences)} sentences.")
        # Simulate embeddings with random numpy arrays
        return np.random.rand(len(sentences), self._dimension)

# Mock for ChromaDB client
class MockChromaClient:
    def __init__(self, path: str):
        self.path = path
        self.collections = {}
        logger.info(f"MockChromaClient initialized at path: {path}")

    def get_or_create_collection(self, name: str) -> 'MockChromaCollection':
        if name not in self.collections:
            self.collections[name] = MockChromaCollection(name)
        return self.collections[name]

class MockChromaCollection:
    def __init__(self, name: str):
        self.name = name
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.embeddings = []
        logger.info(f"MockChromaCollection '{name}' created.")

    def add(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str], embeddings: Optional[np.ndarray] = None):
        logger.info(f"MockChromaCollection '{self.name}' adding {len(documents)} documents.")
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        if embeddings is not None:
            # In a real scenario, embeddings would be stored and used.
            self.embeddings.extend(embeddings.tolist()) # Convert to list for mock storage

    def query(self, query_texts: List[str], n_results: int = 2, **kwargs) -> Dict[str, Any]:
        logger.info(f"MockChromaCollection '{self.name}' querying for: {query_texts[0]}")
        # Simulate retrieval by returning a couple of random documents
        if not self.documents:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        num_docs = len(self.documents)
        if num_docs < n_results:
            n_results = num_docs

        # Simple simulated retrieval: return the first n_results documents
        # In a real system, this would involve similarity search using embeddings
        return {
            "documents": [self.documents[:n_results]],
            "metadatas": [self.metadatas[:n_results]],
            "ids": [self.ids[:n_results]]
        }

# Mock for Langchain components
class MockChatOpenAI:
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        logger.info(f"MockChatOpenAI initialized with model: {model_name}")

    def invoke(self, messages: List[Any], **kwargs) -> 'MockAIMessage':
        logger.info(f"MockChatOpenAI invoked with messages: {messages}")
        # Simple mock response based on the last message content
        last_message_content = messages[-1].content if messages else ""
        response_content = f"LLM processed: '{last_message_content}'. Simulated response: [Generated content]"
        return MockAIMessage(content=response_content)

class MockAIMessage:
    def __init__(self, content: str):
        self.content = content

class MockHumanMessage:
    def __init__(self, content: str):
        self.content = content

class MockSystemMessage:
    def __init__(self, content: str):
        self.content = content

class MockPromptTemplate:
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables
        logger.info(f"MockPromptTemplate initialized with template: {template}")

    def format(self, **kwargs) -> str:
        formatted_template = self.template
        for var in self.input_variables:
            formatted_template = formatted_template.replace(f"{{{var}}}", str(kwargs.get(var, f"[{var} not provided]")))
        return formatted_template

class MockRunnableSequence:
    def __init__(self, steps: List[Any]):
        self.steps = steps
        logger.info(f"MockRunnableSequence initialized with {len(steps)} steps.")

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"MockRunnableSequence invoked with input: {input_data}")
        current_output = input_data
        for step in self.steps:
            if hasattr(step, "invoke"): # Assume it's an LLM or a chain
                # Special handling for RAG and prompt formatting
                if "context" in step.input_variables and "question" in step.input_variables:
                    context = current_output.get("context", "No context")
                    question = current_output.get("question", "No question")
                    formatted_prompt = step.format(context=context, question=question)
                    # Simulate LLM call within a chain
                    mock_messages = [MockHumanMessage(content=formatted_prompt)]
                    llm_response = MockChatOpenAI(model_name="mock-llm").invoke(mock_messages).content
                    current_output = {"answer": llm_response}
                elif hasattr(step, "format"): # Assume it's a prompt template
                    current_output = {"prompt": step.format(**current_output)}
                else:
                    current_output = step.invoke(current_output)
            elif callable(step): # Assume it's a function (e.g., retrieval)
                current_output = step(current_output)
            else:
                logger.warning(f"Unsupported step type in MockRunnableSequence: {type(step)}")
                current_output = {"error": "Unsupported step type"}
        return current_output


class MockBaseRetriever:
    def __init__(self, vectorstore: MockChromaCollection):
        self.vectorstore = vectorstore
        logger.info(f"MockBaseRetriever initialized.")

    def get_relevant_documents(self, query: str) -> List[str]:
        logger.info(f"MockBaseRetriever retrieving documents for query: {query}")
        # Simulate retrieval by calling the mock Chroma query and formatting
        query_result = self.vectorstore.query(query_texts=[query], n_results=2)
        docs = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        formatted_docs = []
        for i, doc_content in enumerate(docs):
            formatted_docs.append(f"Document {i+1}: {doc_content} (Metadata: {metadatas[i]})")
        return formatted_docs

# Langchain specific imports - using mocks
class ChatPromptTemplate: MockPromptTemplate
class HumanMessage: MockHumanMessage
class SystemMessage: MockSystemMessage
class ChatOpenAI: MockChatOpenAI # Using a mock for actual LLM
class StringOutputParser:
    def parse(self, text: str) -> str:
        return text

    def __or__(self, other):
        return MockRunnableSequence(steps=[self, other])

class StrOutputParser(StringOutputParser): pass

class RunnablePassthrough:
    def assign(self, **kwargs):
        def _assign_func(input_data):
            new_data = input_data.copy()
            for key, value_func in kwargs.items():
                if callable(value_func): # Assume it's a runnable or lambda
                    new_data[key] = value_func(input_data)
                else: # Assume it's a direct value
                    new_data[key] = value_func
            return new_data
        return _assign_func

    def __or__(self, other):
        return MockRunnableSequence(steps=[self, other])

# --- Configuration ---
load_dotenv()

# --- Logger Setup ---
logger.add("file.log", rotation="500 MB", level="INFO")
logger.info("Application started.")

# --- Global / Shared Resources ---
# Initialize Mock Embedding Model
embedding_model = MockSentenceTransformer("all-MiniLM-L6-v2")

# Initialize Mock ChromaDB Client and Collection
chroma_client = MockChromaClient(path="./chroma_db")
medical_knowledge_collection = chroma_client.get_or_create_collection(name="medical_knowledge")

# Simulate populating ChromaDB with some dummy medical data
def _populate_medical_knowledge():
    docs = [
        "Symptoms of Diabetes include frequent urination, increased thirst, and unexplained weight loss. Long-term complications can include cardiovascular disease and nerve damage.",
        "Hypertension, or high blood pressure, often has no symptoms. Regular monitoring is crucial. Lifestyle changes like diet and exercise are key to management.",
        "Common cold symptoms are runny nose, sore throat, and sneezing. It's a viral infection and usually resolves on its own within a week.",
        "For acute appendicitis, symptoms typically include sudden pain that begins around the navel and shifts to the lower right abdomen, often accompanied by nausea, vomiting, and loss of appetite.",
        "Migraines are severe headaches often accompanied by throbbing pain, sensitivity to light and sound, and nausea. Triggers vary widely among individuals."
    ]
    metadatas = [
        {"source": "medical_journal_A", "category": "disease_info"},
        {"source": "WHO_guidelines", "category": "disease_info"},
        {"source": "CDC_facts", "category": "common_illness"},
        {"source": "Mayo_Clinic", "category": "emergency"},
        {"source": "NIH_research", "category": "neurology"}
    ]
    ids = [f"doc_{i}" for i in range(len(docs))]
    # In a real scenario, embeddings would be generated here:
    # doc_embeddings = embedding_model.encode(docs).tolist()
    medical_knowledge_collection.add(documents=docs, metadatas=metadatas, ids=ids)
    logger.info("Simulated medical knowledge populated in ChromaDB.")

_populate_medical_knowledge() # Call to populate on startup

# Initialize Mock LLM
llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo"), temperature=0.7)

# Initialize Mock Retriever for Langchain RAG
medical_retriever = MockBaseRetriever(vectorstore=medical_knowledge_collection)

# --- Langchain Chains --- For simplicity, defining functions that simulate chains.

def create_rag_chain():
    template = """You are a medical diagnostic assistant. Use the following context to answer the user's question. If you don't know the answer, state that you don't have enough information.

    Context: {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful medical assistant."),
        HumanMessage(content=template)
    ])

    # Simulate the RAG chain using the mock components
    def rag_chain_invoke(inputs):
        question = inputs["question"]
        context_docs = medical_retriever.get_relevant_documents(question)
        context_str = "\n".join(context_docs)
        formatted_prompt = prompt.format(context=context_str, question=question)
        mock_messages = [MockHumanMessage(content=formatted_prompt)]
        response = llm.invoke(mock_messages).content
        return response
    return rag_chain_invoke

rag_chain = create_rag_chain()


def create_synthetic_data_generator_chain():
    # This chain will directly call the LLM to generate data
    template = """Generate a synthetic patient profile for a medical study. Include age, gender, medical history (2-3 conditions, one rare), current symptoms (3-4 detailed symptoms), lab results (simulate 3-4 key values like glucose, cholesterol, white blood cell count), and a potential diagnosis. Format the output as a JSON object with keys like 'patient_id', 'age', 'gender', 'medical_history', 'symptoms', 'lab_results', 'diagnosis'. The rare condition should be focused on {rare_condition_focus} and symptoms should reflect it. Make sure lab results are consistent with the diagnosis.

    Example for formatting:
    {{
        "patient_id": "SYN_001",
        "age": 45,
        "gender": "Female",
        "medical_history": ["Asthma", "Hypertension", "Ehlers-Danlos Syndrome"],
        "symptoms": ["Joint hypermobility", "Skin hyperextensibility", "Chronic fatigue", "Postural orthostatic tachycardia"],
        "lab_results": {{
            "Glucose": "95 mg/dL",
            "Cholesterol_LDL": "110 mg/dL",
            "WBC_Count": "7.5 x 10^9/L",
            "CRP": "5 mg/L"
        }},
        "diagnosis": "Ehlers-Danlos Syndrome (Hypermobile Type)"
    }}

    Generate the JSON only, no other text.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert medical data generator."),
        HumanMessage(content=template)
    ])
    # This chain will be simplified to directly use the LLM with the formatted prompt.
    def synthetic_data_chain_invoke(inputs):
        formatted_prompt = prompt.format(**inputs)
        mock_messages = [MockHumanMessage(content=formatted_prompt)]
        response = llm.invoke(mock_messages).content
        return response
    return synthetic_data_chain_invoke

synthetic_data_generator_chain = create_synthetic_data_generator_chain()

def create_health_planner_chain():
    # This chain will generate personalized health plans and iteratively refine them
    template = """You are a personalized health planner. Based on the user's conditions, preferences, and previous feedback, generate a comprehensive and actionable health plan. The plan should include dietary recommendations, exercise routines, lifestyle adjustments, and educational content. Ensure the plan is empathetic, easy to understand, and tailored to the user's input. If feedback is provided, adjust the plan accordingly.

    User's Current Health Status/Preferences: {user_input}
    Previous Plan (if any): {previous_plan}
    User Feedback (if any): {feedback}

    Generate the new or refined personalized health plan:
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a compassionate and knowledgeable health planner."),
        HumanMessage(content=template)
    ])

    def health_planner_chain_invoke(inputs):
        user_input = inputs.get("user_input", "")
        previous_plan = inputs.get("previous_plan", "None provided.")
        feedback = inputs.get("feedback", "No feedback.")
        formatted_prompt = prompt.format(user_input=user_input, previous_plan=previous_plan, feedback=feedback)
        mock_messages = [MockHumanMessage(content=formatted_prompt)]
        response = llm.invoke(mock_messages).content
        return response
    return health_planner_chain_invoke

health_planner_chain = create_health_planner_chain()

# --- FastAPI Backend ---
app = FastAPI(
    title="Medical AI Assistant API",
    description="API for Intelligent Medical Diagnostic Assistant and Personalized Health Planner",
    version="1.0.0",
)

class DiagnosisRequest(BaseModel):
    symptoms: str

class DiagnosisResponse(BaseModel):
    diagnosis_insights: str
    relevant_info: List[str]

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    logger.info(f"Received diagnosis request for symptoms: {request.symptoms}")
    try:
        # Simulate RAG chain execution
        result = rag_chain({"question": request.symptoms})
        # Extract the relevant context documents retrieved by the mock retriever
        retrieved_docs_output = medical_retriever.get_relevant_documents(request.symptoms)

        return DiagnosisResponse(
            diagnosis_insights=result,
            relevant_info=retrieved_docs_output
        )
    except Exception as e:
        logger.error(f"Error during diagnosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SyntheticDataRequest(BaseModel):
    rare_condition_focus: str = "Rare Genetic Disorder"
    num_records: int = 1

class SyntheticDataResponse(BaseModel):
    generated_data: List[Dict[str, Any]]
    message: str

@app.post("/generate_synthetic_data", response_model=SyntheticDataResponse)
async def generate_synthetic_data(request: SyntheticDataRequest):
    logger.info(f"Received synthetic data generation request for {request.num_records} records, focusing on: {request.rare_condition_focus}")
    generated_records = []
    for _ in range(request.num_records):
        try:
            # Simulate synthetic data generation chain execution
            raw_llm_output = synthetic_data_generator_chain({"rare_condition_focus": request.rare_condition_focus})
            # The mock LLM will return 'LLM processed: ... Simulated response: [Generated content]' where [Generated content] is the JSON
            # We need to parse this mock output to get the JSON part.
            start_idx = raw_llm_output.find("{")
            end_idx = raw_llm_output.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = raw_llm_output[start_idx : end_idx + 1]
                record = json.loads(json_str)
            else:
                logger.warning(f"Could not parse JSON from mock LLM output: {raw_llm_output}")
                record = {"error": "Failed to parse LLM output", "raw_output": raw_llm_output}

            generated_records.append(record)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error for synthetic data: {e} - Raw output: {raw_llm_output}")
            generated_records.append({"error": "JSON parsing failed", "details": str(e), "raw_output": raw_llm_output})
        except Exception as e:
            logger.error(f"Error generating synthetic data record: {e}")
            generated_records.append({"error": str(e), "focus": request.rare_condition_focus})

    # Basic data structuring/validation (mock)
    df = pd.DataFrame(generated_records)
    # In a real scenario, you'd add more robust validation here
    logger.info(f"Generated {len(df)} synthetic data records.")

    return SyntheticDataResponse(
        generated_data=df.to_dict(orient="records"),
        message=f"Successfully generated {request.num_records} synthetic patient records."
    )


class HealthPlanRequest(BaseModel):
    user_input: str
    previous_plan: Optional[str] = None
    feedback: Optional[str] = None

class HealthPlanResponse(BaseModel):
    health_plan: str
    message: str

@app.post("/health_plan", response_model=HealthPlanResponse)
async def get_health_plan(request: HealthPlanRequest):
    logger.info(f"Received health plan request for user input: {request.user_input}")
    try:
        # Simulate health planner chain execution
        result = health_planner_chain({
            "user_input": request.user_input,
            "previous_plan": request.previous_plan if request.previous_plan else "None provided.",
            "feedback": request.feedback if request.feedback else "No feedback."
        })
        return HealthPlanResponse(
            health_plan=result,
            message="Personalized health plan generated/refined successfully."
        )
    except Exception as e:
        logger.error(f"Error generating health plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Streamlit Frontend (Instructions for separate execution) ---
# To run the Streamlit frontend, save this file as `medical_ai_assistant.py`.
# Then, in your terminal, navigate to the directory containing this file and run:
# `streamlit run medical_ai_assistant.py`
# You will also need the FastAPI backend running, typically with:
# `uvicorn medical_ai_assistant:app --reload` (in a separate terminal)

# The Streamlit code block below is commented out to allow FastAPI to run directly
# without Streamlit trying to execute within the same script when run by uvicorn.
# If you intend to run Streamlit, uncomment the Streamlit-specific imports and the block below.

# if __name__ == "__main__":
#     import streamlit as st # Import here if running Streamlit directly
#     import requests # For making API calls to FastAPI
#     import json # For parsing JSON responses

#     st.set_page_config(layout="wide", page_title="Medical AI Assistant")
#     st.title("🩺 Intelligent Medical Diagnostic Assistant & Health Planner")

#     FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

#     st.sidebar.header("Navigation")
#     page = st.sidebar.radio("Go to", ["Diagnostic Assistant", "Personalized Health Planner", "Synthetic Data Generator (Admin)"])

#     if page == "Diagnostic Assistant":
#         st.header("Medical Diagnostic Assistant")
#         st.write("Enter patient symptoms to get potential diagnostic insights.")

#         symptoms = st.text_area("Describe the patient's symptoms:", height=150)

#         if st.button("Get Diagnosis"):
#             if symptoms:
#                 with st.spinner("Analyzing symptoms..."):
#                     try:
#                         response = requests.post(f"{FASTAPI_BASE_URL}/diagnose", json={"symptoms": symptoms})
#                         response.raise_for_status()
#                         data = response.json()
#                         st.subheader("Diagnostic Insights")
#                         st.write(data["diagnosis_insights"])
#                         st.subheader("Relevant Medical Information")
#                         for info in data["relevant_info"]:
#                             st.markdown(f"- {info}")
#                     except requests.exceptions.ConnectionError:
#                         st.error("Could not connect to the FastAPI backend. Make sure it's running at " + FASTAPI_BASE_URL)
#                     except requests.exceptions.RequestException as e:
#                         st.error(f"Error from API: {e}. Details: {response.text if response else 'No response'}")
#             else:
#                 st.warning("Please enter some symptoms to get a diagnosis.")

#     elif page == "Personalized Health Planner":
#         st.header("Personalized Health Planner")
#         st.write("Get a tailored health plan based on your needs and preferences.")

#         user_input = st.text_area("Describe your health conditions, goals, and preferences (e.g., 'I have mild diabetes and want a low-carb diet plan, I enjoy walking'):", height=150)
#         previous_plan = st.text_area("If you have a previous plan, paste it here for refinement:", height=100)
#         feedback = st.text_area("Provide feedback on the previous plan or specific requests:", height=80)

#         if st.button("Generate/Refine Health Plan"):}
#             if user_input:
#                 with st.spinner("Generating your personalized health plan..."):
#                     try:
#                         payload = {"user_input": user_input}
#                         if previous_plan: payload["previous_plan"] = previous_plan
#                         if feedback: payload["feedback"] = feedback

#                         response = requests.post(f"{FASTAPI_BASE_URL}/health_plan", json=payload)
#                         response.raise_for_status()
#                         data = response.json()
#                         st.subheader("Your Personalized Health Plan")
#                         st.write(data["health_plan"])
#                     except requests.exceptions.ConnectionError:
#                         st.error("Could not connect to the FastAPI backend. Make sure it's running at " + FASTAPI_BASE_URL)
#                     except requests.exceptions.RequestException as e:
#                         st.error(f"Error from API: {e}. Details: {response.text if response else 'No response'}")
#             else:
#                 st.warning("Please provide your health conditions, goals, or preferences.")

#     elif page == "Synthetic Data Generator (Admin)":
#         st.header("Synthetic Patient Data Generator")
#         st.write("Generate synthetic patient data for training and research purposes. (Admin function)")

#         rare_condition_focus = st.text_input("Focus for rare condition (e.g., 'Huntington's Disease', 'Cystic Fibrosis'):", "Rare Genetic Disorder")
#         num_records = st.number_input("Number of records to generate:", min_value=1, max_value=10, value=1)

#         if st.button("Generate Synthetic Data"):}
#             with st.spinner(f"Generating {num_records} synthetic records..."):
#                 try:
#                     response = requests.post(f"{FASTAPI_BASE_URL}/generate_synthetic_data", json={
#                         "rare_condition_focus": rare_condition_focus,
#                         "num_records": num_records
#                     })
#                     response.raise_for_status()
#                     data = response.json()
#                     st.success(data["message"])
#                     st.subheader("Generated Data Preview (First Record)")
#                     if data["generated_data"]:
#                         st.json(data["generated_data"][0])
#                     else:
#                         st.info("No data generated.")
#                 except requests.exceptions.ConnectionError:
#                     st.error("Could not connect to the FastAPI backend. Make sure it's running at " + FASTAPI_BASE_URL)
#                 except requests.exceptions.RequestException as e:
#                     st.error(f"Error from API: {e}. Details: {response.text if response else 'No response'}")

# --- End of Streamlit Frontend ---

# For direct execution of FastAPI (e.g., uvicorn medical_ai_assistant:app --reload)
# The 'if __name__ == "__main__":' block for Streamlit needs to be commented out or handled carefully.
# We include a simple message here for when the script is run directly (not via uvicorn or streamlit).
if __name__ == "__main__":
    logger.info("This script contains both FastAPI backend and Streamlit frontend code.")
    logger.info("To run the FastAPI backend: uvicorn medical_ai_assistant:app --reload")
    logger.info("To run the Streamlit frontend (after starting FastAPI): uncomment Streamlit imports and code block, then run 'streamlit run medical_ai_assistant.py'")
    # Example of how you might run a FastAPI dev server directly for testing (less common than uvicorn CLI)
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)


