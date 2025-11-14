import os
import uuid
from typing import List, Dict, Any, Optional

# External Libraries (as specified in architecture)
# Ensure these are installed:
# pip install fastapi uvicorn transformers sentence-transformers chromadb langchain pydantic

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Suppress HuggingFace warnings for cleaner output in a demo
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. Core LLM & Inference Engine (Simulated for this demo) ---
# In a real application, you would load a model like Mistral-7B-Instruct-v0.2
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import pipeline # For simpler interface

class DummyLLM:
    """
    A dummy LLM to simulate responses without loading a large model.
    In a real scenario, this would load a model like Mistral-7B.
    """
    def generate(self, prompt: str) -> str:
        # Simulate different responses based on prompt keywords
        if "patient history summary" in prompt.lower() and "short-term memory" in prompt.lower():
            return "Based on the provided short-term memory, the patient has a history of hypertension and presented with acute chest pain two hours ago. Recent lab results show elevated troponin levels and ST-segment elevation on ECG."
        elif "differential diagnosis" in prompt.lower():
            return "Considering the patient's symptoms (acute crushing chest pain, radiating to left arm, ST-segment elevation) and retrieved medical knowledge (Myocardial Infarction, Angina Pectoris, Pericarditis), potential differential diagnoses strongly point towards Acute Myocardial Infarction. Other considerations include unstable angina or myocarditis. Further investigation is recommended to confirm."
        elif "treatment recommendation" in prompt.lower() and "myocardial infarction" in prompt.lower():
            return "For suspected Acute Myocardial Infarction, immediate treatment could involve aspirin, nitroglycerin, oxygen (if hypoxic), and urgent reperfusion therapy (e.g., primary PCI or thrombolytics) as per clinical guidelines. Beta-blockers and ACE inhibitors may also be initiated after stabilization. Consult relevant medical protocols."
        elif "fact retrieval" in prompt.lower() and "hypertension" in prompt.lower():
            return "Hypertension (high blood pressure) is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. It's a major risk factor for MI, stroke, and kidney disease."
        else:
            # A more generic fallback for other queries or less specific matches
            if "context:" in prompt.lower() and "question:" in prompt.lower():
                # Extract context and question to provide a more relevant dummy response
                context_start = prompt.lower().find("context:") + len("context:")
                question_start = prompt.lower().find("question:") + len("question:")
                context_part = prompt[context_start:question_start].strip()
                question_part = prompt[question_start:].strip()
                if question_part.startswith("input:"): # Remove 'input:' if present
                    question_part = question_part[len("input:"):].strip()
                return f"LLM generated response for your question: '{question_part}'. Based on the provided context, I can confirm information regarding '{context_part[:50]}...'. This is a simulated response."
            else:
                return f"LLM generated response for: '{prompt[:100]}...'. This is a simulated response, a real LLM would provide a more nuanced answer."

# --- 2. Memory Management System ---
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Initialize Embedding Model
# Using a common sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB Client (in-memory for this demo)
# For persistence: chromadb.PersistentClient(path="./chroma_db_data")
chroma_client = chromadb.Client() 

# ChromaDB Collections
# Short-Term Memory (STM): Patient-specific, dynamic
STM_COLLECTION_NAME = "patient_short_term_memory"
# Long-Term Memory (LTM): General medical knowledge, static
LTM_COLLECTION_NAME = "medical_long_term_knowledge"

def get_or_create_collection(name: str):
    try:
        return chroma_client.get_collection(name=name, embedding_function=chroma_ef)
    except: 
        # chromadb.exceptions.CollectionNotFoundError is not directly exposed
        print(f"Creating collection: {name}")
        return chroma_client.create_collection(name=name, embedding_function=chroma_ef)

stm_collection = get_or_create_collection(STM_COLLECTION_NAME)
ltm_collection = get_or_create_collection(LTM_COLLECTION_NAME)

# --- LangChain Integration for RAG ---
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.chains import create_retrieval_chain # Not directly used with manual retrieval
from langchain.chains.combine_documents import create_stuff_documents_chain # Used conceptually for prompt structuring
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Custom LangChain LLM wrapper for our DummyLLM
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult

class LangChainDummyLLM(BaseLLM):
    """A wrapper for our DummyLLM to integrate with LangChain's BaseLLM interface."""
    dummy_llm: DummyLLM

    @property
    def _llm_type(self) -> str:
        return "dummy_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.dummy_llm.generate(prompt)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.dummy_llm.generate(prompt)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"name": "DummyLLM_for_LangChain"}

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Simulate batch generation."""
        responses = [self.dummy_llm.generate(p) for p in prompts]
        return LLMResult(generations=[[{"text": r}] for r in responses])


langchain_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 3. Query Complexity Classifier (Rule-based for simplicity) ---
class QueryClassifier:
    """
    Classifies query complexity and type using a rule-based approach for this demo.
    In a real system, this could be a fine-tuned small LLM or a specialized model.
    """
    def classify(self, query: str) -> str:
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["what is", "define", "explain", "meaning of", "tell me about"]):
            return "Fact Retrieval"
        elif any(keyword in query_lower for keyword in ["patient history", "summary of patient", "patient details", "tell me about this patient"]):
            return "Patient History Summary"
        elif any(keyword in query_lower for keyword in ["diagnose", "differential diagnosis", "possible conditions", "what could it be"]):
            return "Differential Diagnosis"
        elif any(keyword in query_lower for keyword in ["treatment for", "recommendations for", "how to treat", "manage this condition"]):
            return "Treatment Recommendation"
        else:
            return "General Query"

# --- 4. Adaptive Processing & Orchestration Layer ---
class AdaptiveDiagnosticAssistant:
    def __init__(self):
        self.llm = LangChainDummyLLM(dummy_llm=DummyLLM())
        self.classifier = QueryClassifier()
        
        # Initialize Chroma vector stores using the client and collections
        self.stm_vectorstore = Chroma(
            client=chroma_client,
            collection_name=STM_COLLECTION_NAME,
            embedding_function=langchain_embeddings # Ensure embedding_function is passed
        )
        self.ltm_vectorstore = Chroma(
            client=chroma_client,
            collection_name=LTM_COLLECTION_NAME,
            embedding_function=langchain_embeddings # Ensure embedding_function is passed
        )

    def add_patient_context(self, patient_id: str, context: str, source: str = "doctor_notes"):
        """Adds context to the Short-Term Memory for a specific patient."""
        doc_id = f"stm-{patient_id}-{uuid.uuid4()}"
        # ChromaDB's add_texts expects a list of texts, metadatas, and ids
        self.stm_vectorstore.add_texts(
            texts=[context],
            metadatas=[{"patient_id": patient_id, "source": source, "type": "short_term_memory", "doc_id": doc_id}],
            ids=[doc_id]
        )
        print(f"Added context for patient {patient_id} to STM: {context[:50]}...")

    def add_medical_knowledge(self, content: str, source: str = "medical_guideline", title: str = "Unknown Title"):
        """Adds general medical knowledge to the Long-Term Memory."""
        doc_id = f"ltm-{uuid.uuid4()}"
        self.ltm_vectorstore.add_texts(
            texts=[content],
            metadatas=[{"source": source, "title": title, "type": "long_term_knowledge", "doc_id": doc_id}],
            ids=[doc_id]
        )
        print(f"Added knowledge from '{title}' to LTM: {content[:50]}...")

    def _retrieve_context(self, query: str, patient_id: Optional[str] = None, k: int = 5) -> List[Document]:
        """Retrieves relevant context from STM and LTM."""
        retrieved_docs = []
        
        # Prioritize STM for patient-specific queries if patient_id is provided
        if patient_id:
            # LangChain Chroma .as_retriever() filter syntax for metadata
            stm_retriever = self.stm_vectorstore.as_retriever(search_kwargs={"k": k, "filter": {"patient_id": patient_id}})
            stm_results = stm_retriever.get_relevant_documents(query)
            retrieved_docs.extend(stm_results)
            print(f"Retrieved {len(stm_results)} documents from STM for patient {patient_id}.")

        # Always retrieve from LTM for general medical knowledge
        ltm_retriever = self.ltm_vectorstore.as_retriever(search_kwargs={"k": k})
        ltm_results = ltm_retriever.get_relevant_documents(query)
        retrieved_docs.extend(ltm_results)
        print(f"Retrieved {len(ltm_results)} documents from LTM.")

        # Remove duplicates if any (based on page_content) and limit total
        # Using a dictionary to preserve order and remove duplicates
        unique_docs_map = {doc.page_content: doc for doc in retrieved_docs}
        return list(unique_docs_map.values())[:k] # Return up to k unique documents

    def process_query(self, query: str, patient_id: Optional[str] = None) -> str:
        """
        Processes a doctor's query, adapts retrieval, and generates a response.
        """
        query_type = self.classifier.classify(query)
        print(f"Query classified as: {query_type}")

        # Adapt retrieval strategy based on query type
        # For patient history, retrieve more from STM; for facts, more from LTM.
        retrieval_k = 5
        if query_type == "Patient History Summary":
            retrieval_k = 8 # More context for summaries
        elif query_type == "Differential Diagnosis" or query_type == "Treatment Recommendation":
            retrieval_k = 7 # Balance between STM and LTM

        retrieved_docs = self._retrieve_context(query, patient_id, k=retrieval_k)
        
        # Combine documents into a single context string for the LLM prompt
        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Define base prompt templates and adapt based on query type
        # LangChain's create_stuff_documents_chain is useful for structuring this
        # but for a dummy LLM, we'll format the prompt directly.
        base_instruction = "You are a medical diagnostic assistant. Provide accurate and helpful information based on the provided context."
        
        if query_type == "Patient History Summary":
            prompt_template = f"""{base_instruction}\n\nBased on the following patient-specific context, provide a concise summary of the patient's medical history, recent symptoms, and relevant lab results. Focus on chronological order and key medical facts.\n\nContext:\n{{context}}\n\nQuestion: {{input}}\n\nPatient History Summary:"""
        elif query_type == "Differential Diagnosis":
            prompt_template = f"""{base_instruction}\n\nConsidering the following patient context and general medical knowledge, list potential differential diagnoses for the patient's condition. For each diagnosis, provide a brief rationale based on the provided information.\n\nContext:\n{{context}}\n\nQuestion: {{input}}\n\nDifferential Diagnoses:"""
        elif query_type == "Treatment Recommendation":
            prompt_template = f"""{base_instruction}\n\nBased on the following patient context and general medical guidelines, suggest appropriate treatment strategies for the identified or suspected condition. Include any relevant drug classes, procedures, or lifestyle modifications, citing the context where applicable.\n\nContext:\n{{context}}\n\nQuestion: {{input}}\n\nTreatment Recommendations:"""
        else: # Fact Retrieval, General Query
            prompt_template = f"""{base_instruction}\n\nUse the following pieces of retrieved context to answer the user's question. If the context does not contain enough information, state that you cannot provide a comprehensive answer from the given information.\n\nContext:\n{{context}}\n\nQuestion: {{input}}\n\nAnswer:"""
        
        final_prompt = prompt_template.format(context=context_str, input=query)
        
        # Generate response using the dummy LLM
        response = self.llm.generate(final_prompt)
        return response

# --- 5. Knowledge Update & Fine-tuning Module (Data Ingestion) ---
# Fine-tuning is conceptual for this demo, focusing on LTM updates.
# For actual fine-tuning (e.g., LoRA with TRL), it would involve a separate training script.

# --- 6. API/User Interface (FastAPI Backend) ---
app = FastAPI(
    title="Adaptive LLM Diagnostic Assistant API",
    description="API for a smart diagnostic assistant leveraging adaptive LLM augmentation and memory management."
)

assistant = AdaptiveDiagnosticAssistant()

class QueryRequest(BaseModel):
    query: str
    patient_id: Optional[str] = None

class ContextUpdateRequest(BaseModel):
    patient_id: str
    context: str
    source: str = "doctor_notes"

class KnowledgeUpdateRequest(BaseModel):
    content: str
    source: str = "medical_guideline"
    title: str = "Unknown Title"

@app.post("/query")
async def process_doctor_query(request: QueryRequest):
    """
    Submits a doctor's query to the diagnostic assistant.
    patient_id is optional but recommended for patient-specific queries.
    """
    try:
        response = assistant.process_query(request.query, request.patient_id)
        return {"response": response, "query": request.query, "patient_id": request.patient_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/update_patient_context")
async def update_patient_stm(request: ContextUpdateRequest):
    """
    Updates the short-term memory (STM) for a specific patient.
    This allows adding patient-specific notes, lab results, etc.
    """
    try:
        assistant.add_patient_context(request.patient_id, request.context, request.source)
        return {"message": f"Patient context updated for patient_id: {request.patient_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/update_medical_knowledge")
async def update_ltm(request: KnowledgeUpdateRequest):
    """
    Updates the long-term medical knowledge base (LTM).
    This can be used to ingest new medical guidelines, research, etc.
    """
    try:
        assistant.add_medical_knowledge(request.content, request.source, request.title)
        return {"message": f"Medical knowledge updated from source: {request.source}, title: {request.title}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Example usage (for local testing, run with `uvicorn main:app --reload` in your terminal)
if __name__ == "__main__":
    import uvicorn

    print("Initializing example data...")
    # Clear existing collections for a clean run if using in-memory ChromaDB
    try:
        chroma_client.delete_collection(name=STM_COLLECTION_NAME)
        chroma_client.delete_collection(name=LTM_COLLECTION_NAME)
        print("Cleared existing ChromaDB collections.")
    except Exception as e:
        print(f"Could not clear collections (might not exist yet): {e}")
    
    # Re-initialize collections after potential clearing
    stm_collection = get_or_create_collection(STM_COLLECTION_NAME)
    ltm_collection = get_or_create_collection(LTM_COLLECTION_NAME)
    
    # Add some initial LTM data
    assistant.add_medical_knowledge(
        "Myocardial infarction (MI), commonly known as a heart attack, occurs when blood flow to a part of the heart is blocked for a long enough time, causing heart muscle damage.",
        "Medical Textbook", "Myocardial Infarction Basics"
    )
    assistant.add_medical_knowledge(
        "Symptoms of MI include chest pain, shortness of breath, pain radiating to the arm, and nausea. Diagnosis often involves ECG and blood tests for cardiac enzymes like troponin.",
        "Clinical Guidelines", "MI Diagnosis"
    )
    assistant.add_medical_knowledge(
        "Treatment for acute MI often includes aspirin, nitrates, beta-blockers, ACE inhibitors, and statins. Reperfusion therapy (PCI or thrombolysis) is critical. Early intervention improves outcomes.",
        "Clinical Guidelines", "MI Treatment"
    )
    assistant.add_medical_knowledge(
        "Hypertension is a chronic medical condition in which the blood pressure in the arteries is persistently elevated. Long-term high blood pressure is a major risk factor for coronary artery disease, stroke, heart failure, peripheral arterial disease, vision loss, and chronic kidney disease.",
        "Medical Textbook", "Hypertension Overview"
    )
    assistant.add_medical_knowledge(
        "Pericarditis is an inflammation of the pericardium, the fibrous sac surrounding the heart. Symptoms include sharp, stabbing chest pain that may worsen with deep breath or lying flat, and often improves when leaning forward.",
        "Medical Journal", "Pericarditis Review"
    )
    assistant.add_medical_knowledge(
        "Angina Pectoris is chest pain or discomfort caused by reduced blood flow to the heart muscle. It is a symptom of coronary artery disease. It typically occurs with exertion or stress and subsides with rest or nitroglycerin.",
        "Medical Textbook", "Angina Pectoris"
    )

    # Add some initial STM data for a hypothetical patient "P001"
    assistant.add_patient_context(
        "P001",
        "Patient presented with acute crushing chest pain, radiating to the left arm, onset 2 hours ago. Has a known history of hypertension for 10 years and is currently on Lisinopril.",
        "doctor_notes"
    )
    assistant.add_patient_context(
        "P001",
        "ECG shows ST-segment elevation in leads V2-V5, indicating acute injury. Troponin I: 1.2 ng/mL (elevated, upper limit of normal < 0.04 ng/mL).",
        "lab_results"
    )
    assistant.add_patient_context(
        "P001",
        "Patient is a 65-year-old male, non-smoker, with no known drug allergies. Family history includes paternal MI at age 60.",
        "patient_demographics"
    )
    assistant.add_patient_context(
        "P002",
        "Patient is a 45-year-old female complaining of intermittent chest discomfort, burning sensation, not related to exertion. History of GERD.",
        "doctor_notes"
    )

    print("Example data loaded. Starting FastAPI app...")
    # To run this, save it as main.py and execute: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("FastAPI app stopped.")
