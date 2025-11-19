import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from sentence_transformers import SentenceTransformer
import os
import uvicorn
import threading
import time

# --- 1. Pydantic Models ---

class PatientData(BaseModel):
    symptoms: str = Field(..., example="fever, cough, fatigue, headache")
    medical_history: str = Field(..., example="no significant medical history, non-smoker")
    lab_results: str = Field(..., example="CRP high, WBC normal, throat swab negative")

class ReasoningStep(BaseModel):
    step_number: int
    description: str
    evidence: Optional[str] = None

class VerificationResult(BaseModel):
    path_id: str
    is_consistent: bool
    medical_accuracy_feedback: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)

class DiagnosisOutput(BaseModel):
    final_diagnosis: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning_path: List[ReasoningStep]
    verifier_feedback: List[VerificationResult]

# --- 2. ChromaDB and LlamaIndex Setup ---

# Ensure a directory for medical data exists
if not os.path.exists("medical_data"):  # Creating a dummy medical_data directory
    os.makedirs("medical_data")
    with open("medical_data/common_cold.txt", "w") as f:
        f.write("Common Cold: Symptoms include runny nose, sore throat, cough, congestion, slight body aches, headache. Usually mild and resolves in 7-10 days. Caused by viruses, primarily rhinoviruses.")
    with open("medical_data/influenza.txt", "w") as f:
        f.write("Influenza (Flu): Symptoms include fever, body aches, chills, fatigue, cough, sore throat, runny or stuffy nose. Can be severe and lead to complications like pneumonia. Caused by influenza viruses.")
    with open("medical_data/strep_throat.txt", "w") as f:
        f.write("Strep Throat: Symptoms include sudden sore throat, pain when swallowing, fever, tiny red spots on the roof of the mouth, swollen tonsils (sometimes with white patches). Caused by Streptococcus pyogenes bacteria. Requires antibiotics.")

# Initialize ChromaDB client
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("medical_knowledge")

# Use SentenceTransformer for embeddings (or a more medical-specific one)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

class CustomEmbeddings: # LlamaIndex needs a class for custom embeddings
    def get_query_embedding(self, query): return embed_model.encode(query).tolist()
    def get_text_embedding(self, text): return embed_model.encode(text).tolist()
    def get_text_embeddings(self, texts): return [self.get_text_embedding(text) for text in texts]

# Create a ChromaVectorStore from the collection
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load documents and create index
documents = SimpleDirectoryReader("medical_data").load_data()
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=CustomEmbeddings())

# Configure retriever
retriever = index.as_retriever(similarity_top_k=2)

# --- 3. LLM and Prompt Templates ---

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
verifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

reasoning_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical diagnostic assistant. Provide step-by-step reasoning to arrive at a diagnosis. Consider differential diagnoses and evaluate evidence. Use the provided medical context for factual information."),
    ("user", "Patient Symptoms: {symptoms}\nMedical History: {medical_history}\nLab Results: {lab_results}\nMedical Context: {context}\n\nGenerate 3 distinct reasoning paths for diagnosis. For each path, list step-by-step deductions, potential diagnoses, and evidence evaluation.")
])

verification_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical verifier. Evaluate the given reasoning path for logical consistency and medical accuracy based on the provided context. Provide feedback and a confidence score (0-1). Focus on finding inaccuracies or inconsistencies."),
    ("user", "Original Patient Data:\nSymptoms: {symptoms}\nMedical History: {medical_history}\nLab Results: {lab_results}\n\nMedical Context: {context}\n\nReasoning Path to Verify:\n{reasoning_path}\n\nIs this path logically consistent and medically accurate? Provide specific feedback on any inconsistencies or inaccuracies. Then, give a confidence score (0-1) for this diagnosis path.")
])

# --- 4. LangChain Components ---

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Main Reasoning Agent Chain (simplified Tree-of-Thoughts simulation)
reasoning_chain = (
    RunnablePassthrough.assign(context=RunnableLambda(lambda x: retriever.retrieve(x["symptoms"])) | format_docs)
    | reasoning_prompt
    | llm
    | StrOutputParser()
)

def parse_reasoning_paths(raw_output: str) -> List[str]:
    # Simple parsing to get distinct reasoning paths
    paths = raw_output.split("\n\nReasoning Path ")
    # Filter out any empty strings or non-path intros
    parsed_paths = ["Reasoning Path " + p for p in paths if p.strip().startswith("1:") or p.strip().startswith("2:") or p.strip().startswith("3:")]
    if not parsed_paths and paths:
         # If split didn't work as expected, return the whole thing as one path for fallback
        return [raw_output]
    return parsed_paths


# Verification Chain
verification_chain = (
    RunnablePassthrough.assign(context=RunnableLambda(lambda x: retriever.retrieve(x["symptoms"])) | format_docs)
    | verification_prompt
    | verifier_llm
    | StrOutputParser()
)

def parse_verification_output(raw_output: str, path_id: str) -> VerificationResult:
    is_consistent = "consistent" in raw_output.lower() and "accurate" in raw_output.lower()
    feedback = raw_output
    confidence_match = [float(s) for s in raw_output.split() if s.replace('.', '', 1).isdigit() and 0 <= float(s) <= 1]
    confidence = confidence_match[0] if confidence_match else 0.5 # Default confidence
    return VerificationResult(
        path_id=path_id,
        is_consistent=is_consistent,
        medical_accuracy_feedback=feedback,
        confidence_score=confidence
    )

def parse_final_diagnosis_and_steps(reasoning_text: str) -> (str, List[ReasoningStep]):
    final_diagnosis = "Unknown"
    steps = []
    lines = reasoning_text.split('\n')
    current_step_num = 0
    for line in lines:
        if "Final Diagnosis:" in line:
            final_diagnosis = line.split("Final Diagnosis:")[1].strip()
        elif line.strip().startswith(f"Step ") and ":" in line:
            current_step_num += 1
            desc = line.split(":", 1)[1].strip()
            steps.append(ReasoningStep(step_number=current_step_num, description=desc))
        elif current_step_num > 0 and line.strip() and not (line.strip().startswith("Reasoning Path") or line.strip().startswith("Differential Diagnoses:")):
            # Append to previous step's description if it's a continuation
            if steps:
                steps[-1].description += " " + line.strip()
    return final_diagnosis, steps

# --- 5. FastAPI App ---

app = FastAPI()

@app.post("/diagnose", response_model=DiagnosisOutput)
async def diagnose_patient(patient_data: PatientData):
    # 1. Generate multiple reasoning paths (simulating Tree-of-Thoughts)
    raw_reasoning_output = reasoning_chain.invoke({
        "symptoms": patient_data.symptoms,
        "medical_history": patient_data.medical_history,
        "lab_results": patient_data.lab_results
    })
    
    reasoning_paths_raw = parse_reasoning_paths(raw_reasoning_output)
    
    # 2. Verify each reasoning path
    verification_results = []
    for i, path_raw in enumerate(reasoning_paths_raw):
        verifier_feedback_raw = verification_chain.invoke({
            "symptoms": patient_data.symptoms,
            "medical_history": patient_data.medical_history,
            "lab_results": patient_data.lab_results,
            "reasoning_path": path_raw
        })
        verification_results.append(parse_verification_output(verifier_feedback_raw, f"path_{i+1}"))

    # 3. Ensemble voting/scoring: select the best path based on verification scores
    best_path_result = None
    if verification_results:
        best_path_result = max(verification_results, key=lambda x: x.confidence_score)
    
    final_diagnosis = "Undetermined"
    final_confidence = 0.0
    final_reasoning_steps: List[ReasoningStep] = []

    if best_path_result:
        best_path_index = int(best_path_result.path_id.split('_')[1]) - 1
        chosen_reasoning_text = reasoning_paths_raw[best_path_index]
        final_diagnosis, final_reasoning_steps = parse_final_diagnosis_and_steps(chosen_reasoning_text)
        final_confidence = best_path_result.confidence_score

    return DiagnosisOutput(
        final_diagnosis=final_diagnosis,
        confidence_score=final_confidence,
        reasoning_path=final_reasoning_steps,
        verifier_feedback=verification_results
    )

# --- 6. Gradio Interface ---

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def gradio_interface(symptoms: str, medical_history: str, lab_results: str):
    try:
        import requests
        response = requests.post(
            "http://127.0.0.1:8000/diagnose",
            json={
                "symptoms": symptoms,
                "medical_history": medical_history,
                "lab_results": lab_results
            }
        )
        response.raise_for_status() # Raise an exception for HTTP errors
        diagnosis_output = DiagnosisOutput.model_validate(response.json())

        reasoning_str = "\n".join([f"Step {s.step_number}: {s.description}" for s in diagnosis_output.reasoning_path])
        verifier_str = "\n".join([
            f"Path ID: {v.path_id}\nConsistent: {v.is_consistent}\nConfidence: {v.confidence_score:.2f}\nFeedback: {v.medical_accuracy_feedback}\n------"
            for v in diagnosis_output.verifier_feedback
        ])

        return (
            f"Final Diagnosis: {diagnosis_output.final_diagnosis}\n" +
            f"Overall Confidence: {diagnosis_output.confidence_score:.2f}\n\n" +
            f"--- Detailed Reasoning ---\n{reasoning_str}\n\n" +
            f"--- Verification Feedback ---\n{verifier_str}"
        )

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the FastAPI backend. Make sure it's running at http://127.0.0.1:8000."
    except requests.exceptions.RequestException as e:
        return f"Error during API request: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Patient Symptoms", placeholder="e.g., fever, cough, fatigue, headache"),
        gr.Textbox(label="Medical History", placeholder="e.g., no significant medical history, non-smoker"),
        gr.Textbox(label="Lab Results", placeholder="e.g., CRP high, WBC normal, throat swab negative")
    ],
    outputs=gr.Textbox(label="Diagnostic Output", lines=20),
    title="Medical Diagnostic Assistant with Verified Reasoning",
    description="Enter patient details to get a verified diagnosis with step-by-step reasoning."
)

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True # Allow the main thread to exit even if this thread is running
    fastapi_thread.start()

    # Give FastAPI a moment to start up
    time.sleep(5)
    
    # Start Gradio
    iface.launch(share=True)