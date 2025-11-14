
import os
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

import spacy
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Streamlit for frontend
import streamlit as st

# --- Configuration --- #
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load spaCy model for text preprocessing (if not present, download with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- 1. Data Ingestion and Preprocessing Layer ---
class ImagePreprocessor:
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Mock image preprocessing: resize and convert to numpy array."""
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = image.resize((224, 224)) # Standard size for many vision models
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

class TextPreprocessor:
    def preprocess(self, text: str) -> str:
        """Text preprocessing: basic cleaning and tokenization using spaCy."""
        doc = nlp(text.lower().strip())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)

# --- 2. Multimodal Feature Extraction Layer ---
class MultimodalEncoder:
    def __init__(self):
        # In a real application, this would load actual models like ViT and BERT
        # For this example, we'll use placeholder functions.
        pass

    def encode_image(self, processed_image: np.ndarray) -> List[float]:
        """Mock image encoder: returns a dummy embedding."""
        # In a real scenario, this would pass `processed_image` through a ViT or similar model.
        return np.random.rand(768).tolist() # Dummy 768-dim embedding

    def encode_text(self, preprocessed_text: str) -> List[float]:
        """Mock text encoder: returns a dummy embedding."""
        # In a real scenario, this would pass `preprocessed_text` through a BERT-like model.
        return np.random.rand(768).tolist() # Dummy 768-dim embedding

    def fuse_features(self, image_embedding: List[float], text_embedding: List[float]) -> List[float]:
        """Mock fusion module: concatenates embeddings."""
        return image_embedding + text_embedding

# --- 3. Knowledge Base (RAG Component) --- #
class KnowledgeBase:
    def __init__(self):
        # Simulate a simple in-memory knowledge base for medical terms/conditions
        self.medical_articles = {
            "pneumonia": "Pneumonia is an infection that inflames air sacs in one or both lungs... Symptoms include cough with phlegm or pus, fever, chills, and difficulty breathing.",
            "fracture": "A bone fracture is a medical condition in which there is a partial or complete break in the continuity of a bone...",
            "diabetes": "Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces...",
            "ehr_guideline_fever": "For fever cases, always check patient history, recent travel, and duration of symptoms. Consider blood tests and imaging if persistent."
        }

    def retrieve_medical_context(self, query: str) -> List[str]:
        """Mock retrieval based on keywords in the query."""
        relevant_docs = []
        query_lower = query.lower()
        for term, content in self.medical_articles.items():
            if term in query_lower or any(word in query_lower for word in content.lower().split()):
                relevant_docs.append(content)
        return relevant_docs if relevant_docs else ["No specific medical context found for the query."]

# --- 4. DDCoT Reasoning Engine (Core Logic) --- #
class DDCoTReasoningEngine:
    def __init__(self, llm_api_key: str):
        if not llm_api_key:
            raise ValueError("OPENAI_API_KEY not provided. Please set it in your .env file.")
        self.llm = OpenAI(api_key=llm_api_key, temperature=0.5)
        self.knowledge_base = KnowledgeBase()

    def _create_task_decomposition_chain(self) -> LLMChain:
        prompt = PromptTemplate(
            input_variables=["patient_info", "medical_images_desc", "initial_query"],
            template=(
                "Given the following patient information, description of medical images, and an initial diagnostic query, "
                "break down the complex diagnostic problem into 3-5 distinct, sequential sub-questions. "
                "Focus on duties like 'Analyze image for abnormalities', 'Extract key symptoms', 'Correlate findings'. "
                "Patient Info: {patient_info}\n"
                "Medical Image Description: {medical_images_desc}\n"
                "Initial Query: {initial_query}\n"
                "Sub-questions (numbered list):\n"
            )
        )
        return LLMChain(llm=self.llm, prompt=prompt, output_key="sub_questions")

    def _create_sub_question_answering_chain(self) -> LLMChain:
        prompt = PromptTemplate(
            input_variables=["sub_question", "patient_info", "medical_images_desc", "retrieved_context", "previous_answers"],
            template=(
                "You are a medical assistant. Answer the following sub-question based on the provided information. "
                "\nSub-Question: {sub_question}\n"
                "\nPatient Info: {patient_info}\n"
                "\nMedical Image Description: {medical_images_desc}\n"
                "\nRetrieved Medical Context: {retrieved_context}\n"
                "\nPrevious Answers: {previous_answers}\n"
                "\nAnswer: "
            )
        )
        return LLMChain(llm=self.llm, prompt=prompt, output_key="answer")

    def _create_synthesis_chain(self) -> LLMChain:
        prompt = PromptTemplate(
            input_variables=["initial_query", "patient_info", "medical_images_desc", "all_sub_question_answers"],
            template=(
                "Based on the initial diagnostic query, patient information, medical image descriptions, and the answers to all sub-questions, "
                "synthesize a comprehensive diagnostic assessment. Include differential diagnoses, justification, and potential recommendations. "
                "\nInitial Query: {initial_query}\n"
                "\nPatient Info: {patient_info}\n"
                "\nMedical Image Description: {medical_images_desc}\n"
                "\nSub-Question Answers:\n{all_sub_question_answers}\n"
                "\nComprehensive Diagnostic Assessment: "
            )
        )
        return LLMChain(llm=self.llm, prompt=prompt, output_key="final_diagnosis")

    def run_diagnosis(self, patient_info: str, medical_images_desc: str, initial_query: str) -> Dict[str, Any]:
        """Executes the DDCoT reasoning process."""
        task_decomposition_chain = self._create_task_decomposition_chain()
        sub_question_answering_chain = self._create_sub_question_answering_chain()
        synthesis_chain = self._create_synthesis_chain()

        # Step 1: Decompose the task
        decomposition_output = task_decomposition_chain.invoke({
            "patient_info": patient_info,
            "medical_images_desc": medical_images_desc,
            "initial_query": initial_query
        })
        sub_questions_raw = decomposition_output["sub_questions"]
        sub_questions = [q.strip() for q in sub_questions_raw.split("\n") if q.strip() and q.strip()[0].isdigit()]

        intermediate_steps = []
        all_sub_question_answers = []
        previous_answers = ""

        # Step 2: Solve sub-questions sequentially
        for i, sub_question in enumerate(sub_questions):
            # Retrieve relevant context for the current sub-question
            retrieved_context = self.knowledge_base.retrieve_medical_context(sub_question + " " + patient_info)
            context_str = "\n".join(retrieved_context)

            sub_q_output = sub_question_answering_chain.invoke({
                "sub_question": sub_question,
                "patient_info": patient_info,
                "medical_images_desc": medical_images_desc,
                "retrieved_context": context_str,
                "previous_answers": previous_answers
            })
            answer = sub_q_output["answer"]
            all_sub_question_answers.append(f"Sub-question {i+1}: {sub_question}\nAnswer: {answer}")
            previous_answers += f"Sub-question {i+1}: {sub_question}\nAnswer: {answer}\n"

            intermediate_steps.append({
                "sub_question": sub_question,
                "retrieved_context": retrieved_context,
                "answer": answer
            })

        # Step 3: Synthesize final response
        synthesis_output = synthesis_chain.invoke({
            "initial_query": initial_query,
            "patient_info": patient_info,
            "medical_images_desc": medical_images_desc,
            "all_sub_question_answers": "\n".join(all_sub_question_answers)
        })
        final_diagnosis = synthesis_output["final_diagnosis"]

        return {
            "initial_query": initial_query,
            "patient_info": patient_info,
            "medical_images_desc": medical_images_desc,
            "sub_questions": sub_questions,
            "intermediate_steps": intermediate_steps,
            "final_diagnosis": final_diagnosis
        }

# --- 5. API and User Interface Layer --- #

# FastAPI Backend
app = FastAPI()

image_preprocessor = ImagePreprocessor()
text_preprocessor = TextPreprocessor()
multimodal_encoder = MultimodalEncoder()

try:
    reasoning_engine = DDCoTReasoningEngine(llm_api_key=OPENAI_API_KEY)
except ValueError as e:
    print(f"Error initializing reasoning engine: {e}. FastAPI endpoints will not be fully functional.")
    reasoning_engine = None # Allow app to start even if API key is missing, but diagnosis won't work

class DiagnosisRequest(BaseModel):
    patient_text_data: str
    initial_query: str
    image_base64: Optional[str] = None # Base64 encoded image string

class DiagnosisResponse(BaseModel):
    initial_query: str
    patient_info: str
    medical_images_desc: str
    sub_questions: List[str]
    intermediate_steps: List[Dict[str, Any]]
    final_diagnosis: str

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_patient(request: DiagnosisRequest):
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="AI reasoning engine not initialized. Please configure OPENAI_API_KEY.")

    # Preprocess text data
    preprocessed_patient_text = text_preprocessor.preprocess(request.patient_text_data)

    image_description = "No image provided."
    if request.image_base64:
        try:
            image_bytes = base64.b64decode(request.image_base64)
            processed_image_np = image_preprocessor.preprocess(image_bytes)
            # Mock image analysis to get a description (in a real app, a vision model would do this)
            image_description = "An X-ray image showing potential consolidation in the lower left lung field." # Placeholder

            # For a full multimodal embedding fusion, you would do:
            # image_embedding = multimodal_encoder.encode_image(processed_image_np)
            # text_embedding = multimodal_encoder.encode_text(preprocessed_patient_text)
            # fused_embedding = multimodal_encoder.fuse_features(image_embedding, text_embedding)
            # This fused embedding would then be used by the LLM (e.g., via a multimodal LLM or by providing it as context)
            # For this DDCoT example, we feed descriptive text to the LLM.

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    patient_info_for_llm = f"Patient Reported: {request.patient_text_data}. Preprocessed: {preprocessed_patient_text}"

    try:
        diagnosis_results = reasoning_engine.run_diagnosis(
            patient_info=patient_info_for_llm,
            medical_images_desc=image_description,
            initial_query=request.initial_query
        )
        return DiagnosisResponse(**diagnosis_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {e}")

# Streamlit Frontend (to run separately: streamlit run your_script_name.py)
def streamlit_frontend():
    st.set_page_config(page_title="Multimodal Medical Diagnosis Assistant", layout="wide")
    st.title("🧠 Multimodal Medical Diagnosis Assistant (DDCoT)")
    st.markdown("--- Developed using the **Duty Distinct Chain-of-Thought (DDCoT)** pattern ---")

    st.sidebar.header("Patient Information")
    patient_text_data = st.sidebar.text_area("Enter Patient EHR Notes, Symptoms, Lab Results:", "Patient presents with persistent cough for 2 weeks, shortness of breath, and mild fever. No known allergies. Recent travel to Asia.")
    initial_query = st.sidebar.text_input("Diagnostic Question:", "What is the most likely diagnosis and what further tests are recommended?")
    uploaded_file = st.sidebar.file_uploader("Upload Medical Image (X-ray, MRI, CT):", type=["png", "jpg", "jpeg"])

    st.markdown("## Diagnostic Process and Results")

    if st.sidebar.button("Run Diagnosis"):
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY is not set. Please add it to your .env file or environment variables.")
            return

        if not patient_text_data or not initial_query:
            st.warning("Please provide patient text data and a diagnostic question.")
            return

        image_base64_str = None
        if uploaded_file is not None:
            # Display uploaded image
            st.sidebar.image(uploaded_file, caption="Uploaded Medical Image", use_column_width=True)
            # Convert image to base64 for FastAPI
            image_bytes = uploaded_file.getvalue()
            image_base64_str = base64.b64encode(image_bytes).decode("utf-8")

        with st.spinner("Running DDCoT Diagnosis..."): 
            try:
                # Make a request to the FastAPI backend
                import requests
                FASTAPI_URL = "http://127.0.0.1:8000/diagnose"
                payload = {
                    "patient_text_data": patient_text_data,
                    "initial_query": initial_query,
                    "image_base64": image_base64_str
                }
                headers = {"Content-Type": "application/json"}
                response = requests.post(FASTAPI_URL, json=payload, headers=headers)
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                diagnosis_data = response.json()

                st.success("Diagnosis Complete!")

                st.subheader("Initial Query:")
                st.write(diagnosis_data["initial_query"])

                st.subheader("Patient Information Provided:")
                st.write(diagnosis_data["patient_info"])

                st.subheader("Medical Image Description:")
                st.write(diagnosis_data["medical_images_desc"])

                st.subheader("Decomposed Sub-questions:")
                for i, sq in enumerate(diagnosis_data["sub_questions"]):
                    st.write(f"**{i+1}.** {sq}")

                st.subheader("Sequential Reasoning Steps:")
                for i, step in enumerate(diagnosis_data["intermediate_steps"]):
                    st.markdown(f"#### Step {i+1}: {step["sub_question"]}")
                    st.write(f"**Retrieved Context:** {'; '.join(step['retrieved_context'])}")
                    st.write(f"**Answer:** {step['answer']}")

                st.subheader("Comprehensive Diagnostic Assessment:")
                st.write(diagnosis_data["final_diagnosis"])

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Please ensure the FastAPI server is running at http://127.0.0.1:8000.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error during API request: {e}. Details: {response.text if 'response' in locals() else 'No response body.'}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # To run the FastAPI server:
    # uvicorn multimodal_diagnosis_assistant:app --reload
    # To run the Streamlit frontend (in a separate terminal):
    # streamlit run multimodal_diagnosis_assistant.py

    # You can choose to run one or both. For a full demo, run FastAPI first, then Streamlit.
    # This __main__ block will only run the Streamlit app for simplicity if executed directly.
    # In a real deployment, FastAPI would run as a service, and Streamlit would connect to it.
    streamlit_frontend()
