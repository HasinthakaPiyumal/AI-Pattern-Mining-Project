# Part 1: Database Setup and Models (SQLAlchemy)
import os
from typing import List, Optional, Dict
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_reports.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class MedicalReport(Base):
    __tablename__ = "medical_reports"
    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(Text, nullable=False)
    initial_translation = Column(Text)
    final_translation = Column(Text)
    status = Column(String, default="pending_translation") # pending_translation, needs_clarification, translated

    questions = relationship("ClarificationQuestion", back_populates="report")

class ClarificationQuestion(Base):
    __tablename__ = "clarification_questions"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("medical_reports.id"))
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text)

    report = relationship("MedicalReport", back_populates="questions")

# Create database tables
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Part 2: Backend - FastAPI Application
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
import uvicorn
import httpx # For making requests from frontend to backend
import asyncio # For running async functions in Streamlit

app = FastAPI(title="Medical Report Translator Backend")

# Placeholder for LLM interaction (replace with actual LLM API calls)
class LLMService:
    def initial_translate(self, text: str) -> str:
        # Simulate LLM translation
        print(f"LLM: Initial translation for: {text[:50]}...")
        # Example of an ambiguous term for demonstration
        if "pneumonia" in text.lower() and "consolidation" in text.lower():
            return f"Initial translation of '{text}' (Contains 'consolidation' - possibly ambiguous)."
        return f"Initial translation of '{text}'."

    def generate_clarification_questions(self, translated_text: str) -> List[str]:
        # Simulate LLM detecting ambiguity and generating questions
        if "consolidation" in translated_text:
            print(f"LLM: Generating questions for ambiguity in: {translated_text[:50]}...")
            return [
                "In the context of 'consolidation', are we referring to lung consolidation, financial consolidation, or something else?",
                "If lung consolidation, what is its extent or severity?"
            ]
        print(f"LLM: No obvious ambiguities detected in: {translated_text[:50]}...")
        return []

    def refine_translation(self, original_text: str, questions_answers: Dict[str, str]) -> str:
        # Simulate LLM refining translation based on answers
        print(f"LLM: Refining translation with answers: {questions_answers}")
        refined = original_text
        for q, a in questions_answers.items():
            if "consolidation" in q.lower(): # Specific logic for the example ambiguity
                refined = refined.replace("consolidation", f"consolidation (clarified as: {a})")
        return f"FINAL TRANSLATION of '{refined}' after human clarification: {list(questions_answers.values())}."

llm_service = LLMService()

# Pydantic Models for FastAPI
class ReportUploadRequest(BaseModel):
    report_text: str

class ReportStatusResponse(BaseModel):
    report_id: int
    original_text: str
    initial_translation: Optional[str] = None
    final_translation: Optional[str] = None
    status: str
    clarification_questions: List[Dict[str, Optional[str]]] = []

class ClarificationSubmitRequest(BaseModel):
    report_id: int
    answers: Dict[int, str] # {question_id: answer_text}

# FastAPI Endpoints
@app.post("/upload-report", response_model=ReportStatusResponse)
async def upload_report(request: ReportUploadRequest, db: SessionLocal = Depends(get_db)):
    report = MedicalReport(original_text=request.report_text)
    db.add(report)
    db.commit()
    db.refresh(report)

    initial_translation = llm_service.initial_translate(report.original_text)
    report.initial_translation = initial_translation
    
    questions_texts = llm_service.generate_clarification_questions(initial_translation)
    if questions_texts:
        report.status = "needs_clarification"
        for q_text in questions_texts:
            question = ClarificationQuestion(report_id=report.id, question_text=q_text)
            db.add(question)
        db.commit()
        db.refresh(report) # Refresh to get associated questions
        
        return ReportStatusResponse(
            report_id=report.id,
            original_text=report.original_text,
            initial_translation=report.initial_translation,
            status=report.status,
            clarification_questions=[{"id": q.id, "question_text": q.question_text, "answer_text": q.answer_text} for q in report.questions]
        )
    else:
        report.final_translation = initial_translation
        report.status = "translated"
        db.commit()
        db.refresh(report)
        return ReportStatusResponse(
            report_id=report.id,
            original_text=report.original_text,
            initial_translation=report.initial_translation,
            final_translation=report.final_translation,
            status=report.status
        )

@app.post("/submit-clarification", response_model=ReportStatusResponse)
async def submit_clarification(request: ClarificationSubmitRequest, db: SessionLocal = Depends(get_db)):
    report = db.query(MedicalReport).filter(MedicalReport.id == request.report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if report.status != "needs_clarification":
        raise HTTPException(status_code=400, detail="Report does not require clarification or already translated.")

    questions_answers = {}
    for q_id, answer_text in request.answers.items():
        question = db.query(ClarificationQuestion).filter(ClarificationQuestion.id == q_id, ClarificationQuestion.report_id == report.id).first()
        if not question:
            raise HTTPException(status_code=404, detail=f"Question with ID {q_id} not found for this report.")
        question.answer_text = answer_text
        questions_answers[question.question_text] = answer_text
        db.add(question)
    
    db.commit()
    db.refresh(report) # Refresh to get updated question answers

    final_translation = llm_service.refine_translation(report.original_text, questions_answers)
    report.final_translation = final_translation
    report.status = "translated"
    db.commit()
    db.refresh(report)

    return ReportStatusResponse(
        report_id=report.id,
        original_text=report.original_text,
        initial_translation=report.initial_translation,
        final_translation=report.final_translation,
        status=report.status,
        clarification_questions=[{"id": q.id, "question_text": q.question_text, "answer_text": q.answer_text} for q in report.questions]
    )

@app.get("/report-status/{report_id}", response_model=ReportStatusResponse)
async def get_report_status(report_id: int, db: SessionLocal = Depends(get_db)):
    report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    questions_data = []
    for q in report.questions:
        questions_data.append({"id": q.id, "question_text": q.question_text, "answer_text": q.answer_text})
    
    return ReportStatusResponse(
        report_id=report.id,
        original_text=report.original_text,
        initial_translation=report.initial_translation,
        final_translation=report.final_translation,
        status=report.status,
        clarification_questions=questions_data
    )

# Part 3: Frontend - Streamlit Application
import streamlit as st

# Assuming backend runs on http://127.0.0.1:8000
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("🏥 Medical Report Translator with Ambiguity Clarification")

st.write("Upload a medical report for translation. If ambiguities are detected, you'll be asked to provide clarifications.")

# Initialize session state for report_id and current answers
if "report_id" not in st.session_state:
    st.session_state.report_id = None
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "report_data" not in st.session_state:
    st.session_state.report_data = {}

report_text_input = st.text_area("Enter Medical Report Text:", height=200, key="report_input")

async def upload_and_translate_frontend(report_text: str): # Renamed to avoid conflict if `upload_and_translate` exists elsewhere
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{BACKEND_URL}/upload-report", json={"report_text": report_text})
            response.raise_for_status() # Raises an HTTPStatusError for 4xx/5xx responses
            return response.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Error uploading report: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        st.error(f"Network error during upload: {e}")
        return None

async def submit_clarifications_frontend(report_id: int, answers: Dict[int, str]): # Renamed
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{BACKEND_URL}/submit-clarification", json={"report_id": report_id, "answers": answers})
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Error submitting clarifications: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        st.error(f"Network error during submission: {e}")
        return None

async def get_report_details_frontend(report_id: int): # Renamed
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/report-status/{report_id}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Error fetching report status: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        st.error(f"Network error during status fetch: {e}")
        return None

if st.button("Translate Report", key="translate_button"):
    if report_text_input:
        st.session_state.report_id = None # Reset for new translation
        st.session_state.answers = {}
        st.session_state.report_data = {}
        with st.spinner("Translating report and checking for ambiguities..."):
            report_data = asyncio.run(upload_and_translate_frontend(report_text_input))
            if report_data:
                st.session_state.report_id = report_data["report_id"]
                st.session_state.report_data = report_data
                st.success(f"Report ID: {report_data['report_id']}")
                
                st.subheader("Initial Translation:")
                st.info(report_data.get("initial_translation", "No initial translation available yet."))

                if report_data["status"] == "needs_clarification":
                    st.warning("Ambiguities detected! Please provide clarifications.")
                    for q in report_data["clarification_questions"]:
                        if q["id"] not in st.session_state.answers:
                            st.session_state.answers[q["id"]] = "" # Initialize answer fields
                elif report_data["status"] == "translated":
                    st.subheader("Final Translation:")
                    st.success(report_data["final_translation"])
            else:
                st.error("Failed to process report.")
    else:
        st.warning("Please enter some text to translate.")

# Display clarification questions if available
if st.session_state.report_id and st.session_state.get("report_data", {}).get("status") == "needs_clarification":
    st.subheader("Clarification Questions:")
    questions = st.session_state.report_data["clarification_questions"]
    
    for q in questions:
        key = f"answer_{q['id']}_{st.session_state.report_id}" # Unique key for each question across reports
        st.session_state.answers[q["id"]] = st.text_input(q["question_text"], value=st.session_state.answers.get(q["id"], ""), key=key)
    
    if st.button("Submit Clarifications", key="submit_clarifications_button"):
        if all(st.session_state.answers.values()):
            with st.spinner("Submitting clarifications and refining translation..."):
                updated_report_data = asyncio.run(submit_clarifications_frontend(st.session_state.report_id, st.session_state.answers))
                if updated_report_data:
                    st.session_state.report_data = updated_report_data
                    if updated_report_data["status"] == "translated":
                        st.subheader("Final Translation:")
                        st.success(updated_report_data["final_translation"])
                        # Clear session state for a new report, but keep the current final translation displayed
                        st.session_state.report_id = None 
                        st.session_state.answers = {}
                    else:
                        st.error("Unexpected status after submitting clarifications.")
                else:
                    st.error("Failed to submit clarifications.")
        else:
            st.warning("Please answer all clarification questions.")

# Display final translation if already available
if st.session_state.get("report_data", {}).get("status") == "translated" and st.session_state.get("report_id") is None: # Display only if it's the final state and not waiting for clarification
    st.subheader("Final Translation:")
    st.success(st.session_state.report_data["final_translation"])


# Main entry point for the file
if __name__ == "__main__":
    create_db_and_tables()
    # To run the FastAPI backend:
    # Open a terminal and run: uvicorn main_app:app --reload --host 127.0.0.1 --port 8000
    # The line below would start the FastAPI server if this script was run directly,
    # but for typical development with Streamlit and FastAPI, they are run separately.
    # For this combined file, you would manually start FastAPI in one terminal
    # and Streamlit in another.
    # uvicorn.run(app, host="127.0.0.1", port=8000)

    # To run the Streamlit frontend:
    # Open a separate terminal and run: streamlit run main_app.py
    # The Streamlit part of the code will be executed when run via 'streamlit run'.
    st.sidebar.markdown("### How to run:")
    st.sidebar.markdown("1.  **Start the Backend (in a separate terminal):**")
    st.sidebar.code("uvicorn main_app:app --reload --host 127.0.0.1 --port 8000")
    st.sidebar.markdown("2.  **Start the Frontend (in another terminal):**")
    st.sidebar.code("streamlit run main_app.py")
    st.sidebar.markdown("Make sure you have `uvicorn`, `fastapi`, `streamlit`, `sqlalchemy`, `httpx`, `python-dotenv` installed.")
