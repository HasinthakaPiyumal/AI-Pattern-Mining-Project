import os
import requests
from typing import List, Dict, Any
import uvicorn
import threading

# --- Pydantic Models (Simulated with simple dictionaries for single file execution) ---
class AssignmentInput:
    def __init__(self, assignment_id: str, prompt: str, objectives: List[str]):
        self.assignment_id = assignment_id
        self.prompt = prompt
        self.objectives = objectives

class EvaluationRequest:
    def __init__(self, assignment_id: str, essay_text: str):
        self.assignment_id = assignment_id
        self.essay_text = essay_text

class EvaluationResult:
    def __init__(self, score: float, feedback: str, strengths: List[str], weaknesses: List[str]):
        self.score = score
        self.feedback = feedback
        self.strengths = strengths
        self.weaknesses = weaknesses

# --- Chroma DB Service (Simplified In-Memory) ---
class ChromaClient:
    def __init__(self):
        self.assignments: Dict[str, Dict[str, Any]] = {}
        self.guidelines: Dict[str, str] = {}

    def add_assignment(self, assignment_id: str, prompt: str, objectives: List[str]):
        self.assignments[assignment_id] = {"prompt": prompt, "objectives": objectives}

    def get_assignment(self, assignment_id: str) -> Dict[str, Any]:
        return self.assignments.get(assignment_id, {})

    def store_guidelines(self, assignment_id: str, guidelines: str):
        self.guidelines[assignment_id] = guidelines

    def get_guidelines(self, assignment_id: str) -> str:
        return self.guidelines.get(assignment_id, "")

# --- LLM Service (Simulated LangChain/OpenAI interaction) ---
class LLMService:
    def __init__(self, api_key: str = "dummy_key"):
        self.api_key = api_key

    def generate_evaluation_guidelines(self, assignment_prompt: str, learning_objectives: List[str]) -> str:
        objective_list = "\n- ".join(learning_objectives)
        guidelines_template = f"""Generate detailed, step-by-step evaluation guidelines for an essay based on the following assignment prompt and learning objectives.

Assignment Prompt: {assignment_prompt}
Learning Objectives:
- {objective_list}

Provide criteria covering clarity, argumentation, evidence use, structure, and adherence to objectives.
"""
        return f"""Model-Generated Guidelines for '{assignment_prompt}':\n\n1. Clarity and Cohesion: Evaluate if the essay presents ideas clearly and logically.
2. Argument Strength: Assess the strength and validity of the main argument and supporting points.
3. Evidence Utilization: Check how effectively external evidence (if applicable) is integrated and analyzed.
4. Structure and Organization: Examine the essay's introduction, body paragraphs, and conclusion.
5. Adherence to Objectives: Ensure all learning objectives are addressed thoroughly.
\nDetailed steps derived from objectives: {objective_list}
"""

    def evaluate_essay(self, essay_text: str, guidelines: str) -> Dict[str, Any]:
        # Simulate LLM evaluation based on guidelines
        score = len(essay_text) % 50 + 50 # Dummy score between 50-99
        feedback = f"""Automated feedback based on guidelines:\n\nScore: {score}/100\nOverall impression: This essay demonstrates a good understanding of the topic, adhering to the provided guidelines. \n\nStrengths: 
- Clear introduction. 
- Good use of examples. 
\nWeaknesses: 
- Could improve conclusion depth. 
- Some sentences are overly long.
"""
        strengths = ["Clear introduction", "Good use of examples"]
        weaknesses = ["Could improve conclusion depth", "Some sentences are overly long"]
        return {"score": float(score), "feedback": feedback, "strengths": strengths, "weaknesses": weaknesses}

# --- FastAPI Backend ---
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI()

db_client = ChromaClient()
llm_service = LLMService(api_key=os.getenv("OPENAI_API_KEY", "dummy_key"))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """<html><body><h1>EduGrade AI Backend is Running</h1><p>Use /docs for API documentation.</p></body></html>"""

@app.post("/assignment")
async def create_assignment(assignment: AssignmentInput):
    db_client.add_assignment(assignment.assignment_id, assignment.prompt, assignment.objectives)
    return {"message": "Assignment created successfully", "assignment_id": assignment.assignment_id}

@app.get("/assignment/{assignment_id}")
async def get_assignment(assignment_id: str):
    assignment_data = db_client.get_assignment(assignment_id)
    if not assignment_data:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return assignment_data

@app.post("/generate_guidelines/{assignment_id}")
async def generate_guidelines_endpoint(assignment_id: str):
    assignment_data = db_client.get_assignment(assignment_id)
    if not assignment_data:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    guidelines = llm_service.generate_evaluation_guidelines(
        assignment_prompt=assignment_data["prompt"],
        learning_objectives=assignment_data["objectives"]
    )
    db_client.store_guidelines(assignment_id, guidelines)
    return {"message": "Guidelines generated and stored", "guidelines": guidelines}

@app.post("/evaluate")
async def evaluate_essay_endpoint(request: EvaluationRequest):
    guidelines = db_client.get_guidelines(request.assignment_id)
    if not guidelines:
        raise HTTPException(status_code=404, detail="Guidelines not found for this assignment. Please generate them first.")
    
    evaluation_result = llm_service.evaluate_essay(request.essay_text, guidelines)
    return evaluation_result

# --- Streamlit Frontend ---
import streamlit as st

# Frontend configuration
FASTAPI_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("EduGrade AI: Automated Essay Evaluation")

# Function to run FastAPI in a separate thread
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

# Start FastAPI server in a separate thread if not already running
if "fastapi_thread" not in st.session_state:
    st.session_state.fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    st.session_state.fastapi_thread.start()
    st.success(f"FastAPI backend started at {FASTAPI_BASE_URL}")

# --- Assignment Definition ---
st.header("1. Define New Assignment")
with st.form("new_assignment_form"):
    assignment_id = st.text_input("Assignment ID (e.g., 'essay001')")
    assignment_prompt = st.text_area("Assignment Prompt", height=150)
    learning_objectives_input = st.text_area("Learning Objectives (one per line)", height=100)
    
    submitted_assignment = st.form_submit_button("Create Assignment")
    if submitted_assignment and assignment_id and assignment_prompt and learning_objectives_input:
        objectives = [obj.strip() for obj in learning_objectives_input.split('\n') if obj.strip()]
        assignment_data = {"assignment_id": assignment_id, "prompt": assignment_prompt, "objectives": objectives}
        try:
            response = requests.post(f"{FASTAPI_BASE_URL}/assignment", json=assignment_data)
            response.raise_for_status()
            st.success(f"Assignment '{assignment_id}' created successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error creating assignment: {e}")

# --- Guideline Generation ---
st.header("2. Generate Evaluation Guidelines")
assignment_id_guidelines = st.text_input("Assignment ID for Guideline Generation")
if st.button("Generate Guidelines"):
    if assignment_id_guidelines:
        try:
            response = requests.post(f"{FASTAPI_BASE_URL}/generate_guidelines/{assignment_id_guidelines}")
            response.raise_for_status()
            guidelines_data = response.json()
            st.subheader("Model-Generated Guidelines:")
            st.code(guidelines_data["guidelines"], language="text")
            st.success("Guidelines generated!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error generating guidelines: {e}. Make sure the assignment ID exists.")
    else:
        st.warning("Please enter an Assignment ID.")

# --- Essay Evaluation ---
st.header("3. Evaluate Student Essay")
with st.form("evaluate_essay_form"):
    assignment_id_evaluate = st.text_input("Assignment ID for Evaluation")
    student_essay = st.text_area("Paste Student Essay Here", height=300)
    
    submitted_evaluation = st.form_submit_button("Evaluate Essay")
    if submitted_evaluation and assignment_id_evaluate and student_essay:
        evaluation_request_data = {"assignment_id": assignment_id_evaluate, "essay_text": student_essay}
        try:
            response = requests.post(f"{FASTAPI_BASE_URL}/evaluate", json=evaluation_request_data)
            response.raise_for_status()
            evaluation_result = response.json()
            st.subheader("Evaluation Results:")
            st.metric(label="Score", value=f"{evaluation_result['score']}/100")
            st.write("**Feedback:**")
            st.write(evaluation_result["feedback"])
            st.write("**Strengths:**")
            for s in evaluation_result["strengths"]:
                st.markdown(f"- {s}")
            st.write("**Weaknesses:**")
            for w in evaluation_result["weaknesses"]:
                st.markdown(f"- {w}")
            st.success("Essay evaluated!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error evaluating essay: {e}. Ensure assignment and guidelines exist.")
    else:
        st.warning("Please provide Assignment ID and essay text.")

