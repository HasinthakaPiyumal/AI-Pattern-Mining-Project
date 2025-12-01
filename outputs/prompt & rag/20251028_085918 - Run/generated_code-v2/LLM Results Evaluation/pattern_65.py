import os
import json
import uuid
import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

class Assignment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    prompt: str
    rubric: str

class EssaySubmission(BaseModel):
    assignment_id: str
    student_id: str
    essay_text: str

class CriterionGrade(BaseModel):
    criterion: str
    score: int = Field(..., ge=1, le=5)
    explanation: str

class GradingResult(BaseModel):
    overall_grade: str
    overall_score: int = Field(..., ge=1, le=100)
    criterion_grades: List[CriterionGrade]
    strengths: List[str]
    areas_for_improvement: List[str]
    
class Feedback(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    essay_id: str
    grading_result: GradingResult
    timestamp: str

assignments_db: List[Assignment] = []
essays_db: List[EssaySubmission] = []
feedback_db: List[Feedback] = []

class LLMService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.is_mock = False
        except (ImportError, Exception):
            self.is_mock = True

    def _construct_prompt(self, assignment: Assignment, essay_text: str) -> str:
        return f"""
        You are an experienced essay grader. Grade the following essay based on the provided assignment prompt and rubric.
        
        Assignment Title: {assignment.title}
        Assignment Prompt: {assignment.prompt}
        
        Grading Rubric:
        {assignment.rubric}
        
        Student Essay:
        {essay_text}
        
        Provide a detailed grading result in JSON format. The JSON should contain:
        - "overall_grade": An overall letter grade or descriptive assessment.
        - "overall_score": A numerical score between 1 and 100.
        - "criterion_grades": A list of dictionaries, each with "criterion", "score" (1-5), and "explanation".
        - "strengths": A list of specific strengths.
        - "areas_for_improvement": A list of specific areas for improvement.
        
        Ensure the JSON is perfectly formed. Do not include any other text outside the JSON.
        """

    def grade_essay(self, assignment: Assignment, essay_text: str) -> GradingResult:
        if self.is_mock:
            mock_json_response = {
                "overall_grade": "B+",
                "overall_score": 85,
                "criterion_grades": [
                    {"criterion": "Clarity of Thesis", "score": 4, "explanation": "Thesis is clear but could be more nuanced."},
                    {"criterion": "Argument Strength", "score": 4, "explanation": "Arguments are mostly sound but lack depth in some areas."},
                    {"criterion": "Grammar and Syntax", "score": 5, "explanation": "Excellent command of grammar and syntax."},
                    {"criterion": "Relevance to Prompt", "score": 5, "explanation": "Essay directly addresses the prompt effectively."},
                    {"criterion": "Overall Coherence", "score": 4, "explanation": "Ideas flow logically, but transitions could be smoother."}
                ],
                "strengths": [
                    "Strong understanding of the topic.",
                    "Well-structured paragraphs.",
                    "Excellent grammar."
                ],
                "areas_for_improvement": [
                    "Develop arguments with more detailed evidence.",
                    "Refine thesis for greater analytical depth.",
                    "Improve transitions between paragraphs."
                ]
            }
            return GradingResult(**mock_json_response)
        
        prompt = self._construct_prompt(assignment, essay_text)
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106", 
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.choices[0].message.content
            grading_data = json.loads(content)
            return GradingResult(**grading_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}. Raw response: {content}")
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {e}")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_service = LLMService(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/assignments", response_model=Assignment)
async def create_assignment(assignment: Assignment):
    assignments_db.append(assignment)
    return assignment

@app.get("/assignments", response_model=List[Assignment])
async def get_assignments():
    return assignments_db

@app.post("/submit-essay", response_model=Feedback)
async def submit_essay(submission: EssaySubmission):
    assignment = next((a for a in assignments_db if a.id == submission.assignment_id), None)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    essays_db.append(submission)

    grading_result = llm_service.grade_essay(assignment, submission.essay_text)
    
    essay_unique_id = f"{submission.assignment_id}-{submission.student_id}-{str(uuid.uuid4())[:8]}"
    new_feedback = Feedback(
        essay_id=essay_unique_id,
        grading_result=grading_result,
        timestamp=datetime.datetime.now().isoformat()
    )
    feedback_db.append(new_feedback)
    return new_feedback

@app.get("/feedback/{essay_id}", response_model=Feedback)
async def get_essay_feedback(essay_id: str):
    feedback = next((f for f in feedback_db if f.essay_id == essay_id), None)
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found for this essay ID")
    return feedback

import streamlit as st
import requests

FASTAPI_BASE_URL = "http://localhost:8000"

def streamlit_app():
    st.set_page_config(layout="wide", page_title="Automated Essay Grader")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home", "Define Assignment", "Submit Essay & Get Feedback"])

    if app_mode == "Home":
        st.title("Welcome to the Automated Essay Grader and Feedback System")
        st.write("""
            This system uses a Large Language Model (LLM) to automatically grade student essays
            and provide detailed feedback based on predefined assignment prompts and rubrics.
            
            Use the sidebar to:
            - **Define Assignment**: Instructors can set up new assignments with specific prompts and grading rubrics.
            - **Submit Essay & Get Feedback**: Students can submit their essays and receive instant grades and constructive feedback.
        """)
        st.image("https://via.placeholder.com/600x300.png?text=Automated+Grading+System", use_column_width=True)

    elif app_mode == "Define Assignment":
        st.title("Define New Assignment")
        with st.form("new_assignment_form"):
            title = st.text_input("Assignment Title")
            prompt = st.text_area("Assignment Prompt", height=200)
            rubric = st.text_area("Grading Rubric (e.g., 'Clarity of Thesis: 1-5, Argument Strength: 1-5')", height=300)
            
            submitted = st.form_submit_button("Create Assignment")
            if submitted:
                if title and prompt and rubric:
                    new_assignment_data = {"title": title, "prompt": prompt, "rubric": rubric}
                    try:
                        response = requests.post(f"{FASTAPI_BASE_URL}/assignments", json=new_assignment_data)
                        if response.status_code == 200:
                            st.success(f"Assignment '{title}' created successfully!")
                        else:
                            st.error(f"Error creating assignment: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the FastAPI backend. Make sure it's running at http://localhost:8000.")
                else:
                    st.warning("Please fill in all assignment details.")

        st.subheader("Existing Assignments")
        try:
            response = requests.get(f"{FASTAPI_BASE_URL}/assignments")
            if response.status_code == 200:
                existing_assignments = response.json()
                if existing_assignments:
                    for assign in existing_assignments:
                        st.json(assign)
                else:
                    st.info("No assignments defined yet.")
            else:
                st.error(f"Error fetching assignments: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend to fetch assignments. Make sure it's running.")

    elif app_mode == "Submit Essay & Get Feedback":
        st.title("Submit Your Essay")

        try:
            assignments_response = requests.get(f"{FASTAPI_BASE_URL}/assignments")
            if assignments_response.status_code == 200:
                assignments = assignments_response.json()
                if not assignments:
                    st.warning("No assignments available. Please ask an instructor to define one.")
                    return
                
                assignment_titles = {a["title"]: a["id"] for a in assignments}
                selected_title = st.selectbox("Select Assignment", list(assignment_titles.keys()))
                selected_assignment_id = assignment_titles[selected_title]

                selected_assignment_details = next((a for a in assignments if a["id"] == selected_assignment_id), None)
                if selected_assignment_details:
                    st.subheader("Assignment Prompt:")
                    st.info(selected_assignment_details["prompt"])
                    st.subheader("Grading Rubric:")
                    st.code(selected_assignment_details["rubric"])

                student_id = st.text_input("Your Student ID (e.g., S12345)")
                essay_text = st.text_area("Paste Your Essay Here", height=400)

                if st.button("Submit Essay for Grading"):
                    if not student_id or not essay_text:
                        st.warning("Please enter your Student ID and essay text.")
                    else:
                        submission_data = {
                            "assignment_id": selected_assignment_id,
                            "student_id": student_id,
                            "essay_text": essay_text
                        }
                        with st.spinner("Grading your essay... This may take a moment."):
                            try:
                                response = requests.post(f"{FASTAPI_BASE_URL}/submit-essay", json=submission_data)
                                if response.status_code == 200:
                                    feedback_data = response.json()
                                    st.success("Essay Graded! Here is your feedback:")
                                    
                                    st.subheader(f"Overall Grade: {feedback_data['grading_result']['overall_grade']} (Score: {feedback_data['grading_result']['overall_score']}/100)")
                                    
                                    st.write("---")
                                    st.subheader("Criterion-based Grading:")
                                    for criterion_grade in feedback_data['grading_result']['criterion_grades']:
                                        st.write(f"- **{criterion_grade['criterion']}**: Score {criterion_grade['score']} - {criterion_grade['explanation']}")

                                    st.write("---")
                                    st.subheader("Strengths:")
                                    for strength in feedback_data['grading_result']['strengths']:
                                        st.write(f"- {strength}")

                                    st.write("---")
                                    st.subheader("Areas for Improvement:")
                                    for area in feedback_data['grading_result']['areas_for_improvement']:
                                        st.write(f"- {area}")
                                    
                                    st.info(f"Feedback ID: {feedback_data['id']}")
                                else:
                                    st.error(f"Error submitting essay: {response.text}")
                            except requests.exceptions.ConnectionError:
                                st.error("Could not connect to the FastAPI backend. Make sure it's running at http://localhost:8000.")
                            except Exception as e:
                                st.error(f"An unexpected error occurred during grading: {e}")
            else:
                st.error(f"Error fetching assignments: {assignments_response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend to fetch assignments. Make sure it's running.")


if __name__ == "__main__":
    # This block is executed when the script is run directly, e.g., `python main.py`.
    # It starts the FastAPI backend server.
    print("Starting FastAPI backend...")
    print("API documentation available at http://localhost:8000/docs")
    print("To run the Streamlit frontend, open a separate terminal and execute: `streamlit run main.py`")
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # The `streamlit_app()` function will be implicitly executed by Streamlit when `streamlit run main.py` is invoked.
    # Any code outside of `if __name__ == "__main__":` or a function will also be executed by Streamlit.
    # The logic for the Streamlit frontend is encapsulated in `streamlit_app()`.
    streamlit_app() # This call is primarily for when streamlit is run. If `python main.py` is run, this will not start a streamlit server. This is here for when streamlit itself executes the file.