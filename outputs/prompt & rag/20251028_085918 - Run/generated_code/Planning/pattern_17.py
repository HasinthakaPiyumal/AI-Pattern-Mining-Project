"""AI-Powered Adaptive Project Manager"""

import os
import json
from datetime import date, timedelta
from typing import List, Optional, Dict, Any

import streamlit as st
import requests

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.ext.declarative import declarative_base

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import Field # Using v1 for compatibility with older langchain
import networkx as nx

# --- Configuration --- #
DATABASE_URL = "sqlite:///./project_manager.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FASTAPI_PORT = 8000
FASTAPI_URL = f"http://localhost:{FASTAPI_PORT}"

# --- Database Setup (SQLAlchemy) --- #
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    goal = Column(String)
    created_at = Column(Date, default=date.today)
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    name = Column(String)
    description = Column(String, nullable=True)
    status = Column(String, default="Not Started") # Not Started, In Progress, Completed, Blocked
    estimated_start_date = Column(Date, nullable=True)
    estimated_end_date = Column(Date, nullable=True)
    actual_start_date = Column(Date, nullable=True)
    actual_end_date = Column(Date, nullable=True)
    assigned_resource_id = Column(Integer, ForeignKey("resources.id"), nullable=True)
    priority = Column(Integer, default=5) # 1 (High) - 10 (Low)
    is_critical = Column(Boolean, default=False)

    project = relationship("Project", back_populates="tasks")
    assigned_resource = relationship("Resource", back_populates="tasks")
    dependencies_as_prerequisite = relationship("Dependency", foreign_keys="[Dependency.prerequisite_task_id]", back_populates="prerequisite_task", cascade="all, delete-orphan")
    dependencies_as_dependent = relationship("Dependency", foreign_keys="[Dependency.dependent_task_id]", back_populates="dependent_task", cascade="all, delete-orphan")

class Resource(Base):
    __tablename__ = "resources"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    skill_set = Column(String, nullable=True) # Comma-separated skills
    availability = Column(String, default="Full-time") # Full-time, Part-time, dates of unavailability
    tasks = relationship("Task", back_populates="assigned_resource")

class Dependency(Base):
    __tablename__ = "dependencies"
    id = Column(Integer, primary_key=True, index=True)
    prerequisite_task_id = Column(Integer, ForeignKey("tasks.id"))
    dependent_task_id = Column(Integer, ForeignKey("tasks.id"))
    type = Column(String, default="finish-to-start") # finish-to-start, start-to-start, finish-to-finish, start-to-finish

    prerequisite_task = relationship("Task", foreign_keys="[Dependency.prerequisite_task_id]", back_populates="dependencies_as_prerequisite")
    dependent_task = relationship("Task", foreign_keys="[Dependency.dependent_task_id]", back_populates="dependencies_as_dependent")

def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

# --- Pydantic Models --- #
class ResourceBase(BaseModel):
    name: str
    skill_set: Optional[str] = None
    availability: str = "Full-time"

class ResourceCreate(ResourceBase):
    pass

class ResourceResponse(ResourceBase):
    id: int

    class Config:
        from_attributes = True

class DependencyBase(BaseModel):
    prerequisite_task_id: int
    dependent_task_id: int
    type: str = "finish-to-start"

class DependencyCreate(DependencyBase):
    pass

class DependencyResponse(DependencyBase):
    id: int

    class Config:
        from_attributes = True

class TaskBase(BaseModel):
    name: str
    description: Optional[str] = None
    status: str = "Not Started"
    estimated_start_date: Optional[date] = None
    estimated_end_date: Optional[date] = None
    actual_start_date: Optional[date] = None
    actual_end_date: Optional[date] = None
    assigned_resource_id: Optional[int] = None
    priority: int = 5
    is_critical: bool = False

class TaskCreate(TaskBase):
    pass

class TaskUpdate(TaskBase):
    name: Optional[str] = None
    status: Optional[str] = None
    estimated_start_date: Optional[date] = None
    estimated_end_date: Optional[date] = None
    actual_start_date: Optional[date] = None
    actual_end_date: Optional[date] = None
    assigned_resource_id: Optional[int] = None
    priority: Optional[int] = None
    is_critical: Optional[bool] = None

class TaskResponse(TaskBase):
    id: int
    project_id: int
    dependencies_as_prerequisite: List[DependencyResponse] = []
    dependencies_as_dependent: List[DependencyResponse] = []
    assigned_resource: Optional[ResourceResponse] = None

    class Config:
        from_attributes = True

class ProjectBase(BaseModel):
    name: str
    goal: str

class ProjectCreate(ProjectBase):
    pass

class ProjectResponse(ProjectBase):
    id: int
    created_at: date
    tasks: List[TaskResponse] = []

    class Config:
        from_attributes = True

class PlanResponse(BaseModel):
    project_id: int
    project_name: str
    goal: str
    plan_summary: str
    tasks: List[TaskResponse]
    # Add a Gantt-chart like structure if needed

# --- Dependency Injection for Database Session --- #
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- LangChain LLM Agents (Simplified/Mocked) --- #
if OPENAI_API_KEY:
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
else:
    class MockLLM:
        def invoke(self, prompt):
            if "decompose" in prompt.lower():
                return "{'tasks': [{'name': 'Define Project Scope', 'description': 'Outline the boundaries and deliverables.'}, {'name': 'Gather Requirements', 'description': 'Collect detailed functional and non-functional requirements.'}, {'name': 'Design System Architecture', 'description': 'Create a high-level technical design.'}, {'name': 'Develop Core Features', 'description': 'Implement the main functionalities.'}, {'name': 'Conduct Testing', 'description': 'Perform unit, integration, and user acceptance testing.'}, {'name': 'Deploy Application', 'description': 'Release the application to production.'}, {'name': 'Monitor and Maintain', 'description': 'Continuously track performance and address issues.'}]}"
            elif "plan" in prompt.lower() or "re-plan" in prompt.lower():
                return "{'plan_summary': 'An initial plan has been generated based on the project goal and available resources. Dependencies are critical for successful execution. Adjustments will be made as tasks progress.', 'tasks': [{'name': 'Define Project Scope', 'estimated_start_date': '2023-01-01', 'estimated_end_date': '2023-01-05', 'status': 'Not Started', 'is_critical': True}, {'name': 'Gather Requirements', 'estimated_start_date': '2023-01-06', 'estimated_end_date': '2023-01-15', 'status': 'Not Started', 'is_critical': True, 'prerequisites': ['Define Project Scope']}, {'name': 'Design System Architecture', 'estimated_start_date': '2023-01-16', 'estimated_end_date': '2023-01-30', 'status': 'Not Started', 'is_critical': True, 'prerequisites': ['Gather Requirements']}, {'name': 'Develop Core Features', 'estimated_start_date': '2023-02-01', 'estimated_end_date': '2023-03-15', 'status': 'Not Started', 'prerequisites': ['Design System Architecture']}, {'name': 'Conduct Testing', 'estimated_start_date': '2023-03-16', 'estimated_end_date': '2023-03-30', 'status': 'Not Started', 'prerequisites': ['Develop Core Features']}, {'name': 'Deploy Application', 'estimated_start_date': '2023-04-01', 'estimated_end_date': '2023-04-05', 'status': 'Not Started', 'prerequisites': ['Conduct Testing']}, {'name': 'Monitor and Maintain', 'estimated_start_date': '2023-04-06', 'estimated_end_date': '2023-04-30', 'status': 'Not Started', 'prerequisites': ['Deploy Application']}]}"
            return "{'response': 'I am a mock LLM. This is a default response.'}"

    llm = MockLLM()
    print("Warning: OPENAI_API_KEY not found. Using MockLLM for demonstration.")

class TaskDecomposerAgent:
    def __init__(self, llm_model: Any):
        self.llm = llm_model
        self.prompt = PromptTemplate(
            input_variables=["project_goal"],
            template="""You are an expert project manager. Decompose the following project goal into a list of distinct, actionable sub-tasks. Each task should have a 'name' and a 'description'. Return a JSON array of these tasks.
            Project Goal: {project_goal}
            Example Output: {{'tasks': [{{'name': 'Task 1', 'description': 'Desc 1'}}, {{'name': 'Task 2', 'description': 'Desc 2'}}]}}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def decompose_goal(self, project_goal: str) -> List[Dict[str, str]]:
        response = self.chain.invoke({"project_goal": project_goal})
        try:
            # The invoke method might return a dict with a 'text' key or directly the string
            response_str = response.get('text', str(response))
            # Clean up potential markdown formatting if LLM adds it
            if response_str.startswith("```json") and response_str.endswith("```"):
                response_str = response_str[7:-3].strip()
            return json.loads(response_str).get('tasks', [])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM: {e}")
            print(f"LLM Raw Response: {response}")
            return []

class PlannerAgent:
    def __init__(self, llm_model: Any):
        self.llm = llm_model
        self.prompt = PromptTemplate(
            input_variables=["project_goal", "tasks_json", "resources_json", "constraints_json", "current_plan_json"],
            template="""You are an expert project planner. Given a project goal, a list of tasks, available resources, and any constraints, generate a detailed project plan. 
            The plan should include estimated start/end dates for each task, assigned resources if possible, a brief plan summary, and explicit dependencies between tasks.
            If 'current_plan_json' is provided, re-plan and adjust based on the current status.
            Return the plan as a JSON object with 'plan_summary' and a 'tasks' array. Each task in the array should include 'name', 'estimated_start_date' (YYYY-MM-DD), 'estimated_end_date' (YYYY-MM-DD), 'status', 'is_critical', and 'prerequisites' (a list of task names).

            Project Goal: {project_goal}
            Tasks: {tasks_json}
            Resources: {resources_json}
            Constraints: {constraints_json}
            Current Plan (for re-planning): {current_plan_json}
            Example Output: {{'plan_summary': '...', 'tasks': [{{'name': 'Task 1', 'estimated_start_date': '2023-01-01', 'estimated_end_date': '2023-01-05', 'status': 'Not Started', 'is_critical': True, 'prerequisites': []}}]}}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_plan(self, project_goal: str, tasks: List[TaskResponse], resources: List[ResourceResponse], constraints: Dict[str, Any], current_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tasks_json = json.dumps([t.model_dump(mode='json') for t in tasks])
        resources_json = json.dumps([r.model_dump(mode='json') for r in resources])
        constraints_json = json.dumps(constraints)
        current_plan_json = json.dumps(current_plan) if current_plan else "{}"

        response = self.chain.invoke({
            "project_goal": project_goal,
            "tasks_json": tasks_json,
            "resources_json": resources_json,
            "constraints_json": constraints_json,
            "current_plan_json": current_plan_json
        })
        try:
            response_str = response.get('text', str(response))
            if response_str.startswith("```json") and response_str.endswith("```"):
                response_str = response_str[7:-3].strip()
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM: {e}")
            print(f"LLM Raw Response: {response}")
            return {"plan_summary": "Could not generate a coherent plan.", "tasks": []}

# Initialize Agents
task_decomposer_agent = TaskDecomposerAgent(llm)
planner_agent = PlannerAgent(llm)

# --- FastAPI Application --- #
app = FastAPI(
    title="Adaptive Project Manager API",
    description="API for AI-powered project management, decomposition, and adaptive planning."
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    # Add some initial resources if the database is empty
    db = SessionLocal()
    if not db.query(Resource).first():
        db.add(Resource(name="Alice", skill_set="Backend, DevOps"))
        db.add(Resource(name="Bob", skill_set="Frontend, UI/UX"))
        db.add(Resource(name="Charlie", skill_set="Testing, QA"))
        db.commit()
        db.close()

@app.post("/projects/", response_model=ProjectResponse)
def create_project(project_create: ProjectCreate, db: Session = Depends(get_db)):
    db_project = Project(name=project_create.name, goal=project_create.goal)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)

    # Decompose tasks using LLM
    decomposed_tasks = task_decomposer_agent.decompose_goal(project_create.goal)
    if not decomposed_tasks:
        raise HTTPException(status_code=500, detail="Failed to decompose project goal into tasks.")

    # Convert decomposed tasks to DB models and associate with project
    db_tasks = []
    for task_data in decomposed_tasks:
        db_task = Task(project_id=db_project.id, **task_data)
        db.add(db_task)
        db_tasks.append(db_task)
    db.commit()
    for db_task in db_tasks:
        db.refresh(db_task)
    db_project.tasks = db_tasks # Ensure tasks are loaded into the project object

    # Initial planning using LLM
    all_resources = db.query(Resource).all()
    resources_response = [ResourceResponse.model_validate(r) for r in all_resources]
    tasks_response = [TaskResponse.model_validate(t) for t in db_tasks]
    
    initial_plan = planner_agent.generate_plan(db_project.goal, tasks_response, resources_response, {})
    
    # Apply plan to database (dates, dependencies)
    task_name_to_id = {task.name: task.id for task in db_tasks}
    for planned_task_data in initial_plan.get('tasks', []):
        task_name = planned_task_data.get('name')
        if task_name in task_name_to_id:
            task_id = task_name_to_id[task_name]
            db_task = db.query(Task).filter(Task.id == task_id).first()
            if db_task:
                # Update task properties based on plan
                db_task.estimated_start_date = date.fromisoformat(planned_task_data['estimated_start_date']) if 'estimated_start_date' in planned_task_data else None
                db_task.estimated_end_date = date.fromisoformat(planned_task_data['estimated_end_date']) if 'estimated_end_date' in planned_task_data else None
                db_task.is_critical = planned_task_data.get('is_critical', False)
                db.add(db_task)

                # Add dependencies
                prerequisites = planned_task_data.get('prerequisites', [])
                for prereq_name in prerequisites:
                    prereq_id = task_name_to_id.get(prereq_name)
                    if prereq_id and not db.query(Dependency).filter_by(prerequisite_task_id=prereq_id, dependent_task_id=task_id).first():
                        db_dependency = Dependency(prerequisite_task_id=prereq_id, dependent_task_id=task_id)
                        db.add(db_dependency)
    db.commit()
    db.refresh(db_project)

    return ProjectResponse.model_validate(db_project)


@app.get("/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectResponse.model_validate(project)

@app.put("/tasks/{task_id}", response_model=TaskResponse)
def update_task(task_id: int, task_update: TaskUpdate, db: Session = Depends(get_db)):
    db_task = db.query(Task).filter(Task.id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Update task fields
    update_data = task_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_task, key, value)

    if task_update.status == "Completed" and not db_task.actual_end_date:
        db_task.actual_end_date = date.today()
    if task_update.status == "In Progress" and not db_task.actual_start_date:
        db_task.actual_start_date = date.today()

    db.add(db_task)
    db.commit()
    db.refresh(db_task)

    # Trigger re-planning if a critical task status changes or dates are affected
    if task_update.status in ["Completed", "Blocked"] or (task_update.estimated_end_date and db_task.is_critical):
        project = db.query(Project).filter(Project.id == db_task.project_id).first()
        if project:
            print(f"--- Re-planning triggered for project {project.name} due to task {db_task.name} status change to {db_task.status} ---")
            all_tasks = db.query(Task).filter(Task.project_id == project.id).all()
            all_resources = db.query(Resource).all()
            
            tasks_response = [TaskResponse.model_validate(t) for t in all_tasks]
            resources_response = [ResourceResponse.model_validate(r) for r in all_resources]

            # Create a simplified current plan for LLM context
            current_plan_for_llm = {"tasks": []}
            for task in all_tasks:
                task_data = {
                    "name": task.name,
                    "status": task.status,
                    "estimated_end_date": str(task.estimated_end_date) if task.estimated_end_date else None,
                    "actual_end_date": str(task.actual_end_date) if task.actual_end_date else None,
                    "is_critical": task.is_critical
                }
                current_plan_for_llm["tasks"].append(task_data)

            re_plan_result = planner_agent.generate_plan(
                project.goal, 
                tasks_response, 
                resources_response, 
                {"changed_task": db_task.name, "new_status": db_task.status},
                current_plan=current_plan_for_llm
            )
            
            # Apply re-plan to database (simplified: only update dates for now)
            task_name_to_db_task = {t.name: t for t in all_tasks}
            for planned_task_data in re_plan_result.get('tasks', []):
                task_name = planned_task_data.get('name')
                if task_name in task_name_to_db_task:
                    updated_db_task = task_name_to_db_task[task_name]
                    # Only update estimated dates from re-plan, actuals are user-driven
                    if 'estimated_start_date' in planned_task_data and planned_task_data['estimated_start_date']:
                        updated_db_task.estimated_start_date = date.fromisoformat(planned_task_data['estimated_start_date'])
                    if 'estimated_end_date' in planned_task_data and planned_task_data['estimated_end_date']:
                        updated_db_task.estimated_end_date = date.fromisoformat(planned_task_data['estimated_end_date'])
                    updated_db_task.is_critical = planned_task_data.get('is_critical', updated_db_task.is_critical)
                    db.add(updated_db_task)
            db.commit()
            print(f"--- Re-planning complete for project {project.name} ---")

    return TaskResponse.model_validate(db_task)

@app.post("/resources/", response_model=ResourceResponse)
def create_resource(resource_create: ResourceCreate, db: Session = Depends(get_db)):
    db_resource = Resource(**resource_create.model_dump())
    db.add(db_resource)
    db.commit()
    db.refresh(db_resource)
    return ResourceResponse.model_validate(db_resource)

@app.get("/resources/", response_model=List[ResourceResponse])
def get_resources(db: Session = Depends(get_db)):
    resources = db.query(Resource).all()
    return [ResourceResponse.model_validate(r) for r in resources]

# --- Streamlit Frontend --- #
st.set_page_config(layout="wide")

st.title("AI-Powered Adaptive Project Manager")

if st.sidebar.button("Initialize/Reset Database"): # For easy testing
    if os.path.exists("project_manager.db"):
        os.remove("project_manager.db")
    create_db_and_tables()
    st.sidebar.success("Database initialized!")
    # Re-add default resources after reset
    db = SessionLocal()
    db.add(Resource(name="Alice", skill_set="Backend, DevOps"))
    db.add(Resource(name="Bob", skill_set="Frontend, UI/UX"))
    db.add(Resource(name="Charlie", skill_set="Testing, QA"))
    db.commit()
    db.close()
    st.experimental_rerun()

# --- Project Creation --- #
st.header("1. Create New Project")
with st.form("new_project_form"):
    project_name = st.text_input("Project Name", "Launch New E-commerce Site")
    project_goal = st.text_area("Project Goal", "Develop and launch a new e-commerce website with payment processing, user accounts, product catalog, and an admin panel by Q4.")
    submit_button = st.form_submit_button("Create Project")

    if submit_button:
        try:
            response = requests.post(f"{FASTAPI_URL}/projects/", json={
                "name": project_name,
                "goal": project_goal
            })
            response.raise_for_status() # Raise an exception for HTTP errors
            st.success("Project created and tasks decomposed successfully!")
            st.session_state["current_project_id"] = response.json()["id"]
            st.experimental_rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Error creating project: {e}")
            if response is not None:
                st.error(f"Server response: {response.text}")

# --- Project Selection --- #
st.header("2. Select Existing Project")
projects_response = requests.get(f"{FASTAPI_URL}/projects/")
if projects_response.status_code == 200:
    all_projects = projects_response.json()
    project_names = {p["name"]: p["id"] for p in all_projects}
    selected_project_name = st.selectbox("Select a Project", list(project_names.keys()), key="project_selector")
    if selected_project_name:
        st.session_state["current_project_id"] = project_names[selected_project_name]
    else:
        st.session_state["current_project_id"] = None
else:
    st.error("Could not load projects from API.")
    all_projects = []

current_project_id = st.session_state.get("current_project_id")

if current_project_id:
    st.subheader(f"Project: {selected_project_name}")
    try:
        project_data_response = requests.get(f"{FASTAPI_URL}/projects/{current_project_id}")
        project_data_response.raise_for_status()
        project_data = project_data_response.json()

        st.write(f"**Goal:** {project_data['goal']}")
        st.write(f"**Created At:** {project_data['created_at']}")

        st.subheader("Tasks & Plan")
        if project_data['tasks']:
            tasks_df_data = []
            task_id_to_name = {t['id']: t['name'] for t in project_data['tasks']}
            
            # Fetch resources for display
            resources_response = requests.get(f"{FASTAPI_URL}/resources/")
            all_resources_map = {r['id']: r['name'] for r in resources_response.json()} if resources_response.status_code == 200 else {}

            for task in project_data['tasks']:
                prerequisites = []
                for dep in task.get('dependencies_as_dependent', []) + task.get('dependencies_as_prerequisite', []):
                    if dep['dependent_task_id'] == task['id'] and dep['prerequisite_task_id'] in task_id_to_name:
                        prerequisites.append(task_id_to_name[dep['prerequisite_task_id']])

                tasks_df_data.append({
                    "ID": task['id'],
                    "Task Name": task['name'],
                    "Status": task['status'],
                    "Est. Start": task['estimated_start_date'],
                    "Est. End": task['estimated_end_date'],
                    "Actual Start": task['actual_start_date'],
                    "Actual End": task['actual_end_date'],
                    "Assigned To": all_resources_map.get(task['assigned_resource_id'], "N/A"),
                    "Critical": "✅" if task['is_critical'] else "",
                    "Prerequisites": ", ".join(prerequisites) if prerequisites else "None"
                })
            st.dataframe(tasks_df_data, use_container_width=True)

            st.subheader("Update Task Status")
            task_options = {f"{t['name']} (ID: {t['id']})": t['id'] for t in project_data['tasks']}
            selected_task_for_update_label = st.selectbox("Select Task to Update", list(task_options.keys()))
            
            if selected_task_for_update_label:
                selected_task_id_for_update = task_options[selected_task_for_update_label]
                current_task_status = next((t['status'] for t in project_data['tasks'] if t['id'] == selected_task_id_for_update), "Not Started")
                new_status = st.selectbox(
                    "New Status",
                    ["Not Started", "In Progress", "Completed", "Blocked"],
                    index=["Not Started", "In Progress", "Completed", "Blocked"].index(current_task_status)
                )
                if st.button(f"Update Status for {selected_task_for_update_label}"):
                    try:
                        update_response = requests.put(f"{FASTAPI_URL}/tasks/{selected_task_id_for_update}", json={
                            "status": new_status
                        })
                        update_response.raise_for_status()
                        st.success(f"Task {selected_task_for_update_label} status updated to {new_status}. Re-planning may have occurred.")
                        st.experimental_rerun()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error updating task status: {e}")
                        if update_response is not None:
                            st.error(f"Server response: {update_response.text}")

        else:
            st.write("No tasks found for this project.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching project details: {e}")
        if project_data_response is not None:
            st.error(f"Server response: {project_data_response.text}")

else:
    st.info("Create a new project or select an existing one to view its plan.")

st.sidebar.markdown("""---
**How to Run:**

1.  **Save this file** as `adaptive_project_manager.py`.
2.  **Install dependencies:**
    `pip install fastapi uvicorn sqlalchemy pydantic streamlit requests langchain-openai networkx python-dotenv`
3.  **Create a `.env` file** in the same directory with `OPENAI_API_KEY="your_openai_api_key"` (optional, will use mock LLM if not set).
4.  **Run FastAPI backend:**
    `uvicorn adaptive_project_manager:app --reload --port 8000`
5.  **Run Streamlit frontend** (in a separate terminal):
    `streamlit run adaptive_project_manager.py`
""")
