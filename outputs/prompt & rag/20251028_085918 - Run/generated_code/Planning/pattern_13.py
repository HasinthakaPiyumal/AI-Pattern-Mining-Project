
import streamlit as st
import networkx as nx
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

# --- Enums and Pydantic-like Models (simplified for direct use) ---

class TaskStatus(Enum):
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    BLOCKED = "Blocked"

class ProjectStatus(Enum):
    PLANNING = "Planning"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    ON_HOLD = "On Hold"
    CANCELLED = "Cancelled"

class Task:
    def __init__(
        self, 
        task_id: str, 
        name: str, 
        description: str, 
        status: TaskStatus = TaskStatus.PENDING, 
        assigned_to: Optional[str] = None,
        estimated_duration_days: int = 1,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None
    ):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.status = status
        self.assigned_to = assigned_to
        self.estimated_duration_days = estimated_duration_days
        self.start_date = start_date
        self.end_date = end_date
        self.dependencies = dependencies if dependencies is not None else []

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "estimated_duration_days": self.estimated_duration_days,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "dependencies": self.dependencies
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            assigned_to=data.get("assigned_to"),
            estimated_duration_days=data.get("estimated_duration_days", 1),
            start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            dependencies=data.get("dependencies", [])
        )

class Constraint:
    def __init__(self, name: str, description: str, type: str, value: Any):
        self.name = name
        self.description = description
        self.type = type # e.g., 'budget', 'deadline', 'resource_availability'
        self.value = value

class Project:
    def __init__(
        self, 
        project_id: str, 
        name: str, 
        goal: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        status: ProjectStatus = ProjectStatus.PLANNING
    ):
        self.project_id = project_id
        self.name = name
        self.goal = goal
        self.start_date = start_date
        self.end_date = end_date
        self.status = status
        self.tasks: Dict[str, Task] = {}
        self.constraints: List[Constraint] = []
        self.task_graph = nx.DiGraph() # For dependencies

    def add_task(self, task: Task):
        if task.task_id not in self.tasks:
            self.tasks[task.task_id] = task
            self.task_graph.add_node(task.task_id)
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    self.task_graph.add_edge(dep_id, task.task_id)
        else:
            st.warning(f"Task with ID {task.task_id} already exists.")

    def update_task_status(self, task_id: str, new_status: TaskStatus):
        if task_id in self.tasks:
            self.tasks[task_id].status = new_status
        else:
            st.error(f"Task with ID {task_id} not found.")

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)

    def get_next_executable_tasks(self) -> List[Task]:
        executable_tasks = []
        completed_task_ids = {tid for tid, task in self.tasks.items() if task.status == TaskStatus.COMPLETED}

        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.PENDING, TaskStatus.BLOCKED]: # Consider blocked tasks for re-evaluation
                # Check if all dependencies are completed
                predecessors = list(self.task_graph.predecessors(task_id))
                if all(pred in completed_task_ids for pred in predecessors):
                    executable_tasks.append(task)
        return executable_tasks

    def to_dict(self):
        return {
            "project_id": self.project_id,
            "name": self.name,
            "goal": self.goal,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status.value,
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
            "constraints": [{c.name: c.value} for c in self.constraints] # Simplified representation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        project = cls(
            project_id=data["project_id"],
            name=data["name"],
            goal=data["goal"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            status=ProjectStatus(data["status"])
        )
        for task_dict in data.get("tasks", {}).values():
            project.add_task(Task.from_dict(task_dict))
        # Rebuild graph from tasks already added with their dependencies
        project.task_graph.clear()
        for task_id, task in project.tasks.items():
            project.task_graph.add_node(task_id)
            for dep_id in task.dependencies:
                if dep_id in project.tasks:
                    project.task_graph.add_edge(dep_id, task_id)
        return project


# --- LLM Simulation (Placeholder) ---
class LLMSimulator:
    def parse_goal(self, goal: str) -> Dict[str, Any]:
        st.info(f"Simulating LLM: Parsing goal '{goal}'...")
        # In a real application, this would call an actual LLM API
        # using LangChain, OpenAI, Gemini, etc. to extract entities, constraints, initial tasks.
        keywords = goal.lower().split()
        estimated_tasks = 5
        if "website" in keywords or "web app" in keywords:
            estimated_tasks = 7
        elif "mobile app" in keywords:
            estimated_tasks = 8

        return {
            "initial_understanding": f"Goal is to develop a {goal}.",
            "estimated_tasks_count": estimated_tasks,
            "potential_constraints": [
                {"type": "deadline", "value": (datetime.now() + timedelta(days=30)).isoformat(), "name": "Project Deadline"},
                {"type": "budget", "value": 10000, "name": "Development Budget"}
            ],
            "key_entities": [word for word in keywords if len(word) > 3]
        }

    def decompose_task(self, goal_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        st.info(f"Simulating LLM: Decomposing tasks for goal: {goal_details.get('initial_understanding')}...")
        # This would be a more sophisticated LLM call generating a task breakdown
        base_tasks = [
            {"id": "T1", "name": "Requirement Gathering", "desc": "Gather and document all project requirements.", "duration": 3},
            {"id": "T2", "name": "System Design", "desc": "Design the system architecture and database schema.", "duration": 5, "deps": ["T1"]},
            {"id": "T3", "name": "Frontend Development", "desc": "Develop the user interface.", "duration": 7, "deps": ["T2"]},
            {"id": "T4", "name": "Backend Development", "desc": "Develop the server-side logic and APIs.", "duration": 8, "deps": ["T2"]},
            {"id": "T5", "name": "Database Implementation", "desc": "Set up and populate the database.", "duration": 4, "deps": ["T2"]},
            {"id": "T6", "name": "Integration Testing", "desc": "Test the integration between frontend, backend, and database.", "duration": 3, "deps": ["T3", "T4", "T5"]},
            {"id": "T7", "name": "User Acceptance Testing (UAT)", "desc": "Conduct UAT with stakeholders.", "duration": 2, "deps": ["T6"]},
            {"id": "T8", "name": "Deployment", "desc": "Deploy the application to production.", "duration": 1, "deps": ["T7"]}
        ]

        # Filter or adjust based on goal_details if needed in a real LLM
        num_tasks = goal_details.get("estimated_tasks_count", len(base_tasks))
        return base_tasks[:num_tasks]

    def generate_recommendation(self, project: Project) -> str:
        st.info(f"Simulating LLM: Generating recommendations for project {project.name}...")
        # Analyze project status and suggest actions
        completed_tasks = sum(1 for t in project.tasks.values() if t.status == TaskStatus.COMPLETED)
        total_tasks = len(project.tasks)
        progress_percent = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

        recommendations = []
        if project.status == ProjectStatus.PLANNING:
            recommendations.append("Finalize initial task assignments and kick-off the project.")
        elif project.status == ProjectStatus.IN_PROGRESS:
            if progress_percent < 25:
                recommendations.append("Focus on completing initial setup tasks. Ensure resources are allocated.")
            elif progress_percent < 75:
                recommendations.append("Monitor task dependencies closely. Address any blockers promptly.")
            else:
                recommendations.append("Prepare for final testing and deployment. Communicate with stakeholders.")
            
            if any(task.status == TaskStatus.BLOCKED for task in project.tasks.values()):
                recommendations.append("High priority: Address blocked tasks immediately to prevent project delays.")

            # Check for overdue tasks (simplified)
            overdue_tasks = [t.name for t in project.tasks.values() 
                             if t.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS] and t.end_date and t.end_date < datetime.now()]
            if overdue_tasks:
                recommendations.append(f"Tasks are overdue: {', '.join(overdue_tasks)}. Consider re-planning or re-allocating resources.")

        if not recommendations:
            return "Project appears to be on track. Continue monitoring progress."
        return "\n- " + "\n- ".join(recommendations)


# --- Adaptive Planner Core Logic ---
class AdaptivePlanner:
    def __init__(self, llm_simulator: LLMSimulator):
        self.llm_simulator = llm_simulator
        self.projects: Dict[str, Project] = {}
        self.next_project_id = 1

    def create_project(self, name: str, goal: str, start_date: datetime) -> Project:
        project_id = f"P{self.next_project_id:03d}"
        self.next_project_id += 1
        project = Project(project_id, name, goal, start_date)
        self.projects[project_id] = project
        st.success(f"Project '{name}' created with ID: {project_id}")
        return project

    def initialize_plan(self, project_id: str):
        project = self.projects.get(project_id)
        if not project:
            st.error(f"Project {project_id} not found.")
            return

        st.subheader("1. Goal Parsing & Intent Extraction (LLM)")
        goal_details = self.llm_simulator.parse_goal(project.goal)
        st.json(goal_details)

        st.subheader("2. Task Decomposition (LLM)")
        decomposed_tasks_data = self.llm_simulator.decompose_task(goal_details)
        
        st.subheader("3. Adding Tasks and Dependencies")
        current_date = project.start_date
        for task_data in decomposed_tasks_data:
            task = Task(
                task_id=task_data["id"],
                name=task_data["name"],
                description=task_data["desc"],
                estimated_duration_days=task_data.get("duration", 1),
                dependencies=task_data.get("deps", []),
                start_date=current_date, # Simple sequential assignment for demo
                end_date=current_date + timedelta(days=task_data.get("duration", 1))
            )
            project.add_task(task)
            current_date += timedelta(days=task.estimated_duration_days) # Advance date
        st.success(f"Initial plan generated with {len(project.tasks)} tasks.")

        st.subheader("4. Applying Constraints")
        for const_data in goal_details.get("potential_constraints", []):
            constraint = Constraint(
                name=const_data["name"],
                description=const_data["type"],
                type=const_data["type"],
                value=const_data["value"]
            )
            project.add_constraint(constraint)
            st.write(f"- Added constraint: {constraint.name} ({constraint.type}: {constraint.value})")
        project.status = ProjectStatus.IN_PROGRESS

    def process_feedback_and_adapt(self, project_id: str, feedback: Dict[str, Any]):
        project = self.projects.get(project_id)
        if not project:
            st.error(f"Project {project_id} not found.")
            return

        st.subheader(f"Processing Feedback for {project.name}")
        st.json(feedback)

        # Example feedback processing: update task status
        if "task_updates" in feedback:
            for task_update in feedback["task_updates"]:
                task_id = task_update.get("task_id")
                new_status_str = task_update.get("status")
                if task_id and new_status_str:
                    try:
                        new_status = TaskStatus[new_status_str.upper()]
                        project.update_task_status(task_id, new_status)
                        st.info(f"Task {task_id} status updated to {new_status.value}.")
                    except KeyError:
                        st.warning(f"Invalid status: {new_status_str} for task {task_id}.")

        # Adaptive Re-planning Logic (simplified)
        st.subheader("Adaptive Re-planning")
        current_executable_tasks = project.get_next_executable_tasks()
        if not current_executable_tasks and len([t for t in project.tasks.values() if t.status != TaskStatus.COMPLETED]) > 0:
            st.warning("No executable tasks found, but project is not complete. Potential bottleneck or unhandled dependencies.")
            st.info("Attempting to identify blocked tasks or re-evaluate dependencies...")
            # In a real system, this would trigger more complex reasoning:
            # - Identify tasks stuck in PENDING with unmet dependencies
            # - Use LLM to suggest alternative approaches or resource re-allocation
            # - Check for circular dependencies (NetworkX has cycles detection)
            # For demo, let's just mark some pending tasks as 'in progress' if no blockers are explicit.
            for task in project.tasks.values():
                if task.status == TaskStatus.PENDING:
                    project.update_task_status(task.task_id, TaskStatus.IN_PROGRESS) # Force progress for demo
                    st.info(f"*Demo*: Forcing {task.task_id} into 'In Progress' to simulate progress.")
                    break # Only do one for simplicity

        # Check if all tasks are completed
        if all(task.status == TaskStatus.COMPLETED for task in project.tasks.values()):
            project.status = ProjectStatus.COMPLETED
            project.end_date = datetime.now()
            st.success(f"Project {project.name} (ID: {project.project_id}) has been completed!")

        st.success("Feedback processed and plan adapted (simplified).")


# --- Streamlit UI --- 

st.set_page_config(layout="wide", page_title="Adaptive Project Manager")
st.title("AI-Powered Adaptive Project Management")

# Initialize LLM Simulator and Adaptive Planner
if "llm_simulator" not in st.session_state:
    st.session_state.llm_simulator = LLMSimulator()
if "planner" not in st.session_state:
    st.session_state.planner = AdaptivePlanner(st.session_state.llm_simulator)

# --- Project Creation --- 
st.sidebar.header("Create New Project")
with st.sidebar.form("new_project_form"):
    project_name = st.text_input("Project Name")
    project_goal = st.text_area("Project Goal (e.g., 'Build an e-commerce website with user authentication')")
    project_start_date = st.date_input("Start Date", datetime.now())
    submitted = st.form_submit_button("Create & Initialize Plan")
    if submitted and project_name and project_goal:
        new_project = st.session_state.planner.create_project(project_name, project_goal, datetime.combine(project_start_date, datetime.min.time()))
        st.session_state.current_project_id = new_project.project_id
        st.session_state.planner.initialize_plan(new_project.project_id)
        st.rerun()

# --- Project Selection --- 
st.sidebar.header("Select Project")
project_ids = list(st.session_state.planner.projects.keys())
if project_ids:
    selected_project_id = st.sidebar.selectbox("Choose a project", project_ids, index=project_ids.index(st.session_state.current_project_id) if "current_project_id" in st.session_state and st.session_state.current_project_id in project_ids else 0)
    st.session_state.current_project_id = selected_project_id
else:
    st.sidebar.info("No projects created yet.")
    st.stop()

current_project = st.session_state.planner.projects[st.session_state.current_project_id]
st.header(f"Project: {current_project.name} (ID: {current_project.project_id})")
st.metric("Project Status", current_project.status.value)
st.write(f"**Goal:** {current_project.goal}")
st.write(f"**Start Date:** {current_project.start_date.strftime('%Y-%m-%d')}")
if current_project.end_date:
    st.write(f"**End Date:** {current_project.end_date.strftime('%Y-%m-%d')}")

# --- Project Overview --- 
st.subheader("Project Tasks")
if current_project.tasks:
    tasks_data = [
        {
            "ID": task.task_id,
            "Name": task.name,
            "Description": task.description,
            "Status": task.status.value,
            "Assigned To": task.assigned_to if task.assigned_to else "Unassigned",
            "Estimated Duration (Days)": task.estimated_duration_days,
            "Start Date": task.start_date.strftime('%Y-%m-%d') if task.start_date else 'N/A',
            "End Date": task.end_date.strftime('%Y-%m-%d') if task.end_date else 'N/A',
            "Dependencies": ", ".join(task.dependencies) if task.dependencies else "None",
        }
        for task in current_project.tasks.values()
    ]
    st.dataframe(tasks_data, use_container_width=True)
else:
    st.info("No tasks defined for this project yet. Initialize a plan for a new project.")

# --- Dependencies Visualization (Simplified) ---
st.subheader("Task Dependencies (Graph)")
if current_project.task_graph.nodes():
    # For simplicity, just list edges
    edges = [(u, v) for u, v in current_project.task_graph.edges()]
    if edges:
        st.write("Dependencies (Predecessor -> Successor):")
        for u, v in edges:
            st.write(f"- {current_project.tasks[u].name} (ID: {u}) -> {current_project.tasks[v].name} (ID: {v})")
    else:
        st.info("No explicit dependencies defined.")
else:
    st.info("No tasks to visualize dependencies.")

# --- Constraints --- 
st.subheader("Project Constraints")
if current_project.constraints:
    for const in current_project.constraints:
        st.write(f"- **{const.name}** ({const.type}): {const.value}")
else:
    st.info("No constraints defined.")

# --- Feedback and Adaptation --- 
st.subheader("Simulate Feedback & Adapt Plan")

with st.expander("Update Task Status (Simulated Feedback)"):
    task_ids = list(current_project.tasks.keys())
    if task_ids:
        col1, col2 = st.columns(2)
        with col1:
            task_to_update = st.selectbox("Select Task to Update", task_ids)
        with col2:
            new_status_feedback = st.selectbox("New Status", [s.value for s in TaskStatus])
        
        if st.button("Apply Task Update & Re-plan"):
            feedback_payload = {"task_updates": [{"task_id": task_to_update, "status": new_status_feedback}]}
            st.session_state.planner.process_feedback_and_adapt(current_project.project_id, feedback_payload)
            st.rerun()
    else:
        st.info("No tasks available to update.")

if st.button("Generate LLM Recommendations"): # Button outside expander for direct action
    recommendations = st.session_state.llm_simulator.generate_recommendation(current_project)
    st.success("**LLM Recommendations:**")
    st.write(recommendations)


# --- How to Run --- 
st.sidebar.markdown("""
--- 
### How to Run This App
1.  Save the code as `adaptive_project_manager.py`.
2.  Install necessary libraries: 
    `pip install streamlit networkx`
3.  Run from your terminal: 
    `streamlit run adaptive_project_manager.py`
""")

