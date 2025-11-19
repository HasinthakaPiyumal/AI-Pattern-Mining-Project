import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.tools import BaseTool, tool
from langchain.memory import ConversationBufferWindowMemory


# Environment Variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pydantic Models
class Resource(BaseModel):
    name: str
    role: str
    availability: str = "full-time"

class Task(BaseModel):
    task_id: str
    name: str
    description: str
    assigned_to: Optional[str] = None
    status: str = "pending"
    due_date: Optional[str] = None
    dependencies: List[str] = []
    estimated_effort_hours: Optional[int] = None

class Project(BaseModel):
    project_id: str
    name: str
    description: str
    tasks: List[Task] = []
    resources: List[Resource] = []
    status: str = "new"
    risks: List[str] = []

class ProjectGoal(BaseModel):
    goal: str = Field(..., description="High-level project goal, e.g., 'Develop an e-commerce website'")
    description: str = Field(..., description="Detailed description of the project goal and desired outcomes")

# Simulated Project Database
class SimulatedProjectDB:
    def __init__(self):
        self.projects: Dict[str, Project] = {}
        self.next_task_id = 1

    def add_project(self, project_id: str, name: str, description: str) -> Project:
        if project_id in self.projects:
            raise ValueError(f"Project with ID {project_id} already exists.")
        new_project = Project(project_id=project_id, name=name, description=description)
        self.projects[project_id] = new_project
        return new_project

    def get_project(self, project_id: str) -> Optional[Project]:
        return self.projects.get(project_id)

    def add_task(self, project_id: str, name: str, description: str,
                 assigned_to: Optional[str] = None, due_date: Optional[str] = None,
                 dependencies: List[str] = [], estimated_effort_hours: Optional[int] = None) -> Task:
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found.")

        task_id = f"TASK-{self.next_task_id:03d}"
        self.next_task_id += 1
        new_task = Task(
            task_id=task_id,
            name=name,
            description=description,
            assigned_to=assigned_to,
            due_date=due_date,
            dependencies=dependencies,
            estimated_effort_hours=estimated_effort_hours
        )
        project.tasks.append(new_task)
        return new_task

    def update_task_status(self, project_id: str, task_id: str, status: str) -> Optional[Task]:
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found.")
        for task in project.tasks:
            if task.task_id == task_id:
                task.status = status
                return task
        return None

    def assign_resource_to_task(self, project_id: str, task_id: str, resource_name: str) -> Optional[Task]:
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found.")
        for task in project.tasks:
            if task.task_id == task_id:
                task.assigned_to = resource_name
                return task
        return None
    
    def add_project_risk(self, project_id: str, risk_description: str) -> Project:
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project with ID {project_id} not found.")
        project.risks.append(risk_description)
        return project

    def get_project_status(self, project_id: str) -> Optional[Project]:
        return self.get_project(project_id)

    def get_all_projects(self) -> List[Project]:
        return list(self.projects.values())

project_db_instance = SimulatedProjectDB()

# Custom LangChain Tools
class ProjectDBTool(BaseTool):
    name: str = "Project_Database_Manager"
    description: str = "Tool for managing project data: creating projects, adding/updating tasks, assigning resources, and retrieving project status."
    db: SimulatedProjectDB

    def _run(self, action: str, project_id: str, **kwargs: Any) -> str:
        try:
            if action == "add_project":
                name = kwargs.get("name")
                description = kwargs.get("description")
                if not name or not description:
                    return "Error: name and description are required for adding a project."
                project = self.db.add_project(project_id=project_id, name=name, description=description)
                return f"Project '{project.name}' (ID: {project.project_id}) created successfully."
            elif action == "add_task":
                name = kwargs.get("name")
                description = kwargs.get("description")
                assigned_to = kwargs.get("assigned_to")
                due_date = kwargs.get("due_date")
                dependencies = kwargs.get("dependencies", [])
                estimated_effort_hours = kwargs.get("estimated_effort_hours")
                if not name or not description:
                    return "Error: name and description are required for adding a task."
                task = self.db.add_task(project_id=project_id, name=name, description=description,
                                       assigned_to=assigned_to, due_date=due_date,
                                       dependencies=dependencies, estimated_effort_hours=estimated_effort_hours)
                return f"Task '{task.name}' (ID: {task.task_id}) added to project {project_id}."
            elif action == "update_task_status":
                task_id = kwargs.get("task_id")
                status = kwargs.get("status")
                if not task_id or not status:
                    return "Error: task_id and status are required for updating task status."
                task = self.db.update_task_status(project_id=project_id, task_id=task_id, status=status)
                if task:
                    return f"Task '{task_id}' status updated to '{status}'."
                return f"Error: Task '{task_id}' not found in project {project_id}."
            elif action == "assign_resource_to_task":
                task_id = kwargs.get("task_id")
                resource_name = kwargs.get("resource_name")
                if not task_id or not resource_name:
                    return "Error: task_id and resource_name are required for assigning resource."
                task = self.db.assign_resource_to_task(project_id=project_id, task_id=task_id, resource_name=resource_name)
                if task:
                    return f"Resource '{resource_name}' assigned to task '{task_id}'."
                return f"Error: Task '{task_id}' not found in project {project_id}."
            elif action == "add_project_risk":
                risk_description = kwargs.get("risk_description")
                if not risk_description:
                    return "Error: risk_description is required for adding a project risk."
                project = self.db.add_project_risk(project_id=project_id, risk_description=risk_description)
                return f"Risk '{risk_description}' added to project {project_id}."
            elif action == "get_project_status":
                project = self.db.get_project_status(project_id=project_id)
                if project:
                    return project.json(indent=2)
                return f"Error: Project with ID {project_id} not found."
            else:
                return f"Error: Unknown action '{action}' for Project_Database_Manager."
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    async def _arun(self, action: str, project_id: str, **kwargs: Any) -> str:
        return self._run(action, project_id, **kwargs)

class KnowledgeBaseTool(BaseTool):
    name: str = "Knowledge_Base_Query"
    description: str = "Tool for querying a knowledge base for project best practices, past solutions, or technical guidance."

    def _run(self, query: str) -> str:
        if "best practices for agile" in query.lower():
            return "Agile best practices include daily stand-ups, short sprints, continuous integration, and frequent stakeholder feedback."
        elif "authentication design patterns" in query.lower():
            return "Common authentication design patterns include OAuth2, JWT, and session-based authentication."
        return f"Knowledge Base: No specific guidance found for '{query}'. Please try a different query."

    async def _arun(self, query: str) -> str:
        return self._run(query)

# LangChain Agent Setup
llm = ChatOpenAI(temperature=0.7, model="gpt-4", openai_api_key=OPENAI_API_KEY)

# Initialize tools
project_db_tool = ProjectDBTool(db=project_db_instance)
knowledge_base_tool = KnowledgeBaseTool()
tools = [project_db_tool, knowledge_base_tool]

# Agent System Message
system_message = SystemMessage(
    content=(
        "You are an AI-powered Project Manager Assistant. Your goal is to help users manage complex software development projects.\n"
        "You can decompose high-level goals into sub-tasks, plan execution, assign resources, identify risks, and adapt plans.\n"
        "You have access to a project database tool to manage tasks and project status.\n"
        "Always try to break down the project goal into at least 3-5 initial tasks if it's a new project.\n"
        "For a new project, first use the 'add_project' action to create the project in the database.\n"
        "When adding tasks, try to include a description, estimated effort (in hours), and potential assignee if clear.\n"
        "Identify potential risks for the project and add them using 'add_project_risk'.\n"
        "Be adaptive and ready to modify plans based on new information or simulated feedback.\n"
        "Provide clear, concise updates and recommendations."
    )
)

# Agent Memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input",
    output_key="output",
    k=5
)

# Agent with OpenAI Functions
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True, # Set to True to see the agent's thought process
    agent_kwargs={"system_message": system_message},
    memory=memory,
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

# FastAPI Application
app = FastAPI(
    title="AI Project Manager Assistant",
    description="An AI-powered assistant for intelligent planning and adaptive execution of software projects."
)

class ProjectPlanResponse(BaseModel):
    agent_response: str
    project_status: Optional[Project]

@app.post("/project/plan", response_model=ProjectPlanResponse)
async def create_project_plan(project_goal: ProjectGoal):
    project_id = "PROJ-001" # For simplicity, using a fixed project ID for initial project.

    # Construct the input for the agent
    agent_input = f"Initial project goal: {project_goal.goal}. Detailed description: {project_goal.description}. Please help me create a detailed project plan, including task decomposition, resource suggestions, and risk identification for this project (Project ID: {project_id})."

    try:
        response = await agent_executor.ainvoke({"input": agent_input})
        agent_output = response["output"]
    except Exception as e:
        return ProjectPlanResponse(
            agent_response=f"An error occurred while processing your request: {e}",
            project_status=None
        )

    current_project_status = project_db_instance.get_project(project_id)

    return ProjectPlanResponse(
        agent_response=agent_output,
        project_status=current_project_status
    )
