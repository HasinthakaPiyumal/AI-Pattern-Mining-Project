import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. Data Models (Pydantic) ---

class TaskStatus(str, Enum):
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"

class Task(BaseModel):
    id: uuid.UUID
    description: str
    status: TaskStatus = TaskStatus.TODO
    dependencies: List[uuid.UUID] = []  # IDs of tasks that must be completed before this one
    estimated_effort_hours: int = 8
    assigned_to: Optional[str] = None
    deadline: Optional[datetime] = None
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None

class ProjectPlan(BaseModel):
    project_id: uuid.UUID
    goal: str
    tasks: Dict[uuid.UUID, Task]
    # Using an adjacency list for easy NetworkX graph reconstruction
    task_graph_adj: Dict[uuid.UUID, List[uuid.UUID]]

class CreateProjectRequest(BaseModel):
    goal: str

class UpdateTaskStatusRequest(BaseModel):
    status: TaskStatus
    notes: Optional[str] = None

class BottleneckSuggestion(BaseModel):
    task_id: uuid.UUID
    description: str
    suggestion: str

# --- 2. In-memory Storage (Placeholder for DB) ---
projects: Dict[uuid.UUID, ProjectPlan] = {}

# --- 3. LLM Core (Simulated) ---
class MockLLMCore:
    """
    Simulates the LLM's capabilities for task decomposition, plan generation,
    adaptation, and bottleneck identification using heuristic-based logic.
    """

    @staticmethod
    def _decompose_task(goal: str) -> List[Task]:
        """
        Simulates LLM decomposing a high-level goal into sub-tasks.
        In a real scenario, this would use an actual LLM call.
        """
        print(f"[LLM Core] Decomposing goal: {goal}")
        if "launch new product" in goal.lower():
            task1 = Task(id=uuid.uuid4(), description="Market Research", estimated_effort_hours=40)
            task2 = Task(id=uuid.uuid4(), description="Product Design", dependencies=[task1.id], estimated_effort_hours=80)
            task3 = Task(id=uuid.uuid4(), description="Frontend Development", dependencies=[task2.id], estimated_effort_hours=120)
            task4 = Task(id=uuid.uuid4(), description="Backend Development", dependencies=[task2.id], estimated_effort_hours=120)
            task5 = Task(id=uuid.uuid4(), description="Testing & QA", dependencies=[task3.id, task4.id], estimated_effort_hours=60)
            task6 = Task(id=uuid.uuid4(), description="Marketing Campaign Setup", dependencies=[task1.id], estimated_effort_hours=30)
            task7 = Task(id=uuid.uuid4(), description="Product Launch", dependencies=[task5.id, task6.id], estimated_effort_hours=10)
            return [task1, task2, task3, task4, task5, task6, task7]
        else:
            # Generic decomposition for other goals
            task_a = Task(id=uuid.uuid4(), description=f"Analyze {goal} requirements", estimated_effort_hours=20)
            task_b = Task(id=uuid.uuid4(), description=f"Plan {goal} execution", dependencies=[task_a.id], estimated_effort_hours=15)
            task_c = Task(id=uuid.uuid4(), description=f"Execute {goal} main steps", dependencies=[task_b.id], estimated_effort_hours=50)
            task_d = Task(id=uuid.uuid4(), description=f"Review and finalize {goal}", dependencies=[task_c.id], estimated_effort_hours=10)
            return [task_a, task_b, task_c, task_d]

    @staticmethod
    def _build_dependency_graph(tasks: Dict[uuid.UUID, Task]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for task_id, task in tasks.items():
            graph.add_node(task_id, description=task.description, status=task.status.value)
            for dep_id in task.dependencies:
                if dep_id in tasks: # Ensure dependency exists
                    graph.add_edge(dep_id, task_id) # Edge from dependency to dependent task
        return graph

    @staticmethod
    def _generate_initial_plan(project_id: uuid.UUID, goal: str, tasks: List[Task]) -> ProjectPlan:
        print(f"[LLM Core] Generating initial plan for project {project_id}")
        task_dict = {task.id: task for task in tasks}
        graph = MockLLMCore._build_dependency_graph(task_dict)
        
        # Convert graph to adjacency list for Pydantic model storage
        task_graph_adj = {node: list(graph.successors(node)) for node in graph.nodes}

        return ProjectPlan(
            project_id=project_id,
            goal=goal,
            tasks=task_dict,
            task_graph_adj=task_graph_adj
        )

    @staticmethod
    def _adapt_plan(project_plan: ProjectPlan, updated_task_id: Optional[uuid.UUID] = None, trigger_full_replan: bool = False) -> ProjectPlan:
        print(f"[LLM Core] Adapting plan for project {project_plan.project_id}")
        graph = MockLLMCore._build_dependency_graph(project_plan.tasks)
        tasks_to_update = {} # Collect changes to apply at the end

        # Heuristic 1: If a task is completed, unlock its dependents
        if updated_task_id and project_plan.tasks[updated_task_id].status == TaskStatus.COMPLETED:
            for successor_id in graph.successors(updated_task_id):
                successor_task = project_plan.tasks[successor_id]
                if successor_task.status == TaskStatus.BLOCKED:
                    # Check if all dependencies for the successor are met
                    all_deps_met = True
                    for dep_id in successor_task.dependencies:
                        if project_plan.tasks[dep_id].status != TaskStatus.COMPLETED:
                            all_deps_met = False
                            break
                    if all_deps_met:
                        # Simulate LLM deciding to unblock and set to TODO
                        tasks_to_update[successor_id] = successor_task.copy(update={
                            "status": TaskStatus.TODO,
                            "notes": f"Unlocked by completion of {updated_task_id}"
                        })
                        print(f"[LLM Core] Task {successor_id} unblocked and set to TODO due to {updated_task_id} completion.")

        # Heuristic 2: If a task becomes BLOCKED, its successors might also be affected
        if updated_task_id and project_plan.tasks[updated_task_id].status == TaskStatus.BLOCKED:
            # A real LLM would suggest alternatives or new paths
            # For simulation, just log and leave successors as is for now
            print(f"[LLM Core] Task {updated_task_id} is BLOCKED. Future tasks might need re-evaluation.")

        # Heuristic 3: Full replan logic (e.g., if new constraints or major changes)
        if trigger_full_replan:
            print("[LLM Core] Performing a full project replan based on new constraints/feedback.")
            # For a real LLM, this would involve re-prompting with the current state and new constraints
            # For simulation, we'll re-evaluate the critical path and suggest initial tasks.
            
            # Simple critical path re-evaluation: tasks with no uncompleted dependencies can be TODO
            for task_id, task in project_plan.tasks.items():
                if task.status == TaskStatus.TODO or task.status == TaskStatus.BLOCKED:
                    all_deps_met = True
                    for dep_id in task.dependencies:
                        if project_plan.tasks[dep_id].status != TaskStatus.COMPLETED:
                            all_deps_met = False
                            break
                    if all_deps_met and task.status == TaskStatus.BLOCKED: # Only unblock if truly unblocked
                         tasks_to_update[task_id] = task.copy(update={
                            "status": TaskStatus.TODO,
                            "notes": f"Unblocked during full replan"
                        })
                    elif not all_deps_met and task.status == TaskStatus.TODO: # If not all deps met, block it
                         tasks_to_update[task_id] = task.copy(update={
                            "status": TaskStatus.BLOCKED,
                            "notes": f"Blocked during full replan due to uncompleted dependencies"
                        })

        # Apply collected updates
        for task_id, updated_task in tasks_to_update.items():
            project_plan.tasks[task_id] = updated_task
        
        return project_plan

    @staticmethod
    def _identify_bottlenecks(project_plan: ProjectPlan) -> List[BottleneckSuggestion]:
        print(f"[LLM Core] Identifying bottlenecks for project {project_plan.project_id}")
        bottlenecks: List[BottleneckSuggestion] = []
        graph = MockLLMCore._build_dependency_graph(project_plan.tasks)
        
        # Heuristic 1: Blocked tasks
        for task_id, task in project_plan.tasks.items():
            if task.status == TaskStatus.BLOCKED:
                bottlenecks.append(BottleneckSuggestion(
                    task_id=task_id,
                    description=f"Task '{task.description}' is blocked.",
                    suggestion="Investigate the reason for blocking and try to resolve dependencies or find an alternative approach."
                ))
        
        # Heuristic 2: Tasks with uncompleted dependencies that are themselves ready to start
        # This finds tasks that are TODO but can't proceed because of a specific dependency
        for task_id, task in project_plan.tasks.items():
            if task.status == TaskStatus.TODO:
                for dep_id in task.dependencies:
                    dep_task = project_plan.tasks.get(dep_id)
                    if dep_task and dep_task.status != TaskStatus.COMPLETED:
                        bottlenecks.append(BottleneckSuggestion(
                            task_id=task_id,
                            description=f"Task '{task.description}' (TODO) is waiting on '{dep_task.description}' which is {dep_task.status.value}.",
                            suggestion=f"Prioritize or expedite '{dep_task.description}' to unblock '{task.description}'."
                        ))
        
        # Heuristic 3: Long-running 'IN_PROGRESS' tasks (simplified time check)
        # A real system would track actual start time and compare to estimated.
        for task_id, task in project_plan.tasks.items():
            if task.status == TaskStatus.IN_PROGRESS and task.actual_start_time:
                # Simulate a task being 'long running' if it's been in progress for 2x its estimated effort
                # This is a crude simulation; actual time tracking would be more robust.
                if datetime.now() > task.actual_start_time + timedelta(hours=task.estimated_effort_hours * 2):
                     bottlenecks.append(BottleneckSuggestion(
                        task_id=task_id,
                        description=f"Task '{task.description}' seems to be taking longer than estimated ({task.estimated_effort_hours} hours).",
                        suggestion="Check on progress, identify roadblocks, or consider re-estimating effort/resources."
                    ))

        return bottlenecks

# --- 4. FastAPI Application ---
app = FastAPI(
    title="Intelligent Project Management Assistant",
    description="An AI-powered assistant for dynamic project planning and adaptation."
)

@app.post("/create_project", response_model=ProjectPlan)
async def create_project(request: CreateProjectRequest):
    project_id = uuid.uuid4()
    tasks = MockLLMCore._decompose_task(request.goal)
    project_plan = MockLLMCore._generate_initial_plan(project_id, request.goal, tasks)
    projects[project_id] = project_plan
    return project_plan

@app.get("/get_project_plan/{project_id}", response_model=ProjectPlan)
async def get_project_plan(project_id: uuid.UUID):
    project_plan = projects.get(project_id)
    if not project_plan:
        raise HTTPException(status_code=404, detail="Project not found")
    return project_plan

@app.post("/update_task_status/{project_id}/{task_id}", response_model=ProjectPlan)
async def update_task_status(project_id: uuid.UUID, task_id: uuid.UUID, request: UpdateTaskStatusRequest):
    project_plan = projects.get(project_id)
    if not project_plan:
        raise HTTPException(status_code=404, detail="Project not found")
    
    task = project_plan.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Update task status and times
    original_status = task.status
    task.status = request.status
    if request.status == TaskStatus.IN_PROGRESS and not task.actual_start_time:
        task.actual_start_time = datetime.now()
    elif request.status == TaskStatus.COMPLETED and not task.actual_end_time:
        task.actual_end_time = datetime.now()

    # Trigger plan adaptation based on status change
    projects[project_id] = MockLLMCore._adapt_plan(project_plan, updated_task_id=task_id)
    
    return projects[project_id]

@app.get("/get_bottlenecks/{project_id}", response_model=List[BottleneckSuggestion])
async def get_bottlenecks(project_id: uuid.UUID):
    project_plan = projects.get(project_id)
    if not project_plan:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return MockLLMCore._identify_bottlenecks(project_plan)

@app.post("/replan/{project_id}", response_model=ProjectPlan)
async def replan_project(project_id: uuid.UUID):
    project_plan = projects.get(project_id)
    if not project_plan:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Trigger a full replan
    projects[project_id] = MockLLMCore._adapt_plan(project_plan, trigger_full_replan=True)
    return projects[project_id]

# Example of how to run this (requires `uvicorn`):
# uvicorn project_management_assistant:app --reload
