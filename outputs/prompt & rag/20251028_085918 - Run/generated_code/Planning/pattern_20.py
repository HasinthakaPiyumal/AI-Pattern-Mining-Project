
import datetime
from typing import List, Dict, Optional, Literal

# --- Data Models (Pydantic-like structures for clarity) ---

class Task:
    """Represents a single task within a project."""
    def __init__(
        self, 
        name: str,
        description: str = "",
        duration_days: int = 1,
        dependencies: Optional[List[str]] = None,
        assigned_to: Optional[str] = None,
        status: Literal["not_started", "in_progress", "completed", "blocked"] = "not_started",
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ):
        self.name = name
        self.description = description
        self.duration_days = duration_days
        self.dependencies = dependencies if dependencies is not None else []
        self.assigned_to = assigned_to
        self.status = status
        self.start_date = start_date
        self.end_date = end_date

    def __repr__(self):
        return f"Task(name='{self.name}', status='{self.status}', dependencies={self.dependencies}, start={self.start_date}, end={self.end_date})"


class Project:
    """Represents a project with its tasks, resources, and deadline."""
    def __init__(
        self, 
        name: str,
        description: str,
        deadline: datetime.date,
        tasks: Optional[Dict[str, Task]] = None,
        resources: Optional[Dict[str, int]] = None,
    ):
        self.name = name
        self.description = description
        self.deadline = deadline
        self.tasks = tasks if tasks is not None else {}
        self.resources = resources if resources is not None else {}


# --- LLM Simulation (for task decomposition and initial planning ideas) ---

class LLM_Simulator:
    """Simulates an LLM for project task decomposition and initial planning suggestions.
    In a real application, this would be an actual LLM API call (e.g., using LangChain).
    """
    @staticmethod
    def decompose_project_idea(project_description: str) -> List[Task]:
        """Generates a list of initial tasks based on a project description.
        This is a simplified simulation.
        """
        print(f"[LLM Sim] Decomposing: '{project_description[:50]}...'\n")
        # Simulate a typical breakdown for a software project
        if "website" in project_description.lower():
            return [
                Task("Design UI/UX", description="Create wireframes and mockups", duration_days=5),
                Task("Develop Frontend", description="Implement user interface", duration_days=10, dependencies=["Design UI/UX"]),
                Task("Develop Backend API", description="Build server-side logic and database", duration_days=15),
                Task("Database Setup", description="Configure and populate database", duration_days=3, dependencies=["Develop Backend API"]),
                Task("Integrate Frontend & Backend", description="Connect UI to API", duration_days=7, dependencies=["Develop Frontend", "Develop Backend API"]),
                Task("Testing and QA", description="Perform various tests", duration_days=5, dependencies=["Integrate Frontend & Backend"]),
                Task("Deployment", description="Release to production environment", duration_days=2, dependencies=["Testing and QA"]),
            ]
        else:
            return [
                Task("Initial Research", duration_days=3),
                Task("Phase 1 Development", duration_days=7, dependencies=["Initial Research"]),
                Task("Testing Phase 1", duration_days=3, dependencies=["Phase 1 Development"]),
                Task("Phase 2 Development", duration_days=10, dependencies=["Testing Phase 1"]),
                Task("Final Review", duration_days=2, dependencies=["Phase 2 Development"]),
            ]


# --- Project Planner (Core AI Agent) ---

class ProjectPlanner:
    """An AI-powered assistant for structured project planning and dynamic adjustment.
    Implements the 'Structured AI Problem Solving' pattern.
    """
    def __init__(self, project: Project):
        self.project = project
        self.current_plan: List[Task] = []

    def _check_dependencies_met(self, task: Task) -> bool:
        """Checks if all dependencies for a given task are in 'completed' status."""
        for dep_name in task.dependencies:
            if dep_name not in self.project.tasks or self.project.tasks[dep_name].status != "completed":
                return False
        return True

    def decompose_and_plan_initial(self) -> None:
        """Uses LLM simulation to decompose the project and generate an initial plan."""
        print("\n--- Initial Project Decomposition and Planning ---")
        # Task decomposition
        initial_tasks = LLM_Simulator.decompose_project_idea(self.project.description)
        for task in initial_tasks:
            self.project.tasks[task.name] = task
        print("Project decomposed into initial tasks.")
        self._generate_feasible_plan()

    def _generate_feasible_plan(self) -> None:
        """Generates a feasible plan by ordering tasks based on dependencies and setting start/end dates.
        This simulates multi-step reasoning and constraint integration.
        """
        print("Generating feasible plan...")
        # Topological sort-like approach to plan tasks
        planned_tasks: List[Task] = []
        ready_tasks_names = [name for name, task in self.project.tasks.items() if not task.dependencies and task.status == "not_started"]
        in_progress_tasks_names = [name for name, task in self.project.tasks.items() if task.status == "in_progress"]
        
        # Start with tasks that have no dependencies or are already in progress
        current_date = datetime.date.today()
        
        # Process tasks that are already in progress first (if any)
        for task_name in in_progress_tasks_names:
            task = self.project.tasks[task_name]
            if not task.start_date: # If an in-progress task somehow has no start_date, assign today
                task.start_date = current_date
                task.end_date = task.start_date + datetime.timedelta(days=task.duration_days)
            if task not in planned_tasks:
                planned_tasks.append(task)

        # Prioritize 'ready' tasks and then iteratively find next ready tasks
        while len(planned_tasks) < len(self.project.tasks):
            made_progress_in_iteration = False
            
            # Find tasks that are 'not_started' and whose dependencies are met
            available_tasks_to_start = [
                task for task in self.project.tasks.values()
                if task.status == "not_started" and self._check_dependencies_met(task) and task not in planned_tasks
            ]
            
            if not available_tasks_to_start and len(planned_tasks) < len(self.project.tasks): # Check for blocked tasks
                 blocked_count = len(self.project.tasks) - len(planned_tasks)
                 if blocked_count > 0: # Only report if actually blocked
                    # Identify exactly which tasks are blocked
                    blocked_tasks = [task for task in self.project.tasks.values() if task not in planned_tasks and task.status != "completed"]
                    for blocked_task in blocked_tasks:
                        if not self._check_dependencies_met(blocked_task):
                            blocked_task.status = "blocked"
                            print(f"Warning: Task '{blocked_task.name}' is blocked due to uncompleted dependencies.")
                 break # No more tasks can be planned currently

            # Sort tasks (e.g., by duration, or some heuristic, for simplicity, just take them as is)
            for task in available_tasks_to_start:
                task.status = "not_started" # Ensure status is correct before planning
                if not task.start_date:
                    # Find the latest end date of its dependencies, or use current_date if no dependencies
                    latest_dep_end = current_date
                    if task.dependencies:
                        for dep_name in task.dependencies:
                            dep_task = self.project.tasks[dep_name]
                            if dep_task.end_date and dep_task.end_date > latest_dep_end:
                                latest_dep_end = dep_task.end_date
                    task.start_date = latest_dep_end
                    task.end_date = task.start_date + datetime.timedelta(days=task.duration_days)
                
                planned_tasks.append(task)
                made_progress_in_iteration = True
                # Update current_date to reflect progress of planned tasks if necessary
                if task.end_date > current_date: # Only advance current_date if task finishes later
                    current_date = task.end_date

            if not made_progress_in_iteration and len(planned_tasks) < len(self.project.tasks):
                # This indicates a cycle in dependencies or an unresolvable block
                print("Error: Could not plan all tasks. Possible circular dependencies or unresolvable blocks.")
                break

        # Sort by start date for display
        self.current_plan = sorted(planned_tasks, key=lambda t: t.start_date if t.start_date else datetime.date.min)

    def update_task_status(self, task_name: str, new_status: Literal["not_started", "in_progress", "completed", "blocked"]) -> None:
        """Updates the status of a specific task and triggers plan re-evaluation."""
        if task_name in self.project.tasks:
            print(f"\n--- Updating task '{task_name}' to status '{new_status}' ---")
            self.project.tasks[task_name].status = new_status
            if new_status == "completed":
                # For completed tasks, ensure end_date is set if not already
                if not self.project.tasks[task_name].end_date:
                    self.project.tasks[task_name].end_date = datetime.date.today()
                print(f"Task '{task_name}' marked as completed.")
            elif new_status == "in_progress":
                if not self.project.tasks[task_name].start_date:
                    self.project.tasks[task_name].start_date = datetime.date.today()

            self.re_evaluate_plan()
        else:
            print(f"Error: Task '{task_name}' not found.")

    def re_evaluate_plan(self) -> None:
        """Dynamically adjusts the project plan based on updated task statuses.
        This demonstrates introspective/extrospective adjustment and multi-step reasoning.
        """
        print("\n--- Re-evaluating plan based on updates ---")
        # Reset 'not_started' tasks' start/end dates if their dependencies might have changed
        for task in self.project.tasks.values():
            if task.status == "not_started" or task.status == "blocked":
                task.start_date = None
                task.end_date = None

        self._generate_feasible_plan() # Re-generate the plan from scratch
        print("Plan re-evaluated successfully.")

    def display_plan(self) -> None:
        """Prints the current project plan in a readable format."""
        print("\n--- Current Project Plan ---")
        if not self.current_plan:
            print("No plan generated yet or all tasks completed.")
            return

        for task in self.current_plan:
            deps = f"(Depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
            start = task.start_date.strftime("%Y-%m-%d") if task.start_date else "N/A"
            end = task.end_date.strftime("%Y-%m-%d") if task.end_date else "N/A"
            print(f"- {task.name}: Status: {task.status:<12} Duration: {task.duration_days} days Start: {start} End: {end} {deps}")
        print(f"Project Deadline: {self.project.deadline.strftime("%Y-%m-%d")}")
        latest_project_end = max([t.end_date for t in self.current_plan if t.end_date] or [datetime.date.today()])
        if latest_project_end > self.project.deadline:
            print(f"WARNING: Project projected to finish AFTER deadline by {(latest_project_end - self.project.deadline).days} days!")
        else:
            print(f"Project projected to finish { (self.project.deadline - latest_project_end).days} days before deadline.")


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define a complex project
    project_deadline = datetime.date.today() + datetime.timedelta(days=60)
    my_project = Project(
        name="New E-commerce Website Launch",
        description="Develop and launch a new responsive e-commerce website with payment gateway integration and user accounts.",
        deadline=project_deadline,
        resources={"frontend_dev": 2, "backend_dev": 1, "designer": 1, "qa_engineer": 1}
    )

    # 2. Initialize the Project Planner with the project
    planner = ProjectPlanner(my_project)

    # 3. Decompose the project and generate an initial plan (simulating LLM)
    planner.decompose_and_plan_initial()
    planner.display_plan()

    # 4. Simulate real-time progress and dynamic adjustment
    print("\n--- Simulating Project Progress ---")

    # Task 1 completes faster than expected
    planner.update_task_status("Design UI/UX", "completed")
    planner.display_plan()

    # Task 2 starts (or is reported as in progress)
    planner.update_task_status("Develop Frontend", "in_progress")
    planner.display_plan()

    # Simulate a delay: one task gets blocked
    print("\n--- Simulating a Blockage ---")
    # Manually mark a dependency as incomplete to simulate 'blocked'
    # For this simulation, let's say 'Database Setup' is blocked due to external reasons not reflected in direct dependencies
    # We'll just mark it 'blocked' and re-evaluate
    # In a real system, the AI might identify the root cause.
    planner.project.tasks["Develop Backend API"].status = "in_progress" # Assume it started
    planner.update_task_status("Database Setup", "blocked") # Database setup gets blocked
    planner.display_plan()

    # Resolve blockage: Database Setup is now in progress and then completed
    print("\n--- Resolving Blockage ---")
    planner.update_task_status("Database Setup", "in_progress")
    planner.update_task_status("Database Setup", "completed")
    planner.display_plan()

    # All tasks are completed eventually
    planner.update_task_status("Develop Frontend", "completed")
    planner.update_task_status("Develop Backend API", "completed")
    planner.update_task_status("Integrate Frontend & Backend", "completed")
    planner.update_task_status("Testing and QA", "completed")
    planner.update_task_status("Deployment", "completed")
    planner.display_plan()
