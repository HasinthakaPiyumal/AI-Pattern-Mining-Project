
import networkx as nx
import time
from typing import List, Dict, Optional

class Task:
    """Represents a single task in the project plan."""
    def __init__(
        self,
        name: str,
        description: str,
        estimated_duration: int,  # in simulated minutes
        dependencies: Optional[List[str]] = None,
        status: str = "pending",
        assigned_resources: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.estimated_duration = estimated_duration
        self.dependencies = dependencies if dependencies is not None else []
        self.status = status  # pending, in_progress, completed, failed, blocked
        self.assigned_resources = assigned_resources if assigned_resources is not None else []

    def __repr__(self):
        return f"Task(name='{self.name}', status='{self.status}', duration={self.estimated_duration}min)"


class MockLLMClient:
    """A mock LLM client to simulate task decomposition and plan adjustments."""
    def decompose_goal_into_tasks(self, goal: str) -> List[Task]:
        print(f"[LLM] Decomposing goal: '{goal}' into initial tasks...")
        # Simulate LLM generating a structured plan
        if "marketing campaign" in goal.lower():
            return [
                Task("Market Research", "Analyze target audience and market trends.", 120),
                Task("Content Creation", "Develop campaign creatives and copy.", 180, dependencies=["Market Research"]),
                Task("Channel Selection", "Choose appropriate marketing channels.", 60, dependencies=["Market Research"]),
                Task("Campaign Launch", "Execute the marketing campaign across selected channels.", 30, dependencies=["Content Creation", "Channel Selection"]),
                Task("Performance Monitoring", "Track campaign performance and gather data.", 90, dependencies=["Campaign Launch"]),
                Task("Reporting & Analysis", "Generate reports and analyze campaign effectiveness.", 120, dependencies=["Performance Monitoring"]),
            ]
        else:
            return [
                Task("Initial Assessment", "Understand the problem scope.", 60),
                Task("Planning Phase", "Create a detailed plan of action.", 90, dependencies=["Initial Assessment"]),
                Task("Execution Phase", "Carry out the planned actions.", 240, dependencies=["Planning Phase"]),
                Task("Review & Refine", "Evaluate results and make adjustments.", 120, dependencies=["Execution Phase"]),
            ]

    def suggest_plan_adjustments(self, current_plan: nx.DiGraph, issue_description: str) -> str:
        print(f"[LLM] Analyzing issue: '{issue_description}' and suggesting adjustments...")
        # Simulate LLM providing advice based on the issue
        if "Market Research" in issue_description and "failed" in issue_description:
            return "Suggestion: Re-run market research with a broader scope or consider an external consultant. This may delay dependent tasks by 2-3 days."
        elif "Campaign Launch" in issue_description and "delayed" in issue_description:
            return "Suggestion: Explore options for parallelizing Content Creation and Channel Selection, or prioritize key channels. Adjust dependent tasks' start times."
        return "Suggestion: Re-evaluate task dependencies, allocate more resources, or consider breaking down the failing task further."


class MockToolService:
    """A mock tool service to simulate external resource and calendar interactions."""
    def get_available_resources(self, skill_set: str) -> List[str]:
        print(f"[Tools] Checking for available resources with skill: {skill_set}")
        if skill_set == "marketing":
            return ["Alice (Marketing Lead)", "Bob (Content Creator)"]
        elif skill_set == "data_analysis":
            return ["Charlie (Analyst)"]
        return []

    def check_calendar_availability(self, resource: str, duration: int) -> bool:
        print(f"[Tools] Checking calendar for {resource} for {duration} minutes.")
        # Simulate some availability logic
        return True  # Always available for this mock


class ProjectManager:
    """Orchestrates project planning, execution, and adaptation using LLM and tools."""

    def __init__(self, llm_client: MockLLMClient, tool_service: MockToolService):
        self.llm_client = llm_client
        self.tool_service = tool_service
        self.project_goal = ""
        self.plan_graph = nx.DiGraph()  # Stores tasks and their dependencies
        self.tasks: Dict[str, Task] = {}
        self.current_time_simulated = 0 # In simulated minutes

    def create_project_plan(self, high_level_goal: str):
        self.project_goal = high_level_goal
        print(f"\n--- Creating initial plan for: '{self.project_goal}' ---")
        initial_tasks = self.llm_client.decompose_goal_into_tasks(self.project_goal)

        for task_data in initial_tasks:
            self._add_task_to_graph(task_data)

        print("Initial plan created with the following tasks and dependencies:")
        self._print_plan_summary()

    def _add_task_to_graph(self, task: Task):
        if task.name not in self.tasks:
            self.tasks[task.name] = task
            self.plan_graph.add_node(task.name, task_obj=task)

        for dep_name in task.dependencies:
            if dep_name not in self.tasks:
                # Handle cases where a dependency might not have been added yet
                # For simplicity, we assume dependencies are defined first or in the same batch
                # In a real system, this would require more robust handling or a multi-pass approach
                print(f"Warning: Dependency '{dep_name}' for task '{task.name}' not found. Adding as placeholder.")
                self.tasks[dep_name] = Task(dep_name, f"Placeholder for {dep_name}", 0, status="pending")
                self.plan_graph.add_node(dep_name, task_obj=self.tasks[dep_name])
            self.plan_graph.add_edge(dep_name, task.name)

    def _print_plan_summary(self):
        for task_name in nx.topological_sort(self.plan_graph):
            task = self.tasks[task_name]
            dependencies_str = ', '.join(task.dependencies) if task.dependencies else 'None'
            print(f"  - {task.name} (Status: {task.status}, Duration: {task.estimated_duration}min, Depends on: {dependencies_str})")

    def get_executable_tasks(self) -> List[Task]:
        executable_tasks = []
        for task_name in self.plan_graph.nodes:
            task = self.tasks[task_name]
            if task.status == "pending":
                # Check if all dependencies are completed
                predecessors = list(self.plan_graph.predecessors(task_name))
                all_dependencies_met = all(self.tasks[dep_name].status == "completed" for dep_name in predecessors)
                if all_dependencies_met:
                    executable_tasks.append(task)
        return executable_tasks

    def update_task_status(self, task_name: str, new_status: str):
        if task_name in self.tasks:
            self.tasks[task_name].status = new_status
            print(f"[Project] Task '{task_name}' status updated to '{new_status}'.")
        else:
            print(f"Error: Task '{task_name}' not found.")

    def check_for_issues_and_adapt(self):
        """Simulates detecting issues and triggering LLM-based re-planning."""
        print("\n[Project] Checking for project issues...")
        # Example of a simple issue detection: a task failed
        for task_name, task in self.tasks.items():
            if task.status == "failed":
                issue_description = f"Task '{task_name}' failed during execution."
                print(f"  Issue detected: {issue_description}")
                suggestion = self.llm_client.suggest_plan_adjustments(self.plan_graph, issue_description)
                print(f"  LLM Suggestion: {suggestion}")
                # In a real system, this would involve modifying the plan_graph, 
                # adding new tasks, re-assigning dependencies, or updating estimates.
                # For this demo, we just print the suggestion.
                return True # An issue was found and addressed conceptually
            if task.status == "in_progress" and task.estimated_duration < (self.current_time_simulated - self._get_task_start_time(task.name)):
                 issue_description = f"Task '{task_name}' is still in progress past its estimated duration {task.estimated_duration}min."
                 print(f"  Issue detected: {issue_description}")
                 suggestion = self.llm_client.suggest_plan_adjustments(self.plan_graph, issue_description)
                 print(f"  LLM Suggestion: {suggestion}")
                 return True
        print("  No major issues detected at this time.")
        return False

    def _get_task_start_time(self, task_name: str) -> int:
        """Helper to conceptually track task start times for duration checks."""
        # This is a simplification. In a real system, task execution would be more granular.
        task_obj = self.tasks[task_name]
        if hasattr(task_obj, '_start_time_simulated'):
            return task_obj._start_time_simulated
        return 0 # If not started, assume start time 0 for this conceptual check


    def simulate_project_execution(self):
        print(f"\n--- Simulating project execution for: '{self.project_goal}' ---")
        self.current_time_simulated = 0

        while True:
            executable_tasks = self.get_executable_tasks()
            if not executable_tasks and all(t.status in ["completed", "failed"] for t in self.tasks.values()):
                print("\nAll tasks are either completed or failed. Project simulation finished.")
                break
            if not executable_tasks and any(t.status == "pending" for t in self.tasks.values()):
                print("\nNo executable tasks, but some are still pending (likely blocked). Project might be stuck.")
                self.check_for_issues_and_adapt() # Try to unblock
                time.sleep(1) # Simulate waiting
                self.current_time_simulated += 60 # Advance time to allow for conceptual unblocking
                continue

            print(f"\nSimulated Time: {self.current_time_simulated} minutes. Executing tasks:")
            for task in executable_tasks:
                self.update_task_status(task.name, "in_progress")
                # Assign resources (mocked)
                resources = self.tool_service.get_available_resources("general") # Generic request
                if resources: 
                    task.assigned_resources = [resources[0]] # Assign first available
                    print(f"  Assigned {task.assigned_resources[0]} to '{task.name}'.")
                else:
                    print(f"  No resources found for '{task.name}'.")

                # Simulate task starting time
                task._start_time_simulated = self.current_time_simulated
                print(f"  - Starting '{task.name}'. Estimated duration: {task.estimated_duration} minutes.")

            # Advance time by the minimum duration of currently in-progress tasks
            # For simplicity, we'll advance by a fixed quantum or the shortest task duration for this demo
            # In a real system, this would be more complex, tracking actual start/end times.
            simulation_quantum = 60 # Simulate progress in 60-minute blocks
            self.current_time_simulated += simulation_quantum
            print(f"  ... {simulation_quantum} minutes pass ...")
            time.sleep(0.5) # Pause for readability

            # Check for completed or failed tasks after this time quantum
            for task_name in list(self.tasks.keys()): # Iterate over a copy to allow modification
                task = self.tasks[task_name]
                if task.status == "in_progress":
                    time_spent = self.current_time_simulated - task._start_time_simulated
                    if time_spent >= task.estimated_duration:
                        # Simulate potential failure (e.g., 10% chance)
                        if "Market Research" in task.name and self.current_time_simulated > 180 and self.current_time_simulated % 2 == 0: # Artificial failure condition
                             self.update_task_status(task.name, "failed")
                             print(f"  !!! Task '{task.name}' FAILED after {time_spent} minutes (estimated {task.estimated_duration}).")
                        else:
                            self.update_task_status(task.name, "completed")
                            print(f"  - Task '{task.name}' COMPLETED after {time_spent} minutes (estimated {task.estimated_duration}).")
            
            # After some progress, check for issues and potentially adapt
            if self.check_for_issues_and_adapt():
                print("  Plan adapted based on detected issues.")
                # Re-evaluate executable tasks after adaptation


        print("\n--- Final Project Status ---")
        self._print_plan_summary()
        print(f"Total simulated time: {self.current_time_simulated} minutes.")


# --- Demo Usage ---
if __name__ == "__main__":
    llm_mock = MockLLMClient()
    tool_mock = MockToolService()
    project_manager = ProjectManager(llm_mock, tool_mock)

    # Example 1: Launch a marketing campaign
    project_manager.create_project_plan("Launch a new marketing campaign for product X")
    project_manager.simulate_project_execution()

    print("\n" + "="*80 + "\n")

    # Example 2: General complex project with potential for adaptation
    project_manager_2 = ProjectManager(llm_mock, tool_mock)
    project_manager_2.create_project_plan("Develop a new AI feature for existing platform")
    project_manager_2.simulate_project_execution()
