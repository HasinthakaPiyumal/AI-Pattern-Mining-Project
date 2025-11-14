import uuid
import json
from collections import defaultdict

class Task:
    """Represents a single task within a project."""
    def __init__(
        self, 
        name: str, 
        description: str, 
        estimated_duration_days: int = 1,
        resources_needed: dict = None,
        assigned_to: str = None,
        task_id: str = None
    ):
        self.id = task_id if task_id else str(uuid.uuid4())
        self.name = name
        self.description = description
        self.status = "pending"
        self.dependencies = []  # List of task IDs that must complete before this one
        self.resources_needed = resources_needed if resources_needed else {}
        self.estimated_duration_days = estimated_duration_days
        self.actual_duration_days = None
        self.assigned_to = assigned_to

    def mark_as_complete(self, actual_duration: int = None):
        """Marks the task as completed and sets the actual duration."""
        self.status = "completed"
        self.actual_duration_days = actual_duration
        print(f"Task '{self.name}' ({self.id}) marked as completed.")

    def update_status(self, new_status: str):
        """Updates the status of the task."""
        valid_statuses = ["pending", "in_progress", "completed", "blocked", "on_hold"]
        if new_status in valid_statuses:
            self.status = new_status
            print(f"Task '{self.name}' ({self.id}) status updated to '{new_status}'.")
        else:
            print(f"Invalid status: '{new_status}'. Status not updated.")

    def add_dependency(self, task_id: str):
        """Adds a dependency on another task's ID."""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
            print(f"Added dependency: Task '{self.name}' now depends on task '{task_id}'.")

    def __repr__(self):
        return f"Task(ID={self.id[:4]}..., Name='{self.name}', Status='{self.status}')"


class Constraint:
    """Represents a project constraint."""
    def __init__(self, constraint_type: str, description: str, details: dict = None):
        self.type = constraint_type  # e.g., 'deadline', 'budget', 'resource_availability', 'logical_order'
        self.description = description
        self.details = details if details else {}

    def __repr__(self):
        return f"Constraint(Type='{self.type}', Desc='{self.description[:30]}...')"


class ProjectPlan:
    """Manages the collection of tasks and constraints for a project."""
    def __init__(self, project_goal: str):
        self.project_goal = project_goal
        self.tasks = {}  # dict: task_id -> Task object
        self.constraints = []  # list of Constraint objects
        self.timeline = []  # conceptual, could be a list of task IDs in planned order

    def add_task(self, task: Task):
        """Adds a task to the project plan."""
        if task.id in self.tasks:
            print(f"Warning: Task with ID {task.id} already exists. Updating task.")
        self.tasks[task.id] = task
        if task.id not in self.timeline: # Simple linear timeline for mock
            self.timeline.append(task.id)
        print(f"Task '{task.name}' added to project plan.")

    def get_task(self, task_id: str) -> Task:
        """Retrieves a task by its ID."""
        return self.tasks.get(task_id)

    def get_dependencies_for_task(self, task_id: str) -> list:
        """Returns the list of task IDs that the given task depends on."""
        task = self.get_task(task_id)
        return task.dependencies if task else []

    def add_constraint(self, constraint: Constraint):
        """Adds a constraint to the project plan."""
        self.constraints.append(constraint)
        print(f"Constraint '{constraint.type}' added: {constraint.description}")

    def calculate_critical_path(self):
        """Conceptual: In a real system, this would calculate the critical path."""
        print("Calculating critical path (conceptual)...")
        # This would involve graph traversal algorithms (e.g., topological sort, longest path)
        # For this mock, we'll just indicate the function is called.
        return []

    def visualize_plan(self):
        """Conceptual: In a real system, this would visualize the project plan (e.g., Gantt chart)."""
        print("Visualizing project plan (conceptual)...")
        # This would use libraries like matplotlib, networkx, or dedicated project management tools integration.

    def to_dict(self):
        """Converts the project plan to a dictionary for LLM context."""
        return {
            "project_goal": self.project_goal,
            "tasks": {task_id: {
                "name": task.name,
                "description": task.description,
                "status": task.status,
                "dependencies": task.dependencies,
                "resources_needed": task.resources_needed,
                "estimated_duration_days": task.estimated_duration_days,
                "actual_duration_days": task.actual_duration_days,
                "assigned_to": task.assigned_to
            } for task_id, task in self.tasks.items()},
            "constraints": [{
                "type": c.type,
                "description": c.description,
                "details": c.details
            } for c in self.constraints],
            "timeline": self.timeline
        }


class MockLLM:
    """A mock Large Language Model interface for simulating responses."""
    def generate_decomposition(self, goal_description: str) -> list:
        """Simulates LLM's ability to decompose a goal into sub-tasks."""
        print(f"[MockLLM] Generating decomposition for: '{goal_description}'...")
        # A simple, rule-based decomposition for demonstration
        if "launch new product" in goal_description.lower():
            return [
                {"name": "Market Research", "description": "Conduct market analysis and identify target audience.", "estimated_duration_days": 10, "dependencies": []},
                {"name": "Product Design", "description": "Design product features and user experience.", "estimated_duration_days": 15, "dependencies": []},
                {"name": "Development", "description": "Implement the product features.", "estimated_duration_days": 30, "dependencies": []},
                {"name": "Testing", "description": "Perform quality assurance and bug fixing.", "estimated_duration_days": 10, "dependencies": []},
                {"name": "Marketing Campaign", "description": "Plan and execute product launch marketing.", "estimated_duration_days": 20, "dependencies": []}
            ]
        else:
            return [
                {"name": "Task A", "description": "Generic task A.", "estimated_duration_days": 5, "dependencies": []},
                {"name": "Task B", "description": "Generic task B, depends on A.", "estimated_duration_days": 7, "dependencies": ["Task A"]}
            ]

    def suggest_adaptation(self, current_plan_state: dict, issues: list) -> dict:
        """Simulates LLM's ability to suggest plan adaptations based on issues."""
        print(f"[MockLLM] Suggesting adaptation for issues: {issues}...")
        suggestions = {"plan_modifications": [], "new_tasks": [], "resource_reallocations": []}

        if "Development blocked" in str(issues):
            suggestions["plan_modifications"].append("Prioritize 'Product Design' completion. Reallocate resources to design.")
            suggestions["resource_reallocations"].append({"role": "designer", "quantity": 1, "action": "add"})
            suggestions["new_tasks"].append({"name": "Accelerate Design Review", "description": "Expedite review and approval of design specs.", "estimated_duration_days": 2, "dependencies": []})
        elif "delay in Marketing" in str(issues):
             suggestions["plan_modifications"].append("Consider parallelizing Marketing pre-work with Development.")
        else:
            suggestions["plan_modifications"].append("Review critical path and identify bottlenecks.")

        return suggestions

    def optimize_resources(self, tasks: dict, available_resources: dict) -> dict:
        """Simulates LLM's ability to optimize resource assignments."""
        print("[MockLLM] Optimizing resources...")
        optimized_assignments = {}
        # Simple mock: assign one available developer to first unassigned dev task
        available_devs = available_resources.get("developer", 0)
        for task_id, task_data in tasks.items():
            if task_data.get("assigned_to") is None and "developer" in task_data.get("resources_needed", {}) and available_devs > 0:
                optimized_assignments[task_id] = "Developer_A"
                available_devs -= 1
                print(f"  [MockLLM] Assigned Developer_A to task {task_id}")
                break # Assign only one for simplicity
        return optimized_assignments


class ProjectManagerAI:
    """Orchestrates project planning, monitoring, and adaptation using an LLM."""
    def __init__(self, llm_interface: MockLLM):
        self.project_goal = None
        self.project_plan = None
        self.llm_interface = llm_interface
        self.available_resources = {"developer": 2, "designer": 1, "qa_engineer": 1}

    def initialize_project(self, goal_description: str):
        """Initializes the project with a high-level goal and gets initial task decomposition."""
        self.project_goal = goal_description
        self.project_plan = ProjectPlan(project_goal=goal_description)
        print(f"\n--- Initializing Project: '{self.project_goal}' ---")
        self.decompose_and_plan()

    def decompose_and_plan(self):
        """Uses LLM to decompose the goal and builds the initial project plan."""
        if not self.project_goal:
            print("Error: Project goal not set. Initialize project first.")
            return

        # Get decomposition from LLM
        decomposed_tasks_data = self.llm_interface.generate_decomposition(self.project_goal)
        task_objects = {}
        
        # First pass: create all task objects to resolve IDs for dependencies
        for task_data in decomposed_tasks_data:
            task = Task(
                name=task_data["name"],
                description=task_data["description"],
                estimated_duration_days=task_data.get("estimated_duration_days", 1),
                resources_needed=task_data.get("resources_needed", {})
            )
            self.project_plan.add_task(task)
            task_objects[task.name] = task # Store by name temporarily for dependency resolution

        # Second pass: add dependencies using the created task IDs
        for task_data in decomposed_tasks_data:
            task_name = task_data["name"]
            current_task_obj = task_objects.get(task_name)
            if current_task_obj:
                for dep_name in task_data.get("dependencies", []):
                    dep_task_obj = task_objects.get(dep_name)
                    if dep_task_obj:
                        current_task_obj.add_dependency(dep_task_obj.id)
                    else:
                        print(f"Warning: Dependency '{dep_name}' for task '{task_name}' not found.")
        
        print(f"Project plan initialized with {len(self.project_plan.tasks)} tasks.")

    def apply_constraints(self, constraints_list: list[Constraint]):
        """Applies a list of constraints to the project plan."""
        for constraint in constraints_list:
            self.project_plan.add_constraint(constraint)

    def monitor_progress(self, updated_task_statuses: dict = None) -> list:
        """Monitors project progress, updates task statuses, and identifies issues."""
        print("\n--- Monitoring Project Progress ---")
        issues = []

        if updated_task_statuses:
            for task_id, new_status in updated_task_statuses.items():
                task = self.project_plan.get_task(task_id)
                if task:
                    task.update_status(new_status)

        # Check for blocked tasks or delays
        for task_id, task in self.project_plan.tasks.items():
            if task.status == "pending":
                # Check if all dependencies are met
                all_dependencies_met = True
                for dep_id in task.dependencies:
                    dep_task = self.project_plan.get_task(dep_id)
                    if not dep_task or dep_task.status != "completed":
                        all_dependencies_met = False
                        break
                if not all_dependencies_met and not task.dependencies:
                    # Task is pending with no dependencies, should probably be in_progress
                    print(f"Warning: Task '{task.name}' is pending but has no unmet dependencies.")
                    # issues.append(f"Task '{task.name}' is pending but should be in progress.")
                elif not all_dependencies_met and task.dependencies:
                    task.update_status("blocked")
                    issues.append(f"Task '{task.name}' (ID: {task.id[:4]}...) blocked due to unmet dependencies.")
            elif task.status == "in_progress":
                # Conceptual: In a real system, check against estimated duration.
                pass # For mock, assume progress is fine unless explicit update

        completed_tasks = [t.name for t in self.project_plan.tasks.values() if t.status == "completed"]
        in_progress_tasks = [t.name for t in self.project_plan.tasks.values() if t.status == "in_progress"]
        blocked_tasks = [t.name for t in self.project_plan.tasks.values() if t.status == "blocked"]

        print(f"  Completed: {completed_tasks if completed_tasks else 'None'}")
        print(f"  In Progress: {in_progress_tasks if in_progress_tasks else 'None'}")
        print(f"  Blocked: {blocked_tasks if blocked_tasks else 'None'}")

        return issues

    def adapt_plan_if_needed(self, issues: list):
        """Uses LLM to get adaptation suggestions and applies them to the plan."""
        if not issues:
            print("No issues detected, no plan adaptation needed.")
            return

        print("\n--- Adapting Plan Based on Issues ---")
        current_plan_state = self.project_plan.to_dict()
        adaptation_suggestions = self.llm_interface.suggest_adaptation(current_plan_state, issues)

        print("  Applying suggested adaptations:")
        for modification in adaptation_suggestions.get("plan_modifications", []):
            print(f"    - {modification}")
            # In a real system, parse and apply specific changes (e.g., adjust task priority, re-sequence)

        for new_task_data in adaptation_suggestions.get("new_tasks", []):
            new_task = Task(
                name=new_task_data["name"],
                description=new_task_data["description"],
                estimated_duration_days=new_task_data.get("estimated_duration_days", 1),
                resources_needed=new_task_data.get("resources_needed", {})
            )
            self.project_plan.add_task(new_task)
            for dep_name in new_task_data.get("dependencies", []):
                # Find the ID of the dependent task by name (simple mock assumption)
                for task_id, task_obj in self.project_plan.tasks.items():
                    if task_obj.name == dep_name:
                        new_task.add_dependency(task_id)
                        break

        for resource_realloc in adaptation_suggestions.get("resource_reallocations", []):
            print(f"    - Resource reallocation suggested: {resource_realloc}")
            # In a real system, update self.available_resources and task assignments

        # After adaptation, re-optimize resources
        self.optimize_resource_allocation()


    def optimize_resource_allocation(self):
        """Uses LLM to optimize resource allocation across tasks."""
        print("\n--- Optimizing Resource Allocation ---")
        current_tasks_data = self.project_plan.to_dict()["tasks"]
        optimized_assignments = self.llm_interface.optimize_resources(current_tasks_data, self.available_resources)

        for task_id, assigned_resource in optimized_assignments.items():
            task = self.project_plan.get_task(task_id)
            if task and task.assigned_to is None:
                task.assigned_to = assigned_resource
                print(f"  Assigned {assigned_resource} to task '{task.name}' ({task.id[:4]}...).")

    def get_actionable_insights(self) -> str:
        """Provides human-readable summaries or suggested next steps."""
        print("\n--- Generating Actionable Insights ---")
        insights = []
        pending_tasks = [t.name for t in self.project_plan.tasks.values() if t.status == "pending"]
        in_progress_tasks = [t.name for t in self.project_plan.tasks.values() if t.status == "in_progress"]
        blocked_tasks = [t.name for t in self.project_plan.tasks.values() if t.status == "blocked"]
        completed_tasks = [t.name for t in self.project_plan.tasks.values() if t.status == "completed"]

        if blocked_tasks:
            insights.append(f"Immediate Action Required: Tasks blocked: {', '.join(blocked_tasks)}. Investigate dependencies.")
        if not in_progress_tasks and pending_tasks:
            insights.append(f"Suggestion: Start working on pending tasks: {', '.join(pending_tasks)}.")

        insights.append(f"Project Progress: {len(completed_tasks)}/{len(self.project_plan.tasks)} tasks completed.")
        insights.append(f"Overall Goal: {self.project_goal}")

        return "\n".join(insights)

    def suggest_parallelization(self) -> list:
        """Identifies opportunities for parallel task execution."""
        print("\n--- Suggesting Parallelization Opportunities ---")
        parallelizable_tasks = []
        # Simple heuristic: tasks with no unmet dependencies can potentially run in parallel
        # More advanced: LLM could analyze task descriptions for inherent parallel nature
        for task_id, task in self.project_plan.tasks.items():
            if task.status == "pending" or task.status == "in_progress":
                all_dependencies_met = True
                for dep_id in task.dependencies:
                    dep_task = self.project_plan.get_task(dep_id)
                    if dep_task and dep_task.status != "completed":
                        all_dependencies_met = False
                        break
                if all_dependencies_met and not task.dependencies: # No dependencies, or all met
                     # Check if it's not already in progress or completed and not blocked by resources
                     if task.status != "in_progress" and task.status != "completed" and task.status != "blocked":
                         # Simple check: if resources are available (conceptual)
                         if self.available_resources and any(res_type in self.available_resources for res_type in task.resources_needed):
                             parallelizable_tasks.append(task.name)
                         elif not task.resources_needed:
                             parallelizable_tasks.append(task.name)

        if parallelizable_tasks:
            print(f"  Potential for parallel execution: {', '.join(parallelizable_tasks)}")
        else:
            print("  No immediate parallelization opportunities identified (based on simple heuristic).")

        return parallelizable_tasks


# --- Example Usage --- #
if __name__ == "__main__":
    mock_llm = MockLLM()
    pm_ai = ProjectManagerAI(llm_interface=mock_llm)

    # 1. Initialize Project
    project_goal = "Launch a new product by end of Q4 with a $500k budget."
    pm_ai.initialize_project(project_goal)

    # 2. Apply Constraints
    deadline_constraint = Constraint("deadline", "Product must launch by December 31st", {"date": "2024-12-31"})
    budget_constraint = Constraint("budget", "Total budget not to exceed $500,000", {"amount": 500000})
    pm_ai.apply_constraints([deadline_constraint, budget_constraint])

    # Get task IDs for easier reference
    task_ids_by_name = {task.name: task.id for task in pm_ai.project_plan.tasks.values()}
    market_research_id = task_ids_by_name.get("Market Research")
    product_design_id = task_ids_by_name.get("Product Design")
    development_id = task_ids_by_name.get("Development")
    testing_id = task_ids_by_name.get("Testing")
    marketing_campaign_id = task_ids_by_name.get("Marketing Campaign")

    # Manually set some dependencies that the mock LLM might not fully infer
    if product_design_id and market_research_id:
        pm_ai.project_plan.get_task(product_design_id).add_dependency(market_research_id)
    if development_id and product_design_id:
        pm_ai.project_plan.get_task(development_id).add_dependency(product_design_id)
    if testing_id and development_id:
        pm_ai.project_plan.get_task(testing_id).add_dependency(development_id)
    if marketing_campaign_id and testing_id:
        pm_ai.project_plan.get_task(marketing_campaign_id).add_dependency(testing_id)

    print("\n--- Initial Project Plan Tasks ---")
    for task in pm_ai.project_plan.tasks.values():
        print(task)

    # 3. Simulate Progress and Monitoring
    print("\n--- Simulating Progress: Market Research Completed ---")
    updated_statuses_1 = {market_research_id: "completed"}
    pm_ai.monitor_progress(updated_statuses_1)
    pm_ai.optimize_resource_allocation() # Re-optimize after progress
    pm_ai.suggest_parallelization()
    print(pm_ai.get_actionable_insights())

    print("\n--- Simulating Progress: Product Design In Progress ---")
    updated_statuses_2 = {product_design_id: "in_progress"}
    issues = pm_ai.monitor_progress(updated_statuses_2)
    pm_ai.optimize_resource_allocation()
    pm_ai.suggest_parallelization()
    print(pm_ai.get_actionable_insights())

    print("\n--- Simulating a Problem: Development Blocked by Design Specs ---")
    updated_statuses_3 = {product_design_id: "in_progress"} # Still in progress, but we'll manually block dev
    issues = pm_ai.monitor_progress(updated_statuses_3) # This will detect development is blocked
    pm_ai.project_plan.get_task(development_id).update_status("blocked") # Explicitly block for demo
    issues = pm_ai.monitor_progress() # Re-monitor to pick up the explicit block

    # 4. Adapt Plan if Needed
    pm_ai.adapt_plan_if_needed(issues)
    print("\n--- Project Plan After Adaptation ---")
    for task in pm_ai.project_plan.tasks.values():
        print(task)

    pm_ai.suggest_parallelization()
    print(pm_ai.get_actionable_insights())

    # Further simulation (e.g., design completes, development starts)
    print("\n--- Simulating Progress: Product Design Completed, Development Starts ---")
    pm_ai.project_plan.get_task(product_design_id).mark_as_complete(actual_duration=18)
    pm_ai.project_plan.get_task(development_id).update_status("in_progress")
    issues = pm_ai.monitor_progress()
    pm_ai.optimize_resource_allocation()
    print(pm_ai.get_actionable_insights())

    print("\n--- Simulating Final Progress ---")
    pm_ai.project_plan.get_task(development_id).mark_as_complete(actual_duration=35)
    pm_ai.project_plan.get_task(testing_id).update_status("in_progress")
    pm_ai.monitor_progress()
    pm_ai.project_plan.get_task(testing_id).mark_as_complete(actual_duration=12)
    pm_ai.project_plan.get_task(marketing_campaign_id).update_status("in_progress")
    pm_ai.monitor_progress()
    pm_ai.project_plan.get_task(marketing_campaign_id).mark_as_complete(actual_duration=22)

    final_issues = pm_ai.monitor_progress()
    print(pm_ai.get_actionable_insights())
    pm_ai.project_plan.calculate_critical_path()
    pm_ai.project_plan.visualize_plan()

    print("\n--- Final Project Plan Status ---")
    for task in pm_ai.project_plan.tasks.values():
        print(task)
