
class Task:
    def __init__(self, name: str, description: str, duration_days: int, dependencies: list, required_skills: list):
        self.name = name
        self.description = description
        self.duration_days = duration_days
        self.dependencies = dependencies
        self.required_skills = required_skills
        self.assigned_resource = None
        self.start_date = None
        self.end_date = None

    def __repr__(self):
        return f"Task(name='{self.name}', assigned='{self.assigned_resource}')"

class Resource:
    def __init__(self, name: str, skills: list, availability_days_per_week: int):
        self.name = name
        self.skills = skills
        self.availability_days_per_week = availability_days_per_week

    def __repr__(self):
        return f"Resource(name='{self.name}', skills={self.skills})"

class ProjectDetails:
    def __init__(self, project_name: str, total_budget: float, deadline: str, tasks: list, resources: list):
        self.project_name = project_name
        self.total_budget = total_budget
        self.deadline = deadline
        self.tasks = tasks  # List of Task objects
        self.resources = resources # List of Resource objects

    def __repr__(self):
        return f"ProjectDetails(name='{self.project_name}', tasks={len(self.tasks)}, resources={len(self.resources)})"

class ProjectPlan:
    def __init__(self, project_name: str, planned_tasks: list, total_estimated_cost: float, estimated_completion_date: str, bottlenecks: list = None):
        self.project_name = project_name
        self.planned_tasks = planned_tasks # List of Task objects with assignments and dates
        self.total_estimated_cost = total_estimated_cost
        self.estimated_completion_date = estimated_completion_date
        self.bottlenecks = bottlenecks if bottlenecks is not None else []

    def __repr__(self):
        return f"ProjectPlan(name='{self.project_name}', completion='{self.estimated_completion_date}')"

class ProjectManagerAssistant:
    def __init__(self):
        print("Project Manager Assistant initialized. Ready for two-stage planning.")

    def _simulate_info_gathering(self, project_id: str) -> dict:
        """
        Simulates gathering comprehensive project information from various sources.
        In a real application, this would involve database queries, API calls, document parsing, etc.
        """
        print(f"STAGE 1: Collecting information for project '{project_id}'...")
        # Mock data for demonstration
        if project_id == "AI_Agent_Deployment":
            tasks_data = [
                Task("Requirement Analysis", "Define AI agent features", 5, [], ["Software Engineering", "AI/ML Design"]),
                Task("Data Collection Setup", "Configure data pipelines for training", 7, ["Requirement Analysis"], ["Data Engineering", "Cloud Infrastructure"]),
                Task("Model Training", "Train the core AI model", 10, ["Data Collection Setup"], ["Machine Learning", "Data Science"]),
                Task("Deployment Strategy", "Plan the deployment infrastructure", 4, ["Requirement Analysis"], ["DevOps", "Cloud Infrastructure"]),
                Task("Testing & Validation", "Rigorously test the deployed agent", 8, ["Model Training", "Deployment Strategy"], ["QA", "AI/ML Testing"]),
                Task("Monitoring & Maintenance", "Set up monitoring and maintenance procedures", 6, ["Deployment Strategy", "Testing & Validation"], ["DevOps", "Site Reliability"])
            ]
            resources_data = [
                Resource("Alice", ["Software Engineering", "AI/ML Design"], 5),
                Resource("Bob", ["Data Engineering", "Cloud Infrastructure", "DevOps"], 4),
                Resource("Charlie", ["Machine Learning", "Data Science", "AI/ML Testing"], 5),
                Resource("David", ["QA", "Software Engineering"], 3),
            ]
            return {
                "project_name": "AI Agent Deployment",
                "total_budget": 50000.0,
                "deadline": "2024-12-31",
                "tasks": tasks_data,
                "resources": resources_data
            }
        else:
            return None # Or raise an error for unknown project

    def _plan_project_logic(self, project_details: ProjectDetails) -> ProjectPlan:
        """
        Generates an optimized project plan based on collected information.
        This simplified logic assigns tasks sequentially and checks basic constraints.
        A real system would use more complex algorithms (e.g., Gantt charts, critical path method, LLM-based planning).
        """
        print(f"STAGE 2: Generating plan for project '{project_details.project_name}'...")
        planned_tasks = []
        current_date = "2024-07-01" # Mock start date
        total_estimated_cost = 0.0
        bottlenecks = []

        # Simple resource pool for assignment
        available_resources = {r.name: r for r in project_details.resources}
        resource_schedules = {r.name: [] for r in project_details.resources} # Tracks assigned days

        # Sort tasks (a very simple approach, a real one would consider dependencies)
        # For simplicity, we'll assume a basic topological sort is handled or tasks are mostly independent for this demo
        # A more robust system would need a proper dependency graph.
        remaining_tasks = list(project_details.tasks)
        processed_task_names = set()

        # Simple iterative assignment (not truly optimal, but demonstrates the concept)
        while remaining_tasks:
            assigned_this_iteration = False
            for task in list(remaining_tasks): # Iterate over a copy to modify the original list
                # Check if dependencies are met
                dependencies_met = True
                for dep_name in task.dependencies:
                    if dep_name not in processed_task_names:
                        dependencies_met = False
                        break

                if not dependencies_met:
                    continue # Skip this task for now, dependencies not met

                # Try to find an available resource with required skills
                best_resource = None
                for res_name, resource in available_resources.items():
                    if all(skill in resource.skills for skill in task.required_skills):
                        # Simple availability check: if resource has capacity (e.g., less than 5 tasks concurrently, or simple day-based check)
                        # For this demo, we just assign if skills match and resource is generally 'available'
                        best_resource = resource
                        break # Found a suitable resource

                if best_resource:
                    task.assigned_resource = best_resource.name
                    task.start_date = current_date # Simplified: all tasks start on current_date if assigned
                    task.end_date = "N/A" # Would need date calculations in a real scenario
                    planned_tasks.append(task)
                    processed_task_names.add(task.name)
                    remaining_tasks.remove(task)
                    assigned_this_iteration = True
                    print(f"  - Assigned '{task.name}' to '{best_resource.name}'")
                    # Update cost: simple flat rate per day per task for demo
                    total_estimated_cost += task.duration_days * 500 # Assume $500/day/task
                else:
                    # If no resource found for a task, it's a bottleneck or skill gap
                    bottlenecks.append(f"No suitable resource found for task '{task.name}' requiring skills: {', '.join(task.required_skills)}")

            if not assigned_this_iteration and remaining_tasks:
                # If we iterated through all remaining tasks but couldn't assign any new one,
                # it means we're stuck due to unresolvable dependencies or lack of resources.
                # This indicates a planning failure or complex bottleneck.
                for task in remaining_tasks:
                    bottlenecks.append(f"Task '{task.name}' could not be assigned due to unmet dependencies or resource unavailability.")
                break # Exit the loop to prevent infinite loop for unassignable tasks

        # Estimate completion date (very rough for demo)
        estimated_completion_date = "Unknown (complex scheduling required)"
        if planned_tasks:
            # In a real scenario, this would be based on critical path analysis
            estimated_completion_date = f"Approx. {len(project_details.tasks) * 7} days from start (simplified)" # Roughly duration of all tasks * average overlap

        print("Planning complete.")
        return ProjectPlan(
            project_name=project_details.project_name,
            planned_tasks=planned_tasks,
            total_estimated_cost=total_estimated_cost,
            estimated_completion_date=estimated_completion_date,
            bottlenecks=bottlenecks
        )

    def plan_project(self, project_id: str) -> ProjectPlan:
        """
        Orchestrates the two-stage project planning process:
        1. Information Collection
        2. Project Planning
        """
        # Stage 1: Information Collection
        project_data = self._simulate_info_gathering(project_id)
        if not project_data:
            print(f"Error: Could not retrieve information for project '{project_id}'.")
            return None

        project_details = ProjectDetails(**project_data)
        print(f"Collected Project Details: {project_details}")

        # Stage 2: Planning
        project_plan = self._plan_project_logic(project_details)
        print(f"Generated Project Plan: {project_plan}")

        return project_plan

# Example Usage:
if __name__ == "__main__":
    assistant = ProjectManagerAssistant()
    final_plan = assistant.plan_project("AI_Agent_Deployment")

    if final_plan:
        print("\n--- Final Project Plan Summary ---")
        print(f"Project Name: {final_plan.project_name}")
        print(f"Estimated Completion: {final_plan.estimated_completion_date}")
        print(f"Estimated Cost: ${final_plan.total_estimated_cost:,.2f}")
        print("\nPlanned Tasks:")
        for task in final_plan.planned_tasks:
            print(f"  - {task.name} (Assigned to: {task.assigned_resource})")
        if final_plan.bottlenecks:
            print("\nPotential Bottlenecks:")
            for bottleneck in final_plan.bottlenecks:
                print(f"  - {bottleneck}")
    else:
        print("Project planning failed.")
