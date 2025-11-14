import uuid
import datetime

def simulate_llm_decompose(project_goal: str) -> list:
    """Simulates LLM decomposing a project goal into sub-tasks."""
    print(f"LLM: Decomposing goal: '{project_goal}'")
    if "website" in project_goal.lower():
        return [
            {"name": "Define Website Requirements", "description": "Gather user stories and functional specifications.", "dependencies": [], "estimated_duration_days": 3},
            {"name": "Design UI/UX Mockups", "description": "Create wireframes and mockups for the website.", "dependencies": ["Define Website Requirements"], "estimated_duration_days": 5},
            {"name": "Develop Backend API", "description": "Build server-side logic and database.", "dependencies": ["Define Website Requirements"], "estimated_duration_days": 10},
            {"name": "Develop Frontend Interface", "description": "Implement UI based on mockups and connect to API.", "dependencies": ["Design UI/UX Mockups", "Develop Backend API"], "estimated_duration_days": 12},
            {"name": "Conduct Testing", "description": "Perform unit, integration, and user acceptance testing.", "dependencies": ["Develop Frontend Interface", "Develop Backend API"], "estimated_duration_days": 7},
            {"name": "Deploy Website", "description": "Publish the website to production servers.", "dependencies": ["Conduct Testing"], "estimated_duration_days": 2}
        ]
    elif "marketing campaign" in project_goal.lower():
        return [
            {"name": "Define Campaign Goals", "description": "Set clear objectives and target audience.", "dependencies": [], "estimated_duration_days": 2},
            {"name": "Research Market & Competitors", "description": "Analyze trends and competitor strategies.", "dependencies": ["Define Campaign Goals"], "estimated_duration_days": 4},
            {"name": "Develop Marketing Strategy", "description": "Outline channels, messaging, and budget.", "dependencies": ["Research Market & Competitors"], "estimated_duration_days": 5},
            {"name": "Create Campaign Assets", "description": "Design creatives, write copy, prepare materials.", "dependencies": ["Develop Marketing Strategy"], "estimated_duration_days": 8},
            {"name": "Launch Campaign", "description": "Execute the campaign across chosen channels.", "dependencies": ["Create Campaign Assets"], "estimated_duration_days": 1},
            {"name": "Monitor & Optimize Campaign", "description": "Track performance and make adjustments.", "dependencies": ["Launch Campaign"], "estimated_duration_days": 10}
        ]
    else:
        return [
            {"name": "Phase 1: Research", "description": "Initial data gathering and analysis.", "dependencies": [], "estimated_duration_days": 5},
            {"name": "Phase 2: Planning", "description": "Outline strategy and detailed steps.", "dependencies": ["Phase 1: Research"], "estimated_duration_days": 7},
            {"name": "Phase 3: Execution", "description": "Carry out the main work.", "dependencies": ["Phase 2: Planning"], "estimated_duration_days": 15},
            {"name": "Phase 4: Review", "description": "Evaluate outcomes and make final adjustments.", "dependencies": ["Phase 3: Execution"], "estimated_duration_days": 3}
        ]

def simulate_llm_replan(current_plan_state: dict, feedback: str) -> list:
    """Simulates LLM adjusting a plan based on feedback."""
    print(f"LLM: Re-planning based on feedback: '{feedback}'")
    if "stuck" in feedback.lower() or "blocked" in feedback.lower():
        problem_task_id = None
        for task_id, task_data in current_plan_state["tasks"].items():
            if "stuck" in task_data["status"].lower() or "blocked" in task_data["status"].lower():
                problem_task_id = task_id
                break
        
        if problem_task_id:
            problem_task_name = current_plan_state["tasks"][problem_task_id]["name"]
            new_subtask = {
                "id": str(uuid.uuid4()),
                "name": f"Troubleshoot {problem_task_name}",
                "description": f"Investigate and resolve issues blocking '{problem_task_name}'.",
                "dependencies": [problem_task_id],
                "estimated_duration_days": 2,
                "status": "Not Started"
            }
            return [new_subtask]
    
    return []


class Task:
    def __init__(self, name: str, description: str, dependencies: list = None, estimated_duration_days: int = 1):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.dependencies = dependencies if dependencies is not None else []
        self.estimated_duration_days = estimated_duration_days
        self.status = "Not Started"
        self.actual_start_date = None
        self.actual_end_date = None
        self.assigned_resources = []
        self.notes = []

    def update_status(self, new_status: str, note: str = ""):
        valid_statuses = ["Not Started", "In Progress", "Blocked", "Completed"]
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status: {new_status}. Must be one of {valid_statuses}")
        
        if new_status == "In Progress" and self.actual_start_date is None:
            self.actual_start_date = datetime.date.today()
        elif new_status == "Completed" and self.actual_end_date is None:
            self.actual_end_date = datetime.date.today()
        
        self.status = new_status
        if note:
            self.notes.append(f"{datetime.date.today()}: {note}")

    def add_dependency(self, task_id: str):
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "dependencies": self.dependencies,
            "estimated_duration_days": self.estimated_duration_days,
            "status": self.status,
            "actual_start_date": str(self.actual_start_date) if self.actual_start_date else None,
            "actual_end_date": str(self.actual_end_date) if self.actual_end_date else None,
            "assigned_resources": self.assigned_resources,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict):
        task = cls(
            name=data["name"],
            description=data["description"],
            dependencies=data.get("dependencies", []),
            estimated_duration_days=data.get("estimated_duration_days", 1)
        )
        task.id = data["id"]
        task.status = data.get("status", "Not Started")
        task.actual_start_date = datetime.date.fromisoformat(data["actual_start_date"]) if data.get("actual_start_date") else None
        task.actual_end_date = datetime.date.fromisoformat(data["actual_end_date"]) if data.get("actual_end_date") else None
        task.assigned_resources = data.get("assigned_resources", [])
        task.notes = data.get("notes", [])
        return task


class ProjectPlanner:
    def __init__(self, project_goal: str):
        self.project_goal = project_goal
        self.tasks = {}
        self.plan_history = []

    def decompose_and_plan(self):
        """
        Uses an LLM to decompose the project goal into initial tasks
        and generate an initial plan.
        """
        print(f"\n--- Initial Decomposition and Planning for: {self.project_goal} ---")
        llm_suggested_tasks_data = simulate_llm_decompose(self.project_goal)
        
        temp_tasks = {}
        for task_data in llm_suggested_tasks_data:
            task = Task(
                name=task_data["name"],
                description=task_data["description"],
                estimated_duration_days=task_data["estimated_duration_days"]
            )
            self.tasks[task.id] = task
            temp_tasks[task.name] = task

        for task_data in llm_suggested_tasks_data:
            current_task = temp_tasks[task_data["name"]]
            for dep_name in task_data.get("dependencies", []):
                dep_task = temp_tasks.get(dep_name)
                if dep_task:
                    current_task.add_dependency(dep_task.id)
                else:
                    print(f"Warning: Dependency '{dep_name}' for task '{current_task.name}' not found.")
        
        self._record_plan_state("Initial Plan Generated")
        print("Initial plan generated successfully.")
        return self.get_current_plan()

    def update_task_status(self, task_id: str, new_status: str, feedback: str = ""):
        """
        Updates the status of a specific task and potentially triggers replanning.
        """
        if task_id not in self.tasks:
            print(f"Error: Task with ID '{task_id}' not found.")
            return

        task = self.tasks[task_id]
        old_status = task.status
        try:
            task.update_status(new_status, feedback)
            print(f"Task '{task.name}' ({task_id}) status updated from '{old_status}' to '{new_status}'.")
            if feedback:
                print(f"Feedback provided: '{feedback}'")
            
            self._dynamic_replan(task_id, new_status, feedback)
            self._record_plan_state(f"Task '{task.name}' status changed to '{new_status}'")
            
        except ValueError as e:
            print(f"Error updating task status: {e}")

    def _dynamic_replan(self, changed_task_id: str, new_status: str, feedback: str):
        """
        Internal method to trigger LLM-based replanning.
        """
        print("\n--- Initiating Dynamic Re-planning ---")
        
        if new_status == "Blocked":
            print(f"Task '{self.tasks[changed_task_id].name}' is blocked. Seeking LLM assistance for resolution.")
            llm_suggestions = simulate_llm_replan(self.get_current_plan(), f"Task {self.tasks[changed_task_id].name} is blocked. User feedback: {feedback}")
            for suggestion_data in llm_suggestions:
                new_task = Task.from_dict(suggestion_data)
                self.tasks[new_task.id] = new_task
                print(f"LLM suggested new task: '{new_task.name}' (ID: {new_task.id})")
                if changed_task_id not in new_task.dependencies:
                    new_task.add_dependency(changed_task_id)

        for task_id, task in self.tasks.items():
            if task_id != changed_task_id and task.status == "Not Started":
                can_start = True
                for dep_id in task.dependencies:
                    if dep_id in self.tasks and self.tasks[dep_id].status != "Completed":
                        can_start = False
                        break
                
                if can_start and task.status == "Not Started":
                    print(f"Note: Task '{task.name}' (ID: {task.id}) can now potentially start as its dependencies are met or progressed.")
                elif not can_start and task.status == "In Progress":
                     print(f"Warning: Task '{task.name}' (ID: {task.id}) is In Progress but a dependency is not yet completed. This might indicate an issue.")

        print("--- Dynamic Re-planning Complete ---")


    def get_current_plan(self):
        """Returns a snapshot of the current project plan."""
        return {
            "project_goal": self.project_goal,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "plan_status": self._get_overall_plan_status()
        }

    def _get_overall_plan_status(self):
        """Calculates the overall status of the project."""
        if not self.tasks:
            return "No tasks defined"
        
        completed_count = sum(1 for task in self.tasks.values() if task.status == "Completed")
        total_count = len(self.tasks)
        
        if completed_count == total_count:
            return "Completed"
        elif any(task.status == "In Progress" for task in self.tasks.values()):
            return f"In Progress ({completed_count}/{total_count} tasks completed)"
        elif any(task.status == "Blocked" for task in self.tasks.values()):
            return f"Blocked ({completed_count}/{total_count} tasks completed)"
        else:
            return f"Not Started ({completed_count}/{total_count} tasks completed)"

    def _record_plan_state(self, event_description: str):
        """Records the current state of the plan for history tracking."""
        self.plan_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event_description,
            "state": self.get_current_plan()
        })

    def get_actionable_tasks(self):
        """Identifies tasks that can be started or are in progress."""
        actionable = []
        for task_id, task in self.tasks.items():
            if task.status in ["Not Started", "In Progress", "Blocked"]:
                dependencies_met = True
                for dep_id in task.dependencies:
                    if dep_id in self.tasks and self.tasks[dep_id].status != "Completed":
                        dependencies_met = False
                        break
                
                if dependencies_met or task.status == "In Progress":
                    actionable.append(task)
        
        actionable.sort(key=lambda t: (
            0 if t.status == "Blocked" else
            1 if t.status == "In Progress" else
            2 if t.status == "Not Started" else 3
        ))
        return actionable

    def visualize_plan(self):
        """Prints a simplified textual representation of the plan."""
        print("\n--- Project Plan Visualization ---")
        print(f"Project Goal: {self.project_goal}")
        print(f"Overall Status: {self._get_overall_plan_status()}")
        print("-" * 30)

        sorted_tasks = sorted(self.tasks.values(), key=lambda t: t.name)

        for task in sorted_tasks:
            dep_names = [self.tasks[dep_id].name for dep_id in task.dependencies if dep_id in self.tasks]
            deps_str = f" (Depends on: {', '.join(dep_names)})" if dep_names else ""
            print(f"[{task.status}] {task.name}{deps_str}")
            print(f"  ID: {task.id}")
            print(f"  Description: {task.description}")
            print(f"  Est. Duration: {task.estimated_duration_days} days")
            if task.actual_start_date:
                print(f"  Started: {task.actual_start_date}")
            if task.actual_end_date:
                print(f"  Completed: {task.actual_end_date}")
            if task.notes:
                print(f"  Notes: {'; '.join(task.notes)}")
            print("-" * 30)

        print("\n--- Actionable Tasks ---")
        actionable_tasks = self.get_actionable_tasks()
        if actionable_tasks:
            for i, task in enumerate(actionable_tasks):
                print(f"{i+1}. [{task.status}] {task.name} (ID: {task.id})")
                if task.status == "Blocked":
                    print(f"   Reason/Feedback: {'; '.join(task.notes)}")
        else:
            print("No immediate actionable tasks. All tasks might be completed or waiting on dependencies.")
        print("-" * 30)

