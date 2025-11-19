import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# -----------------------------------------------------------
# 1. Task & Project Data Models
# -----------------------------------------------------------

class Constraint(BaseModel):
    type: str  # e.g., "budget", "deadline", "resource"
    value: Any
    unit: Optional[str] = None # e.g., "USD", "days", "person-hours"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: str = "to-do"  # "to-do", "in-progress", "done", "blocked"
    dependencies: List[str] = Field(default_factory=list)
    assigned_to: Optional[str] = None
    estimated_effort_hours: int = 8
    actual_effort_hours: int = 0
    due_date: Optional[datetime] = None
    comments: List[str] = Field(default_factory=list)

class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    tasks: Dict[str, Task] = Field(default_factory=dict)
    plan: List[str] = Field(default_factory=list)  # Ordered list of task IDs
    status: str = "initialized" # "initialized", "in-progress", "completed", "halted"
    constraints: List[Constraint] = Field(default_factory=list)
    start_date: datetime = Field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    # ConstraintManager is an external dependency, not part of the model itself
    # but we will assign it after Project instantiation.
    constraint_manager: Any = Field(default=None, exclude=True) 

# -----------------------------------------------------------
# 2. LLM Integration Layer (Simulated/Placeholder)
# -----------------------------------------------------------

class SimulatedLLM:
    def __init__(self, model_name: str = "gpt-4-simulated"):
        self.model_name = model_name

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        prompt_text = messages[-1]["content"].lower()

        if "decompose" in prompt_text and "goal" in prompt_text:
            return """
            [
                {"description": "Define project scope and requirements", "dependencies": []},
                {"description": "Setup development environment", "dependencies": []},
                {"description": "Design core architecture", "dependencies": ["Define project scope and requirements"]},
                {"description": "Develop user authentication module", "dependencies": ["Design core architecture", "Setup development environment"]},
                {"description": "Develop task management module", "dependencies": ["Design core architecture", "Setup development environment"]},
                {"description": "Integrate modules and test", "dependencies": ["Develop user authentication module", "Develop task management module"]},
                {"description": "Deploy to staging environment", "dependencies": ["Integrate modules and test"]},
                {"description": "User acceptance testing (UAT)", "dependencies": ["Deploy to staging environment"]},
                {"description": "Final deployment", "dependencies": ["User acceptance testing (UAT)"]}
            ]
            """
        elif "generate initial plan" in prompt_text and "tasks" in prompt_text:
            return """
            [
                "Define project scope and requirements",
                "Setup development environment",
                "Design core architecture",
                "Develop user authentication module",
                "Develop task management module",
                "Integrate modules and test",
                "Deploy to staging environment",
                "User acceptance testing (UAT)",
                "Final deployment"
            ]
            """
        elif "introspect plan" in prompt_text:
            if "authentication" in prompt_text and "task management" in prompt_text:
                return "The plan looks solid, but ensure sufficient security testing for the authentication module."
            return "The initial plan seems reasonable, no immediate major issues detected. Consider adding a dedicated 'performance testing' phase."
        elif "adapt plan" in prompt_text and "feedback" in prompt_text:
            if "task management module is blocked" in prompt_text:
                return """
                Adaptation: Prioritize 'Develop user authentication module' while investigating the blocker for 'Develop task management module'.
                Consider reassigning resources if possible.
                New plan suggestion:
                [
                    "Define project scope and requirements",
                    "Setup development environment",
                    "Design core architecture",
                    "Develop user authentication module",
                    "Investigate task management blocker",
                    "Develop task management module",
                    "Integrate modules and test",
                    "Deploy to staging environment",
                    "User acceptance testing (UAT)",
                    "Final deployment"
                ]
                """
            return "Plan adapted: Added a buffer day to the 'Integrate modules and test' phase due to minor delays."
        elif "filter irrelevant information" in prompt_text:
            return "Filtered: Focus on 'task status' and 'dependencies' for planning."
        elif "check constraints" in prompt_text:
            if "budget_exceeded" in prompt_text:
                return "Constraint check: Budget potentially exceeded. Suggest reducing scope or seeking additional funds."
            if "deadline_missed" in prompt_text:
                return "Constraint check: Deadline at risk. Suggest re-prioritizing tasks or adding more resources."
            return "Constraint check: All current constraints appear to be satisfied."
        else:
            return "Simulated LLM response: Understood. How can I assist further with planning?"

    def generate_response(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

# -----------------------------------------------------------
# 3. Constraint Manager
# -----------------------------------------------------------

class ConstraintManager:
    def __init__(self):
        self.constraints: List[Constraint] = []

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)

    def validate_plan(self, project: 'Project') -> List[str]:
        violations = []
        for constraint in project.constraints:
            if constraint.type == "deadline":
                # This is a simplified check. A real system would calculate project end date based on task durations.
                if project.end_date and project.end_date > constraint.value:
                    violations.append(f"Deadline constraint violated: Project estimated to finish after {constraint.value.strftime('%Y-%m-%d')}")
            # Add more complex checks here for budget, resources, etc.
        return violations

# -----------------------------------------------------------
# 4. Project Planner (Orchestration & Reasoning Engine)
# -----------------------------------------------------------

class ProjectPlanner:
    def __init__(self, llm: SimulatedLLM, constraint_manager: ConstraintManager):
        self.llm = llm
        self.constraint_manager = constraint_manager

    def decompose_goal(self, project: Project) -> List[Task]:
        prompt = f"Decompose the high-level project goal '{project.goal}' into a list of specific, actionable sub-tasks. Provide the output as a JSON array of objects, where each object has 'description' and 'dependencies' (a list of task descriptions)."
        llm_response_str = self.llm.generate_response(prompt)
        try:
            task_data = eval(llm_response_str.strip()) # Using eval for simplicity, use json.loads in real app
            new_tasks = []
            for item in task_data:
                task = Task(description=item["description"], dependencies=item["dependencies"])
                new_tasks.append(task)
            return new_tasks
        except Exception as e:
            print(f"Error parsing LLM response for goal decomposition: {e}")
            return []

    def generate_initial_plan(self, project: Project) -> List[str]:
        tasks_str = ", ".join([f"'{t.description}'" for t in project.tasks.values()])
        prompt = f"Generate an initial project plan (ordered list of task descriptions) for the following tasks, considering their dependencies: {tasks_str}. Output as a JSON array of task descriptions."
        llm_response_str = self.llm.generate_response(prompt)
        try:
            plan_descriptions = eval(llm_response_str.strip()) # Using eval for simplicity
            plan_ids = []
            description_to_id = {task.description: task.id for task in project.tasks.values()}
            for desc in plan_descriptions:
                if desc in description_to_id:
                    plan_ids.append(description_to_id[desc])
            return plan_ids
        except Exception as e:
            print(f"Error parsing LLM response for plan generation: {e}")
            return []

    def introspect_plan(self, project: Project) -> str:
        current_plan_descriptions = [project.tasks[task_id].description for task_id in project.plan]
        plan_str = ", ".join(current_plan_descriptions)
        prompt = f"Introspect the following project plan for '{project.goal}' and identify potential issues, missing steps, or areas for refinement: {plan_str}. Provide a concise summary or suggested changes."
        return self.llm.generate_response(prompt)

    def adapt_plan(self, project: Project, feedback: str) -> List[str]:
        current_plan_descriptions = [project.tasks[task_id].description for task_id in project.plan]
        plan_str = ", ".join(current_plan_descriptions)
        prompt = f"Given the current project plan: {plan_str} and the following feedback: '{feedback}', adapt the plan to address the feedback. Provide a new ordered list of task descriptions as a JSON array."
        llm_response_str = self.llm.generate_response(prompt)
        try:
            new_plan_descriptions = eval(llm_response_str.strip())
            new_plan_ids = []
            description_to_id = {task.description: task.id for task in project.tasks.values()}
            for desc in new_plan_descriptions:
                if desc not in description_to_id:
                    new_task = Task(description=desc, dependencies=[])
                    project.tasks[new_task.id] = new_task
                    description_to_id[desc] = new_task.id
                new_plan_ids.append(description_to_id[desc])
            return new_plan_ids
        except Exception as e:
            print(f"Error parsing LLM response for plan adaptation: {e}")
            return project.plan

    def filter_information(self, prompt_text: str) -> str:
        keywords = ["task status", "dependencies", "due date", "assigned to", "blocker"]
        filtered_parts = [part for part in prompt_text.split() if any(k in part.lower() for k in keywords)]
        if not filtered_parts:
            return f"Refined prompt focusing on key project metrics from: {prompt_text}"
        return " ".join(filtered_parts)

    def get_next_actionable_task(self, project: Project) -> Optional[Task]:
        completed_task_ids = {tid for tid, task in project.tasks.items() if task.status == "done"}
        in_progress_task_ids = {tid for tid, task in project.tasks.items() if task.status == "in-progress"}

        for task_id in project.plan:
            task = project.tasks.get(task_id)
            if task and task.status == "to-do":
                dependencies_met = True
                for dep_desc in task.dependencies:
                    dep_task_id = next((tid for tid, t in project.tasks.items() if t.description == dep_desc), None)
                    if dep_task_id and dep_task_id not in completed_task_ids:
                        dependencies_met = False
                        break
                if dependencies_met:
                    return task
        return None

# -----------------------------------------------------------
# 5. Feedback & Monitoring System (Simulated)
# -----------------------------------------------------------

class FeedbackMonitor:
    def __init__(self, project: Project):
        self.project = project
        self.simulation_step = 0

    def simulate_task_completion(self, task: Task) -> str:
        self.simulation_step += 1
        if self.simulation_step % 5 == 0:
            task.status = "blocked"
            task.comments.append(f"Simulated: Task '{task.description}' encountered a blocker.")
            return f"Task '{task.description}' is BLOCKED."
        elif self.simulation_step % 3 == 0:
            task.estimated_effort_hours += 4
            task.comments.append(f"Simulated: Task '{task.description}' is experiencing a minor delay, estimated effort increased.")
            return f"Task '{task.description}' is IN-PROGRESS (delayed)."
        else:
            task.status = "done"
            task.actual_effort_hours = task.estimated_effort_hours
            task.comments.append(f"Simulated: Task '{task.description}' completed successfully.")
            return f"Task '{task.description}' completed."

    def get_project_status_summary(self) -> str:
        total_tasks = len(self.project.tasks)
        completed_tasks = sum(1 for t in self.project.tasks.values() if t.status == "done")
        in_progress_tasks = sum(1 for t in self.project.tasks.values() if t.status == "in-progress")
        blocked_tasks = sum(1 for t in self.project.tasks.values() if t.status == "blocked")
        to_do_tasks = sum(1 for t in self.project.tasks.values() if t.status == "to-do")

        summary = f"Project Status for '{self.project.goal}':\n"
        summary += f"  Total tasks: {total_tasks}\n"
        summary += f"  Completed: {completed_tasks}\n"
        summary += f"  In Progress: {in_progress_tasks}\n"
        summary += f"  Blocked: {blocked_tasks}\n"
        summary += f"  To-Do: {to_do_tasks}\n"

        if self.project.end_date:
            summary += f"  Estimated completion: {self.project.end_date.strftime('%Y-%m-%d')}\n"

        violations = self.project.constraint_manager.validate_plan(self.project)
        if violations:
            summary += "  !!! Constraint Violations !!!\n"
            for viol in violations:
                summary += f"    - {viol}\n"

        return summary

# -----------------------------------------------------------
# 6. Communication & Reporting Module
# -----------------------------------------------------------

class CommunicationReporter:
    def post_update(self, message: str):
        print(f"[COMMUNICATION] {message}")

    def generate_report(self, project: Project) -> str:
        report = f"\n--- Project Report: {project.goal} ---\n"
        report += f"Project ID: {project.id}\n"
        report += f"Status: {project.status}\n"
        report += f"Start Date: {project.start_date.strftime('%Y-%m-%d')}\n"
        if project.end_date:
            report += f"Estimated End Date: {project.end_date.strftime('%Y-%m-%d')}\n"
        report += "\nTasks:\n"
        for task_id in project.plan:
            task = project.tasks[task_id]
            report += f"- {task.description} [{task.status.upper()}] (Effort: {task.actual_effort_hours}/{task.estimated_effort_hours}h)\n"
            if task.comments:
                for comment in task.comments:
                    report += f"    Comment: {comment}\n"
        report += "\nConstraints:\n"
        for constraint in project.constraints:
            report += f"- {constraint.type}: {constraint.value} {constraint.unit or ''}\n"

        violations = project.constraint_manager.validate_plan(project)
        if violations:
            report += "\nConstraint Violations:\n"
            for viol in violations:
                report += f"- {viol}\n"
        report += "\n--- End Report ---\n"
        return report

# -----------------------------------------------------------
# Main Workflow Execution
# -----------------------------------------------------------

def run_project_manager_simulation():
    print("--- AI-Powered Project Manager Simulation Started ---")

    llm = SimulatedLLM()
    constraint_manager = ConstraintManager()
    planner = ProjectPlanner(llm, constraint_manager)
    reporter = CommunicationReporter()

    project_goal = "Develop a new customer feedback portal with AI-powered sentiment analysis."
    initial_project = Project(goal=project_goal)
    initial_project.constraint_manager = constraint_manager

    constraint_manager.add_constraint(Constraint(type="deadline", value=datetime.now() + timedelta(days=60), unit="days"))
    constraint_manager.add_constraint(Constraint(type="budget", value=50000, unit="USD"))
    initial_project.constraints = constraint_manager.constraints

    reporter.post_update(f"Project initialized: '{initial_project.goal}'")
    reporter.post_update(f"Initial constraints: {', '.join([f'{c.type} {c.value}{c.unit or ''}' for c in initial_project.constraints])}")

    reporter.post_update("\n--- Decomposing Goal ---")
    decomposed_tasks = planner.decompose_goal(initial_project)
    for task in decomposed_tasks:
        initial_project.tasks[task.id] = task
    reporter.post_update(f"Decomposed into {len(decomposed_tasks)} tasks.")

    reporter.post_update("\n--- Generating Initial Plan ---")
    initial_project.plan = planner.generate_initial_plan(initial_project)
    reporter.post_update("Initial plan generated:")
    for i, task_id in enumerate(initial_project.plan):
        reporter.post_update(f"  {i+1}. {initial_project.tasks[task_id].description}")

    reporter.post_update("\n--- Initial Constraint Validation ---")
    violations = constraint_manager.validate_plan(initial_project)
    if violations:
        reporter.post_update("Initial plan has constraint violations:")
        for viol in violations:
            reporter.post_update(f"- {viol}")
    else:
        reporter.post_update("Initial plan passes constraint validation.")

    reporter.post_update("\n--- Introspecting Plan ---")
    introspection_feedback = planner.introspect_plan(initial_project)
    reporter.post_update(f"LLM Introspection: {introspection_feedback}")

    reporter.post_update("\n--- Starting Execution Loop ---")
    initial_project.status = "in-progress"
    monitor = FeedbackMonitor(initial_project)
    max_iterations = 20

    iteration = 0
    while initial_project.status == "in-progress" and iteration < max_iterations:
        iteration += 1
        reporter.post_update(f"\n[ITERATION {iteration}]")

        actionable_task = planner.get_next_actionable_task(initial_project)

        if actionable_task:
            reporter.post_update(f"Attempting to work on: '{actionable_task.description}'")
            actionable_task.status = "in-progress"
            feedback_message = monitor.simulate_task_completion(actionable_task)
            reporter.post_update(f"Feedback: {feedback_message}")

            if "BLOCKED" in feedback_message or "delayed" in feedback_message:
                reporter.post_update(f"--- Adapting Plan due to '{actionable_task.description}' being {actionable_task.status} ---")
                new_plan_ids = planner.adapt_plan(initial_project, feedback_message)
                if new_plan_ids != initial_project.plan:
                    initial_project.plan = new_plan_ids
                    reporter.post_update("Plan was adapted. New plan order:")
                    for i, task_id in enumerate(initial_project.plan):
                        reporter.post_update(f"  {i+1}. {initial_project.tasks[task_id].description}")
                else:
                    reporter.post_update("Plan was not significantly adapted.")

            violations = constraint_manager.validate_plan(initial_project)
            if violations:
                reporter.post_update("!!! Constraint Violations after adaptation !!!")
                for viol in violations:
                    reporter.post_update(f"- {viol}")

        else:
            all_done = all(task.status == "done" for task in initial_project.tasks.values())
            if all_done:
                reporter.post_update("All tasks completed!")
                initial_project.status = "completed"
            else:
                reporter.post_update("No actionable tasks currently available. Checking for blocked tasks...")
                blocked_tasks = [t for t in initial_project.tasks.values() if t.status == "blocked"]
                if blocked_tasks:
                    reporter.post_update(f"Found {len(blocked_tasks)} blocked tasks. LLM should intervene for resolution.")
                    initial_project.status = "halted"
                else:
                    reporter.post_update("No 'to-do' tasks with met dependencies. Project might be stuck or implicitly completed.")
                    initial_project.status = "completed"

    initial_project.end_date = datetime.now()
    reporter.post_update("\n--- Execution Loop Finished ---")
    reporter.post_update(monitor.get_project_status_summary())

    reporter.post_update("\n--- Generating Final Project Report ---")
    final_report = reporter.generate_report(initial_project)
    print(final_report)

    print("\n--- AI-Powered Project Manager Simulation Ended ---")

if __name__ == "__main__":
    run_project_manager_simulation()
