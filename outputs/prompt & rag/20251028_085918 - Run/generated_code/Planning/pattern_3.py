import networkx as nx
from datetime import datetime, timedelta

class ProjectManagerAI:
    def __init__(self, llm_model_name="simulated-llm"):
        self.llm_model_name = llm_model_name
        self.project_plan = {}
        self.task_graph = nx.DiGraph()

    def _simulate_llm_response(self, prompt, context=""):
        """
        Simulates an LLM's response based on the prompt.
        In a real application, this would involve calling an actual LLM API.
        """
        if "decompose" in prompt.lower():
            if "build a website" in context.lower():
                return {
                    "task": "Build a Website",
                    "sub_tasks": [
                        {"id": "t1", "name": "Define Website Requirements", "duration": 2, "dependencies": []},
                        {"id": "t2", "name": "Design UI/UX", "duration": 3, "dependencies": ["t1"]},
                        {"id": "t3", "name": "Develop Frontend", "duration": 5, "dependencies": ["t2"]},
                        {"id": "t4", "name": "Develop Backend", "duration": 6, "dependencies": ["t2"]},
                        {"id": "t5", "name": "Database Setup", "duration": 4, "dependencies": ["t4"]},
                        {"id": "t6", "name": "Integrate Frontend & Backend", "duration": 3, "dependencies": ["t3", "t5"]},
                        {"id": "t7", "name": "Testing & Debugging", "duration": 4, "dependencies": ["t6"]},
                        {"id": "t8", "name": "Deployment", "duration": 2, "dependencies": ["t7"]},
                        {"id": "t9", "name": "Launch & Monitoring", "duration": 1, "dependencies": ["t8"]},
                    ]
                }
            elif "organize a conference" in context.lower():
                return {
                    "task": "Organize a Conference",
                    "sub_tasks": [
                        {"id": "c1", "name": "Define Conference Theme & Scope", "duration": 3, "dependencies": []},
                        {"id": "c2", "name": "Budget Planning", "duration": 2, "dependencies": ["c1"]},
                        {"id": "c3", "name": "Venue Selection", "duration": 5, "dependencies": ["c2"]},
                        {"id": "c4", "name": "Speaker Outreach", "duration": 8, "dependencies": ["c1"]},
                        {"id": "c5", "name": "Marketing & Registration", "duration": 7, "dependencies": ["c2", "c4"]},
                        {"id": "c6", "name": "Logistics & Setup", "duration": 4, "dependencies": ["c3", "c5"]},
                        {"id": "c7", "name": "Event Execution", "duration": 2, "dependencies": ["c6"]},
                        {"id": "c8", "name": "Post-Conference Follow-up", "duration": 3, "dependencies": ["c7"]},
                    ]
                }
        elif "constraints" in prompt.lower() or "optimize" in prompt.lower():
             if "deadline" in context.lower() and "website" in context.lower():
                 return {
                    "optimization_suggestions": [
                        "Consider parallelizing frontend and backend development if resources allow.",
                        "Prioritize critical features for MVP release if deadline is tight.",
                        "Allocate more testing resources to catch bugs early."
                    ]
                 }
             elif "budget" in context.lower() and "conference" in context.lower():
                 return {
                    "optimization_suggestions": [
                        "Look for sponsorship opportunities to offset costs.",
                        "Negotiate with venues for better rates.",
                        "Utilize free marketing channels where possible."
                    ]
                 }
        elif "adapt" in prompt.lower():
            if "t3" in context.lower() and "delay" in context.lower(): # Develop Frontend
                return {
                    "adaptation_action": "Reschedule dependent tasks (t6, t7, t8, t9). Assess impact on overall deadline.",
                    "new_durations": {"t3": 7} # Simulate actual delay
                }
            elif "c4" in context.lower() and "speaker declined" in context.lower(): # Speaker Outreach
                 return {
                    "adaptation_action": "Identify alternative speakers for the same topic. Extend Speaker Outreach task duration if needed.",
                    "new_durations": {"c4": 10}
                 }
        return {}


    def decompose_task(self, high_level_goal: str) -> dict:
        """
        Decomposes a high-level project goal into smaller, manageable sub-tasks.
        Leverages LLM for intelligent task breakdown and initial dependency identification.
        """
        print(f"AI: Decomposing '{high_level_goal}'...")
        prompt = f"Decompose the project goal '{high_level_goal}' into a list of detailed sub-tasks, including estimated durations (in days) and their direct dependencies. Provide the output in a structured JSON format."
        llm_output = self._simulate_llm_response(prompt, context=high_level_goal)

        if not llm_output or "sub_tasks" not in llm_output:
            print("AI: Could not decompose task effectively. Please refine the goal.")
            return {"main_task": high_level_goal, "sub_tasks": []}

        self.project_plan["main_task"] = high_level_goal
        self.project_plan["sub_tasks"] = llm_output["sub_tasks"]

        # Build the task graph
        self.task_graph.clear()
        for task in self.project_plan["sub_tasks"]:
            self.task_graph.add_node(task["id"], name=task["name"], duration=task["duration"])
            for dep_id in task.get("dependencies", []):
                self.task_graph.add_edge(dep_id, task["id"])

        print(f"AI: Decomposed '{high_level_goal}' into {len(self.project_plan['sub_tasks'])} sub-tasks.")
        return self.project_plan

    def generate_plan(self, start_date: datetime, constraints: dict = None) -> dict:
        """
        Generates a structured project plan including start/end dates for each task,
        considering dependencies and optional constraints (e.g., deadlines, resource availability).
        """
        if not self.project_plan or not self.project_plan.get("sub_tasks"):
            print("AI: No sub-tasks defined. Please decompose a task first.")
            return {}

        print(f"AI: Generating plan starting from {start_date.strftime('%Y-%m-%d')} with constraints: {constraints}...")

        # Calculate earliest start and end dates for each task
        task_schedules = {}
        for task_id in nx.topological_sort(self.task_graph):
            task_data = self.task_graph.nodes[task_id]
            earliest_start = start_date

            # Consider dependencies
            for pred_id in self.task_graph.predecessors(task_id):
                if pred_id in task_schedules:
                    earliest_start = max(earliest_start, task_schedules[pred_id]["end_date"])

            end_date = earliest_start + timedelta(days=task_data["duration"])
            task_schedules[task_id] = {
                "name": task_data["name"],
                "start_date": earliest_start,
                "end_date": end_date,
                "duration": task_data["duration"],
                "dependencies": list(self.task_graph.predecessors(task_id))
            }

        self.project_plan["schedule"] = task_schedules

        # Apply constraints and optimize (simulated with LLM)
        if constraints:
            prompt = f"Given the current plan {task_schedules} and constraints {constraints}, suggest optimizations to meet these constraints. Consider resource allocation, parallelization, and critical path adjustments."
            optimization_output = self._simulate_llm_response(prompt, context=f"project: {self.project_plan['main_task']}, constraints: {constraints}")
            if optimization_output and "optimization_suggestions" in optimization_output:
                self.project_plan["optimization_suggestions"] = optimization_output["optimization_suggestions"]
                print("AI: Optimization suggestions generated based on constraints.")

        print("AI: Project plan generated successfully.")
        return self.project_plan


    def adapt_plan(self, task_id: str, feedback: str) -> dict:
        """
        Dynamically adapts the project plan based on real-time feedback or changes.
        Uses LLM to understand feedback and propose modifications to tasks, dependencies, or schedule.
        """
        if task_id not in self.task_graph.nodes:
            print(f"AI: Task ID '{task_id}' not found in the plan.")
            return {}

        print(f"AI: Adapting plan for task '{self.task_graph.nodes[task_id]['name']}' with feedback: '{feedback}'...")

        prompt = f"The project task '{self.task_graph.nodes[task_id]['name']}' (ID: {task_id}) encountered the following feedback: '{feedback}'. Propose adaptation actions, including potential changes to task durations, dependencies, or rescheduling. Consider the impact on subsequent tasks."
        adaptation_output = self._simulate_llm_response(prompt, context=f"task: {task_id}, feedback: {feedback}, current_plan: {self.project_plan['schedule']}")

        if adaptation_output and "adaptation_action" in adaptation_output:
            print(f"AI: Adaptation proposed: {adaptation_output['adaptation_action']}")
            # Apply changes (simulated for demonstration)
            if "new_durations" in adaptation_output:
                for tid, new_duration in adaptation_output["new_durations"].items():
                    if tid in self.task_graph.nodes:
                        old_duration = self.task_graph.nodes[tid]["duration"]
                        self.task_graph.nodes[tid]["duration"] = new_duration
                        print(f"    - Task '{self.task_graph.nodes[tid]['name']}' duration changed from {old_duration} to {new_duration} days.")

            # Re-generate the schedule to reflect changes
            # We use the start_date of the first task in the original schedule as the base for rescheduling.
            if self.project_plan.get("schedule"):
                first_task_id = list(self.project_plan["schedule"].keys())[0]
                original_start_date = self.project_plan["schedule"][first_task_id]["start_date"]
                self.generate_plan(original_start_date)
            else:
                print("AI: Cannot re-generate plan; no initial schedule found.")

        print("AI: Plan adaptation complete.")
        return self.project_plan

    def get_current_plan(self) -> dict:
        """Returns the current state of the project plan."""
        return self.project_plan