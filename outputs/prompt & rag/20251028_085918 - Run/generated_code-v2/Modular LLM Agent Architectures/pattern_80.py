import time
import random
from datetime import datetime, timedelta

class ProjectManagementAssistant:
    def __init__(self, project_name):
        self.project_name = project_name
        self.collected_data = {}
        self.project_plan = {}

    def _fetch_jira_tasks(self):
        print(f"[{datetime.now()}] Stage 1: Collecting Jira task data...")
        time.sleep(1) # Simulate API call delay
        tasks = {
            "task_A": {"status": random.choice(["To Do", "In Progress", "Done"]), "assignee": "Alice", "due_date": (datetime.now() + timedelta(days=random.randint(1, 10))).strftime("%Y-%m-%d")},
            "task_B": {"status": random.choice(["To Do", "In Progress", "Done"]), "assignee": "Bob", "due_date": (datetime.now() + timedelta(days=random.randint(5, 15))).strftime("%Y-%m-%d")},
            "task_C": {"status": random.choice(["To Do", "In Progress", "Done"]), "assignee": "Alice", "due_date": (datetime.now() + timedelta(days=random.randint(2, 8))).strftime("%Y-%m-%d")},
        }
        print(f"[{datetime.now()}] Jira data collected.")
        return tasks

    def _fetch_calendar_availability(self):
        print(f"[{datetime.now()}] Stage 1: Collecting team calendar availability...")
        time.sleep(0.8) # Simulate API call delay
        availability = {
            "Alice": ["Mon", "Wed", "Fri"],
            "Bob": ["Tue", "Thu", "Fri"],
            "Charlie": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        }
        print(f"[{datetime.now()}] Calendar data collected.")
        return availability

    def _fetch_external_dependencies(self):
        print(f"[{datetime.now()}] Stage 1: Checking external dependencies...")
        time.sleep(1.2) # Simulate API call delay
        dependencies = {
            "task_A": [],
            "task_B": ["external_service_ready"],
            "task_C": ["task_A"],
        }
        print(f"[{datetime.now()}] External dependency data collected.")
        return dependencies

    def _fetch_historical_data(self):
        print(f"[{datetime.now()}] Stage 1: Retrieving historical project data...")
        time.sleep(0.7) # Simulate database query delay
        historical_completion_times = {
            "Alice": {"average_task_days": 3},
            "Bob": {"average_task_days": 4},
            "Charlie": {"average_task_days": 2},
        }
        print(f"[{datetime.now()}] Historical data collected.")
        return historical_completion_times

    def information_collection_stage(self):
        print("\n--- Entering Information Collection Stage ---")
        self.collected_data["jira_tasks"] = self._fetch_jira_tasks()
        self.collected_data["calendar_availability"] = self._fetch_calendar_availability()
        self.collected_data["external_dependencies"] = self._fetch_external_dependencies()
        self.collected_data["historical_data"] = self._fetch_historical_data()
        print("--- Information Collection Stage Complete ---")
        return self.collected_data

    def _synthesize_data(self, data):
        print(f"[{datetime.now()}] Stage 2: Synthesizing collected data...")
        synthesized = {
            "tasks_overview": {},
            "team_load": {},
            "critical_path_candidates": []
        }

        # Process tasks
        for task_name, details in data["jira_tasks"].items():
            synthesized["tasks_overview"][task_name] = {
                "status": details["status"],
                "assignee": details["assignee"],
                "due_date": details["due_date"],
                "dependencies": data["external_dependencies"].get(task_name, [])
            }
            # Simple load estimation
            synthesized["team_load"][details["assignee"]] = synthesized["team_load"].get(details["assignee"], 0) + 1

            # Identify potential critical path tasks (very basic)
            if "external_service_ready" in details["dependencies"] or details["status"] == "To Do":
                synthesized["critical_path_candidates"].append(task_name)

        print(f"[{datetime.now()}] Data synthesis complete.")
        return synthesized

    def planning_and_optimization_stage(self):
        print("\n--- Entering Planning and Optimization Stage ---")
        if not self.collected_data:
            print("No data collected. Please run information_collection_stage first.")
            return {}

        synthesized_data = self._synthesize_data(self.collected_data)
        print(f"[{datetime.now()}] Stage 2: Generating project plan and optimizing...")
        time.sleep(2) # Simulate complex planning logic

        plan = {
            "project_name": self.project_name,
            "status_summary": {
                "total_tasks": len(synthesized_data["tasks_overview"]),
                "to_do": sum(1 for t in synthesized_data["tasks_overview"].values() if t["status"] == "To Do"),
                "in_progress": sum(1 for t in synthesized_data["tasks_overview"].values() if t["status"] == "In Progress"),
                "done": sum(1 for t in synthesized_data["tasks_overview"].values() if t["status"] == "Done"),
            },
            "task_assignments": {},
            "bottlenecks_identified": synthesized_data["critical_path_candidates"],
            "suggested_adjustments": []
        }

        # Simple assignment based on current (simulated) load
        for task_name, details in synthesized_data["tasks_overview"].items():
            plan["task_assignments"][task_name] = details["assignee"]

        # Example of a simple adjustment
        if plan["status_summary"]["to_do"] > 1 and synthesized_data["team_load"].get("Alice", 0) > 1:
            plan["suggested_adjustments"].append("Consider re-assigning one 'To Do' task from Alice to Charlie for better load balancing.")

        self.project_plan = plan
        print(f"[{datetime.now()}] Project plan generated and optimized.\n")
        print("--- Planning and Optimization Stage Complete ---")
        return self.project_plan

    def run_full_cycle(self):
        print(f"--- Starting AI Project Manager for '{self.project_name}' ---")
        self.information_collection_stage()
        self.planning_and_optimization_stage()
        print("--- AI Project Manager Cycle Complete ---")
        return self.project_plan

if __name__ == "__main__":
    manager = ProjectManagementAssistant("Website Relaunch Project")
    final_plan = manager.run_full_cycle()

    print("\n--- Final Project Plan Summary ---")
    print(f"Project: {final_plan.get('project_name', 'N/A')}")
    print(f"Status: {final_plan.get('status_summary', 'N/A')}")
    print(f"Task Assignments: {final_plan.get('task_assignments', 'N/A')}")
    print(f"Bottlenecks: {final_plan.get('bottlenecks_identified', 'N/A')}")
    print(f"Suggested Adjustments: {final_plan.get('suggested_adjustments', 'N/A')}")
