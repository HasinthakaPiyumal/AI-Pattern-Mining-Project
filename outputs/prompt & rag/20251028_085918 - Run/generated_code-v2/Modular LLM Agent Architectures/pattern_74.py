import json

class ProjectPlanningAssistant:
    """
    An AI-powered Project Planning Assistant that manages cognitive load
    by operating in a two-stage mode: Information Collection and Planning & Optimization.
    """

    def __init__(self):
        self.project_data = {}

    def _get_team_availability(self):
        """
        Mocks fetching team members' availability from an external system.
        In a real application, this would involve API calls to HR systems or calendars.
        """
        print("\t[INFO] Querying team members' availability...")
        # Mock data: {team_member: [available_days_of_week]}
        return {"Alice": ["Mon", "Tue", "Wed", "Fri"], "Bob": ["Tue", "Wed", "Thu", "Sat"], "Charlie": ["Mon", "Wed", "Thu", "Fri"]}

    def _get_task_dependencies(self):
        """
        Mocks fetching task dependencies from a project management database.
        """
        print("\t[INFO] Fetching task dependencies from database...")
        # Mock data: {task: [list_of_dependencies]}
        return {
            "Task A": [],
            "Task B": ["Task A"],
            "Task C": ["Task A"],
            "Task D": ["Task B", "Task C"],
            "Task E": ["Task D"]
        }

    def _get_resource_constraints(self):
        """
        Mocks identifying project-specific resource constraints.
        """
        print("\t[INFO] Identifying resource constraints...")
        # Mock data: {constraint_name: value}
        return {"Max_Tasks_Per_Day_Per_Person": 1, "Max_Concurrent_Tasks": 3}

    def _get_public_holidays(self):
        """
        Mocks collecting public holiday calendars from an external service.
        """
        print("\t[INFO] Collecting public holiday calendars...")
        # Mock data: list of holiday dates (YYYY-MM-DD)
        return ["2023-12-25", "2024-01-01", "2024-03-29"]

    def information_collection_stage(self):
        """
        Stage 1: Focuses solely on gathering all necessary project data.
        """
        print("\n--- Stage 1: Information Collection ---")
        self.project_data["team_availability"] = self._get_team_availability()
        self.project_data["task_dependencies"] = self._get_task_dependencies()
        self.project_data["resource_constraints"] = self._get_resource_constraints()
        self.project_data["public_holidays"] = self._get_public_holidays()
        print("\t[SUCCESS] Information collection complete. Data collected:\n", json.dumps(self.project_data, indent=2))
        return self.project_data

    def planning_and_optimization_stage(self, collected_data):
        """
        Stage 2: Uses the gathered data to construct an optimized project schedule.
        """
        print("\n--- Stage 2: Planning and Optimization ---")
        team_availability = collected_data["team_availability"]
        task_dependencies = collected_data["task_dependencies"]
        resource_constraints = collected_data["resource_constraints"]
        public_holidays = collected_data["public_holidays"]

        schedule = {}
        current_day_idx = 0
        assigned_tasks = set()
        available_tasks = set(task_dependencies.keys())
        task_start_dates = {}

        print("\t[INFO] Generating project schedule based on collected data...")

        import datetime
        start_date = datetime.date(2023, 12, 18) # Arbitrary start date for demonstration

        while len(assigned_tasks) < len(available_tasks):
            current_date = start_date + datetime.timedelta(days=current_day_idx)
            current_date_str = current_date.strftime("%Y-%m-%d")
            day_of_week = current_date.strftime("%a")

            schedule[current_date_str] = {"tasks": [], "assigned_to": {}, "notes": []}

            if current_date_str in public_holidays:
                schedule[current_date_str]["notes"].append("Public Holiday - No work scheduled")
                current_day_idx += 1
                continue

            # Identify tasks that are ready to be scheduled
            ready_tasks = []
            for task, deps in task_dependencies.items():
                if task not in assigned_tasks and task not in [t for day_info in schedule.values() for t in day_info["tasks"]]: # Not yet assigned or currently in schedule
                    if all(dep in assigned_tasks for dep in deps):
                        ready_tasks.append(task)
            
            # Prioritize tasks if needed (e.g., shortest duration, critical path) - simple FIFO for this example
            ready_tasks.sort()

            tasks_assigned_today = 0
            available_team_members = list(team_availability.keys())
            available_team_members.sort()

            for task in ready_tasks:
                if tasks_assigned_today >= resource_constraints["Max_Concurrent_Tasks"]:
                    break

                assigned = False
                for member in available_team_members:
                    if day_of_week[:3] in team_availability[member]: # Check if member is available on this day of week
                        # Check if member already has a task for today (Max_Tasks_Per_Day_Per_Person)
                        if schedule[current_date_str]["assigned_to"].get(member) is None:
                            schedule[current_date_str]["tasks"].append(task)
                            schedule[current_date_str]["assigned_to"][member] = task
                            assigned_tasks.add(task)
                            task_start_dates[task] = current_date_str
                            tasks_assigned_today += 1
                            print(f"\t  [SCHEDULED] '{task}' assigned to {member} on {current_date_str}")
                            assigned = True
                            break
                if assigned:
                    break # Move to next day if a task was assigned to avoid over-assigning within one day (simplified)

            current_day_idx += 1

        print("\t[SUCCESS] Planning and optimization complete. Generated schedule:")
        return schedule

    def run_assistant(self):
        """
        Executes the two-stage project planning process.
        """
        print("\n===== Starting Project Planning Assistant =====")
        collected_data = self.information_collection_stage()
        final_schedule = self.planning_and_optimization_stage(collected_data)
        print("\n===== Project Planning Assistant Finished =====")
        return final_schedule

# Example Usage:
if __name__ == "__main__":
    assistant = ProjectPlanningAssistant()
    schedule = assistant.run_assistant()
    print("\nFinal Project Schedule:")
    print(json.dumps(schedule, indent=2))

    # You can also inspect specific parts of the collected data
    # print("\nCollected Team Availability:", assistant.project_data.get("team_availability"))
    # print("\nCollected Task Dependencies:", assistant.project_data.get("task_dependencies"))

