import uuid
from typing import Dict, List, Optional

class TaskManagementSystem:
    """A simplified in-memory task management system."""

    def __init__(self):
        self.tasks: Dict[str, Dict] = {}

    def create_task(self, name: str, description: str, dependencies: Optional[List[str]] = None) -> Dict:
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "name": name,
            "description": description,
            "status": "pending", # pending, in_progress, completed, blocked
            "dependencies": dependencies if dependencies is not None else [],
            "assigned_to": None,
            "priority": "medium"
        }
        self.tasks[task_id] = task
        print(f"Task created: {name} (ID: {task_id})")
        return task

    def update_task_status(self, task_id: str, status: str) -> Optional[Dict]:
        if task_id not in self.tasks:
            print(f"Error: Task with ID {task_id} not found.")
            return None
        
        valid_statuses = ["pending", "in_progress", "completed", "blocked"]
        if status not in valid_statuses:
            print(f"Error: Invalid status '{status}'. Must be one of {valid_statuses}.")
            return None

        self.tasks[task_id]["status"] = status
        print(f"Task {task_id} status updated to {status}")
        return self.tasks[task_id]

    def get_task(self, task_id: str) -> Optional[Dict]:
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[Dict]:
        return list(self.tasks.values())

    def add_dependency(self, task_id: str, dependency_task_id: str) -> bool:
        if task_id not in self.tasks or dependency_task_id not in self.tasks:
            print(f"Error: One or both tasks not found for dependency {task_id} -> {dependency_task_id}")
            return False
        if dependency_task_id not in self.tasks[task_id]["dependencies"]:
            self.tasks[task_id]["dependencies"].append(dependency_task_id)
            print(f"Added dependency: Task {task_id} depends on Task {dependency_task_id}")
            return True
        print(f"Dependency already exists: Task {task_id} depends on Task {dependency_task_id}")
        return False

    def remove_dependency(self, task_id: str, dependency_task_id: str) -> bool:
        if task_id not in self.tasks or dependency_task_id not in self.tasks:
            print(f"Error: One or both tasks not found for dependency {task_id} -> {dependency_task_id}")
            return False
        try:
            self.tasks[task_id]["dependencies"].remove(dependency_task_id)
            print(f"Removed dependency: Task {task_id} no longer depends on Task {dependency_task_id}")
            return True
        except ValueError:
            print(f"Dependency not found to remove: Task {task_id} does not depend on Task {dependency_task_id}")
            return False

    def __str__(self):
        if not self.tasks:
            return "No tasks in the system."
        output = "Current Tasks:\n"
        for task_id, task in self.tasks.items():
            output += f"  ID: {task_id}\n"
            output += f"    Name: {task['name']}\n"
            output += f"    Description: {task['description']}\n"
            output += f"    Status: {task['status']}\n"
            output += f"    Dependencies: {', '.join(task['dependencies']) if task['dependencies'] else 'None'}\n"
            output += f"    Assigned To: {task['assigned_to'] if task['assigned_to'] else 'Unassigned'}\n"
            output += f"    Priority: {task['priority']}\n"
            output += "---\n"
        return output
