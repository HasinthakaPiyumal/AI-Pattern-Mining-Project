import networkx as nx
import datetime
from collections import deque

# --- 1. Data Structures (Simplified Data Storage) ---

class Task:
    def __init__(self, id, name, description, duration=0, dependencies=None, assigned_resources=None, status="Not Started", start_date=None, end_date=None):
        self.id = id
        self.name = name
        self.description = description
        self.duration = duration  # in days
        self.dependencies = dependencies if dependencies is not None else []  # List of Task IDs
        self.assigned_resources = assigned_resources if assigned_resources is not None else [] # List of Resource IDs
        self.status = status  # "Not Started", "In Progress", "Completed", "Blocked"
        self.start_date = start_date
        self.end_date = end_date

    def __repr__(self):
        return f"Task(ID: {self.id}, Name: {self.name}, Status: {self.status}, Duration: {self.duration} days)"

class Resource:
    def __init__(self, id, name, availability=1.0): # availability as a fraction of full-time (e.g., 1.0 for full-time)
        self.id = id
        self.name = name
        self.availability = availability

    def __repr__(self):
        return f"Resource(ID: {self.id}, Name: {self.name}, Availability: {self.availability})"

class ProjectPlan:
    def __init__(self, project_goal, initial_constraints):
        self.project_goal = project_goal
        self.initial_constraints = initial_constraints
        self.tasks = {}
        self.resources = {}
        self.critical_path = []
        self.estimated_start_date = None
        self.estimated_end_date = None

    def add_task(self, task):
        self.tasks[task.id] = task

    def add_resource(self, resource):
        self.resources[resource.id] = resource

    def __repr__(self):
        return f"ProjectPlan(Goal: {self.project_goal}, Tasks: {len(self.tasks)}, Resources: {len(self.resources)})"

# --- 2. Core AI Planning Engine (Mock LLM Integration) ---

class CoreAIPlanningEngine:
    def __init__(self, llm_model="MockLLM"):
        self.llm_model = llm_model

    def _mock_llm_response(self, prompt):
        # A simple mock for LLM responses based on keywords
        if "decompose" in prompt.lower() and "website" in prompt.lower():
            return {
                "tasks": [
                    {"id": "T1", "name": "Gather Requirements", "description": "Understand client needs and scope.", "duration": 3},
                    {"id": "T2", "name": "Design UI/UX", "description": "Create wireframes and mockups.", "duration": 5},
                    {"id": "T3", "name": "Develop Frontend", "description": "Code the user interface.", "duration": 10},
                    {"id": "T4", "name": "Develop Backend", "description": "Build server-side logic and database.", "duration": 12},
                    {"id": "T5", "name": "Database Setup", "description": "Configure and populate the database.", "duration": 4},
                    {"id": "T6", "name": "Integrate Frontend/Backend", "description": "Connect UI with API endpoints.", "duration": 7},
                    {"id": "T7", "name": "Testing and QA", "description": "Perform unit, integration, and user acceptance tests.", "duration": 6},
                    {"id": "T8", "name": "Deployment", "description": "Publish the website to production.", "duration": 2},
                    {"id": "T9", "name": "Client Review and Feedback", "description": "Present to client and gather feedback.", "duration": 3},
                ]
            }
        elif "dependencies" in prompt.lower():
            return {
                "dependencies": [
                    {"task_id": "T2", "depends_on": ["T1"]},
                    {"task_id": "T3", "depends_on": ["T2"]},
                    {"task_id": "T4", "depends_on": ["T1"]},
                    {"task_id": "T5", "depends_on": ["T4"]},
                    {"task_id": "T6", "depends_on": ["T3", "T4", "T5"]},
                    {"task_id": "T7", "depends_on": ["T6"]},
                    {"task_id": "T8", "depends_on": ["T7", "T9"]},
                    {"task_id": "T9", "depends_on": ["T7"]},
                ]
            }
        elif "resources" in prompt.lower():
            return {
                "resources": [
                    {"id": "R1", "name": "Project Manager", "availability": 0.5},
                    {"id": "R2", "name": "UI/UX Designer", "availability": 1.0},
                    {"id": "R3", "name": "Frontend Dev", "availability": 1.0},
                    {"id": "R4", "name": "Backend Dev", "availability": 1.0},
                    {"id": "R5", "name": "QA Engineer", "availability": 1.0},
                ],
                "assignments": {
                    "T1": ["R1"],
                    "T2": ["R2"],
                    "T3": ["R3"],
                    "T4": ["R4"],
                    "T5": ["R4"],
                    "T6": ["R3", "R4"],
                    "T7": ["R5"],
                    "T8": ["R1"],
                    "T9": ["R1"],
                }
            }
        elif "re-plan" in prompt.lower():
             # Simplified re-planning: just adjust duration for a specific task if it's delayed
            if "T3" in prompt: # If task T3 was delayed
                print("\n--- LLM suggested re-planning: Adjusting T3 duration due to delay ---")
                return {"action": "update_task", "task_id": "T3", "new_duration": 15}
            return {"action": "no_change"}
        return {}

    def decompose_goal_into_tasks(self, project_goal):
        print(f"[LLM] Decomposing project goal: '{project_goal}'")
        response = self._mock_llm_response(f"decompose the project goal '{project_goal}' into detailed tasks, durations, and descriptions.")
        tasks = []
        if "tasks" in response:
            for t_data in response["tasks"]:
                tasks.append(Task(**t_data))
        return tasks

    def identify_dependencies(self, tasks):
        print("[LLM] Identifying task dependencies...")
        task_names = ", ".join([t.name for t in tasks])
        response = self._mock_llm_response(f"identify dependencies between the following tasks: {task_names}")
        dependencies = [] # List of tuples (task_id, depends_on_task_id)
        if "dependencies" in response:
            for dep_data in response["dependencies"]:
                for depends_on_id in dep_data["depends_on"]:
                    dependencies.append((dep_data["task_id"], depends_on_id))
        return dependencies

    def estimate_timelines_resources(self, project_goal, tasks):
        print("[LLM] Estimating resources and initial assignments...")
        task_info = "; ".join([f"{t.id}: {t.name} ({t.duration} days)" for t in tasks])
        response = self._mock_llm_response(f"estimate resources needed and initial assignments for project '{project_goal}' with tasks: {task_info}")
        
        resources = []
        if "resources" in response:
            for r_data in response["resources"]:
                resources.append(Resource(**r_data))
        
        assignments = response.get("assignments", {})
        
        return resources, assignments

    def generate_recommendations(self, project_plan, issue_description):
        print(f"[LLM] Generating recommendations for issue: {issue_description}")
        # In a real scenario, this would use a more sophisticated prompt
        response = self._mock_llm_response(f"Given the project plan '{project_plan.project_goal}' and the issue '{issue_description}', suggest actions to re-plan or mitigate.")
        return response

# --- 3. Planning & Optimization Module ---

class PlanningOptimizationModule:
    def build_task_graph(self, tasks, dependencies):
        graph = nx.DiGraph()
        for task_id in tasks.keys():
            graph.add_node(task_id, duration=tasks[task_id].duration)
        for task_id, depends_on_id in dependencies:
            if task_id in tasks and depends_on_id in tasks:
                graph.add_edge(depends_on_id, task_id) # Edge from dependency to dependent task
        return graph

    def calculate_critical_path(self, graph, tasks):
        try:
            # Calculate earliest start/finish times
            earliest_start = {node: 0 for node in graph.nodes}
            earliest_finish = {node: tasks[node].duration for node in graph.nodes}

            for node in nx.topological_sort(graph):
                for successor in graph.successors(node):
                    earliest_start[successor] = max(earliest_start[successor], earliest_finish[node])
                    earliest_finish[successor] = earliest_start[successor] + tasks[successor].duration
            
            project_duration = max(earliest_finish.values()) if earliest_finish else 0

            # Calculate latest start/finish times (for backward pass)
            latest_finish = {node: project_duration for node in graph.nodes}
            latest_start = {node: project_duration - tasks[node].duration for node in graph.nodes}

            # Iterate in reverse topological order
            reverse_topo_order = list(nx.topological_sort(graph))
            reverse_topo_order.reverse()
            
            for node in reverse_topo_order:
                for predecessor in graph.predecessors(node):
                    latest_finish[predecessor] = min(latest_finish[predecessor], latest_start[node])
                    latest_start[predecessor] = latest_finish[predecessor] - tasks[predecessor].duration

            critical_path_tasks = []
            for node in graph.nodes:
                if earliest_start[node] == latest_start[node] and earliest_finish[node] == latest_finish[node]:
                    critical_path_tasks.append(node)
            
            # Sort critical path tasks based on their earliest start for a more coherent display
            critical_path_tasks.sort(key=lambda task_id: earliest_start[task_id])

            return critical_path_tasks, project_duration

        except nx.NetworkXNoCycle:
            print("Error: The task graph contains a cycle, critical path cannot be calculated.")
            return [], 0
        except Exception as e:
            print(f"An error occurred during critical path calculation: {e}")
            return [], 0

    def allocate_resources(self, project_plan, assignments):
        print("\n[Optimizer] Allocating resources...")
        # Simple allocation based on LLM suggestions
        for task_id, resource_ids in assignments.items():
            if task_id in project_plan.tasks:
                project_plan.tasks[task_id].assigned_resources = resource_ids
                for res_id in resource_ids:
                    if res_id not in project_plan.resources:
                        # Add resource if it doesn't exist (basic handling for mock scenario)
                        print(f"Warning: Resource {res_id} not found in project_plan. Adding mock resource.")
                        project_plan.add_resource(Resource(res_id, f"Mock Resource {res_id}", availability=1.0))
        return project_plan

    def solve_constraints(self, project_plan):
        print("\n[Optimizer] Checking project constraints...")
        # Example: Check deadline constraint
        deadline_str = project_plan.initial_constraints.get("deadline")
        if deadline_str and project_plan.estimated_end_date:
            deadline = datetime.datetime.strptime(deadline_str, "%Y-%m-%d").date()
            if project_plan.estimated_end_date > deadline:
                print(f"ALERT: Project estimated end date ({project_plan.estimated_end_date}) exceeds deadline ({deadline}).")
                return False
        print("Constraints checked. Plan looks feasible (based on current simple checks).")
        return True

# --- 4. Monitoring & Adaptation Module ---

class MonitoringAdaptationModule:
    def track_progress(self, project_plan, current_date, task_updates=None):
        print(f"\n[Monitor] Tracking progress as of {current_date.strftime('%Y-%m-%d')}...")
        if task_updates:
            for task_id, status in task_updates.items():
                if task_id in project_plan.tasks:
                    project_plan.tasks[task_id].status = status
                    if status == "Completed":
                        project_plan.tasks[task_id].end_date = current_date # Simulate completion date
                    print(f"  - Task {task_id}: Status updated to '{status}'")
        return project_plan

    def detect_deviations(self, project_plan, current_date):
        print("[Monitor] Detecting deviations...")
        deviations = []
        # Simple deviation: any 'In Progress' task that should have completed by now based on initial plan
        for task_id, task in project_plan.tasks.items():
            if task.status == "In Progress" and task.estimated_end_date and current_date > task.estimated_end_date:
                deviations.append(f"Task {task.name} ({task.id}) is 'In Progress' but should have completed by {task.estimated_end_date.strftime('%Y-%m-%d')}.")
            elif task.status == "Not Started" and task.estimated_start_date and current_date > task.estimated_start_date + datetime.timedelta(days=task.duration):
                deviations.append(f"Task {task.name} ({task.id}) has not started and is significantly delayed.")

        if deviations:
            print("  --- DEVIATION DETECTED! ---")
            for dev in deviations:
                print(f"    - {dev}")
            return {"has_deviation": True, "description": "; ".join(deviations)}
        print("  No significant deviations detected.")
        return {"has_deviation": False}

    def adapt_plan(self, project_plan, deviation_info, ai_engine, optimizer):
        if deviation_info.get("has_deviation"):
            print("\n[Monitor] Triggering re-planning due to deviations.")
            recommendations = ai_engine.generate_recommendations(project_plan, deviation_info["description"])

            if recommendations.get("action") == "update_task":
                task_id = recommendations["task_id"]
                new_duration = recommendations["new_duration"]
                if task_id in project_plan.tasks:
                    print(f"Applying recommendation: Updating duration of {project_plan.tasks[task_id].name} to {new_duration} days.")
                    project_plan.tasks[task_id].duration = new_duration
                    # Recalculate critical path and dates after duration change
                    project_plan = self._recalculate_plan_dates(project_plan, optimizer)
                else:
                    print(f"Warning: Task {task_id} not found for update.")
            else:
                print("No specific actionable recommendation from LLM for adaptation. Manual intervention might be needed.")
            return True # Plan adapted
        return False # No adaptation needed

    def _recalculate_plan_dates(self, project_plan, optimizer):
        # Helper to re-calculate project dates after a change
        graph = optimizer.build_task_graph(project_plan.tasks, [(t_id, dep_id) for t_id, task in project_plan.tasks.items() for dep_id in task.dependencies])
        critical_path_ids, project_duration = optimizer.calculate_critical_path(graph, project_plan.tasks)
        project_plan.critical_path = critical_path_ids
        
        # Update estimated project end date
        if project_plan.estimated_start_date:
            project_plan.estimated_end_date = project_plan.estimated_start_date + datetime.timedelta(days=project_duration)
        
        # Update individual task dates (simplified, based on earliest start/finish)
        earliest_start = {node: 0 for node in graph.nodes}
        earliest_finish = {node: project_plan.tasks[node].duration for node in graph.nodes}
        for node in nx.topological_sort(graph):
            for successor in graph.successors(node):
                earliest_start[successor] = max(earliest_start[successor], earliest_finish[node])
                earliest_finish[successor] = earliest_start[successor] + project_plan.tasks[successor].duration
        
        for task_id, task in project_plan.tasks.items():
            task.estimated_start_date = project_plan.estimated_start_date + datetime.timedelta(days=earliest_start.get(task_id, 0))
            task.estimated_end_date = project_plan.estimated_start_date + datetime.timedelta(days=earliest_finish.get(task_id, 0))

        print("  Project plan dates and critical path re-calculated.")
        return project_plan

# --- 5. Adaptive Project Planner (Main Orchestrator) ---

class AdaptiveProjectPlanner:
    def __init__(self, project_goal, initial_constraints):
        self.project_plan = ProjectPlan(project_goal, initial_constraints)
        self.ai_engine = CoreAIPlanningEngine()
        self.optimizer = PlanningOptimizationModule()
        self.monitor = MonitoringAdaptationModule()

    def create_initial_plan(self, start_date=None):
        print("\n=== Creating Initial Project Plan ===")
        self.project_plan.estimated_start_date = start_date if start_date else datetime.date.today()

        # Step 1: LLM decomposes goal into tasks
        initial_tasks = self.ai_engine.decompose_goal_into_tasks(self.project_plan.project_goal)
        for task in initial_tasks:
            self.project_plan.add_task(task)

        # Step 2: LLM identifies dependencies
        dependencies_list = self.ai_engine.identify_dependencies(list(self.project_plan.tasks.values()))
        # Convert list of (task_id, depends_on_id) to update Task objects
        for task_id, depends_on_id in dependencies_list:
            if task_id in self.project_plan.tasks:
                self.project_plan.tasks[task_id].dependencies.append(depends_on_id)

        # Step 3: LLM estimates resources and initial assignments
        initial_resources, assignments = self.ai_engine.estimate_timelines_resources(self.project_plan.project_goal, list(self.project_plan.tasks.values()))
        for res in initial_resources:
            self.project_plan.add_resource(res)
        self.project_plan = self.optimizer.allocate_resources(self.project_plan, assignments)

        # Step 4: Build task graph and calculate critical path
        task_durations = {t_id: task.duration for t_id, task in self.project_plan.tasks.items()}
        graph = self.optimizer.build_task_graph(self.project_plan.tasks, dependencies_list)

        self.project_plan.critical_path, project_duration = self.optimizer.calculate_critical_path(graph, self.project_plan.tasks)
        self.project_plan.estimated_end_date = self.project_plan.estimated_start_date + datetime.timedelta(days=project_duration)

        # Assign estimated start/end dates to tasks (simplified, based on critical path logic)
        earliest_start = {node: 0 for node in graph.nodes}
        earliest_finish = {node: self.project_plan.tasks[node].duration for node in graph.nodes}

        for node in nx.topological_sort(graph):
            for successor in graph.successors(node):
                earliest_start[successor] = max(earliest_start[successor], earliest_finish[node])
                earliest_finish[successor] = earliest_start[successor] + self.project_plan.tasks[successor].duration
        
        for task_id, task in self.project_plan.tasks.items():
            task.estimated_start_date = self.project_plan.estimated_start_date + datetime.timedelta(days=earliest_start.get(task_id, 0))
            task.estimated_end_date = self.project_plan.estimated_start_date + datetime.timedelta(days=earliest_finish.get(task_id, 0))


        # Step 5: Solve constraints
        self.optimizer.solve_constraints(self.project_plan)

        print("\nInitial Project Plan Created:")
        print(f"  Goal: {self.project_plan.project_goal}")
        print(f"  Estimated Start: {self.project_plan.estimated_start_date.strftime('%Y-%m-%d')}")
        print(f"  Estimated End: {self.project_plan.estimated_end_date.strftime('%Y-%m-%d')}")
        print(f"  Total Tasks: {len(self.project_plan.tasks)}")
        print(f"  Critical Path: {[self.project_plan.tasks[t_id].name for t_id in self.project_plan.critical_path]}")
        print("  Tasks Details:")
        for task_id in sorted(self.project_plan.tasks.keys()):
            task = self.project_plan.tasks[task_id]
            dependencies_names = [self.project_plan.tasks[dep_id].name for dep_id in task.dependencies if dep_id in self.project_plan.tasks]
            resources_names = [self.project_plan.resources[res_id].name for res_id in task.assigned_resources if res_id in self.project_plan.resources]
            print(f"    - {task.id}: {task.name} (Dur: {task.duration} days, Status: {task.status})\n      Starts: {task.estimated_start_date.strftime('%Y-%m-%d') if task.estimated_start_date else 'N/A'}, Ends: {task.estimated_end_date.strftime('%Y-%m-%d') if task.estimated_end_date else 'N/A'}\n      Deps: {', '.join(dependencies_names) or 'None'}, Res: {', '.join(resources_names) or 'None'}")

    def monitor_and_adapt(self, current_date, task_updates=None):
        print(f"\n=== Monitoring and Adaptation Cycle ({current_date.strftime('%Y-%m-%d')}) ===")
        self.project_plan = self.monitor.track_progress(self.project_plan, current_date, task_updates)
        deviation_info = self.monitor.detect_deviations(self.project_plan, current_date)
        
        if self.monitor.adapt_plan(self.project_plan, deviation_info, self.ai_engine, self.optimizer):
            print("\n--- Project Plan AFTER Adaptation ---")
            print(f"  Goal: {self.project_plan.project_goal}")
            print(f"  Estimated Start: {self.project_plan.estimated_start_date.strftime('%Y-%m-%d')}")
            print(f"  Estimated End: {self.project_plan.estimated_end_date.strftime('%Y-%m-%d')}")
            print(f"  Critical Path: {[self.project_plan.tasks[t_id].name for t_id in self.project_plan.critical_path]}")
            print("  Tasks Details:")
            for task_id in sorted(self.project_plan.tasks.keys()):
                task = self.project_plan.tasks[task_id]
                dependencies_names = [self.project_plan.tasks[dep_id].name for dep_id in task.dependencies if dep_id in self.project_plan.tasks]
                resources_names = [self.project_plan.resources[res_id].name for res_id in task.assigned_resources if res_id in self.project_plan.resources]
                print(f"    - {task.id}: {task.name} (Dur: {task.duration} days, Status: {task.status})\n      Starts: {task.estimated_start_date.strftime('%Y-%m-%d') if task.estimated_start_date else 'N/A'}, Ends: {task.estimated_end_date.strftime('%Y-%m-%d') if task.estimated_end_date else 'N/A'}\n      Deps: {', '.join(dependencies_names) or 'None'}, Res: {', '.join(resources_names) or 'None'}")


# --- Main Demonstration --- 
if __name__ == "__main__":
    project_goal = "Develop and Deploy a New E-commerce Website"
    initial_constraints = {"budget": "$50,000", "deadline": "2024-09-30", "team_size": 5}

    planner = AdaptiveProjectPlanner(project_goal, initial_constraints)
    
    # Simulate project start today
    project_start_date = datetime.date.today()
    planner.create_initial_plan(start_date=project_start_date)

    print("\n\n=========== Simulating Project Execution & Adaptation ===========")

    # Scenario 1: Simulate some tasks completing on time
    print("\n--- Week 1 Progress ---")
    current_date = project_start_date + datetime.timedelta(days=7)
    planner.monitor_and_adapt(current_date, task_updates={
        "T1": "Completed",
        "T2": "In Progress"
    })

    # Scenario 2: Simulate another week, T2 completes, but T3 (Frontend Dev) gets delayed
    print("\n--- Week 2 Progress ---")
    current_date = project_start_date + datetime.timedelta(days=14)
    planner.monitor_and_adapt(current_date, task_updates={
        "T2": "Completed",
        "T3": "In Progress",
        "T4": "In Progress"
    })

    # Scenario 3: T3 is significantly delayed, LLM suggests duration change
    print("\n--- Week 3 Progress (T3 is delayed) ---")
    current_date = project_start_date + datetime.timedelta(days=21) # T3 should have finished around day 18-20 based on initial 10 days duration
    planner.monitor_and_adapt(current_date, task_updates={
        "T3": "In Progress", # Still in progress
        "T4": "In Progress" # Still in progress
    })

    # Scenario 4: Acknowledge adaptation and continue, T4 completes, T5 starts
    print("\n--- Week 4 Progress (After T3 duration adjustment) ---")
    current_date = project_start_date + datetime.timedelta(days=28) 
    planner.monitor_and_adapt(current_date, task_updates={
        "T4": "Completed",
        "T5": "In Progress",
        "T3": "In Progress" # T3 is still ongoing with new longer duration
    })

    print("\n\n=========== End of Demonstration ===========")
