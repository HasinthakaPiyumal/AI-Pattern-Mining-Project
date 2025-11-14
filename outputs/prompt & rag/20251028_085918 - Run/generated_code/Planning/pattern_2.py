import gradio as gr
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
import networkx as nx
import time

# 1. Pydantic Models for structured data

class Task(BaseModel):
    id: str = Field(..., description="Unique identifier for the task")
    name: str = Field(..., description="Name of the task")
    description: str = Field(..., description="Detailed description of the task")
    estimated_duration: int = Field(..., description="Estimated duration in hours")
    dependencies: List[str] = Field(default_factory=list, description="List of task IDs this task depends on")
    status: str = Field("pending", description="Current status of the task (pending, in_progress, completed, failed)")

class Constraint(BaseModel):
    id: str = Field(..., description="Unique identifier for the constraint")
    type: str = Field(..., description="Type of constraint (e.g., HARD, COMMONSENSE, ENVIRONMENTAL)")
    description: str = Field(..., description="Description of the constraint")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details for the constraint")

class ProjectPlan(BaseModel):
    project_goal: str = Field(..., description="The overall goal of the project")
    tasks: List[Task] = Field(default_factory=list, description="List of decomposed tasks")
    constraints: List[Constraint] = Field(default_factory=list, description="List of applicable constraints")
    plan_steps: List[str] = Field(default_factory=list, description="Ordered steps of the plan")
    status: str = Field("draft", description="Status of the plan (draft, active, completed)")

# 2. LLM Simulation (Placeholder for Langchain integration)
# In a real application, this would use a Langchain LLMChain or Agent

class LLMSimulator:
    def __init__(self):
        pass

    def decompose_project_goal(self, project_goal: str) -> List[Dict[str, Any]]:
        print(f"[LLM Sim] Decomposing: {project_goal}")
        # Simulate LLM breaking down a project goal into sub-tasks
        # In a real scenario, this would involve a complex prompt to an LLM
        if "website" in project_goal.lower():
            return [
                {"id": "t1", "name": "Gather Requirements", "description": "Understand client needs and specifications", "estimated_duration": 8, "dependencies": []},
                {"id": "t2", "name": "Design UI/UX", "description": "Create wireframes and mockups", "estimated_duration": 16, "dependencies": ["t1"]},
                {"id": "t3", "name": "Develop Frontend", "description": "Code the user interface", "estimated_duration": 40, "dependencies": ["t2"]},
                {"id": "t4", "name": "Develop Backend", "description": "Implement server-side logic and database", "estimated_duration": 30, "dependencies": ["t1"]},
                {"id": "t5", "name": "Integrate Frontend/Backend", "description": "Connect UI with API", "estimated_duration": 10, "dependencies": ["t3", "t4"]},
                {"id": "t6", "name": "Testing", "description": "Perform unit, integration, and user acceptance testing", "estimated_duration": 20, "dependencies": ["t5"]},
                {"id": "t7", "name": "Deployment", "description": "Deploy the website to production server", "estimated_duration": 5, "dependencies": ["t6"]},
            ]
        elif "mobile app" in project_goal.lower():
             return [
                {"id": "ma1", "name": "Market Research", "description": "Analyze competitor apps and user needs", "estimated_duration": 12, "dependencies": []},
                {"id": "ma2", "name": "App Design", "description": "Create app flows and UI mockups", "estimated_duration": 24, "dependencies": ["ma1"]},
                {"id": "ma3", "name": "Frontend Development", "description": "Build iOS and Android interfaces", "estimated_duration": 60, "dependencies": ["ma2"]},
                {"id": "ma4", "name": "API Development", "description": "Create backend APIs", "estimated_duration": 40, "dependencies": ["ma1"]},
                {"id": "ma5", "name": "Database Setup", "description": "Configure cloud database", "estimated_duration": 15, "dependencies": ["ma4"]},
                {"id": "ma6", "name": "Integration & Testing", "description": "Connect app to API and perform tests", "estimated_duration": 25, "dependencies": ["ma3", "ma5"]},
                {"id": "ma7", "name": "App Store Submission", "description": "Prepare for release on Apple App Store and Google Play", "estimated_duration": 8, "dependencies": ["ma6"]},
            ]
        else:
            return [
                {"id": "g1", "name": "Initial Brainstorming", "description": "Generate ideas for the project", "estimated_duration": 4, "dependencies": []},
                {"id": "g2", "name": "Resource Allocation", "description": "Assign team members and resources", "estimated_duration": 6, "dependencies": ["g1"]},
                {"id": "g3", "name": "Execution Phase", "description": "Carry out core project activities", "estimated_duration": 20, "dependencies": ["g2"]},
                {"id": "g4", "name": "Review and Refine", "description": "Evaluate results and make adjustments", "estimated_duration": 8, "dependencies": ["g3"]},
            ]

    def generate_plan_steps(self, tasks: List[Task], constraints: List[Constraint]) -> List[str]:
        print(f"[LLM Sim] Generating plan for {len(tasks)} tasks with {len(constraints)} constraints")
        # Simulate LLM generating a plan from tasks and constraints
        # This would be a topological sort + heuristic based ordering in a real LLM application
        sorted_task_names = [task.name for task in tasks]
        
        # Apply a very basic topological sort if networkx is available for a better order
        try:
            graph = nx.DiGraph()
            for task in tasks:
                graph.add_node(task.id, name=task.name)
                for dep_id in task.dependencies:
                    if dep_id in [t.id for t in tasks]: # Ensure dependency exists in the current task list
                        graph.add_edge(dep_id, task.id)
            
            # Attempt topological sort; if cycle, fall back to simple order
            try:
                topo_sorted_ids = list(nx.topological_sort(graph))
                sorted_task_names = [graph.nodes[node_id]['name'] for node_id in topo_sorted_ids]
            except nx.NetworkXUnfeasible:
                sorted_task_names = [task.name for task in tasks] # Fallback
                print("[LLM Sim] Detected cycle in dependencies, falling back to basic order.")

        except Exception as e:
            print(f"[LLM Sim] Error during topological sort: {e}. Falling back to basic order.")
            sorted_task_names = [task.name for task in tasks] # Fallback if networkx fails for some reason

        plan = [f"Step {i+1}: {name}" for i, name in enumerate(sorted_task_names)]
        if constraints:
            plan.append("\n--- Constraints to consider ---")
            for con in constraints:
                plan.append(f"- [{con.type}] {con.description}")
        return plan


# 3. Core Project Assistant Logic

class ProjectAssistant:
    def __init__(self):
        self.llm_simulator = LLMSimulator()
        self.current_project_plan: Optional[ProjectPlan] = None
        self.dependency_graph = nx.DiGraph()

    def _reset_graph(self):
        self.dependency_graph = nx.DiGraph()

    def decompose_and_plan(self, project_goal: str, include_dynamic_constraints: bool = False) -> str:
        self._reset_graph()
        try:
            # Intent Analysis & Task Decomposition
            task_dicts = self.llm_simulator.decompose_project_goal(project_goal)
            tasks = [Task(**td) for td in task_dicts]

            for task in tasks:
                self.dependency_graph.add_node(task.id, name=task.name)
                for dep_id in task.dependencies:
                    if dep_id in [t.id for t in tasks]: # Ensure dependency exists in current task list
                        self.dependency_graph.add_edge(dep_id, task.id)
            
            # Add default constraints
            constraints = [
                Constraint(id="c1", type="HARD", description="Project must be completed within 100 working hours.", details={"max_hours": 100}),
                Constraint(id="c2", type="COMMONSENSE", description="Design tasks must precede development tasks."),
            ]

            if include_dynamic_constraints:
                constraints.append(Constraint(id="c3", type="ENVIRONMENTAL", description="Team member 'Alice' is unavailable next week.", details={"unavailability": {"Alice": "next_week"}}))

            # Planning & Orchestration
            # Check for cycles before planning
            try:
                nx.find_cycle(self.dependency_graph)
                return "Error: Detected a cyclic dependency in tasks. Cannot generate a valid plan. Please refine your project goal or task dependencies."
            except nx.NetworkXNoCycle:
                pass # No cycle, proceed

            plan_steps = self.llm_simulator.generate_plan_steps(tasks, constraints)

            self.current_project_plan = ProjectPlan(
                project_goal=project_goal,
                tasks=tasks,
                constraints=constraints,
                plan_steps=plan_steps,
                status="active"
            )

            validation_issues = self.check_constraints(self.current_project_plan)
            if validation_issues:
                return f"Plan generated with potential issues:\n" + "\n".join(validation_issues) + f"\n\nProposed Plan:\n" + "\n".join(self.current_project_plan.plan_steps)
            else:
                return f"Successfully generated plan for '{project_goal}':\n" + "\n".join(self.current_project_plan.plan_steps)

        except ValidationError as e:
            return f"Data validation error: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    def check_constraints(self, plan: ProjectPlan) -> List[str]:
        issues = []

        # Check total estimated duration against a hard constraint
        total_duration = sum(task.estimated_duration for task in plan.tasks)
        for con in plan.constraints:
            if con.id == "c1" and con.type == "HARD":
                max_hours = con.details.get("max_hours", float('inf'))
                if total_duration > max_hours:
                    issues.append(f"HARD CONSTRAINT VIOLATION: Total estimated duration ({total_duration} hours) exceeds maximum allowed ({max_hours} hours).")
            
            # Basic check for commonsense (design before dev)
            if con.id == "c2" and con.type == "COMMONSENSE":
                design_tasks = [t for t in plan.tasks if "design" in t.name.lower() or "ui/ux" in t.name.lower()]
                dev_tasks = [t for t in plan.tasks if "develop" in t.name.lower() or "frontend" in t.name.lower() or "backend" in t.name.lower()]
                
                for dev_task in dev_tasks:
                    dev_task_deps = set(dev_task.dependencies)
                    # Check if any design task is a direct dependency
                    if not any(design_task.id in dev_task_deps for design_task in design_tasks):
                         # More robust check: use graph to see if there's a path
                        for design_task in design_tasks:
                            if design_task.id in self.dependency_graph and dev_task.id in self.dependency_graph:
                                if not nx.has_path(self.dependency_graph, design_task.id, dev_task.id):
                                    issues.append(f"COMMONSENSE CONSTRAINT VIOLATION: Development task '{dev_task.name}' might not be properly preceded by a design task '{design_task.name}'. Missing dependency or incorrect ordering.")
                                    break # Only report once per dev task

        # Check if all dependencies can be met (simplified)
        for task in plan.tasks:
            for dep_id in task.dependencies:
                if dep_id not in [t.id for t in plan.tasks]:
                    issues.append(f"DEPENDENCY ERROR: Task '{task.name}' depends on unknown task ID '{dep_id}'.")
        
        # Check for environmental constraints (simplified)
        for con in plan.constraints:
            if con.id == "c3" and con.type == "ENVIRONMENTAL":
                unavailability = con.details.get("unavailability", {})
                if "Alice" in unavailability and unavailability["Alice"] == "next_week":
                    # This is a very simplistic check; real implementation would map tasks to resources
                    issues.append("ENVIRONMENTAL CONSTRAINT: 'Alice' is unavailable next week. Plan might need adjustment if she's a critical resource.")

        return issues

    def simulate_execution_and_adapt(self, user_feedback: str) -> str:
        if not self.current_project_plan:
            return "No active plan to simulate or adapt. Please generate a plan first."

        original_plan_steps = "\n".join(self.current_project_plan.plan_steps)
        adaptation_report = []

        adaptation_report.append(f"--- Simulating Execution and Adapting Plan ---")
        adaptation_report.append(f"Original Plan Status: {self.current_project_plan.status}")
        adaptation_report.append(f"User Feedback: {user_feedback}")

        # Simulate a task failing or a new requirement emerging
        if "task failed" in user_feedback.lower() and self.current_project_plan.tasks:
            failed_task_id = self.current_project_plan.tasks[0].id # Simulate first task failing
            for task in self.current_project_plan.tasks:
                if task.id == failed_task_id:
                    task.status = "failed"
                    adaptation_report.append(f"Simulating failure for task: {task.name}")
                    # Basic backtracking: identify dependents and mark them for re-evaluation
                    dependents = [t for t in self.current_project_plan.tasks if failed_task_id in t.dependencies]
                    for dep in dependents:
                        adaptation_report.append(f"Marking dependent task '{dep.name}' for re-evaluation.")
                        # In a real system, this would trigger re-planning for these branches
                    break

        if "new requirement" in user_feedback.lower():
            new_task_id = f"new_t{len(self.current_project_plan.tasks) + 1}"
            new_task = Task(
                id=new_task_id,
                name="Implement new Feature X",
                description="Add a critical new feature as per feedback",
                estimated_duration=15,
                dependencies=[self.current_project_plan.tasks[-1].id if self.current_project_plan.tasks else "g1"]
            )
            self.current_project_plan.tasks.append(new_task)
            self.dependency_graph.add_node(new_task.id, name=new_task.name)
            for dep_id in new_task.dependencies:
                if dep_id in [t.id for t in self.current_project_plan.tasks]:
                    self.dependency_graph.add_edge(dep_id, new_task.id)
            adaptation_report.append(f"Added new task: {new_task.name} based on feedback.")
        
        # Re-plan based on simulated changes (simplified heuristic search)
        # This would call LLM again with updated context in a real app
        adaptation_report.append("Re-generating plan based on adaptations...")
        self.current_project_plan.plan_steps = self.llm_simulator.generate_plan_steps(self.current_project_plan.tasks, self.current_project_plan.constraints)
        self.current_project_plan.status = "active_adapted"

        adaptation_report.append(f"New Plan Status: {self.current_project_plan.status}")
        adaptation_report.append(f"\n--- Adapted Plan ---\n" + "\n".join(self.current_project_plan.plan_steps))
        
        # Check constraints again for the adapted plan
        validation_issues = self.check_constraints(self.current_project_plan)
        if validation_issues:
            adaptation_report.append("\n--- Adapted Plan Issues ---")
            adaptation_report.extend(validation_issues)

        return "\n".join(adaptation_report)


# 4. Gradio Interface

def run_assistant(project_goal: str, include_dynamic_constraints: bool) -> str:
    assistant = ProjectAssistant()
    initial_plan = assistant.decompose_and_plan(project_goal, include_dynamic_constraints)
    
    # Simulate dynamic feedback and adaptation only if an initial plan was successfully created
    if "Error:" not in initial_plan and "Plan generated with potential issues" not in initial_plan:
        # For demonstration, let's hardcode some feedback
        # In a real app, this would come from user interaction or external systems
        feedback = "Simulating that 'Gather Requirements' task failed and there's a new requirement: 'Implement user authentication'."
        adapted_plan_report = assistant.simulate_execution_and_adapt(feedback)
        return initial_plan + "\n\n" + adapted_plan_report
    else:
        return initial_plan


if __name__ == "__main__":
    # Example usage for direct testing (without Gradio UI)
    # assistant = ProjectAssistant()
    # project_goal_example = "Develop an e-commerce website with payment integration."
    # print(assistant.decompose_and_plan(project_goal_example))
    # print("\n" + "-"*50 + "\n")
    # project_goal_example_mobile = "Create a mobile app for fitness tracking."
    # print(assistant.decompose_and_plan(project_goal_example_mobile, include_dynamic_constraints=True))
    # print("\n" + "-"*50 + "\n")
    # if assistant.current_project_plan:
    #     print(assistant.simulate_execution_and_adapt("The 'Market Research' task took longer than expected and we need to add a 'User Onboarding Tutorial' task."))


    print("Starting Gradio UI...")
    iface = gr.Interface(
        fn=run_assistant,
        inputs=[
            gr.Textbox(lines=5, label="Enter Project Goal (e.g., 'Develop an e-commerce website' or 'Create a mobile app for fitness tracking')"),
            gr.Checkbox(label="Include Dynamic/Environmental Constraints (for demo purposes)", value=False)
        ],
        outputs=gr.Textbox(label="Project Plan & Adaptation Report", lines=30),
        title="AI-Powered Project Assistant (Adaptive Planning)",
        description="This assistant helps decompose complex project goals, generate plans, and adapt to changes using simulated LLM and planning logic."
    )
    iface.launch(share=False)
