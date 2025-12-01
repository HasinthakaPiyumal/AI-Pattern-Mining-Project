import json
import time
import random

class Goal:
    def __init__(self, name, description, subgoals=None):
        self.name = name
        self.description = description
        self.subgoals = subgoals if subgoals is not None else []
        self.status = "pending"

    def add_subgoal(self, subgoal):
        self.subgoals.append(subgoal)

    def __repr__(self):
        return f"Goal(Name: {self.name}, Status: {self.status})"

class GoalManagementModule:
    def decompose_goal(self, high_level_goal_name):
        # For demonstration, a predefined decomposition for a common project goal
        if high_level_goal_name == "Develop and launch a new e-commerce platform by Q4":
            goal = Goal(high_level_goal_name, "Overall project to develop and launch e-commerce platform")

            frontend = Goal("Frontend Development", "Develop user interface")
            frontend.add_subgoal(Goal("Design UI/UX Mockups", "Create visual designs"))
            frontend.add_subgoal(Goal("Implement React Components", "Code frontend features"))

            backend = Goal("Backend Development", "Develop server-side logic and APIs")
            backend.add_subgoal(Goal("Design Database Schema", "Define database structure"))
            backend.add_subgoal(Goal("Implement RESTful APIs", "Code backend endpoints"))
            backend.add_subgoal(Goal("Integrate Payment Gateway", "Connect with payment providers"))

            database = Goal("Database Setup", "Set up and configure database")
            database.add_subgoal(Goal("Provision Database Server", "Allocate server resources"))
            database.add_subgoal(Goal("Populate Initial Data", "Add initial product data"))

            testing = Goal("Testing", "Ensure quality and functionality")
            testing.add_subgoal(Goal("Unit Testing", "Test individual components"))
            testing.add_subgoal(Goal("Integration Testing", "Test component interactions"))
            testing.add_subgoal(Goal("User Acceptance Testing (UAT)", "Validate with end-users"))

            deployment = Goal("Deployment", "Release to production environment")
            deployment.add_subgoal(Goal("Configure Production Servers", "Set up live environment"))
            deployment.add_subgoal(Goal("Deploy Application", "Push code to production"))
            deployment.add_subgoal(Goal("Post-Deployment Monitoring", "Monitor for issues"))

            goal.add_subgoal(frontend)
            goal.add_subgoal(backend)
            goal.add_subgoal(database)
            goal.add_subgoal(testing)
            goal.add_subgoal(deployment)
            return goal
        else:
            return Goal(high_level_goal_name, "Generic project goal")

    def get_goal_hierarchy_text(self, goal, indent=0):
        text = "  " * indent + f"- {goal.name} (Status: {goal.status})\n"
        for sub in goal.subgoals:
            text += self.get_goal_hierarchy_text(sub, indent + 1)
        return text

class KnowledgeBase:
    def __init__(self):
        self.best_practices = {"software_development": "Follow Agile Scrum methodologies for iterative development.",
                                 "database_design": "Normalize database schemas to reduce data redundancy.",
                                 "testing": "Prioritize critical path testing and automate regression tests."}
    
    def get_best_practice(self, domain):
        return self.best_practices.get(domain, "No specific best practice found.")

class PlanningEngine:
    def __init__(self, kb):
        self.kb = kb

    def generate_plan(self, subgoal):
        domain_hint = "software_development" # Simple heuristic
        if "database" in subgoal.name.lower():
            domain_hint = "database_design"
        elif "test" in subgoal.name.lower():
            domain_hint = "testing"

        best_practice = self.kb.get_best_practice(domain_hint)
        
        plan = f"Action Plan for '{subgoal.name}':\n"
        plan += f"  Description: {subgoal.description}\n"
        plan += f"  Estimated Duration: {random.randint(1, 10)} days\n"
        plan += f"  Key Steps:\n"
        plan += f"    1. Analyze requirements for {subgoal.name}.\n"
        plan += f"    2. Implement core functionality for {subgoal.name}.\n"
        plan += f"    3. Review and refine {subgoal.name} implementation.\n"
        plan += f"  Best Practice from KB: {best_practice}\n"
        return plan

class ExecutionMonitoringSimulator:
    def simulate_execution(self, plan_steps):
        feedback = []
        for step in plan_steps.split('\n')[:-1]: # Skip the last newline
            if "Key Steps:" in step or "Action Plan for" in step or "Description:" in step or "Estimated Duration:" in step or "Best Practice from KB:" in step:
                continue
            task_status = random.choice(["completed", "delayed", "blocked"])
            if task_status == "completed":
                feedback.append(f"  [SIMULATED] Task '{step.strip()}' completed successfully.")
            elif task_status == "delayed":
                feedback.append(f"  [SIMULATED] Task '{step.strip()}' delayed by {random.randint(1, 3)} days.")
            else:
                feedback.append(f"  [SIMULATED] Task '{step.strip()}' blocked due to dependency issue.")
            time.sleep(0.1) # Simulate some work
        return "\n".join(feedback)

class MemoryLearningModule:
    def __init__(self):
        self.past_projects = []

    def record_project_data(self, project_data):
        self.past_projects.append(project_data)

    def get_past_experiences(self, num_experiences=1):
        return self.past_projects[-num_experiences:]

class AdaptationEngine:
    def adapt_plan(self, current_plan, feedback, past_experiences):
        adaptation_suggestions = ""
        if "delayed" in feedback:
            adaptation_suggestions += "  [ADAPTATION] Consider re-prioritizing dependent tasks or allocating additional resources.\n"
        if "blocked" in feedback:
            adaptation_suggestions += "  [ADAPTATION] Investigate and resolve the dependency issue immediately. Review past project data for similar blocking issues.\n"
        if not adaptation_suggestions:
            adaptation_suggestions = "  [ADAPTATION] No immediate adaptations needed. Continue monitoring.\n"
        return current_plan + "\n" + "--- Adaptation Suggestions ---\n" + adaptation_suggestions


class AutonomousProjectManagerAgent:
    def __init__(self):
        self.gmm = GoalManagementModule()
        self.kb = KnowledgeBase()
        self.pe = PlanningEngine(self.kb)
        self.ems = ExecutionMonitoringSimulator()
        self.mlm = MemoryLearningModule()
        self.ae = AdaptationEngine()
        self.current_project_goal = None

    def start_project(self, high_level_goal_name):
        print(f"\n--- Starting Project: {high_level_goal_name} ---\n")
        self.current_project_goal = self.gmm.decompose_goal(high_level_goal_name)
        print("Initial Goal Hierarchy:\n")
        print(self.gmm.get_goal_hierarchy_text(self.current_project_goal))

        self._process_goal(self.current_project_goal)
        print(f"\n--- Project '{high_level_goal_name}' Finished ---\n")
        print("Final Goal Hierarchy:\n")
        print(self.gmm.get_goal_hierarchy_text(self.current_project_goal))

    def _process_goal(self, goal):
        if not goal.subgoals:
            print(f"\nProcessing leaf goal: {goal.name}")
            plan = self.pe.generate_plan(goal)
            print(f"Generated Plan for '{goal.name}':\n{plan}")

            # Simulate iterative planning and execution
            for i in range(3): # Simulate a few iterations of plan-execute-adapt
                print(f"\n  --- Iteration {i+1} for '{goal.name}' ---")
                feedback = self.ems.simulate_execution(plan)
                print(f"  Execution Feedback:\n{feedback}")

                past_experiences = self.mlm.get_past_experiences()
                adapted_plan = self.ae.adapt_plan(plan, feedback, past_experiences)
                print(f"  Adapted Plan/Suggestions:\n{adapted_plan}")
                
                # For simplicity, we just show the adaptation. In a real system, the plan would be genuinely updated.
                plan = adapted_plan # Update plan for next iteration (conceptual)

                # Record current project state for learning
                self.mlm.record_project_data({"goal": goal.name, "iteration": i+1, "plan": plan, "feedback": feedback, "adapted_plan": adapted_plan})
                
                if "completed successfully" in feedback:
                    goal.status = "completed"
                    print(f"Goal '{goal.name}' marked as {goal.status}.")
                    break # Assume completion after successful feedback
                else:
                    goal.status = "in progress (needs adaptation)"
                    print(f"Goal '{goal.name}' still {goal.status}.")
            
            if goal.status != "completed":
                 goal.status = "partially completed" # Or failed, depending on logic

        else:
            print(f"\nProcessing parent goal: {goal.name}")
            for sub_goal in goal.subgoals:
                self._process_goal(sub_goal)
            
            # After processing all subgoals, check their status to update parent goal
            if all(sg.status == "completed" for sg in goal.subgoals):
                goal.status = "completed"
            elif any(sg.status == "in progress (needs adaptation)" or sg.status == "partially completed" for sg in goal.subgoals):
                goal.status = "in progress"
            else:
                goal.status = "pending"
            print(f"Parent goal '{goal.name}' status updated to: {goal.status}")

if __name__ == "__main__":
    agent = AutonomousProjectManagerAgent()
    agent.start_project("Develop and launch a new e-commerce platform by Q4")