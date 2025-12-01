from dataclasses import dataclass, field
import random

@dataclass
class ProjectData:
    requirements: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    team_availability: dict = field(default_factory=dict)
    historical_data: dict = field(default_factory=dict)
    external_dependencies: list = field(default_factory=list)
    identified_risks: list = field(default_factory=list)
    inconsistencies: list = field(default_factory=list)

@dataclass
class ProjectPlan:
    tasks: list = field(default_factory=list)
    resource_allocation: dict = field(default_factory=dict)
    timeline: dict = field(default_factory=dict)
    risk_mitigation_strategies: list = field(default_factory=list)
    optimized: bool = False

class ResearchAgent:
    def collect_requirements(self, raw_input: str) -> list:
        return [req.strip() for req in raw_input.split(',') if req.strip()]

    def collect_constraints(self, raw_input: str) -> dict:
        constraints = {}
        for item in raw_input.split(','):
            if ':' in item:
                key, value = item.split(':', 1)
                constraints[key.strip()] = value.strip()
        return constraints

    def analyze_team_availability(self, team_data: dict) -> dict:
        return team_data

    def fetch_historical_data(self, project_type: str) -> dict:
        if project_type == "software_development":
            return {"avg_dev_time": "6 months", "common_risks": ["scope creep", "resource contention"]}
        return {}

    def identify_external_dependencies(self, project_scope: str) -> list:
        if "api integration" in project_scope.lower():
            return ["External API availability", "Third-party service reliability"]
        return []

    def detect_inconsistencies(self, project_data: ProjectData) -> list:
        inconsistencies = []
        if "budget" in project_data.constraints and "high-end features" in project_data.requirements:
            inconsistencies.append("Potential budget vs. feature inconsistency.")
        return inconsistencies

    def assess_risks(self, project_data: ProjectData) -> list:
        risks = []
        if "tight deadline" in project_data.constraints.values():
            risks.append("Risk of not meeting deadlines due to aggressive schedule.")
        risks.extend(project_data.historical_data.get("common_risks", []))
        return risks

    def conduct_research(self, raw_project_description: str, team_data: dict, project_type: str) -> ProjectData:
        project_data = ProjectData()
        project_data.requirements = self.collect_requirements(raw_project_description.get("requirements", ""))
        project_data.constraints = self.collect_constraints(raw_project_description.get("constraints", ""))
        project_data.team_availability = self.analyze_team_availability(team_data)
        project_data.historical_data = self.fetch_historical_data(project_type)
        project_data.external_dependencies = self.identify_external_dependencies(raw_project_description.get("scope", ""))
        project_data.inconsistencies = self.detect_inconsistencies(project_data)
        project_data.identified_risks = self.assess_risks(project_data)
        return project_data

class PlanningAgent:
    def create_task_breakdown(self, project_data: ProjectData) -> list:
        tasks = []
        for req in project_data.requirements:
            tasks.append(f"Implement {req.lower().replace(' ', '_')}_feature")
        tasks.append("Project Management Overhead")
        return tasks

    def allocate_resources(self, project_data: ProjectData, tasks: list) -> dict:
        resource_allocation = {}
        available_team = list(project_data.team_availability.keys())
        if not available_team:
            return {}

        for i, task in enumerate(tasks):
            resource_allocation[task] = random.choice(available_team)
        return resource_allocation

    def generate_timeline(self, tasks: list, resources: dict) -> dict:
        timeline = {"start_date": "2023-10-26", "end_date": "2024-04-26"}
        for task in tasks:
            timeline[task] = f"Estimated {random.randint(5, 20)} days"
        return timeline

    def develop_risk_mitigation(self, project_data: ProjectData) -> list:
        mitigation_strategies = []
        for risk in project_data.identified_risks:
            mitigation_strategies.append(f"Plan for '{risk}' by having a contingency.")
        if "scope creep" in project_data.historical_data.get("common_risks", []):
            mitigation_strategies.append("Implement strict change control process for scope creep.")
        return mitigation_strategies

    def optimize_plan(self, plan_draft: ProjectPlan) -> ProjectPlan:
        plan_draft.optimized = True
        plan_draft.timeline["overall_duration_optimized"] = "Slightly reduced"
        return plan_draft

    def generate_plan(self, project_data: ProjectData) -> ProjectPlan:
        project_plan = ProjectPlan()
        project_plan.tasks = self.create_task_breakdown(project_data)
        project_plan.resource_allocation = self.allocate_resources(project_data, project_plan.tasks)
        project_plan.timeline = self.generate_timeline(project_plan.tasks, project_plan.resource_allocation)
        project_plan.risk_mitigation_strategies = self.develop_risk_mitigation(project_data)
        project_plan = self.optimize_plan(project_plan)
        return project_plan

class ProjectManagerAssistant:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.planning_agent = PlanningAgent()

    def assist_project_planning(self, raw_project_input: dict, team_data: dict, project_type: str) -> ProjectPlan:
        print("--- Stage 1: Information Collection and Analysis (Research Mode) ---")
        project_data = self.research_agent.conduct_research(raw_project_input, team_data, project_type)
        print("Research complete. Identified data:")
        print(f"  Requirements: {project_data.requirements}")
        print(f"  Constraints: {project_data.constraints}")
        print(f"  Risks: {project_data.identified_risks}")
        print(f"  Inconsistencies: {project_data.inconsistencies}")
        print("\n--- Stage 2: Planning and Optimization (Planning Mode) ---")
        project_plan = self.planning_agent.generate_plan(project_data)
        print("Planning complete. Generated plan:")
        return project_plan

if __name__ == "__main__":
    pma = ProjectManagerAssistant()

    raw_input_1 = {
        "requirements": "User authentication, Dashboard, Data visualization, Reporting features",
        "constraints": "budget: $50000, deadline: 6 months, technology: Python Flask",
        "scope": "Develop a web application with API integration"
    }
    team_info_1 = {"Alice": "available", "Bob": "available", "Charlie": "busy"}
    project_type_1 = "software_development"

    print("\n### Project 1: Web Application Development ###")
    plan_1 = pma.assist_project_planning(raw_input_1, team_info_1, project_type_1)
    print(f"  Tasks: {plan_1.tasks}")
    print(f"  Resource Allocation: {plan_1.resource_allocation}")
    print(f"  Timeline: {plan_1.timeline}")
    print(f"  Risk Mitigation: {plan_1.risk_mitigation_strategies}")
    print(f"  Optimized: {plan_1.optimized}")

    raw_input_2 = {
        "requirements": "Mobile app with push notifications, Offline mode",
        "constraints": "budget: $30000, deadline: 3 months, high-end features",
        "scope": "Develop a native mobile application"
    }
    team_info_2 = {"David": "available", "Eve": "available"}
    project_type_2 = "mobile_app_development"

    print("\n### Project 2: Mobile App Development ###")
    plan_2 = pma.assist_project_planning(raw_input_2, team_info_2, project_type_2)
    print(f"  Tasks: {plan_2.tasks}")
    print(f"  Resource Allocation: {plan_2.resource_allocation}")
    print(f"  Timeline: {plan_2.timeline}")
    print(f"  Risk Mitigation: {plan_2.risk_mitigation_strategies}")
    print(f"  Optimized: {plan_2.optimized}")