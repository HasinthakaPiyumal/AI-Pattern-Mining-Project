from typing import List, Dict
from project_data_models import Task, Resource, Risk, ProjectData

class PlanningRecommender:
    """Provides planning and recommendations based on gathered project data."""

    def __init__(self):
        pass

    def _identify_critical_tasks(self, tasks: List[Task]) -> List[Task]:
        # A simple critical path identification: tasks with upcoming deadlines and dependencies
        critical_tasks = []
        today = date.today()
        for task in tasks:
            if task.status != "Completed" and task.due_date <= today + timedelta(days=7) and task.dependencies:
                critical_tasks.append(task)
        return critical_tasks

    def _propose_schedule_adjustments(self, tasks: List[Task], critical_tasks: List[Task]) -> List[str]:
        adjustments = []
        for task in critical_tasks:
            adjustments.append(f"Consider prioritizing Task \'{task.name}\' (ID: {task.id}) as its deadline is approaching and it has dependencies.")
            if task.progress_percentage < 50: # Example heuristic
                adjustments.append(f"Investigate reasons for slow progress on Task \'{task.name}\' and allocate more resources if possible.")
        
        # Identify potential bottlenecks (e.g., tasks not started, but due soon)
        for task in tasks:
            if task.status == "Not Started" and task.due_date < date.today() + timedelta(days=5):
                adjustments.append(f"Task \'{task.name}\' (ID: {task.id}) is not started and due soon. Immediate action required.")
        return adjustments

    def _suggest_resource_reallocations(self, tasks: List[Task], resources: List[Resource]) -> List[str]:
        reallocations = []
        resource_load: Dict[str, int] = {res.id: 0 for res in resources}
        for task in tasks:
            if task.assigned_resource_id:
                resource_load[task.assigned_resource_id] += 1 # Simple count of assigned tasks
        
        # Identify potentially over-allocated resources (e.g., assigned more than 2 tasks and low availability)
        for resource in resources:
            if resource_load.get(resource.id, 0) > 2 and resource.availability_percentage < 50:
                reallocations.append(f"Resource \'{resource.name}\' (ID: {resource.id}) appears over-allocated. Consider reassigning some tasks.")
        
        # Identify under-utilized resources (e.g., high availability and no critical tasks)
        for resource in resources:
            if resource.availability_percentage > 80 and resource_load.get(resource.id, 0) == 0: # No tasks assigned for simplicity
                reallocations.append(f"Resource \'{resource.name}\' (ID: {resource.id}) is under-utilized. They could potentially take on more tasks or assist critical paths.")
        return reallocations

    def _generate_risk_mitigation_strategies(self, risks: List[Risk]) -> List[str]:
        strategies = []
        for risk in risks:
            if risk.severity == "High" and not risk.mitigation_plan:
                strategies.append(f"Urgent: No mitigation plan for high-severity risk: \'{risk.description}\' (ID: {risk.id}). Develop one immediately.")
            elif risk.mitigation_plan:
                strategies.append(f"Existing mitigation plan for \'{risk.description}\' (ID: {risk.id}): {risk.mitigation_plan}.")
        return strategies

    def generate_recommendations(self, project_data: ProjectData) -> Dict[str, List[str]]:
        """Generates comprehensive recommendations based on the gathered project data."""
        print("\n[PLANNING AND RECOMMENDATION STAGE] Generating insights and recommendations...")

        critical_tasks = self._identify_critical_tasks(project_data.tasks)
        schedule_adjustments = self._propose_schedule_adjustments(project_data.tasks, critical_tasks)
        resource_reallocations = self._suggest_resource_reallocations(project_data.tasks, project_data.resources)
        risk_strategies = self._generate_risk_mitigation_strategies(project_data.risks)

        recommendations = {
            "schedule_adjustments": schedule_adjustments,
            "resource_reallocations": resource_reallocations,
            "risk_mitigation_strategies": risk_strategies,
            "overall_project_status_summary": [
                f"Total tasks: {project_data.project_status.total_tasks}",
                f"Completed tasks: {project_data.project_status.completed_tasks}",
                f"Overdue tasks: {project_data.project_status.overdue_tasks}",
                f"Overall progress: {project_data.project_status.overall_progress_percentage}%"
            ],
            "external_contextual_notes": [
                f"External market information: {project_data.external_dependencies_info}"
            ]
        }

        print("Recommendations generated.")
        return recommendations
