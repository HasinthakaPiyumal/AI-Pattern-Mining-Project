import os
from information_collector import InformationCollector
from planner_recommender import PlannerRecommender

class SmartProjectAssistant:
    def __init__(self, project_name):
        self.project_name = project_name
        self.information_collector = InformationCollector()
        self.planner_recommender = PlannerRecommender()

    def assist(self, project_context):
        print(f"\n--- Smart Project Assistant for '{self.project_name}' ---")
        print("Stage 1: Information Collection")
        # Stage 1: Information Collection
        synthesized_info = self.information_collector.gather_and_synthesize(project_context)
        print("Information Collection Complete. Synthesized Data:")
        print(synthesized_info)

        print("\nStage 2: Planning & Recommendation")
        # Stage 2: Planning & Recommendation
        recommendations, communication_draft = self.planner_recommender.analyze_and_recommend(synthesized_info)
        print("Planning & Recommendation Complete.")
        print("Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")
        print("\nCommunication Draft:")
        print(communication_draft)
        return recommendations, communication_draft

if __name__ == "__main__":
    # Example Usage
    project_context = {
        "current_date": "2023-10-27",
        "tasks": [
            {"id": "T1", "name": "Design UI/UX", "status": "Completed", "due_date": "2023-10-20", "assignee": "Alice"},
            {"id": "T2", "name": "Develop Backend API", "status": "In Progress", "due_date": "2023-10-28", "assignee": "Bob"},
            {"id": "T3", "name": "Integrate Payment Gateway", "status": "Pending", "due_date": "2023-11-05", "assignee": "Alice"},
            {"id": "T4", "name": "Prepare Deployment Scripts", "status": "Pending", "due_date": "2023-10-25", "assignee": "Charlie"} # Overdue
        ],
        "team_availability": {
            "Alice": "Full-time",
            "Bob": "Full-time",
            "Charlie": "Part-time (50% capacity)"
        },
        "risks": [
            {"description": "API latency issues", "severity": "Medium", "status": "Open"},
            {"description": "Charlie is overloaded", "severity": "High", "status": "Open"}
        ]
    }

    assistant = SmartProjectAssistant("Website Relaunch Project")
    assistant.assist(project_context)

    print("\n---")
    print("Demonstrating another scenario with more complex data (simulated)")
    project_context_2 = {
        "current_date": "2023-11-10",
        "tasks": [
            {"id": "P1", "name": "Phase 1 Report", "status": "Completed", "due_date": "2023-11-01", "assignee": "Alice"},
            {"id": "P2", "name": "Phase 2 Development", "status": "In Progress", "due_date": "2023-11-15", "assignee": "Bob"},
            {"id": "P3", "name": "Client Review Meeting", "status": "Pending", "due_date": "2023-11-12", "assignee": "Alice"},
            {"id": "P4", "name": "Documentation Update", "status": "Pending", "due_date": "2023-11-05", "assignee": "Charlie"} # Overdue
        ],
        "team_availability": {
            "Alice": "Full-time",
            "Bob": "Full-time",
            "Charlie": "On Leave until 2023-11-13" # Charlie is on leave
        },
        "risks": [
            {"description": "Key dependency delay", "severity": "High", "status": "Open"},
            {"description": "Budget overrun risk", "severity": "Medium", "status": "Monitoring"}
        ]
    }
    assistant2 = SmartProjectAssistant("Product Expansion Initiative")
    assistant2.assist(project_context_2)
