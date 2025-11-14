import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel

from data_models import ProjectPlan, Risk, AdaptiveSuggestion, Task

class LLMTools:
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def _get_parser_and_prompt(self, pydantic_object: BaseModel, system_message: str, human_message: str):
        parser = PydanticOutputParser(pydantic_object=pydantic_object)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message + "\n{format_instructions}"),
            ("human", human_message)
        ]).partial(format_instructions=parser.get_format_instructions())
        return parser, prompt

    def decompose_project_goal(self, project_goal: str) -> ProjectPlan:
        system_message = (
            "You are an expert project manager AI. Your task is to take a high-level project goal " 
            "and decompose it into a list of detailed, actionable sub-tasks. " 
            "Each task should have an estimated duration and clearly defined dependencies. "
            "Ensure the tasks collectively achieve the project goal."
        )
        human_message = f"Decompose the following project goal into a structured plan: {project_goal}"
        
        parser, prompt = self._get_parser_and_prompt(ProjectPlan, system_message, human_message)
        chain = prompt | self.llm | parser
        
        try:
            result = chain.invoke({"project_goal": project_goal})
            # Ensure the project_goal in the result matches the input if LLM modifies it
            result.project_goal = project_goal 
            return result
        except Exception as e:
            print(f"Error decomposing project goal: {e}")
            return ProjectPlan(project_goal=project_goal, tasks=[])

    def identify_risks_and_suggest_mitigations(self, project_plan: ProjectPlan) -> List[Risk]:
        tasks_description = "\n".join([
            f"- {task.name} (Duration: {task.estimated_duration_hours}h, Dependencies: {', '.join(task.dependencies) if task.dependencies else 'None'})"
            for task in project_plan.tasks
        ])
        system_message = (
            "You are an AI risk assessment expert. Analyze the provided project plan and identify potential risks. "
            "For each risk, describe it, assign a severity (low, medium, high), and suggest a clear mitigation strategy."
        )
        human_message = f"Identify risks and suggest mitigation strategies for the following project plan with goal '{project_plan.project_goal}':\n{tasks_description}"
        
        parser, prompt = self._get_parser_and_prompt(List[Risk], system_message, human_message)
        chain = prompt | self.llm | parser
        
        try:
            return chain.invoke({"project_plan": project_plan})
        except Exception as e:
            print(f"Error identifying risks: {e}")
            return []

    def suggest_adaptive_strategy(self, project_plan: ProjectPlan, current_status: Dict[str, str], identified_risks: List[Risk], issue_description: str) -> AdaptiveSuggestion:
        tasks_status = "\n".join([f"- {task.name}: {current_status.get(task.name, task.status)}" for task in project_plan.tasks])
        risks_summary = "\n".join([f"- {risk.description} (Severity: {risk.severity}): {risk.mitigation_strategy}" for risk in identified_risks])
        
        system_message = (
            "You are an AI project strategist. Given a project plan, its current status, identified risks, "
            "and a specific issue or change, propose an adaptive strategy. "
            "The strategy should include suggested changes to the plan (e.g., task durations, new tasks, reordering) "
            "to address the issue and keep the project on track or adapt to new circumstances."
        )
        human_message = (
            f"Project Goal: {project_plan.project_goal}\n\n"
            f"Current Plan:\n{', '.join([task.name for task in project_plan.tasks])}\n\n"
            f"Current Task Status:\n{tasks_status}\n\n"
            f"Identified Risks:\n{risks_summary if risks_summary else 'No major risks identified.'}\n\n"
            f"Issue/Change to Address: {issue_description}\n\n"
            f"Suggest an adaptive strategy and proposed changes to the project plan."
        )
        
        parser, prompt = self._get_parser_and_prompt(AdaptiveSuggestion, system_message, human_message)
        chain = prompt | self.llm | parser
        
        try:
            return chain.invoke({"project_plan": project_plan, "current_status": current_status, "identified_risks": identified_risks, "issue_description": issue_description})
        except Exception as e:
            print(f"Error suggesting adaptive strategy: {e}")
            return AdaptiveSuggestion(reason="Failed to generate adaptive strategy.", suggested_changes={})
