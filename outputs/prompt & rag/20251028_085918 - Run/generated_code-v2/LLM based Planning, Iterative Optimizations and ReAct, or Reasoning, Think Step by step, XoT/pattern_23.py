import streamlit as st
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# --- Pydantic Data Models ---
class Resource(BaseModel):
    name: str = Field(description="Name of the resource (e.g., 'Python developer', 'Jira software')")
    type: str = Field(description="Type of the resource (e.g., 'skill', 'tool', 'personnel')")

class Dependency(BaseModel):
    task_description: str = Field(description="Description of the sub-task this task depends on")

class SubTask(BaseModel):
    description: str = Field(description="Detailed description of the sub-task")
    deadline: str = Field(description="Suggested realistic deadline (e.g., '2 weeks', 'next Friday')")
    dependencies: List[Dependency] = Field(default_factory=list, description="List of sub-tasks this task depends on")
    resources: List[Resource] = Field(default_factory=list, description="List of recommended resources for this sub-task")

class ProjectPlan(BaseModel):
    goal: str = Field(description="The high-level project goal")
    sub_tasks: List[SubTask] = Field(description="A list of decomposed sub-tasks with details")

# --- Langchain Backend --- 
# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Task Decomposition Chain
decomposition_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert project manager. Your task is to break down a high-level project goal into a list of concise, actionable sub-tasks. Provide only the sub-task descriptions, one per line, without any numbering or additional text."),
    ("user", "Project Goal: {project_goal}")
])
decomposition_chain = decomposition_prompt | llm | StrOutputParser()

# Planning & Suggestion Engine Chain
enrichment_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a meticulous project planner. Given a project goal and its decomposed sub-tasks, your job is to enrich each sub-task with a realistic deadline, identify potential dependencies between tasks (referencing other sub-tasks by their description), and recommend necessary resources. The output must strictly follow the provided JSON schema."),
    ("user", "Project Goal: {project_goal}\n\nSub-tasks:\n{sub_tasks_list}")
])
enrichment_chain = enrichment_prompt | llm.with_structured_output(ProjectPlan)

# --- Streamlit Frontend ---
st.set_page_config(page_title="AI Project Management Assistant", layout="wide")
st.title("🧠 AI Project Management Assistant")
st.write("Enter your high-level project goal, and I will help you decompose it into manageable tasks, suggest deadlines, identify dependencies, and recommend resources.")

project_goal_input = st.text_area("Describe your project goal:", height=100)

if st.button("Generate Project Plan"):
    if not project_goal_input:
        st.warning("Please enter a project goal to get started!")
    else:
        with st.spinner("Decomposing task and generating plan..."):
            try:
                # Step 1: Decompose the main task
                raw_sub_tasks_str = decomposition_chain.invoke({"project_goal": project_goal_input})
                sub_tasks_list = [task.strip() for task in raw_sub_tasks_str.split('\n') if task.strip()]

                if not sub_tasks_list:
                    st.error("Could not decompose the project goal into sub-tasks. Please try a different goal.")
                else:
                    # Prepare a readable list of sub-tasks for the enrichment prompt
                    formatted_sub_tasks = "\n".join([f"- {task}" for task in sub_tasks_list])

                    # Step 2: Enrich the sub-tasks with details
                    project_plan = enrichment_chain.invoke({
                        "project_goal": project_goal_input,
                        "sub_tasks_list": formatted_sub_tasks
                    })

                    st.success("Project Plan Generated!")

                    st.subheader(f"Project Goal: {project_plan.goal}")

                    for i, task in enumerate(project_plan.sub_tasks):
                        st.markdown(f"### Task {i+1}: {task.description}")
                        st.markdown(f"**Deadline:** {task.deadline}")
                        
                        if task.dependencies:
                            st.markdown("**Dependencies:**")
                            for dep in task.dependencies:
                                st.markdown(f"- Depends on: {dep.task_description}")
                        else:
                            st.markdown("**Dependencies:** None")

                        if task.resources:
                            st.markdown("**Recommended Resources:**")
                            for res in task.resources:
                                st.markdown(f"- {res.name} ({res.type})")
                        else:
                            st.markdown("**Recommended Resources:** None")
                        st.markdown("--- athletic")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please ensure your OpenAI API key is correctly set in a .env file.")

