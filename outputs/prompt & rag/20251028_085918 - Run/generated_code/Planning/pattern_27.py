import os
from typing import List, Dict, Any

import streamlit as st
from pydantic import BaseModel, Field
from loguru import logger

# LangChain/LlamaIndex imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# --- Configuration and Environment Variables ---
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Set your OpenAI API key
logger.add("ai_project_manager.log", rotation="1 week")

# --- Pydantic Models for Structured Output ---
class Task(LCBaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    name: str = Field(..., description="Name of the task")
    description: str = Field(..., description="Detailed description of the task")
    assigned_to: str = Field(..., description="Person or team assigned to the task")
    due_date: str = Field(..., description="Due date for the task in YYYY-MM-DD format")
    status: str = Field("Not Started", description="Current status of the task (e.g., Not Started, In Progress, Completed)")
    dependencies: List[str] = Field([], description="List of task_ids this task depends on")
    estimated_hours: int = Field(..., description="Estimated hours to complete the task")

class ProjectPlan(LCBaseModel):
    project_name: str = Field(..., description="Name of the project")
    overall_goal: str = Field(..., description="High-level goal of the project")
    tasks: List[Task] = Field(..., description="List of detailed tasks for the project")
    timeline_summary: str = Field(..., description="Overall timeline summary and key milestones")
    resource_allocation_summary: str = Field(..., description="Summary of resources allocated")
    key_constraints: List[str] = Field(..., description="List of identified project constraints")

# --- Knowledge Base and RAG Setup ---
class KnowledgeBase:
    def __init__(self):
        self.vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db")
        self.store = InMemoryStore()
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )
        self.seed_knowledge()

    def seed_knowledge(self):
        docs = [
            Document(page_content="Software development projects typically follow phases: Requirements, Design, Implementation, Testing, Deployment."),
            Document(page_content="Common software development roles include: Project Manager, Software Engineer, QA Engineer, DevOps Engineer, UI/UX Designer."),
            Document(page_content="Agile methodology emphasizes iterative development, collaboration, and responding to change."),
            Document(page_content="Waterfall methodology is a linear sequential design approach."),
            Document(page_content="Budget constraints directly impact resource allocation and project scope."),
            Document(page_content="Deadline constraints require efficient task scheduling and prioritization."),
            Document(page_content="Technical debt can significantly slow down future development."),
            Document(page_content="Effective communication is crucial for project success."),
            Document(page_content="Risk management involves identifying, assessing, and mitigating potential project risks."),
        ]
        self.retriever.add_documents(docs)
        logger.info("Knowledge base seeded with initial documents.")

    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        doc = Document(page_content=text, metadata=metadata or {})
        self.retriever.add_documents([doc])
        logger.info(f"Added document to knowledge base: {text[:50]}...")

# --- External Integration Layer (Mocks) ---
class ExternalIntegrations:
    def update_pm_tool(self, project_name: str, task: Task):
        logger.info(f"[MOCK] Updating PM tool for project '{project_name}': Task '{task.name}' set to '{task.status}'.")
        # In a real app, this would call Jira/Asana API
        st.sidebar.info(f"Updated PM Tool: {task.name} ({task.status})")

    def send_communication(self, recipient: str, message: str):
        logger.info(f"[MOCK] Sending communication to '{recipient}': {message}")
        # In a real app, this would call Slack/Email API
        st.sidebar.info(f"Sent notification to {recipient}")

# --- AI Project Manager Core Logic ---
class AIProjectManager:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        self.knowledge_base = KnowledgeBase()
        self.integrations = ExternalIntegrations()

    def _get_rag_chain(self, prompt_template):
        return (
            RunnablePassthrough.assign(context=RunnableLambda(lambda x: self.knowledge_base.retriever.invoke(x["query"]))) |
            prompt_template |
            self.llm
        )

    def generate_initial_plan(self, project_goal: str, constraints: List[str]) -> ProjectPlan:
        logger.info(f"Generating initial plan for goal: {project_goal} with constraints: {constraints}")
        
        # Prompt for initial plan generation with RAG context
        plan_prompt = ChatPromptTemplate.from_messages([
            SystemMessage("You are an expert AI Project Manager. Your task is to create a detailed project plan including task decomposition, estimated effort, and resource allocation based on the project goal and constraints."),
            HumanMessage(
                "Given the project goal: {project_goal}\n"
                "And the following constraints: {constraints}\n"
                "Use the following project management knowledge to help create a comprehensive plan: {context}\n\n"
                "Decompose the project goal into a list of specific, actionable tasks, assigning an estimated effort (in hours) and a due date (YYYY-MM-DD), and a logical resource (e.g., 'Software Engineer', 'QA Engineer', 'DevOps'). "
                "Ensure tasks have clear dependencies. Provide a summary of the overall timeline and resource allocation. "
                "Output the plan strictly as a JSON object adhering to the ProjectPlan pydantic schema. "
                "Ensure task_ids are unique and dependencies refer to existing task_ids."
                "Do NOT include any other text or explanation outside the JSON."
            )
        ])

        plan_chain = self._get_rag_chain(plan_prompt) | JsonOutputParser()

        try:
            raw_output = plan_chain.invoke({"query": f"Project planning for: {project_goal}", "project_goal": project_goal, "constraints": constraints})
            logger.debug(f"Raw LLM output for plan: {raw_output}")
            if isinstance(raw_output, str):
                 parsed_output = ProjectPlan.parse_raw(raw_output)
            else:
                parsed_output = ProjectPlan.parse_obj(raw_output)
            return parsed_output
        except Exception as e:
            logger.error(f"Error parsing initial plan: {e}")
            st.error(f"Failed to generate plan: {e}")
            return None

    def adapt_plan(self, current_plan: ProjectPlan, feedback: str, new_constraints: List[str]) -> ProjectPlan:
        logger.info(f"Adapting plan for project '{current_plan.project_name}' with feedback: {feedback} and new constraints: {new_constraints}")
        
        # Prompt for plan adaptation with RAG context
        adapt_prompt = ChatPromptTemplate.from_messages([
            SystemMessage("You are an expert AI Project Manager. Your task is to adapt an existing project plan based on new feedback and constraints. "
                          "You must update tasks, timelines, and resource allocations as needed, while maintaining overall project coherence. "
                          "Ensure all updates are reflected in the provided JSON schema."),
            HumanMessage(
                "Current Project Plan (in JSON format):\n{current_plan_json}\n\n"
                "New Feedback/Issues: {feedback}\n"
                "Updated/New Constraints: {new_constraints}\n\n"
                "Use the following project management knowledge to help adapt the plan: {context}\n\n"
                "Revise the current plan by modifying existing tasks, adding new ones, or updating their status, due dates, or assignments based on the feedback and new constraints. "
                "Ensure consistency and logical flow. Provide an updated plan strictly as a JSON object adhering to the ProjectPlan pydantic schema. "
                "Do NOT include any other text or explanation outside the JSON."
            )
        ])

        adapt_chain = self._get_rag_chain(adapt_prompt) | JsonOutputParser()
        
        try:
            raw_output = adapt_chain.invoke({
                "query": f"Adapt project plan for {current_plan.project_name} based on feedback: {feedback}",
                "current_plan_json": current_plan.json(), 
                "feedback": feedback,
                "new_constraints": new_constraints
            })
            logger.debug(f"Raw LLM output for adapted plan: {raw_output}")
            if isinstance(raw_output, str):
                 parsed_output = ProjectPlan.parse_raw(raw_output)
            else:
                parsed_output = ProjectPlan.parse_obj(raw_output)
            
            # Simulate updating external PM tool for changed tasks
            for new_task in parsed_output.tasks:
                old_task = next((t for t in current_plan.tasks if t.task_id == new_task.task_id), None)
                if old_task is None or old_task.status != new_task.status or old_task.assigned_to != new_task.assigned_to:
                    self.integrations.update_pm_tool(parsed_output.project_name, new_task)

            return parsed_output
        except Exception as e:
            logger.error(f"Error parsing adapted plan: {e}")
            st.error(f"Failed to adapt plan: {e}")
            return None

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Intelligent AI Project Manager")
st.title("Intelligent AI Project Manager")

if "project_manager" not in st.session_state:
    st.session_state.project_manager = AIProjectManager()
if "current_plan" not in st.session_state:
    st.session_state.current_plan = None

pm_agent = st.session_state.project_manager

with st.sidebar:
    st.header("Project Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY"))
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    else:
        st.warning("Please enter your OpenAI API Key to proceed.")

    st.subheader("Seed Additional Knowledge")
    new_knowledge = st.text_area("Add new project management best practices, company policies, etc.")
    if st.button("Add Knowledge") and new_knowledge:
        pm_agent.knowledge_base.add_document(new_knowledge)
        st.success("Knowledge added!")


st.header("1. Define Project Goal & Constraints")
project_goal = st.text_area("Describe the overall project goal:", 
                            value="Develop a new e-commerce website with user authentication, product catalog, shopping cart, and payment processing.", 
                            height=100)
constraints_input = st.text_area("Enter key constraints (one per line, e.g., 'Budget: $50,000', 'Deadline: 2024-12-31', 'Team Size: 5 developers'):", 
                                 value="Budget: $50,000\nDeadline: 2024-12-31\nTeam Size: 5 developers")
constraints = [c.strip() for c in constraints_input.split('\n') if c.strip()]

if st.button("Generate Initial Plan") and openai_api_key:
    with st.spinner("Generating initial project plan..."):
        st.session_state.current_plan = pm_agent.generate_initial_plan(project_goal, constraints)
    if st.session_state.current_plan:
        st.success("Initial plan generated successfully!")
        logger.info("Initial plan generated.")

st.markdown("---विकास---")
st.header("2. Current Project Plan")

if st.session_state.current_plan:
    plan = st.session_state.current_plan
    st.subheader(f"Project: {plan.project_name}")
    st.write(f"**Goal:** {plan.overall_goal}")
    st.write(f"**Key Constraints:** {', '.join(plan.key_constraints)}")
    st.write(f"**Timeline Summary:** {plan.timeline_summary}")
    st.write(f"**Resource Allocation Summary:** {plan.resource_allocation_summary}")

    st.subheader("Tasks")
    task_data = []
    for task in plan.tasks:
        task_data.append({
            "Task ID": task.task_id,
            "Name": task.name,
            "Description": task.description,
            "Assigned To": task.assigned_to,
            "Due Date": task.due_date,
            "Status": task.status,
            "Dependencies": ", ".join(task.dependencies),
            "Estimated Hours": task.estimated_hours,
        })
    st.dataframe(task_data, use_container_width=True)
else:
    st.info("No project plan generated yet. Define a goal and click 'Generate Initial Plan'.")

st.markdown("---विकास---")
st.header("3. Adapt Plan Based on Feedback/Changes")

if st.session_state.current_plan:
    feedback_input = st.text_area("Enter new feedback, issues, or changes (e.g., 'Payment gateway integration is delayed', 'New feature request: admin dashboard'):", height=100)
    new_constraints_input = st.text_area("Enter updated/new constraints (one per line, e.g., 'New Budget: $60,000', 'New Deadline: 2025-01-15'):", height=50)
    new_constraints = [c.strip() for c in new_constraints_input.split('\n') if c.strip()]

    if st.button("Adapt Plan") and openai_api_key:
        if feedback_input or new_constraints:
            with st.spinner("Adapting project plan..."):
                st.session_state.current_plan = pm_agent.adapt_plan(st.session_state.current_plan, feedback_input, new_constraints)
            if st.session_state.current_plan:
                st.success("Project plan adapted successfully!")
                logger.info("Project plan adapted.")
            else:
                st.error("Failed to adapt plan.")
        else:
            st.warning("Please provide feedback or new constraints to adapt the plan.")
else:
    st.info("Generate an initial plan first before adapting it.")
