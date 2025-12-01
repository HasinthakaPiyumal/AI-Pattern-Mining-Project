import streamlit as st
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
import datetime

# Load environment variables from .env file
load_dotenv()

# --- 1. Tool Integration Layer (Simulated API Connectors) ---

class ProjectManagementToolAPI:
    def create_task(self, task_name: str, assignee: str, due_date: str, project_name: str = "Default Project") -> str:
        # Simulate API call to a project management tool
        print(f"Simulating API call: Creating task '{task_name}' for {assignee} in {project_name} due by {due_date}")
        # In a real scenario, this would involve HTTP requests to Jira, Asana, etc.
        return f"Successfully created task '{task_name}' in {project_name} for {assignee} with due date {due_date}."

    def schedule_meeting(self, title: str, start_time: str, end_time: str, attendees: list, calendar_name: str = "Default Calendar") -> str:
        # Simulate API call to a scheduling tool
        print(f"Simulating API call: Scheduling meeting '{title}' in {calendar_name} from {start_time} to {end_time} with {', '.join(attendees)}")
        # In a real scenario, this would involve Google Calendar or Outlook Calendar API calls.
        return f"Successfully scheduled meeting '{title}' in {calendar_name} from {start_time} to {end_time} with attendees: {', '.join(attendees)}."


pm_api = ProjectManagementToolAPI()

# --- 2. Tool Definitions (LangChain Tools with Pydantic) ---

class CreateTaskInput(BaseModel):
    task_name: str = Field(description="The name or title of the task to be created.")
    assignee: str = Field(description="The person assigned to the task.")
    due_date: str = Field(description="The due date of the task in YYYY-MM-DD format.")
    project_name: str = Field(description="The name of the project where the task will be created. Defaults to 'Default Project'.")

@tool("create_project_task", args_schema=CreateTaskInput)
def create_project_task_tool(
    task_name: str,
    assignee: str,
    due_date: str,
    project_name: str = "Default Project"
) -> str:
    """Creates a new task in a project management system with a given name, assignee, and due date."""
    try:
        # Basic date validation
        datetime.datetime.strptime(due_date, "%Y-%m-%d")
        return pm_api.create_task(task_name, assignee, due_date, project_name)
    except ValueError:
        return "Error: Due date must be in YYYY-MM-DD format."


class ScheduleMeetingInput(BaseModel):
    title: str = Field(description="The title of the meeting.")
    start_time: str = Field(description="The start time of the meeting in YYYY-MM-DD HH:MM format.")
    end_time: str = Field(description="The end time of the meeting in YYYY-MM-DD HH:MM format.")
    attendees: list = Field(description="A list of email addresses or names of attendees.")
    calendar_name: str = Field(description="The name of the calendar to schedule the meeting in. Defaults to 'Default Calendar'.")

@tool("schedule_meeting", args_schema=ScheduleMeetingInput)
def schedule_meeting_tool(
    title: str,
    start_time: str,
    end_time: str,
    attendees: list,
    calendar_name: str = "Default Calendar"
) -> str:
    """Schedules a new meeting in a calendar system with a title, start time, end time, and a list of attendees."""
    try:
        # Basic datetime validation
        datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M")
        return pm_api.schedule_meeting(title, start_time, end_time, attendees, calendar_name)
    except ValueError:
        return "Error: Start and end times must be in YYYY-MM-DD HH:MM format."


# --- 3. LLM Agent Orchestrator (LangChain Agent) ---

# Initialize LLM (Ensure OPENAI_API_KEY is set in your .env file)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

tools = [
    create_project_task_tool,
    schedule_meeting_tool,
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)

# --- 4. User Interface (Streamlit) ---

st.set_page_config(page_title="Smart Project Manager Assistant", layout="centered")
st.title("🧠 Smart Project Manager Assistant")
st.markdown("Hello! I can help you manage your projects and schedule meetings.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})