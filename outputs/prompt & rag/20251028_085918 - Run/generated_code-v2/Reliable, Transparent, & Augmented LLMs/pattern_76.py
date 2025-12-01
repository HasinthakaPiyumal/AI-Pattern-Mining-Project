import os
from datetime import datetime, timedelta
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import BaseModel, Field

# --- 1. Pydantic Models for Tool Parameters ---

class CreateTaskInput(BaseModel):
    """Input schema for creating a new project management task."""
    description: str = Field(description="A detailed description of the task.")
    assignee: str = Field(description="The name of the person to whom the task is assigned.")
    due_date: str = Field(description="The due date of the task in YYYY-MM-DD format.")
    project_tool: str = Field(description="The project management tool to use (e.g., Jira, Trello, Asana).")

class SendReminderInput(BaseModel):
    """Input schema for sending a reminder message to a team via a communication platform."""
    message: str = Field(description="The reminder message to send.")
    channel: str = Field(description="The communication channel or team name (e.g., #general, Engineering Team).")
    platform: str = Field(description="The communication platform to use (e.g., Slack, Microsoft Teams).")
    time: str = Field(description="The specific time for the reminder, e.g., '10 AM every weekday' or 'next Friday at 2 PM'.")

# --- 2. Mock Tool Functions (Simulating external API calls) ---

def _get_today_date_str():
    return datetime.now().strftime("%Y-%m-%d")

def _get_next_friday_date_str():
    today = datetime.now()
    # Calculate days until next Friday (Friday is weekday 4, Mon=0)
    days_until_friday = (4 - today.weekday() + 7) % 7 
    if days_until_friday == 0: # If today is Friday, get next Friday
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday)
    return next_friday.strftime("%Y-%m-%d")

@tool(args_schema=CreateTaskInput)
def create_project_task(description: str, assignee: str, due_date: str, project_tool: str) -> str:
    """
    Creates a new task in a specified project management tool.
    Example usage: create_project_task(description=\'Research LLM frameworks\', assignee=\'John Doe\', due_date=\'2023-12-31\', project_tool=\'Jira\')
    """
    print(f"[MOCK TOOL] Creating task in {project_tool}...")
    # In a real application, this would involve API calls to Jira, Trello, Asana, etc.
    return f"Successfully created task \'{description}\' for {assignee} in {project_tool} with due date {due_date}."

@tool(args_schema=SendReminderInput)
def send_team_reminder(message: str, channel: str, platform: str, time: str) -> str:
    """
    Sends a reminder message to a team via a specified communication platform.
    Example usage: send_team_reminder(message=\'Daily stand-up\', channel=\'#general\', platform=\'Slack\', time=\'10 AM every weekday\')
    """
    print(f"[MOCK TOOL] Sending reminder via {platform}...")
    # In a real application, this would involve API calls to Slack, Microsoft Teams, etc.
    return f"Successfully sent reminder \'{message}\' to {channel} on {platform} for {time}."

# --- 3. LLM and Agent Setup ---

# Set your OpenAI API key as an environment variable (e.g., OPENAI_API_KEY="your_key_here")
# For local development, you might set it directly or use a .env file.
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

# Define the tools available to the agent
tools = [create_project_task, send_team_reminder]

# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI-powered project management assistant. Your goal is to help users automate tasks "
        "by interacting with project management and communication tools. "
        "Accurately extract parameters from user requests and invoke the appropriate tools. "
        "Always infer specific dates like 'next Friday' or 'tomorrow' into YYYY-MM-DD format. "
        f"Today's date is {_get_today_date_str()}. Next Friday is {_get_next_friday_date_str()}. "
        "If a due date is specified like 'next Friday', use that; otherwise, if a general timeframe is given like 'upcoming sprint', default to next Friday's date. "
        "If no due date is provided for a task, explicitly state that you need one."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. User Interaction Loop ---

def main():
    print("\n--- AI Project Management Assistant ---")
    print("Type 'exit' to quit.")
    print("Examples:")
    print("  - Create a new task to research LLM frameworks for the upcoming sprint, due next Friday, assign to John in Jira.")
    print("  - Remind the team about the daily stand-up meeting at 10 AM every weekday in #general Slack channel.")
    print("  - Create a task to review Q3 reports, assign to Sarah in Trello, due tomorrow.")

    chat_history = []

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            break

        try:
            result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            print(f"Agent: {result['output']}")
            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=result['output'])
            ])
        except Exception as e:
            print(f"Agent Error: {e}")
            print("Please try rephrasing your request or check your API key if you encounter repeated errors.")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment variables
    if os.getenv("OPENAI_API_KEY") is None:
        print("WARNING: OPENAI_API_KEY environment variable not set. Please set it to use the OpenAI LLM.")
        print("You can run `export OPENAI_API_KEY=\'your_key_here\'` in your terminal or add to a .env file.")
    main()